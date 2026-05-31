"""
AST-based SQL structural comparison using sqlglot.

Parses SQL queries into ASTs, extracts structural components (SELECT columns,
FROM tables, WHERE conditions, GROUP BY, aggregations), and computes weighted
Jaccard similarity for clustering in semantic entropy analysis.

Supports two modes:
  - Plain AST: parse → extract components → weighted Jaccard (default)
  - Canonicalized AST: parse → sqlglot.optimizer.optimize() → extract components
    Normalizes: predicate reordering, BETWEEN→>=AND<=, CTE inlining, expression
    simplification, double negation elimination. Instant, no DB needed.

Fallback: normalized string comparison when sqlglot parsing fails (common with
high-temperature LLM samples that produce malformed SQL).
"""

import re
import logging
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.optimizer import optimize

logger = logging.getLogger(__name__)

# Original SELECT-oriented component weights (back-compat name).
COMPONENT_WEIGHTS = {
    "select_columns": 0.30,
    "from_tables": 0.25,
    "where_conditions": 0.25,
    "group_by": 0.10,
    "aggregations": 0.10,
}

# Statement-type-aware weights. structural_similarity() picks the row matching
# the AST root type. Unlisted types fall back to "Select" weights.
#
# Each row sums to 1.0 so similarity stays in [0, 1] for that type. SELECT
# rows are unchanged from the original weights — pure SELECT clustering
# behavior is preserved bit-for-bit.
COMPONENT_WEIGHTS_BY_TYPE = {
    "Select": COMPONENT_WEIGHTS,
    "Update": {
        "from_tables": 0.25,
        "set_assignments": 0.45,
        "where_conditions": 0.30,
    },
    "Delete": {
        "from_tables": 0.40,
        "where_conditions": 0.60,
    },
    "Alter": {
        "from_tables": 0.20,
        "alter_actions": 0.80,
    },
    "AlterTable": {
        "from_tables": 0.20,
        "alter_actions": 0.80,
    },
    "Insert": {
        "from_tables": 0.30,
        "insert_targets": 0.30,
        "insert_values": 0.40,
    },
    # CREATE — kind disambiguated inside structural_similarity() by inspecting
    # which of (create_columns, function_body) is populated.
    "Create_TABLE": {
        "from_tables": 0.20,
        "create_columns": 0.80,
    },
    # CREATE FUNCTION: only the signature (name + parameter names/types) is
    # used. Body text (Heredoc) varies wildly in surface form across LLM
    # samples even when the semantics are identical (variable renames,
    # formatting), so including it floods clear samples with noise. Signature
    # captures the cases where the LLM disagrees about parameters — which is
    # what happens when the formula is masked. (Also matches the original
    # behavior for same-signature pairs: they merge.)
    "Create_FUNCTION": {
        "create_columns": 1.00,   # signature only
    },
    # Command (sqlglot fallback for unsupported syntax — DO blocks, CREATE
    # TYPE ENUM, ALTER ... USING <expr>). The token bag captures coarse
    # difference between *kinds* of commands but doesn't try to compare the
    # internals — two structurally similar DO blocks shouldn't split just
    # because variable names differ.
    "Command": {
        "command_tokens": 1.00,
    },
}


def parse_and_normalize(sql: str, dialect: str = "postgres") -> Optional[exp.Expression]:
    """Parse SQL string to AST. Returns None on failure."""
    if not sql or not sql.strip():
        return None
    try:
        # Strip trailing semicolons and whitespace
        sql = sql.strip().rstrip(";").strip()
        ast = sqlglot.parse_one(sql, dialect=dialect, error_level=sqlglot.ErrorLevel.WARN)
        return ast
    except BaseException as e:
        # BaseException catches Rust panics (pyo3_runtime.PanicException)
        # which are not subclasses of Exception
        logger.debug(f"sqlglot parse failed: {e}")
        return None


def extract_components(ast: exp.Expression) -> dict:
    """
    Extract structural signature components from a parsed SQL AST.

    Returns dict with sets/frozensets of normalized component strings:
      SELECT-side (populated for SELECT/CTE):
        - select_columns: normalized column/expression strings
        - from_tables: table names (including JOINs)
        - where_conditions: individual predicate strings
        - group_by: GROUP BY expression strings
        - aggregations: set of aggregate function names (COUNT, AVG, etc.)
        - has_limit: bool
        - has_order_by: bool
      Non-SELECT components (populated for UPDATE/ALTER/INSERT/CREATE/Command):
        - set_assignments: UPDATE SET col = expr
        - alter_actions: ALTER TABLE actions (ADD/DROP/ALTER COLUMN ...)
        - create_columns: CREATE TABLE column defs / CREATE FUNCTION signature
        - function_body: CREATE FUNCTION body chunks (Heredoc text n-grams)
        - command_tokens: sqlglot Command fallback tokens (for unsupported syntax)
        - insert_targets: INSERT target column names
        - insert_values: INSERT VALUES tuple expressions
        - stmt_type: AST root class name (used by structural_similarity for type-mismatch penalty)
    """
    components = {
        "stmt_type": type(ast).__name__,
        "select_columns": set(),
        "from_tables": set(),
        "where_conditions": set(),
        "group_by": set(),
        "aggregations": set(),
        "has_limit": False,
        "has_order_by": False,
        "set_assignments": set(),
        "alter_actions": set(),
        "create_columns": set(),
        "function_body": set(),
        "command_tokens": set(),
        "insert_targets": set(),
        "insert_values": set(),
    }

    # SELECT columns
    for select in ast.find_all(exp.Select):
        for expr in select.expressions:
            # Normalize: strip aliases, lowercase, remove table prefixes
            if isinstance(expr, exp.Alias):
                col_sql = expr.this.sql(dialect="postgres").lower()
            else:
                col_sql = expr.sql(dialect="postgres").lower()
            # Strip table alias prefixes (o.col -> col, "a"."col" -> col)
            col_sql = re.sub(r'"[^"]+"\."([^"]+)"', r'\1', col_sql)
            col_sql = re.sub(r"\b\w+\.(\w+)", r"\1", col_sql)
            components["select_columns"].add(col_sql)

    # FROM tables (including JOINs)
    for table in ast.find_all(exp.Table):
        name = table.name.lower() if table.name else ""
        if name:
            components["from_tables"].add(name)

    # WHERE conditions — split into individual predicates
    for where in ast.find_all(exp.Where):
        _extract_predicates(where.this, components["where_conditions"])

    # GROUP BY
    for group in ast.find_all(exp.Group):
        for expr in group.expressions:
            gb = expr.sql(dialect="postgres").lower()
            gb = re.sub(r'"[^"]+"\."([^"]+)"', r'\1', gb)
            gb = re.sub(r"\b\w+\.(\w+)", r"\1", gb)
            components["group_by"].add(gb)

    # Aggregations (COUNT, AVG, SUM, MIN, MAX, etc.)
    agg_types = (exp.Count, exp.Avg, exp.Sum, exp.Min, exp.Max, exp.StddevPop,
                 exp.StddevSamp, exp.Variance, exp.VariancePop)
    for node in ast.walk():
        if isinstance(node, agg_types):
            components["aggregations"].add(type(node).__name__.upper())
        # Also catch generic function calls that are aggregates
        if isinstance(node, exp.Anonymous):
            fname = node.name.upper() if hasattr(node, "name") else ""
            if fname in ("PERCENTILE_CONT", "PERCENTILE_DISC", "STDDEV",
                         "VARIANCE", "MEDIAN", "STRING_AGG", "ARRAY_AGG"):
                components["aggregations"].add(fname)

    # LIMIT / ORDER BY
    components["has_limit"] = ast.find(exp.Limit) is not None
    components["has_order_by"] = ast.find(exp.Order) is not None

    # UPDATE SET clauses (each is exp.EQ with column on left, expression on right)
    if isinstance(ast, exp.Update):
        for set_expr in ast.expressions or []:
            try:
                s = set_expr.sql(dialect="postgres").lower()
                s = _normalize_predicate(s)
                components["set_assignments"].add(s)
            except Exception:
                pass

    # ALTER actions: walk children, skip the target Table node
    if isinstance(ast, getattr(exp, "Alter", exp.Expression)):
        if type(ast).__name__ in ("Alter", "AlterTable"):
            target_table = ast.this if hasattr(ast, "this") else None
            for child in ast.iter_expressions() if hasattr(ast, "iter_expressions") else []:
                if child is target_table:
                    continue
                if isinstance(child, exp.Table):
                    continue
                try:
                    a = child.sql(dialect="postgres").lower()
                    a = re.sub(r'"[^"]+"\."([^"]+)"', r'\1', a)
                    components["alter_actions"].add(a)
                except Exception:
                    pass

    # CREATE TABLE columns / CREATE FUNCTION signature + body
    if isinstance(ast, exp.Create):
        kind = (ast.args.get("kind") or "").upper() if ast.args else ""
        if kind == "TABLE" and ast.this is not None:
            schema = ast.this
            for col_def in getattr(schema, "expressions", []) or []:
                try:
                    cd = col_def.sql(dialect="postgres").lower()
                    cd = re.sub(r'"([^"]+)"', r'\1', cd)
                    components["create_columns"].add(cd)
                except Exception:
                    pass
        elif kind == "FUNCTION":
            # Function signature (name + params)
            if ast.this is not None:
                try:
                    sig = ast.this.sql(dialect="postgres").lower()
                    components["create_columns"].add(sig)
                except Exception:
                    pass
            # Function body — Heredoc node sits at ast.args['expression']
            body_node = ast.args.get("expression") if ast.args else None
            if body_node is not None and type(body_node).__name__ == "Heredoc":
                try:
                    body = str(body_node.this) if body_node.this is not None else ""
                    body_norm = re.sub(r"\s+", " ", body.lower()).strip()
                    # n-gram chunks (size 80, stride 40) so partial overlap still
                    # produces matching elements in the Jaccard.
                    chunk_size, stride = 80, 40
                    for i in range(0, max(1, len(body_norm)), stride):
                        chunk = body_norm[i:i + chunk_size]
                        if len(chunk) >= 20:
                            components["function_body"].add(chunk)
                except Exception:
                    pass

    # INSERT target columns + VALUES tuples
    if isinstance(ast, exp.Insert) and ast.this is not None:
        target = ast.this
        # Schema node holds the (col1, col2, ...) part
        if hasattr(target, "expressions") and target.expressions:
            for col in target.expressions:
                try:
                    components["insert_targets"].add(
                        col.name.lower() if hasattr(col, "name") and col.name
                        else col.sql(dialect="postgres").lower()
                    )
                except Exception:
                    pass
        for values in ast.find_all(exp.Values):
            for tup in values.expressions or []:
                try:
                    components["insert_values"].add(tup.sql(dialect="postgres").lower())
                except Exception:
                    pass

    # Command fallback: sqlglot couldn't fully parse (e.g., DO $$..$$,
    # CREATE TYPE ENUM, ALTER ... USING <expr>). Take only the leading
    # keywords (first N non-quoted, non-numeric tokens) so we distinguish
    # DO / ALTER / CREATE-TYPE coarsely, but don't split same-kind commands
    # on internal body variation.
    if type(ast).__name__ == "Command":
        try:
            text = ast.sql(dialect="postgres").lower()
            text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)  # drop heredoc bodies
            text = re.sub(r"\s+", " ", text)
            tokens = re.findall(r"[a-z_][a-z0-9_]*", text)
            for tok in tokens[:6]:
                if len(tok) >= 3:
                    components["command_tokens"].add(tok)
        except Exception:
            pass

    return components


def _extract_predicates(node: exp.Expression, predicates: set):
    """Recursively extract individual predicates from a WHERE clause."""
    if node is None:
        return
    if isinstance(node, exp.And):
        _extract_predicates(node.left, predicates)
        _extract_predicates(node.right, predicates)
    elif isinstance(node, exp.Or):
        # Treat OR as a single compound predicate
        try:
            predicates.add(_normalize_predicate(node.sql(dialect="postgres").lower()))
        except Exception:
            pass
    else:
        try:
            predicates.add(_normalize_predicate(node.sql(dialect="postgres").lower()))
        except Exception:
            pass


def _normalize_predicate(pred: str) -> str:
    """Normalize a predicate string by stripping table alias prefixes.

    Converts 'o.lunarstage' and 'obs.lunarstage' both to just 'lunarstage'
    so alias differences don't affect comparison.

    Also handles quoted identifiers from sqlglot optimizer:
    '"a"."lunarstage"' → 'lunarstage'
    """
    # Strip quoted alias.column patterns first: "table"."column" → column
    pred = re.sub(r'"[^"]+"\."([^"]+)"', r'\1', pred)
    # Strip unquoted alias.column patterns: word.word → column
    pred = re.sub(r"\b\w+\.(\w+)", r"\1", pred)
    return pred


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets. Returns 1.0 if both empty."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _resolve_weights(comp: dict) -> dict:
    """Pick the weight row matching this AST's statement type.

    For exp.Create the kind (TABLE / FUNCTION / ...) further disambiguates,
    inferred from which slot is populated.
    """
    t = comp.get("stmt_type", "Select")
    if t == "Create":
        if comp.get("function_body"):
            return COMPONENT_WEIGHTS_BY_TYPE["Create_FUNCTION"]
        return COMPONENT_WEIGHTS_BY_TYPE.get("Create_TABLE", COMPONENT_WEIGHTS)
    return COMPONENT_WEIGHTS_BY_TYPE.get(t, COMPONENT_WEIGHTS)


def _component_similarity(comp1: dict, comp2: dict) -> float:
    """Weighted Jaccard over the component bags relevant to this statement type.

    Empty-vs-empty pairs return Jaccard 1.0 (preserves SELECT-side behavior),
    so weights from the type-specific row sum to 1.0 and no renormalization is
    required.
    """
    weights = _resolve_weights(comp1)
    score = 0.0
    for key, weight in weights.items():
        s1 = comp1.get(key)
        s2 = comp2.get(key)
        if isinstance(s1, set) and isinstance(s2, set):
            score += weight * _jaccard(s1, s2)
        elif isinstance(s1, bool) and isinstance(s2, bool):
            score += weight * (1.0 if s1 == s2 else 0.0)
    return score


def structural_similarity(sql1: str, sql2: str) -> float:
    """
    Compute weighted structural similarity between two SQL queries.

    Returns float in [0.0, 1.0]. Uses AST comparison when both parse
    successfully; falls back to normalized string comparison otherwise.

    Two SQLs with different statement-type roots (e.g., SELECT vs UPDATE)
    fall through to string similarity — they're different operations, not
    structurally comparable component-wise.
    """
    ast1 = parse_and_normalize(sql1)
    ast2 = parse_and_normalize(sql2)

    if ast1 is None or ast2 is None:
        return _string_similarity(sql1, sql2)

    comp1 = extract_components(ast1)
    comp2 = extract_components(ast2)

    # Different statement types — not structurally comparable
    if comp1.get("stmt_type") != comp2.get("stmt_type"):
        return _string_similarity(sql1, sql2)

    return _component_similarity(comp1, comp2)


def _normalize_sql_string(sql: str) -> str:
    """Normalize SQL string for fallback comparison."""
    s = sql.lower().strip().rstrip(";").strip()
    s = re.sub(r"\s+", " ", s)  # Collapse whitespace
    s = re.sub(r"--.*$", "", s, flags=re.MULTILINE)  # Strip line comments
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)  # Strip block comments
    s = re.sub(r'\s*as\s+"[^"]*"', "", s)  # Strip aliases
    s = re.sub(r"\s*as\s+\w+", "", s)  # Strip simple aliases
    return s.strip()


def _string_similarity(sql1: str, sql2: str) -> float:
    """Normalized string similarity as fallback when AST parsing fails."""
    s1 = _normalize_sql_string(sql1)
    s2 = _normalize_sql_string(sql2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    # Token-level Jaccard
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    return _jaccard(tokens1, tokens2)


def are_structurally_equivalent(sql1: str, sql2: str, threshold: float = 0.85) -> bool:
    """Binary structural equivalence check."""
    return structural_similarity(sql1, sql2) >= threshold


# ---------------------------------------------------------------------------
# Canonicalized AST comparison
# ---------------------------------------------------------------------------

def canonicalize(sql: str, dialect: str = "postgres") -> Optional[exp.Expression]:
    """
    Parse and canonicalize SQL via sqlglot optimizer.

    Normalizations applied:
      - Predicate reordering (A AND B → canonical order)
      - BETWEEN → >= AND <= normalization
      - CTE inlining (WITH ... AS → direct subquery)
      - Expression simplification (1+1 → 2)
      - Double negation elimination (NOT NOT x → x)
      - Redundant subquery removal

    Returns optimized AST, or None on failure.
    Falls back to plain parse if optimizer fails (e.g. on complex/unusual SQL).
    """
    ast = parse_and_normalize(sql, dialect=dialect)
    if ast is None:
        return None
    try:
        return optimize(ast, dialect=dialect)
    except Exception as e:
        logger.debug(f"sqlglot optimize failed, using plain AST: {e}")
        return ast


def canonical_similarity(sql1: str, sql2: str) -> float:
    """
    Compute weighted structural similarity using canonicalized ASTs.

    Same algorithm as structural_similarity() but runs sqlglot.optimizer.optimize()
    before extracting components. This merges queries that differ only in:
      - Predicate ordering
      - BETWEEN vs >= AND <=
      - CTE vs inline subquery
      - Redundant expressions
    """
    ast1 = canonicalize(sql1)
    ast2 = canonicalize(sql2)

    if ast1 is None or ast2 is None:
        return _string_similarity(sql1, sql2)

    comp1 = extract_components(ast1)
    comp2 = extract_components(ast2)

    if comp1.get("stmt_type") != comp2.get("stmt_type"):
        return _string_similarity(sql1, sql2)

    score = _component_similarity(comp1, comp2)
    if score != score:
        return _string_similarity(sql1, sql2)
    return score


def are_canonically_equivalent(sql1: str, sql2: str, threshold: float = 0.85) -> bool:
    """Binary canonical equivalence check."""
    return canonical_similarity(sql1, sql2) >= threshold
