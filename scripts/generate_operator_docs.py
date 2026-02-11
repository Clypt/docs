#!/usr/bin/env python3
"""Auto-generate operator reference MDX pages from ClyptQ source code.

Traverses the ClyptQ operator source tree, extracts class metadata via AST
parsing (class name, docstring, role, __init__ signature, ephemeral flag),
and generates Mintlify-compatible .mdx files organized by category.

Usage:
    python docs/scripts/generate_operator_docs.py

    # Preview without writing files
    python docs/scripts/generate_operator_docs.py --dry-run

    # Generate only a specific category
    python docs/scripts/generate_operator_docs.py --category indicators
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_DIR = SCRIPT_DIR.parent  # docs/
PROJECT_ROOT = DOCS_DIR.parent  # clypt/

# ClyptQ source locations
CLYPTQ_BASE = PROJECT_ROOT / "clyptq-advanced" / "clyptq" / "clyptq"
TRADING_OPS = CLYPTQ_BASE / "apps" / "trading" / "operators"
CORE_OPS = CLYPTQ_BASE / "operator"
SYSTEM_OPS = CLYPTQ_BASE / "system"

# Output directory
OUTPUT_DIR = DOCS_DIR / "operators"

# Partials directory (optional hand-written intros to merge)
PARTIALS_DIR = DOCS_DIR / "partials"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParamInfo:
    name: str
    type_hint: str = "Any"
    default: Optional[str] = None
    description: str = ""


@dataclass
class OperatorInfo:
    class_name: str
    module_path: str  # relative to CLYPTQ_BASE
    docstring: str = ""
    role: str = ""
    ephemeral: bool = False
    params: list[ParamInfo] = field(default_factory=list)
    inputs_desc: str = ""  # extracted from docstring Args section
    usage_example: str = ""  # extracted from docstring Usage section
    category: str = ""  # doc category (e.g., "indicators/momentum")
    subcategory: str = ""  # for grouping within a page
    compute_source: str = ""  # full compute() method source code


# ---------------------------------------------------------------------------
# Category mapping: source path pattern -> doc output
# ---------------------------------------------------------------------------

CATEGORY_MAP = {
    # Trading operators
    "indicator": {
        "doc_dir": "indicators",
        "subcategories": {
            "sma": "moving-averages", "ema": "moving-averages", "dema": "moving-averages",
            "tema": "moving-averages", "wma": "moving-averages", "trima": "moving-averages",
            "t3": "moving-averages", "kama": "moving-averages", "mama": "moving-averages",
            "ma": "moving-averages",
            "rsi": "momentum", "macd": "momentum", "mom": "momentum", "roc": "momentum",
            "cmo": "momentum", "apo": "momentum", "ppo": "momentum", "stoch": "momentum",
            "stochrsi": "momentum", "willr": "momentum", "ultosc": "momentum",
            "bop": "momentum", "trix": "momentum",
            "adx": "trend", "di": "trend", "aroon": "trend", "sar": "trend",
            "ichimoku": "trend",
            "atr": "volatility", "bbands": "volatility", "stddev": "volatility",
            "volume": "volume",
            "stats": "statistics", "minmax": "statistics", "price": "statistics",
            "hilbert": "statistics",
            "patterns": "patterns",
        },
    },
    "signal/alpha": {
        "doc_dir": "signals",
        "subcategories": {
            "momentum": "alphas", "mean_reversion": "alphas", "volatility": "alphas",
            "volume": "alphas", "liquidity": "alphas", "quality": "alphas",
            "size": "alphas", "value": "alphas",
        },
    },
    "signal/alpha/alpha_101": {
        "doc_dir": "signals",
        "subcategories": {"_default": "alpha-101"},
    },
    "signal/factor": {
        "doc_dir": "signals",
        "subcategories": {"_default": "factors"},
    },
    "transform": {
        "doc_dir": "transforms",
        "subcategories": {
            "scalers": "scalers",
            "neutralizers": "neutralizers",
            "optimizers": "optimizers",
            "position": "position",
        },
    },
    "universe/filter": {
        "doc_dir": "universe",
        "subcategories": {"_default": "filters"},
    },
    "universe/score": {
        "doc_dir": "universe",
        "subcategories": {"_default": "scores"},
    },
    "metrics/rolling": {
        "doc_dir": "",
        "subcategories": {"_default": "metrics"},
    },
    "metrics/accum": {
        "doc_dir": "",
        "subcategories": {"_default": "metrics"},
    },
    "order": {
        "doc_dir": "",
        "subcategories": {"_default": "order"},
    },
    "semantic": {
        "doc_dir": "",
        "subcategories": {"_default": "semantic"},
    },
}


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------

def extract_operators_from_file(filepath: Path) -> list[OperatorInfo]:
    """Parse a Python file and extract operator class info via AST."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  WARNING: Could not parse {filepath}: {e}")
        return []

    # First pass: collect roles from all classes (including private base classes)
    class_roles: dict[str, str] = {}
    class_params: dict[str, list[ParamInfo]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            role = _extract_role(node)
            if role != "UNKNOWN":
                class_roles[node.name] = role
            params = _extract_init_params(node)
            if params:
                class_params[node.name] = params

    operators: list[OperatorInfo] = []
    module_path = str(filepath.relative_to(CLYPTQ_BASE))

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check if it inherits from BaseOperator or has a role attribute
        is_operator = _is_operator_class(node)
        if not is_operator:
            continue

        op = OperatorInfo(
            class_name=node.name,
            module_path=module_path,
        )

        # Extract docstring
        op.docstring = ast.get_docstring(node) or ""

        # Extract role from class body, falling back to parent class role
        op.role = _extract_role(node)
        if op.role == "UNKNOWN":
            for base in node.bases:
                base_name = base.id if isinstance(base, ast.Name) else (
                    base.attr if isinstance(base, ast.Attribute) else ""
                )
                if base_name in class_roles:
                    op.role = class_roles[base_name]
                    break

        # Extract ephemeral flag
        op.ephemeral = _extract_ephemeral(node, source)

        # Extract __init__ parameters (fall back to parent if none)
        op.params = _extract_init_params(node)
        if not op.params:
            for base in node.bases:
                base_name = base.id if isinstance(base, ast.Name) else (
                    base.attr if isinstance(base, ast.Attribute) else ""
                )
                if base_name in class_params:
                    op.params = class_params[base_name]
                    break

        # Parse docstring sections
        _parse_docstring_sections(op)

        # Extract compute() source code
        op.compute_source = _extract_compute_source(node, source)
        if not op.compute_source:
            op.compute_source = _extract_compute_source_from_parents(
                node, tree, source
            )

        operators.append(op)

    return operators


def _is_operator_class(node: ast.ClassDef) -> bool:
    """Check if a class definition is an operator.

    Matches classes that:
    - Inherit from BaseOperator or SemanticOperator
    - Inherit from a private base class (e.g., _BaseCandlePattern) that
      is itself an operator — detected by having a compute() method
    - Have a ``role`` class variable
    - Have a ``compute`` method and are NOT private (no leading ``_``)
    """
    # Skip private classes (internal base classes like _BaseCandlePattern)
    if node.name.startswith("_"):
        return False

    # Check base classes
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in ("BaseOperator", "SemanticOperator"):
            return True
        if isinstance(base, ast.Attribute) and base.attr in ("BaseOperator", "SemanticOperator"):
            return True

    # Check for role class variable
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "role":
                    return True

    # Duck-typing: has compute() method → treat as operator
    has_compute = any(
        isinstance(item, ast.FunctionDef) and item.name == "compute"
        for item in node.body
    )
    # Also check if parent class (in the same file) has compute
    if not has_compute:
        for base in node.bases:
            base_name = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else "")
            if base_name.startswith("_"):
                # Private base in same file — likely an operator hierarchy
                has_compute = True
                break

    return has_compute


def _extract_role(node: ast.ClassDef) -> str:
    """Extract the OperatorRole from a class definition."""
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "role":
                    val = item.value
                    if isinstance(val, ast.Attribute):
                        return val.attr  # e.g., "INDICATOR"
                    if isinstance(val, ast.Constant):
                        return str(val.value)
    return "UNKNOWN"


def _extract_ephemeral(node: ast.ClassDef, source: str) -> bool:
    """Check if the operator is ephemeral."""
    # Check for ephemeral=True in super().__init__() call
    for item in ast.walk(node):
        if isinstance(item, ast.Call):
            for kw in getattr(item, "keywords", []):
                if kw.arg == "ephemeral":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        return True

    # Check if class inherits from SemanticOperator (always ephemeral)
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "SemanticOperator":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "SemanticOperator":
            return True

    return False


def _extract_init_params(node: ast.ClassDef) -> list[ParamInfo]:
    """Extract __init__ parameters from a class definition."""
    params: list[ParamInfo] = []

    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            args = item.args
            defaults = args.defaults
            n_args = len(args.args)
            n_defaults = len(defaults)

            for i, arg in enumerate(args.args):
                if arg.arg == "self":
                    continue

                param = ParamInfo(name=arg.arg)

                # Type hint
                if arg.annotation:
                    param.type_hint = _annotation_to_str(arg.annotation)

                # Default value
                default_idx = i - (n_args - n_defaults)
                if default_idx >= 0:
                    param.default = _node_to_str(defaults[default_idx])

                params.append(param)

            # Handle **kwargs and *args
            if args.kwonlyargs:
                kw_defaults = args.kw_defaults
                for j, kwarg in enumerate(args.kwonlyargs):
                    param = ParamInfo(name=kwarg.arg)
                    if kwarg.annotation:
                        param.type_hint = _annotation_to_str(kwarg.annotation)
                    if j < len(kw_defaults) and kw_defaults[j] is not None:
                        param.default = _node_to_str(kw_defaults[j])
                    params.append(param)

            break

    return params


def _extract_compute_source(node: ast.ClassDef, source: str) -> str:
    """Extract the full compute() method source code from a class definition.

    Uses AST line numbers (end_lineno available in Python 3.8+) to slice
    the raw source text.  Falls back to searching parent private base
    classes defined in the same file when the class itself doesn't define
    compute().
    """
    source_lines = source.splitlines()

    # Direct compute() on this class
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "compute":
            start = item.lineno - 1  # 0-indexed
            end = getattr(item, "end_lineno", None)
            if end is None:
                # Fallback: read until next def/class or end of class
                end = start + 1
                indent = len(source_lines[start]) - len(source_lines[start].lstrip())
                for idx in range(start + 1, min(len(source_lines), node.end_lineno)):
                    line = source_lines[idx]
                    if line.strip() == "":
                        continue
                    line_indent = len(line) - len(line.lstrip())
                    if line_indent <= indent and line.strip():
                        break
                    end = idx + 1
            raw = source_lines[start:end]
            return textwrap.dedent("\n".join(raw)).strip()

    return ""


def _extract_compute_source_from_parents(
    node: ast.ClassDef, tree: ast.Module, source: str
) -> str:
    """If compute() is inherited from a private base class in the same file,
    extract it from there."""
    for base in node.bases:
        base_name = base.id if isinstance(base, ast.Name) else (
            base.attr if isinstance(base, ast.Attribute) else ""
        )
        if not base_name:
            continue
        # Find the base class definition in the same file
        for top_node in ast.walk(tree):
            if (isinstance(top_node, ast.ClassDef)
                    and top_node.name == base_name):
                src = _extract_compute_source(top_node, source)
                if src:
                    return src
    return ""


def _annotation_to_str(node: ast.expr) -> str:
    """Convert an AST annotation node to a string representation."""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_annotation_to_str(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        value = _annotation_to_str(node.value)
        slice_val = _annotation_to_str(node.slice)
        return f"{value}[{slice_val}]"
    if isinstance(node, ast.Tuple):
        elts = ", ".join(_annotation_to_str(e) for e in node.elts)
        return elts
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _annotation_to_str(node.left)
        right = _annotation_to_str(node.right)
        return f"{left} | {right}"
    return ast.dump(node)


def _node_to_str(node: ast.expr) -> str:
    """Convert an AST expression node to its string representation."""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_node_to_str(node.value)}.{node.attr}"
    if isinstance(node, ast.List):
        elts = ", ".join(_node_to_str(e) for e in node.elts)
        return f"[{elts}]"
    if isinstance(node, ast.Dict):
        pairs = []
        for k, v in zip(node.keys, node.values):
            pairs.append(f"{_node_to_str(k)}: {_node_to_str(v)}")
        return "{" + ", ".join(pairs) + "}"
    if isinstance(node, ast.Call):
        func = _node_to_str(node.func)
        args = ", ".join(_node_to_str(a) for a in node.args)
        return f"{func}({args})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"-{_node_to_str(node.operand)}"
    return "..."


def _parse_docstring_sections(op: OperatorInfo) -> None:
    """Parse the docstring to extract Args and Usage sections."""
    if not op.docstring:
        return

    lines = op.docstring.split("\n")
    current_section = "description"
    sections: dict[str, list[str]] = {"description": [], "args": [], "usage": []}

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            current_section = "args"
            continue
        elif stripped.lower().startswith("usage:") or stripped.lower().startswith("example:"):
            current_section = "usage"
            continue
        elif stripped.lower().startswith("returns:"):
            current_section = "returns"
            continue
        sections.setdefault(current_section, []).append(line)

    # Extract input descriptions from Args
    if sections.get("args"):
        for arg_line in sections["args"]:
            arg_stripped = arg_line.strip()
            if arg_stripped.startswith("input"):
                op.inputs_desc = arg_stripped.split(":", 1)[-1].strip() if ":" in arg_stripped else ""

            # Match params by name
            for param in op.params:
                if arg_stripped.startswith(param.name):
                    desc_part = arg_stripped.split(":", 1)[-1].strip() if ":" in arg_stripped else ""
                    # Remove default info from docstring (we get it from AST)
                    if "(default:" in desc_part:
                        desc_part = desc_part.split("(default:")[0].strip()
                    param.description = desc_part

    # Extract usage example
    if sections.get("usage"):
        op.usage_example = textwrap.dedent("\n".join(sections["usage"])).strip()


# ---------------------------------------------------------------------------
# Source tree traversal
# ---------------------------------------------------------------------------

def collect_all_operators() -> dict[str, list[OperatorInfo]]:
    """Walk the source tree and collect all operators, grouped by doc page."""
    pages: dict[str, list[OperatorInfo]] = {}

    # 1. Trading operators
    if TRADING_OPS.exists():
        _collect_trading_ops(pages)

    # 2. Core operators (operator/)
    if CORE_OPS.exists():
        _collect_core_ops(pages)

    return pages


def _collect_trading_ops(pages: dict[str, list[OperatorInfo]]) -> None:
    """Collect operators from apps/trading/operators/."""
    for py_file in sorted(TRADING_OPS.rglob("*.py")):
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        if py_file.name == "__init__.py":
            continue
        if py_file.name == "base.py" or py_file.name == "futures.py":
            continue

        rel = py_file.relative_to(TRADING_OPS)
        rel_parts = list(rel.parts)

        # Determine category
        category_key = _match_category(rel_parts)
        if not category_key:
            continue

        cat_config = CATEGORY_MAP[category_key]
        doc_dir = cat_config["doc_dir"]
        subcats = cat_config["subcategories"]

        # Determine subcategory (page filename)
        file_stem = py_file.stem
        if "_default" in subcats:
            subcat = subcats["_default"]
        elif file_stem in subcats:
            subcat = subcats[file_stem]
        else:
            # Try parent dir name
            parent_stem = rel_parts[-2] if len(rel_parts) > 1 else ""
            subcat = subcats.get(parent_stem, file_stem)

        # Page key
        if doc_dir:
            page_key = f"{doc_dir}/{subcat}"
        else:
            page_key = subcat

        # Extract operators
        operators = extract_operators_from_file(py_file)
        for op in operators:
            op.category = page_key
            op.subcategory = file_stem

        pages.setdefault(page_key, []).extend(operators)


def _collect_core_ops(pages: dict[str, list[OperatorInfo]]) -> None:
    """Collect operators from operator/ (core utilities)."""
    # These map to the utility page
    for py_file in sorted(CORE_OPS.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name == "__init__.py":
            continue

        operators = extract_operators_from_file(py_file)
        for op in operators:
            op.category = "utility"
        pages.setdefault("utility", []).extend(operators)


def _match_category(rel_parts: list[str]) -> Optional[str]:
    """Match a relative path to a category key."""
    # Build progressively longer keys
    for length in range(len(rel_parts) - 1, 0, -1):
        key = "/".join(rel_parts[:length])
        if key in CATEGORY_MAP:
            return key
    # Single directory match
    if rel_parts[0] in CATEGORY_MAP:
        return rel_parts[0]
    return None


# ---------------------------------------------------------------------------
# MDX generation
# ---------------------------------------------------------------------------

# Page title/description mapping
PAGE_META = {
    "indicators/moving-averages": ("Moving Averages", "SMA, EMA, DEMA, TEMA, WMA, TRIMA, T3, KAMA, MAMA, MA"),
    "indicators/momentum": ("Momentum Indicators", "RSI, MACD, MOM, ROC, CMO, APO, PPO, STOCH, STOCHRSI, WILLR, ULTOSC, BOP, TRIX, CCI"),
    "indicators/trend": ("Trend Indicators", "ADX, DI, AROON, SAR, ICHIMOKU"),
    "indicators/volatility": ("Volatility Indicators", "ATR, NATR, BBANDS, STDDEV"),
    "indicators/volume": ("Volume Indicators", "OBV, AD, ADOSC, MFI, VWAP, CCI"),
    "indicators/statistics": ("Statistics and Math", "LINEARREG, BETA, CORREL, VAR, TSF, MINMAX, Hilbert Transform"),
    "indicators/patterns": ("Candlestick Patterns", "48+ CDL pattern recognition operators"),
    "signals/alphas": ("Alpha Signals", "21 alpha signals across momentum, mean reversion, volatility, volume, and liquidity categories"),
    "signals/factors": ("Cross-Sectional Factors", "8 risk factors for portfolio construction"),
    "signals/alpha-101": ("Alpha 101", "101 Formulaic Alphas (Kakushadze, 2016)"),
    "transforms/scalers": ("Scalers", "ZScore, Rank, MinMaxScale, L1Norm, L2Norm, Softmax, Clip, Winsorize"),
    "transforms/neutralizers": ("Neutralizers", "Demean, Neutralize, GroupNeutralize, BetaNeutralize, FactorNeutralize, BarraNeutralizer"),
    "transforms/optimizers": ("Portfolio Optimizers", "MeanVarianceOptimizer, RiskParityOptimizer, EqualWeight, ClipWeights, MaxPositions"),
    "transforms/position": ("Position Operators", "WeightsToPositions, TurnoverConstraint, LotRounder, PositionLimits"),
    "universe/filters": ("Universe Filters", "VolumeFilter, VolatilityFilter, PriceFilter, LiquidityFilter, DataAvailabilityFilter"),
    "universe/scores": ("Universe Scores", "VolumeScore, LiquidityScore, VolatilityScore"),
    "metrics": ("Performance Metrics", "Rolling and accumulated performance metrics"),
    "utility": ("Utility Operators", "Identity, Resample, FieldMerge, SymbolSelect, SymbolDrop, Constant, IntervalGate"),
    "order": ("Order Operators", "TargetPositionIntention, FuturesTargetPositionIntention, DynamicUniverseIntention, ArbitrageIntention"),
    "semantic": ("Semantic Operators", "SentimentParser, WebSearchOperator, LLMScorer"),
}


def generate_mdx_page(page_key: str, operators: list[OperatorInfo]) -> str:
    """Generate an MDX page for a group of operators."""
    meta = PAGE_META.get(page_key, (page_key.replace("/", " - ").title(), ""))
    title, description = meta

    # Check for manual partial
    partial_path = PARTIALS_DIR / f"{page_key.replace('/', '_')}_intro.mdx"
    partial_content = ""
    if partial_path.exists():
        partial_content = partial_path.read_text(encoding="utf-8").strip()

    lines: list[str] = []

    # Frontmatter
    lines.append("---")
    lines.append(f'title: "{title}"')
    lines.append(f'description: "{_escape_mdx(description)}"')
    lines.append("---")
    lines.append("")

    # Auto-generated notice
    lines.append("{/* AUTO-GENERATED — do not edit manually. Run generate_operator_docs.py */}")
    lines.append("")

    # Partial intro
    if partial_content:
        lines.append(partial_content)
        lines.append("")
        lines.append("---")
        lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    role_set = sorted(set(op.role for op in operators if op.role != "UNKNOWN"))
    role_str = ", ".join(f"`{r}`" for r in role_set) if role_set else "various"
    lines.append(f"This page documents **{len(operators)} operators** (role: {role_str}).")
    lines.append("")

    # Quick reference table
    if len(operators) > 3:
        lines.append("## Quick Reference")
        lines.append("")
        lines.append("| Operator | Role | Key Parameters | Ephemeral |")
        lines.append("|----------|------|----------------|-----------|")
        for op in operators:
            key_params = _format_key_params(op.params)
            eph = "Yes" if op.ephemeral else "No"
            lines.append(f"| **{op.class_name}** | `{op.role}` | {key_params} | {eph} |")
        lines.append("")

    # Individual operator sections
    for op in operators:
        lines.extend(_generate_operator_section(op))
        lines.append("")

    # Related pages
    lines.append("## Related Pages")
    lines.append("")
    lines.append('<CardGroup cols={2}>')
    lines.append('  <Card title="Operator Protocol" icon="gear" href="/engine/operator-protocol">')
    lines.append("    How operators implement the compute() interface")
    lines.append("  </Card>")
    lines.append('  <Card title="StatefulGraph" icon="diagram-project" href="/engine/stateful-graph">')
    lines.append("    How operators compose into a DAG")
    lines.append("  </Card>")
    lines.append("</CardGroup>")
    lines.append("")

    return "\n".join(lines)


def _generate_operator_section(op: OperatorInfo) -> list[str]:
    """Generate the MDX section for a single operator."""
    lines: list[str] = []

    lines.append("---")
    lines.append("")
    lines.append(f"## {op.class_name}")
    lines.append("")

    # Description from docstring
    if op.docstring:
        desc = _extract_description(op.docstring)
        if desc:
            # Escape MDX-unsafe characters
            desc = _escape_mdx_content(desc)
            lines.append(desc)
            lines.append("")

    # Role and ephemeral badge
    badges = f"**Role**: `{op.role}`"
    if op.ephemeral:
        badges += " | **Ephemeral**: Yes"
    else:
        badges += " | **Ephemeral**: No"
    lines.append(badges)
    lines.append("")

    # Parameter table
    user_params = [p for p in op.params if p.name not in ("self",)]
    if user_params:
        lines.append("### Parameters")
        lines.append("")
        lines.append("| Parameter | Type | Default | Description |")
        lines.append("|-----------|------|---------|-------------|")
        for p in user_params:
            type_str = f"`{p.type_hint}`" if p.type_hint != "Any" else ""
            default_str = f"`{p.default}`" if p.default is not None else "Required"
            desc_str = _escape_mdx(p.description) if p.description else ""
            lines.append(f"| `{p.name}` | {type_str} | {default_str} | {desc_str} |")
        lines.append("")

    # Usage example from docstring
    if op.usage_example:
        lines.append("### Usage")
        lines.append("")
        lines.append("```python")
        lines.append(op.usage_example)
        lines.append("```")
        lines.append("")

    # Full compute() source code
    if op.compute_source:
        lines.append("### Source Code")
        lines.append("")
        lines.append(f"Full `compute()` implementation — no hidden logic.")
        lines.append("")
        lines.append("```python")
        lines.append(op.compute_source)
        lines.append("```")
        lines.append("")

    # Source location
    lines.append(f'<sub>Source: `{op.module_path}`</sub>')
    lines.append("")

    return lines


def _extract_description(docstring: str) -> str:
    """Extract the description portion of a docstring (before Args/Usage)."""
    lines = docstring.split("\n")
    desc_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith(("args:", "usage:", "returns:", "raises:")):
            break
        desc_lines.append(line)

    return textwrap.dedent("\n".join(desc_lines)).strip()


def _format_key_params(params: list[ParamInfo]) -> str:
    """Format key parameters for the quick reference table."""
    user_params = [p for p in params if p.name not in ("self", "input", "inputs")]
    if not user_params:
        return "—"
    parts = []
    for p in user_params[:3]:  # Max 3 params in quick ref
        if p.default is not None:
            parts.append(f"`{p.name}={p.default}`")
        else:
            parts.append(f"`{p.name}`")
    return ", ".join(parts)


def _escape_mdx(text: str) -> str:
    """Escape characters that break MDX rendering in table cells."""
    # Replace < and > with HTML entities to prevent JSX parsing
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    # Replace { and } with HTML entities to prevent JSX expression parsing
    text = text.replace("{", "&#123;").replace("}", "&#125;")
    return text


def _escape_mdx_content(text: str) -> str:
    """Escape MDX-unsafe characters in content blocks.

    Handles characters that MDX interprets as JSX:
    - < and > in formulas (e.g., RSI > 70, x <= threshold)
    - { and } in descriptions (e.g., dict literals, format strings)
    - Patterns like <-50 which MDX parses as JSX elements
    """
    import re

    # Replace { and } with HTML entities to prevent JSX expression parsing
    text = text.replace("{", "&#123;").replace("}", "&#125;")

    # Replace <-NUMBER patterns first (e.g., "<-50" → "below -50")
    text = re.sub(r'<-(\d+)', r'below -\1', text)

    # Replace <= and >= operators in prose
    text = text.replace("<=", "&lt;=").replace(">=", "&gt;=")

    # Replace comparison operators in prose (not inside code blocks)
    # Pattern: word/number followed by < or > followed by word/number
    text = re.sub(r'(?<!\`)\b(\w+)\s*>\s*(\d+)', r'\1 above \2', text)
    text = re.sub(r'(?<!\`)\b(\w+)\s*<\s*(\d+)', r'\1 below \2', text)

    # Replace standalone < and > that look like JSX tags (not already escaped)
    text = re.sub(r'<(?!/?\w|!--|&)', '&lt;', text)

    # Replace standalone > at start of line followed by number (MDX blockquote)
    text = re.sub(r'^>\s*(\d+)', r'Above \1', text, flags=re.MULTILINE)

    return text


# ---------------------------------------------------------------------------
# Alpha 101 special handling
# ---------------------------------------------------------------------------

def generate_alpha101_page(operators: list[OperatorInfo]) -> str:
    """Generate a compact table-format page for Alpha 101 operators."""
    lines: list[str] = []

    lines.append("---")
    lines.append('title: "Alpha 101"')
    lines.append('description: "101 Formulaic Alphas — systematic alpha generation from Kakushadze (2016)"')
    lines.append("---")
    lines.append("")
    lines.append("{/* AUTO-GENERATED — do not edit manually. Run generate_operator_docs.py */}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("Implementation of [101 Formulaic Alphas](https://arxiv.org/abs/1601.00991) (Kakushadze, Z., 2016).")
    lines.append(f"**{len(operators)} alphas** available, each as a standalone operator.")
    lines.append("")
    lines.append("All Alpha 101 operators share: **Role**: `ALPHA` | **Ephemeral**: No")
    lines.append("")

    # Table format
    lines.append("## Alpha Catalog")
    lines.append("")
    lines.append("| Alpha | Description | Key Parameters |")
    lines.append("|-------|-------------|----------------|")

    for op in sorted(operators, key=lambda o: o.class_name):
        # Use only the first line of the docstring for the table
        desc = op.docstring.split("\n")[0].strip() if op.docstring else ""
        desc = _escape_mdx(desc)
        key_params = _format_key_params(op.params)
        lines.append(f"| **{op.class_name}** | {desc} | {key_params} |")

    lines.append("")

    # Usage pattern
    lines.append("## Usage Pattern")
    lines.append("")
    lines.append("All Alpha 101 operators follow the same pattern:")
    lines.append("")
    lines.append("```python")
    lines.append("from clyptq.apps.trading.operators.signal.alpha.alpha_101 import Alpha101_001")
    lines.append("")
    lines.append('graph.add_node("alpha_001", Alpha101_001(')
    lines.append('    Input("FIELD:binance:futures:ohlcv:close", timeframe="1m", lookback=20)')
    lines.append("))")
    lines.append("```")
    lines.append("")

    # Full source code for each alpha
    alphas_with_source = [op for op in sorted(operators, key=lambda o: o.class_name) if op.compute_source]
    if alphas_with_source:
        lines.append("## Source Code")
        lines.append("")
        lines.append("Full `compute()` implementations — no hidden logic.")
        lines.append("")
        lines.append("<AccordionGroup>")
        for op in alphas_with_source:
            desc = _extract_description(op.docstring).split("\n")[0] if op.docstring else ""
            desc = _escape_mdx(desc)
            lines.append(f'<Accordion title="{op.class_name}">')
            if desc:
                lines.append(f"")
                lines.append(f"{desc}")
                lines.append("")
            lines.append("```python")
            lines.append(op.compute_source)
            lines.append("```")
            lines.append("</Accordion>")
        lines.append("</AccordionGroup>")
        lines.append("")

    lines.append("## Related Pages")
    lines.append("")
    lines.append('<CardGroup cols={2}>')
    lines.append('  <Card title="Alpha Signals" icon="chart-line" href="/operators/signals/alphas">')
    lines.append("    Hand-crafted alpha signals")
    lines.append("  </Card>")
    lines.append('  <Card title="Factors" icon="layer-group" href="/operators/signals/factors">')
    lines.append("    Cross-sectional risk factors")
    lines.append("  </Card>")
    lines.append("</CardGroup>")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics special handling
# ---------------------------------------------------------------------------

def generate_metrics_page(operators: list[OperatorInfo]) -> str:
    """Generate a page that separates rolling vs accumulated metrics."""
    rolling = [op for op in operators if "Rolling" in op.class_name or "rolling" in op.module_path]
    accum = [op for op in operators if "Accum" in op.class_name or "accum" in op.module_path]
    other = [op for op in operators if op not in rolling and op not in accum]

    lines: list[str] = []

    lines.append("---")
    lines.append('title: "Performance Metrics"')
    lines.append('description: "Rolling and accumulated performance metrics for strategy evaluation"')
    lines.append("---")
    lines.append("")
    lines.append("{/* AUTO-GENERATED — do not edit manually. Run generate_operator_docs.py */}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"ClyptQ provides **{len(rolling)} rolling** and **{len(accum)} accumulated** performance metrics.")
    lines.append("All metrics have **Role**: `METRIC` and read from `STATE:` inputs.")
    lines.append("")

    if rolling:
        lines.append("## Rolling Metrics")
        lines.append("")
        lines.append("Computed over a sliding window of recent data.")
        lines.append("")
        lines.append("| Metric | Key Parameters | Description |")
        lines.append("|--------|----------------|-------------|")
        for op in sorted(rolling, key=lambda o: o.class_name):
            desc = _extract_description(op.docstring).split("\n")[0] if op.docstring else ""
            desc = _escape_mdx(desc)
            key_params = _format_key_params(op.params)
            lines.append(f"| **{op.class_name}** | {key_params} | {desc} |")
        lines.append("")

    if accum:
        lines.append("## Accumulated Metrics")
        lines.append("")
        lines.append("Computed over the entire strategy history (since inception).")
        lines.append("")
        lines.append("| Metric | Key Parameters | Description |")
        lines.append("|--------|----------------|-------------|")
        for op in sorted(accum, key=lambda o: o.class_name):
            desc = _extract_description(op.docstring).split("\n")[0] if op.docstring else ""
            desc = _escape_mdx(desc)
            key_params = _format_key_params(op.params)
            lines.append(f"| **{op.class_name}** | {key_params} | {desc} |")
        lines.append("")

    if other:
        lines.append("## Other Metrics")
        lines.append("")
        for op in other:
            lines.extend(_generate_operator_section(op))
            lines.append("")

    # Detailed sections for each metric
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Reference")
    lines.append("")

    for op in sorted(operators, key=lambda o: o.class_name):
        lines.extend(_generate_operator_section(op))
        lines.append("")

    lines.append("## Related Pages")
    lines.append("")
    lines.append('<CardGroup cols={2}>')
    lines.append('  <Card title="STATE Principle" icon="database" href="/engine/state-principle">')
    lines.append("    How metrics access portfolio state")
    lines.append("  </Card>")
    lines.append('  <Card title="Backtesting Overview" icon="flask" href="/backtesting/overview">')
    lines.append("    Using metrics to evaluate backtest results")
    lines.append("  </Card>")
    lines.append("</CardGroup>")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate operator reference MDX pages")
    parser.add_argument("--dry-run", action="store_true", help="Print output without writing files")
    parser.add_argument("--category", type=str, help="Generate only a specific category (e.g., 'indicators')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("ClyptQ Operator Docs Generator")
    print("=" * 60)
    print()

    # Validate paths
    if not TRADING_OPS.exists():
        print(f"ERROR: Trading operators directory not found: {TRADING_OPS}")
        sys.exit(1)

    print(f"Source: {CLYPTQ_BASE}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Collect all operators
    print("Scanning source files...")
    pages = collect_all_operators()

    total_ops = sum(len(ops) for ops in pages.values())
    print(f"Found {total_ops} operators across {len(pages)} pages")
    print()

    # Filter by category if specified
    if args.category:
        pages = {k: v for k, v in pages.items() if args.category in k}
        if not pages:
            print(f"No pages matching category '{args.category}'")
            sys.exit(1)

    # Generate pages
    generated = 0
    for page_key, operators in sorted(pages.items()):
        if not operators:
            continue

        # Special handling for Alpha 101
        if page_key == "signals/alpha-101":
            content = generate_alpha101_page(operators)
        elif page_key == "metrics":
            content = generate_metrics_page(operators)
        else:
            content = generate_mdx_page(page_key, operators)

        # Output path
        output_path = OUTPUT_DIR / f"{page_key}.mdx"

        if args.dry_run:
            print(f"[DRY RUN] Would write {output_path}")
            print(f"          {len(operators)} operators, {len(content)} chars")
            if args.verbose:
                print(content[:500])
                print("...")
            print()
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
            print(f"  WROTE {output_path} ({len(operators)} operators)")
            generated += 1

    print()
    print(f"Done. {'Would generate' if args.dry_run else 'Generated'} {generated if not args.dry_run else len(pages)} pages.")

    # Summary table
    print()
    print("Page Summary:")
    print(f"{'Page':<35} {'Operators':>10}")
    print("-" * 47)
    for page_key, operators in sorted(pages.items()):
        if operators:
            print(f"  {page_key:<33} {len(operators):>10}")
    print("-" * 47)
    print(f"  {'TOTAL':<33} {total_ops:>10}")


if __name__ == "__main__":
    main()
