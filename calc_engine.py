"""
calc_engine.py — Stage 3 Python calculation engine
Provides run(rec) for financial metrics and rule_flags(rec) for Stage 7 rule-based flags.
"""

from typing import Any

# ---------------------------------------------------------------------------
# Severity matrix for rule flags
# ---------------------------------------------------------------------------
SEVERITY_MATRIX = {
    "negative_ebitda":          {"severity": "HIGH",   "label": "Negative EBITDA"},
    "ebitda_margin_low":        {"severity": "MEDIUM", "label": "EBITDA Margin < 10%"},
    "revenue_decline":          {"severity": "HIGH",   "label": "Year-on-Year Revenue Decline"},
    "net_margin_negative":      {"severity": "HIGH",   "label": "Negative Net Profit Margin"},
    "high_debt_equity":         {"severity": "HIGH",   "label": "Debt-to-Equity > 2.0"},
    "current_ratio_low":        {"severity": "HIGH",   "label": "Current Ratio < 1.0"},
    "current_ratio_warning":    {"severity": "MEDIUM", "label": "Current Ratio 1.0–1.5"},
    "interest_coverage_low":    {"severity": "HIGH",   "label": "Interest Coverage < 1.5x"},
    "operating_cashflow_neg":   {"severity": "HIGH",   "label": "Negative Operating Cash Flow"},
    "capex_revenue_high":       {"severity": "MEDIUM", "label": "CapEx > 15% of Revenue"},
    "gross_margin_low":         {"severity": "MEDIUM", "label": "Gross Margin < 20%"},
    "revenue_growth_high":      {"severity": "LOW",    "label": "Revenue Growth > 50% (verify)"},
}


def _safe(val: Any, default: float = 0.0) -> float:
    """Return float or default if val is None / non-numeric."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _pct(numerator: float, denominator: float) -> float | None:
    """Return percentage (0–100 scale) or None if denominator is zero."""
    if denominator == 0:
        return None
    return round((numerator / denominator) * 100, 2)


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


# ---------------------------------------------------------------------------
# Main calculation function
# ---------------------------------------------------------------------------
def run(rec: dict) -> dict:
    """
    Calculate financial metrics from extracted record.

    Parameters
    ----------
    rec : dict
        Extracted financials with keys (all optional, default 0):
          revenue, revenue_prev, cost_of_goods_sold, gross_profit,
          operating_expenses, ebitda, depreciation_amortisation,
          ebit, interest_expense, tax_expense, net_profit,
          total_assets, total_liabilities, total_equity,
          current_assets, current_liabilities, total_debt,
          operating_cashflow, capex, free_cashflow

    Returns
    -------
    dict
        Calculated metrics plus a 'reconciliation' block noting any
        derived values and data-quality warnings.
    """
    r = {k: _safe(v) for k, v in rec.items()}

    revenue        = r.get("revenue", 0)
    revenue_prev   = r.get("revenue_prev", 0)
    cogs           = r.get("cost_of_goods_sold", 0)
    gross_profit   = r.get("gross_profit", 0) or (revenue - cogs)
    op_expenses    = r.get("operating_expenses", 0)
    ebitda         = r.get("ebitda", 0)
    da             = r.get("depreciation_amortisation", 0)
    ebit           = r.get("ebit", 0) or (ebitda - da)
    interest       = r.get("interest_expense", 0)
    tax            = r.get("tax_expense", 0)
    net_profit     = r.get("net_profit", 0)
    total_assets   = r.get("total_assets", 0)
    total_liab     = r.get("total_liabilities", 0)
    total_equity   = r.get("total_equity", 0) or (total_assets - total_liab)
    current_assets = r.get("current_assets", 0)
    current_liab   = r.get("current_liabilities", 0)
    total_debt     = r.get("total_debt", 0)
    op_cashflow    = r.get("operating_cashflow", 0)
    capex          = r.get("capex", 0)
    free_cashflow  = r.get("free_cashflow", 0) or (op_cashflow - capex)

    # Derive EBITDA if not provided
    derived_ebitda = False
    if not ebitda and gross_profit:
        ebitda = gross_profit - op_expenses + da
        derived_ebitda = True

    reconciliation = {
        "derived_ebitda":    derived_ebitda,
        "derived_ebit":      not r.get("ebit"),
        "derived_equity":    not r.get("total_equity"),
        "derived_fcf":       not r.get("free_cashflow"),
        "warnings":          [],
    }
    if not revenue:
        reconciliation["warnings"].append("revenue is zero or missing")
    if not total_assets:
        reconciliation["warnings"].append("total_assets is zero or missing — leverage ratios unreliable")

    metrics = {
        # ── Profitability ──────────────────────────────────────────────
        "gross_margin_pct":          _pct(gross_profit, revenue),
        "ebitda_margin_pct":         _pct(ebitda, revenue),
        "ebit_margin_pct":           _pct(ebit, revenue),
        "net_profit_margin_pct":     _pct(net_profit, revenue),

        # ── Year-on-Year growth ─────────────────────────────────────────
        "revenue_yoy_pct":           _pct(revenue - revenue_prev, revenue_prev),

        # ── Leverage / solvency ─────────────────────────────────────────
        "debt_to_equity":            _ratio(total_debt, total_equity),
        "total_liabilities_equity":  _ratio(total_liab, total_equity),
        "interest_coverage":         _ratio(ebit, interest),

        # ── Liquidity ───────────────────────────────────────────────────
        "current_ratio":             _ratio(current_assets, current_liab),

        # ── Efficiency / returns ────────────────────────────────────────
        "return_on_assets_pct":      _pct(net_profit, total_assets),
        "return_on_equity_pct":      _pct(net_profit, total_equity),

        # ── Cash flow ───────────────────────────────────────────────────
        "capex_to_revenue_pct":      _pct(capex, revenue),
        "free_cashflow":             round(free_cashflow, 2),
        "operating_cashflow":        round(op_cashflow, 2),

        # ── Absolute values passed through for context ──────────────────
        "revenue":                   round(revenue, 2),
        "ebitda":                    round(ebitda, 2),
        "net_profit":                round(net_profit, 2),
        "total_debt":                round(total_debt, 2),
        "total_equity":              round(total_equity, 2),

        # ── Reconciliation notes ─────────────────────────────────────────
        "reconciliation":            reconciliation,
    }

    return metrics


# ---------------------------------------------------------------------------
# Rule-based flag function (Stage 7)
# ---------------------------------------------------------------------------
def rule_flags(rec: dict) -> list[dict]:
    """
    Apply deterministic rules to extracted + calculated metrics.

    Parameters
    ----------
    rec : dict
        Should include both raw extracted values AND the output of run(rec).
        Pass {**extracted, **run(extracted)} for full coverage.

    Returns
    -------
    list[dict]
        Each entry: { "flag": str, "severity": str, "label": str, "value": float|None }
    """
    m = {k: _safe(v, None) for k, v in rec.items()}

    def _get(key):
        v = rec.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    flags = []

    def _flag(key, value):
        info = SEVERITY_MATRIX.get(key, {"severity": "LOW", "label": key})
        flags.append({
            "flag":     key,
            "severity": info["severity"],
            "label":    info["label"],
            "value":    round(value, 4) if value is not None else None,
        })

    ebitda          = _get("ebitda")
    ebitda_margin   = _get("ebitda_margin_pct")
    revenue_yoy     = _get("revenue_yoy_pct")
    net_margin      = _get("net_profit_margin_pct")
    dte             = _get("debt_to_equity")
    current         = _get("current_ratio")
    coverage        = _get("interest_coverage")
    op_cf           = _get("operating_cashflow")
    capex_rev       = _get("capex_to_revenue_pct")
    gross_margin    = _get("gross_margin_pct")

    if ebitda is not None and ebitda < 0:
        _flag("negative_ebitda", ebitda)

    if ebitda_margin is not None and 0 <= ebitda_margin < 10:
        _flag("ebitda_margin_low", ebitda_margin)

    if revenue_yoy is not None and revenue_yoy < 0:
        _flag("revenue_decline", revenue_yoy)

    if net_margin is not None and net_margin < 0:
        _flag("net_margin_negative", net_margin)

    if dte is not None and dte > 2.0:
        _flag("high_debt_equity", dte)

    if current is not None:
        if current < 1.0:
            _flag("current_ratio_low", current)
        elif current < 1.5:
            _flag("current_ratio_warning", current)

    if coverage is not None and 0 < coverage < 1.5:
        _flag("interest_coverage_low", coverage)

    if op_cf is not None and op_cf < 0:
        _flag("operating_cashflow_neg", op_cf)

    if capex_rev is not None and capex_rev > 15:
        _flag("capex_revenue_high", capex_rev)

    if gross_margin is not None and 0 <= gross_margin < 20:
        _flag("gross_margin_low", gross_margin)

    if revenue_yoy is not None and revenue_yoy > 50:
        _flag("revenue_growth_high", revenue_yoy)

    # Sort: HIGH first, then MEDIUM, then LOW
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    flags.sort(key=lambda f: order.get(f["severity"], 9))

    return flags
