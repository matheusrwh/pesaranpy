from __future__ import annotations

import numpy as np
from scipy import stats

_WO = 80     # outer width (title + metadata borders)
_WI = 78     # inner width (parameter table borders)
_NC = 16     # variable name column width
_NW = [11, 10, 10, 10, 11, 10]  # numeric col widths: coef, se, t, p, lower, upper (sum=62; 16+62=78)
_Z95 = 1.96  # 95% normal CI multiplier


def summary(result) -> str:
    """Return a formatted summary string for MeanGroup, PooledMeanGroup, or HausmanResult."""
    from pesaranpy.estimators.mg import MeanGroup
    from pesaranpy.estimators.pmg import PooledMeanGroup
    from pesaranpy.diagnostics.hausman import HausmanResult

    if isinstance(result, MeanGroup):
        return _summary_mg(result)
    if isinstance(result, PooledMeanGroup):
        return _summary_pmg(result)
    if isinstance(result, HausmanResult):
        return _summary_hausman(result)
    raise TypeError(
        f"Expected MeanGroup, PooledMeanGroup, or HausmanResult, got {type(result).__name__}"
    )


# ── layout primitives ─────────────────────────────────────────────────────────

def _sep(n, char):
    return char * n


def _meta_row(ll, lv, rl="", rv=""):
    """
    One metadata row: left cell (37 chars) + 3-char gap + right cell (40 chars) = 80.
    Pass rl="" to leave the right side blank.
    """
    left = f"{ll:<20}{str(lv):>17}"
    right = f"   {rl:<22}{str(rv):>18}" if rl else " " * 43
    return left + right


def _param_header():
    labels = ["", "Parameter", "Std. Err.", "T-stat", "P-value", "Lower CI", "Upper CI"]
    widths = [_NC] + _NW
    return "".join(f"{lbl:>{w}}" for lbl, w in zip(labels, widths))


def _param_row(name, coef, se):
    t = coef / se if se > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(t))) if not np.isnan(t) else np.nan
    lo = coef - _Z95 * se
    hi = coef + _Z95 * se
    name_str = (name[:_NC - 1] + ">") if len(name) >= _NC else name
    row = f"{name_str:<{_NC}}"
    for val, w in zip([coef, se, t, p, lo, hi], _NW):
        row += f"{'      ---':>{w}}" if np.isnan(val) else f"{val:>{w}.4f}"
    return row


def _param_section(title, names, coefs, ses):
    """One parameter block: title (above border), = header - rows =."""
    lines = [
        title.center(_WI),
        _sep(_WI, "="),
        _param_header(),
        _sep(_WI, "-"),
    ]
    for n, c, s in zip(names, coefs, ses):
        lines.append(_param_row(n, c, s))
    lines.append(_sep(_WI, "="))
    return lines


def _build(title, meta_rows, sections, notes):
    lines = [
        _sep(_WO, "="),
        title.center(_WO),
        _sep(_WO, "="),
        *meta_rows,
        _sep(_WO, "="),
    ]
    for i, sec in enumerate(sections):
        if i > 0:
            lines.append("")
        lines.extend(sec)
    lines.append("")
    lines.extend(notes)
    return "\n".join(lines)


# ── MG ────────────────────────────────────────────────────────────────────────

def _summary_mg(r) -> str:
    avg_t = r.n_obs / r.n_units if r.n_units else 0

    meta = [
        _meta_row("Dep. Variable:", r.dep_var,       "Estimator:",          "MG"),
        _meta_row("No. Observations:", r.n_obs,       "No. Cross-sections:", r.n_units),
        _meta_row("",                  "",             "Avg. Obs per unit:",  f"{avg_t:.1f}"),
    ]

    sections = [
        _param_section(
            "Long-run Coefficients",
            r.long_run_names, r.long_run_coef, r.long_run_se,
        ),
        _param_section(
            "Speed of Adjustment",
            ["phi (EC)"], [r.speed_adjust], [r.speed_adjust_se],
        ),
    ]
    if len(r.short_run_names) > 0:
        sections.append(_param_section(
            "Short-run Coefficients",
            r.short_run_names, r.short_run_coef, r.short_run_se,
        ))

    notes = ["Standard errors: cross-sectional variance (Pesaran & Smith, 1995)"]
    return _build("Mean Group Estimation Summary", meta, sections, notes)


# ── PMG ───────────────────────────────────────────────────────────────────────

def _summary_pmg(r) -> str:
    avg_t = r.n_obs / r.n_units if r.n_units else 0
    conv = "Yes" if r.converged else f"No ({r.n_iterations} iter)"

    meta = [
        _meta_row("Dep. Variable:", r.dep_var,              "Estimator:",          "PMG"),
        _meta_row("No. Observations:", r.n_obs,              "No. Cross-sections:", r.n_units),
        _meta_row("Log-likelihood:", f"{r.log_likelihood:.4f}", "Avg. Obs per unit:", f"{avg_t:.1f}"),
        _meta_row("No. iterations:", r.n_iterations,         "Converged:",          conv),
    ]

    sections = [
        _param_section(
            "Long-run Coefficients",
            r.long_run_names, r.long_run_coef, r.long_run_se,
        ),
        _param_section(
            "Speed of Adjustment",
            ["phi (EC)"], [r.speed_adjust], [r.speed_adjust_se],
        ),
    ]
    if len(r.short_run_names) > 0:
        sections.append(_param_section(
            "Short-run Coefficients (mean across units)",
            r.short_run_names, r.short_run_coef, r.short_run_se,
        ))

    notes = [
        "Standard errors: concentrated Hessian (long-run); "
        "cross-sectional variance (phi, short-run)",
    ]
    return _build("Pooled Mean Group Estimation Summary", meta, sections, notes)


# ── Hausman ───────────────────────────────────────────────────────────────────

def _summary_hausman(r) -> str:
    alpha = 0.05
    reject = r.p_value < alpha
    conclusion = (
        f"Reject H0 at {int(alpha*100)}% -- slope heterogeneity detected; prefer MG."
        if reject else
        f"Fail to reject H0 at {int(alpha*100)}% -- long-run homogeneity supported; PMG efficient."
    )

    # Metadata block
    lines = [
        _sep(_WO, "="),
        "Hausman Specification Test".center(_WO),
        _sep(_WO, "="),
        _meta_row("Efficient estimator:", "PMG",  "Consistent estimator:", "MG"),
        _meta_row(
            "H0:", "Long-run slope homogeneity (PMG efficient)",
        ),
        _meta_row(
            "H1:", "Slope heterogeneity (only MG consistent)",
        ),
        _sep(_WO, "="),
        _meta_row(
            "Chi-sq statistic:", f"{r.statistic:.4f}",
            "Degrees of freedom:", str(r.df),
        ),
        _meta_row("P-value:", f"{r.p_value:.4f}"),
        _sep(_WO, "="),
    ]

    # Coefficient differences table (no SE available — display only values)
    _NC_H = max(16, max((len(n) for n in r.long_run_names), default=8) + 2)
    _col_w = [14, 14]  # MG-PMG diff col only

    hdr = f"{'Variable':<{_NC_H}}{'MG - PMG':>14}"
    sep_inner = "-" * (_NC_H + 14)
    diff_lines = [
        "Coefficient Differences (MG - PMG)".center(_WI),
        _sep(_WI, "="),
        hdr,
        sep_inner,
    ]
    for name, d in zip(r.long_run_names, r.diff_coefs):
        diff_lines.append(f"{name:<{_NC_H}}{d:>14.6f}")
    diff_lines.append(_sep(_WI, "="))

    lines.append("")
    lines.extend(diff_lines)
    lines.append("")
    lines.append(f"  Conclusion: {conclusion}")

    if r.note:
        lines.append("")
        lines.append(f"  Note: {r.note}")

    return "\n".join(lines)
