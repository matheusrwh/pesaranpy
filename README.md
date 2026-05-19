# pesaranpy

Python implementation of the **Mean Group (MG)** and **Pooled Mean Group (PMG)** estimators for cointegrated heterogeneous panels, replicating the behavior of Stata's `xtpmg`.

**References**
- Pesaran, M. H., & Smith, R. (1995). Estimating long-run relationships from dynamic heterogeneous panels. *Journal of Econometrics*, 68(1), 79–113.
- Pesaran, M. H., Shin, Y., & Smith, R. P. (1999). Pooled mean group estimation of dynamic heterogeneous panels. *JASA*, 94(446), 621–634.
- Blackburne, E. F., & Frank, M. W. (2007). Estimation of nonstationary heterogeneous panels. *Stata Journal*, 7(2), 197–208.

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10. Dependencies: `numpy`, `scipy`, `pandas`.

---

## Workflow

```
prepare_panel()      →    build_ecm_design()    →    estimate_mean_group()
                                                  →    estimate_pmg()
                                                            ↓
                                                      hausman_test(pmg, mg)
```

```python
import pesaranpy as pp

panel  = pp.prepare_panel(df, id_unit="country", id_time="year", variables=["y", "x1", "x2"])
design = pp.build_ecm_design(panel, dep_var="y", long_run_vars=["x1", "x2"], p=1, q=1)
mg     = pp.estimate_mean_group(design)
pmg    = pp.estimate_pmg(design)
test   = pp.hausman_test(pmg, mg)
```

---

## API Reference

### `prepare_panel`

```python
prepare_panel(
    df: pd.DataFrame,
    id_unit: str,
    id_time: str,
    variables: list[str],
    drop_missing: bool = True,
) -> PanelData
```

Validates and sorts a long-format panel. Raises `ValueError` on duplicate `(id_unit, id_time)` pairs or internal time gaps within any unit — both invalidate lag construction. With `drop_missing=True`, units with any `NaN` in `variables` are dropped entirely.

Returns a `PanelData` dataclass with fields `df`, `id_unit`, `id_time`, `units`, `n_units`, `t_per_unit`, `is_balanced`.

---

### `build_ecm_design`

```python
build_ecm_design(
    panel: PanelData,
    dep_var: str,
    long_run_vars: list[str],
    short_run_vars: list[str] | None = None,
    p: int = 1,
    q: int = 1,
) -> dict
```

Constructs per-unit ARDL(p, q) design matrices in ECM form:

$$\Delta y_{it} = \varphi_i(y_{i,t-1} - \theta_i' x_{it}) + \sum_{j=1}^{p-1}\lambda^*_{ij}\,\Delta y_{i,t-j} + \sum_{j=0}^{q-1}\delta^{*\prime}_{ij}\,\Delta x_{i,t-j} + \mu_i + \varepsilon_{it}$$

Follows the `xtpmg` convention: the EC term uses $x_{it}$ (contemporaneous levels), not $x_{i,t-1}$. Variables in `short_run_vars` enter only as first differences and are not restricted to a long-run relationship.

Returns a `dict` with keys `by_unit`, `long_run_names`, `short_run_names`, `dep_var`, consumed directly by both estimators.

---

### `estimate_mean_group`

```python
estimate_mean_group(design: dict) -> MeanGroup
```

MG estimator (Pesaran & Smith, 1995). Fits unit-specific OLS regressions on the ECM parameterization and averages across units. Long-run coefficients are recovered as $\hat\theta_i = -\hat\psi_i / \hat\varphi_i$.

Standard errors follow the Pesaran-Smith cross-sectional variance formula:

$$\widehat{SE}(\hat\theta_{MG}) = \sqrt{\frac{\sum_i(\hat\theta_i - \hat\theta_{MG})^2}{N(N-1)}}$$

| Field | Description |
|---|---|
| `long_run_coef` | $\hat\theta_{MG}$ |
| `long_run_se` | Cross-sectional SE |
| `speed_adjust` | $\bar{\hat\varphi}$ |
| `speed_adjust_se` | Cross-sectional SE of $\hat\varphi_i$ |
| `short_run_coef` | Mean short-run coefficients |
| `short_run_se` | Cross-sectional SE of short-run coefficients |
| `individual_coef` | Per-unit `dict` with keys `phi`, `theta`, `sr` |
| `n_units` | Effective N after dropping degenerate units |

---

### `estimate_pmg`

```python
estimate_pmg(
    design: dict,
    max_iter: int = 100,
    tol: float = 1e-6,
    initial_theta: np.ndarray | None = None,
) -> PooledMeanGroup
```

PMG estimator (Pesaran, Shin & Smith, 1999). Maximizes the concentrated log-likelihood by alternating between closed-form updates of $(\varphi_i, \sigma^2_i)$ and a linear solve for the pooled $\theta$. Initializes from $\hat\theta_{MG}$ if `initial_theta` is not provided.

Long-run SE and VCOV are derived from the inverse of the concentrated Hessian $A = \sum_i (\hat\varphi_i^2 / \hat\sigma_i^2)\, X_i' M_{W_i} X_i$.

| Field | Description |
|---|---|
| `long_run_coef` | $\hat\theta_{PMG}$ |
| `long_run_se` | $\sqrt{\text{diag}(A^{-1})}$ |
| `long_run_vcov` | $A^{-1}$ |
| `speed_adjust` | Mean $\hat\varphi_i$ across units |
| `speed_adjust_individual` | Per-unit $\hat\varphi_i$ |
| `sigma2_individual` | Per-unit $\hat\sigma^2_i$ |
| `log_likelihood` | Concentrated log-likelihood at convergence |
| `n_iterations` | Iterations to convergence |
| `converged` | Whether tolerance criterion was met |

---

### `hausman_test`

```python
hausman_test(
    efficient_results,      # PMG
    consistent_results,     # MG
) -> HausmanResult
```

Hausman specification test for long-run slope homogeneity:

$$H = (\hat\theta_{MG} - \hat\theta_{PMG})'\,[\hat V_{MG} - \hat V_{PMG}]^{-1}\,(\hat\theta_{MG} - \hat\theta_{PMG}) \overset{d}{\to} \chi^2(k)$$

$\hat V_{MG}$ is the full cross-sectional covariance matrix constructed from the unit-level $\hat\theta_i$. If $\hat V_{MG} - \hat V_{PMG}$ is not positive semi-definite (finite-sample occurrence), the Moore-Penrose pseudoinverse is used and degrees of freedom are adjusted to the effective rank.

| Field | Description |
|---|---|
| `statistic` | $H$ |
| `p_value` | $p$-value from $\chi^2(k)$ |
| `df` | Degrees of freedom (effective rank of $\hat V_{MG} - \hat V_{PMG}$) |
| `diff_coefs` | $\hat\theta_{MG} - \hat\theta_{PMG}$ |
| `note` | Non-empty if pseudoinverse was used |
