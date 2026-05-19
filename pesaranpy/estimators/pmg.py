import numpy as np
from dataclasses import dataclass
import warnings


@dataclass(frozen=True)
class PooledMeanGroup:
    long_run_coef: np.ndarray
    long_run_se: np.ndarray
    long_run_vcov: np.ndarray
    speed_adjust: float
    speed_adjust_se: float
    speed_adjust_individual: dict
    short_run_coef: np.ndarray
    short_run_se: np.ndarray
    sigma2_individual: dict
    log_likelihood: float
    n_iterations: int
    converged: bool
    n_units: int
    n_obs: int
    dep_var: str
    long_run_names: list
    short_run_names: list


def _annihilator_projection(W: np.ndarray, v: np.ndarray) -> np.ndarray:
    if W.shape[1] == 0:
        return v
    Q, _ = np.linalg.qr(W)
    return v - Q @ (Q.T @ v)


def estimate_pmg(
        design: dict,
        max_iter: int = 100,
        tol: float = 1e-6,
        initial_theta: np.ndarray | None = None) -> PooledMeanGroup:

    by_unit = design['by_unit']
    units = list(by_unit.keys())
    k_long = len(design["long_run_names"])
    N = len(units)

    pre = {}
    for u in units:
        d = by_unit[u]
        T_i = d["T_eff"]
        W_i = np.column_stack([d["SR"], np.ones(T_i)])
        H_dy = _annihilator_projection(W_i, d["dy"])
        H_ylag = _annihilator_projection(W_i, d["y_lag"])
        H_X = _annihilator_projection(W_i, d["X_long"])
        pre[u] = {
            "T_i": T_i, "W_i": W_i,
            "dy": d["dy"], "y_lag": d["y_lag"], "X": d["X_long"],
            "H_dy": H_dy, "H_ylag": H_ylag, "H_X": H_X,
        }

    if initial_theta is None:
        try:
            from pesaranpy.estimators.mg import estimate_mean_group
            mg = estimate_mean_group(design)
            theta = mg.long_run_coef.copy()
        except RuntimeError as exc:
            warnings.warn(
                f"MG initialization failed ({exc}). Using zero initialization for PMG.",
                RuntimeWarning,
                stacklevel=2,
            )
            theta = np.zeros(k_long)
    else:
        theta = np.asarray(initial_theta, dtype=float).ravel()
        if theta.shape != (k_long,):
            raise ValueError(
                f"initial_theta must have shape ({k_long},), got {theta.shape}"
            )

    converged = False
    for it in range(max_iter):
        theta_old = theta.copy()

        phi = {}
        sigma2 = {}
        for u in units:
            p = pre[u]
            xi = p["y_lag"] - p["X"] @ theta
            H_xi = p["H_ylag"] - p["H_X"] @ theta
            num = xi @ p["H_dy"]
            den = xi @ H_xi
            if abs(den) < 1e-12:
                phi[u] = 0.0
                sigma2[u] = np.var(p["dy"]) + 1e-8
                continue
            phi_i = num / den
            e_i = p["dy"] - phi_i * xi
            H_e = p["H_dy"] - phi_i * H_xi
            sigma2_i = max(e_i @ H_e / p["T_i"], 1e-10)
            phi[u] = phi_i
            sigma2[u] = sigma2_i

        A = np.zeros((k_long, k_long))
        b = np.zeros(k_long)
        for u in units:
            p = pre[u]
            phi_i, s2_i = phi[u], sigma2[u]
            A += (phi_i**2 / s2_i) * (p["X"].T @ p["H_X"])
            rhs_inner = p["H_dy"] - phi_i * p["H_ylag"]
            b += (phi_i / s2_i) * (p["X"].T @ rhs_inner)

        try:
            theta = -np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            warnings.warn("Matriz A singular na iteração {}. Parando.".format(it))
            break

        delta = np.max(np.abs(theta - theta_old) / (np.abs(theta_old) + 1e-8))
        if delta < tol:
            converged = True
            break

    # Coeficientes de curto prazo por unidade após convergência
    sr_individual = []
    for u in units:
        p = pre[u]
        xi = p["y_lag"] - p["X"] @ theta
        residual = p["dy"] - phi[u] * xi
        if p["W_i"].shape[1] > 0:
            sr_full, *_ = np.linalg.lstsq(p["W_i"], residual, rcond=None)
            sr_individual.append(sr_full[:-1])

    sr_matrix = np.array(sr_individual) if sr_individual else np.empty((N, 0))
    sr_mean = sr_matrix.mean(axis=0) if sr_matrix.size else np.array([])
    sr_se = (
        np.sqrt(((sr_matrix - sr_mean) ** 2).sum(axis=0) / (N * (N - 1)))
        if sr_matrix.size else np.array([])
    )

    # VCOV de θ: inversa da Hessiana concentrada
    try:
        vcov_theta = np.linalg.inv(A)
        theta_se = np.sqrt(np.diag(vcov_theta))
    except np.linalg.LinAlgError:
        vcov_theta = np.full((k_long, k_long), np.nan)
        theta_se = np.full(k_long, np.nan)

    # SE de φ via variância empírica entre unidades
    phi_arr = np.array([phi[u] for u in units])
    phi_mean = phi_arr.mean()
    phi_se = np.sqrt(((phi_arr - phi_mean) ** 2).sum() / (N * (N - 1)))

    # Log-verossimilhança concentrada no ótimo
    ll = 0.0
    for u in units:
        p = pre[u]
        s2 = sigma2[u]
        T_i = p["T_i"]
        xi = p["y_lag"] - p["X"] @ theta
        e = p["dy"] - phi[u] * xi
        H_e = p["H_dy"] - phi[u] * (p["H_ylag"] - p["H_X"] @ theta)
        ll += -0.5 * T_i * np.log(2 * np.pi * s2) - 0.5 * (e @ H_e) / s2

    n_obs = sum(pre[u]["T_i"] for u in units)

    return PooledMeanGroup(
        long_run_coef=theta,
        long_run_se=theta_se,
        long_run_vcov=vcov_theta,
        speed_adjust=phi_mean,
        speed_adjust_se=phi_se,
        speed_adjust_individual=phi,
        short_run_coef=sr_mean,
        short_run_se=sr_se,
        sigma2_individual=sigma2,
        log_likelihood=ll,
        n_iterations=it + 1,
        converged=converged,
        n_units=N,
        n_obs=n_obs,
        dep_var=design["dep_var"],
        long_run_names=design["long_run_names"],
        short_run_names=design["short_run_names"],
    )
