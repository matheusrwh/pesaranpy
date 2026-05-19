import warnings
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class MeanGroup:
    long_run_coef: np.ndarray
    long_run_se: np.ndarray
    speed_adjust: float
    speed_adjust_se: float
    short_run_coef: np.ndarray
    short_run_se: np.ndarray
    individual_coef: dict
    n_units: int
    n_obs: int
    dep_var: str
    long_run_names: list
    short_run_names: list


def estimate_mean_group(design: dict) -> MeanGroup:
    """
    Calcula o estimador de mean group a partir das matrizes de regressores.
    """

    by_unit = design['by_unit']
    units = list(by_unit.keys())
    k_long = len(design["long_run_names"])

    individual_coef = {}
    long_run_matrix = []
    speed_adjust_list = []
    short_run_matrix = []

    for u in units:
        d = by_unit[u]
        X = np.column_stack([
            np.ones(d['T_eff']),
            d['y_lag'][:, None],
            d['X_long'],
            d['SR'],
        ])

        y = d['dy']

        k = X.shape[1]
        if d['T_eff'] <= k:
            warnings.warn(
                f"Unidade '{u}' descartada: T_eff={d['T_eff']} não excede o número de regressores k={k}.",
                RuntimeWarning,
                stacklevel=2
            )
            continue

        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        phi_i = beta[1]
        psi_i = beta[2:2 + k_long]
        sr_i = beta[2 + k_long: 2 + k_long + d['SR'].shape[1]]

        if abs(phi_i) < 9e-10:
            warnings.warn(
                f"Unidade '{u}' descartada: |φ_i| = {abs(phi_i):.2e} está abaixo do limiar numérico.",
                RuntimeWarning,
                stacklevel=2
            )
            continue

        theta_i = -psi_i / phi_i

        individual_coef[u] = {
            'phi': phi_i,
            'theta': theta_i,
            'sr': sr_i
        }

        long_run_matrix.append(theta_i)
        speed_adjust_list.append(phi_i)
        short_run_matrix.append(sr_i)

    N = len(long_run_matrix)
    if N == 0:
        k_regs = 1 + 1 + k_long + (by_unit[units[0]]["SR"].shape[1] if units else 0)
        raise RuntimeError(
            f"MG estimator: all {len(units)} units were discarded. "
            f"Every unit has T_eff <= k={k_regs} regressors. "
            f"Reduce p or q, add more time periods, or use fewer long_run_vars."
        )

    long_run_matrix = np.array(long_run_matrix)
    speed_adjust_arr = np.array(speed_adjust_list)
    short_run_matrix = np.array(short_run_matrix)

    theta_mg = long_run_matrix.mean(axis=0)
    phi_mg = speed_adjust_arr.mean()
    sr_mg = short_run_matrix.mean(axis=0)

    # Erro-padrão de Pesaran-Smith (1995): variância empírica entre unidades / N
    theta_se = np.sqrt(((long_run_matrix - theta_mg) ** 2).sum(axis=0) / (N * (N - 1)))
    phi_se = np.sqrt(((speed_adjust_arr - phi_mg) ** 2).sum() / (N * (N - 1)))
    sr_se = np.sqrt(((short_run_matrix - sr_mg) ** 2).sum(axis=0) / (N * (N - 1)))

    n_obs = sum(by_unit[u]["T_eff"] for u in individual_coef)

    return MeanGroup(
        long_run_coef=theta_mg,
        long_run_se=theta_se,
        speed_adjust=phi_mg,
        speed_adjust_se=phi_se,
        short_run_coef=sr_mg,
        short_run_se=sr_se,
        individual_coef=individual_coef,
        n_units=N,
        n_obs=n_obs,
        dep_var=design["dep_var"],
        long_run_names=design["long_run_names"],
        short_run_names=design["short_run_names"]
    )
