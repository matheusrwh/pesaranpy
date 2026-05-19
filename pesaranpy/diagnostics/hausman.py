"""
Teste de Hausman para escolha entre estimadores PMG, MG e DFE.

H₀ : θ_MG = θ_PMG (homogeneidade de longo prazo válida → PMG eficiente)
H₁ : θ_MG ≠ θ_PMG (heterogeneidade → apenas MG consistente)

Estatística:
    H = (θ̂_MG - θ̂_PMG)' [V(θ̂_MG) - V(θ̂_PMG)]⁻¹ (θ̂_MG - θ̂_PMG)
    ~ χ²(k)  sob H₀, onde k = dim(θ).

Tratamento de matriz não-PSD: usa pseudo-inversa de Moore-Penrose com
ajuste do rank nos graus de liberdade.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class HausmanResult:
    statistic: float
    p_value: float
    df: int
    diff_coefs: np.ndarray
    long_run_names: list
    note: str


def hausman_test(
    efficient_results,
    consistent_results,
    label_efficient: str = "PMG",
    label_consistent: str = "MG",
) -> HausmanResult:
    """Aplica o teste de Hausman.

    Parâmetros
    ----------
    efficient_results : objeto com .long_run_coef e .long_run_vcov (PMG ou DFE).
    consistent_results : objeto com .long_run_coef e variância empírica
        entre unidades (MG).

    Para o MG, a matriz de covariância dos coeficientes médios não é
    fornecida diretamente; ela deve ser construída a partir dos coeficientes
    individuais. Esta função aceita um atributo `.long_run_vcov` ou,
    em sua ausência, calcula a matriz a partir de `.individual_coef`.
    """
    theta_eff = efficient_results.long_run_coef
    theta_con = consistent_results.long_run_coef

    V_eff = efficient_results.long_run_vcov

    # Construir V_MG empírica entre unidades
    if hasattr(consistent_results, "long_run_vcov"):
        V_con = consistent_results.long_run_vcov
    else:
        thetas = np.array([
            d["theta"] for d in consistent_results.individual_coef.values()
        ])
        N = len(thetas)
        diff = thetas - thetas.mean(axis=0)
        V_con = (diff.T @ diff) / (N * (N - 1))

    diff = theta_con - theta_eff
    M = V_con - V_eff

    # Verificar se M é positiva definida
    eigvals = np.linalg.eigvalsh(M)
    note = ""
    if np.any(eigvals < -1e-10):
        # Matriz não-PSD: usar pseudo-inversa
        M_inv = np.linalg.pinv(M)
        rank = np.sum(eigvals > 1e-10)
        note = (
            f"Matriz V_MG - V_eff não positiva definida (autovalores mínimos = "
            f"{eigvals.min():.4e}). Usada pseudo-inversa com rank {rank}."
        )
    else:
        M_inv = np.linalg.inv(M)
        rank = len(eigvals)

    H = float(diff @ M_inv @ diff)
    p_value = 1.0 - stats.chi2.cdf(H, df=rank)

    return HausmanResult(
        statistic=H,
        p_value=p_value,
        df=rank,
        diff_coefs=diff,
        long_run_names=list(efficient_results.long_run_names),
        note=note,
    )
