"""
Validação e preparação de painéis dinâmicos para estimação ECM/ARDL.

Replica a lógica de xtset + tsset do Stata, garantindo que cada unidade
tenha série temporal contígua (sem gaps internos), o que é requisito para
construção correta dos lags em modelos dinâmicos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PanelData:
    """Container imutável para um painel longitudinal preparado.

    Atributos
    ---------
    df : pd.DataFrame
        Painel ordenado por (id_unit, id_time).
    id_unit : str
        Nome da coluna identificadora da unidade cross-section (ex: 'pais', 'uf').
    id_time : str
        Nome da coluna temporal (ex: 'ano').
    units : np.ndarray
        Vetor de identificadores únicos das unidades.
    n_units : int
        Número de unidades (N).
    t_per_unit : dict
        Mapeia cada unidade ao número de observações disponíveis.
    is_balanced : bool
        True se todas as unidades têm o mesmo número de observações.
    """

    df: pd.DataFrame
    id_unit: str
    id_time: str
    units: np.ndarray
    n_units: int
    t_per_unit: dict
    is_balanced: bool


def prepare_panel(
    df: pd.DataFrame,
    id_unit: str,
    id_time: str,
    variables: list[str],
    drop_missing: bool = True,
) -> PanelData:
    """Valida e prepara um DataFrame para estimação.

    Verifica:
      1. Unicidade do par (id_unit, id_time) — não pode haver duplicatas.
      2. Regularidade temporal de cada unidade — não pode haver gaps internos
         na sequência temporal, pois isso quebra a construção dos lags.
      3. Existência de variáveis especificadas.

    Parâmetros
    ----------
    df : DataFrame em formato longo.
    id_unit : nome da coluna de identificação cross-section.
    id_time : nome da coluna temporal (deve ser numérica ou ordenável).
    variables : lista das variáveis a serem usadas (dependente + regressores).
    drop_missing : se True, remove unidades com qualquer NaN nas variáveis.

    Retorna
    -------
    PanelData : container com o painel limpo e metadados.

    Notas
    -----
    Unidades com gaps internos na dimensão temporal geram um ValueError.
    Diferenças calculadas através de um gap representam variações multi-período
    e invalidam a construção do ECM. O comportamento seguro é rejeitar o painel
    e exigir tratamento explícito pelo usuário antes de prosseguir.
    """

    # 1. Cópia defensiva e validação básica
    if id_unit not in df.columns or id_time not in df.columns:
        raise KeyError(f"Colunas '{id_unit}' ou '{id_time}' ausentes no DataFrame.")

    missing_vars = [v for v in variables if v not in df.columns]
    if missing_vars:
        raise KeyError(f"Variáveis ausentes: {missing_vars}")

    panel = df[[id_unit, id_time] + variables].copy()
    panel = panel.sort_values([id_unit, id_time]).reset_index(drop=True)

    # 2. Unicidade do par (i, t)
    if panel.duplicated(subset=[id_unit, id_time]).any():
        dups = panel[panel.duplicated(subset=[id_unit, id_time], keep=False)]
        raise ValueError(f"Painel possui duplicatas em ({id_unit}, {id_time}):\n{dups.head()}")

    # 3. Regularidade temporal: detectar gaps internos
    def _has_internal_gap(series_t: pd.Series) -> bool:
        """Detecta gaps em série temporal (passo regular, qualquer frequência)."""
        if len(series_t) < 2:
            return False
        diffs = np.diff(np.sort(series_t.values).astype(float))
        return not np.allclose(diffs, diffs[0])

    units_with_gaps = []
    for u, grp in panel.groupby(id_unit):
        if _has_internal_gap(grp[id_time]):
            units_with_gaps.append(u)
    if units_with_gaps:
        raise ValueError(
            f"Unidades com gaps internos na dimensão temporal: {units_with_gaps}. "
            "Modelos ECM/ARDL exigem séries contíguas. Considere preencher "
            "ou remover estas unidades antes de prosseguir."
        )

    # 4. Tratamento de NaN
    if drop_missing:
        units_complete = panel.groupby(id_unit)[variables].apply(
            lambda g: g.notna().all().all()
        )
        keep_units = units_complete[units_complete].index.tolist()
        panel = panel[panel[id_unit].isin(keep_units)].reset_index(drop=True)

    units = panel[id_unit].unique()
    t_per_unit = panel.groupby(id_unit).size().to_dict()
    is_balanced = len(set(t_per_unit.values())) == 1

    return PanelData(
        df=panel,
        id_unit=id_unit,
        id_time=id_time,
        units=units,
        n_units=len(units),
        t_per_unit=t_per_unit,
        is_balanced=is_balanced,
    )


def build_ecm_design(
    panel: PanelData,
    dep_var: str,
    long_run_vars: list[str],
    short_run_vars: list[str] | None = None,
    p: int = 1,
    q: int = 1,
) -> dict:
    """Constrói as matrizes de design para a forma ECM da ARDL(p, q).

    A reparametrização adotada é a canônica de Pesaran, Shin e Smith (1999):

        Δy_{it} = φ_i (y_{i,t-1} - θ_i' x_{it}) + Σ λ*_{ij} Δy_{i,t-j}
                  + Σ δ*_{ij}' Δx_{i,t-j} + μ_i + ε_{it}

    Parâmetros
    ----------
    panel : PanelData previamente validado.
    dep_var : nome da variável dependente y.
    long_run_vars : vetor de regressores que entram no termo de cointegração.
    short_run_vars : regressores adicionais que entram apenas nas diferenças
        (úteis para variáveis exógenas como dummies de evento). Default: None.
    p : ordem AR da especificação ARDL na variável dependente.
    q : ordem do polinômio de defasagens das variáveis explicativas.

    Retorna
    -------
    dict com:
        'by_unit': dicionário {unidade: (Δy, EC_lag, ΔX_short, μ_const)}
        'var_names': nomes das colunas das matrizes de curto e longo prazo.

    Notas
    -----
    A construção é feita por unidade i para preservar a estrutura cross-section
    necessária aos estimadores PMG/MG (que estimam coeficientes individuais
    como passo intermediário). Após a diferenciação, cada unidade perde
    max(p, q) observações iniciais.
    """
    df = panel.df
    by_unit = {}
    n_lost = max(p, q)

    short_run_vars = short_run_vars or []

    for unit, grp in df.groupby(panel.id_unit):
        grp = grp.sort_values(panel.id_time).reset_index(drop=True)
        T = len(grp)
        if T <= n_lost + 1:
            continue

        y = grp[dep_var].values
        X_long = grp[long_run_vars].values  # (T, k_long)

        dy = np.diff(y)  # (T-1,)

        # Componente EC: y_{i,t-1} e x_{i,t} (convenção xtpmg — Blackburne & Frank 2007)
        y_lag = y[:-1]
        X_long_curr = X_long[1:, :]

        # Diferenças defasadas de y: Δy_{t-1}, ..., Δy_{t-(p-1)}
        dy_lags = np.column_stack([
            np.concatenate([np.full(j, np.nan), dy[:-j]])
            for j in range(1, p)
        ]) if p > 1 else np.empty((len(dy), 0))

        # Diferenças contemporâneas e defasadas de x: Δx_t, ..., Δx_{t-(q-1)}
        dX = np.diff(X_long, axis=0)  # (T-1, k_long)
        dX_lags_list = [dX]
        for j in range(1, q):
            lagged = np.vstack([np.full((j, dX.shape[1]), np.nan), dX[:-j, :]])
            dX_lags_list.append(lagged)
        dX_block = np.column_stack(dX_lags_list)

        # Variáveis exógenas em curto prazo (entram só em diferenças)
        if short_run_vars:
            X_short_levels = grp[short_run_vars].values
            dX_short = np.diff(X_short_levels, axis=0)
        else:
            dX_short = np.empty((len(dy), 0))

        SR = np.column_stack([dy_lags, dX_block, dX_short])

        valid = ~np.isnan(np.column_stack([dy, y_lag, X_long_curr, SR])).any(axis=1)
        by_unit[unit] = {
            "dy": dy[valid],
            "y_lag": y_lag[valid],
            "X_long": X_long_curr[valid],
            "SR": SR[valid],
            "T_eff": int(valid.sum()),
        }

    sr_names = (
        [f"D{dep_var}_L{j}" for j in range(1, p)]
        + [f"D{v}_L{j}" for j in range(0, q) for v in long_run_vars]
        + [f"D{v}" for v in short_run_vars]
    )

    return {
        "by_unit": by_unit,
        "long_run_names": list(long_run_vars),
        "short_run_names": sr_names,
        "dep_var": dep_var,
    }
