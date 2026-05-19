"""
pesaranpy — Estimadores MG e PMG de Pesaran para painéis cointegrados.

Referências principais
----------------------
Pesaran, M. H., & Smith, R. (1995). Estimating long-run relationships from
    dynamic heterogeneous panels. Journal of Econometrics, 68(1), 79–113.

Pesaran, M. H., Shin, Y., & Smith, R. P. (1999). Pooled mean group estimation
    of dynamic heterogeneous panels. Journal of the American Statistical
    Association, 94(446), 621–634.
"""

from pesaranpy.utils.data import PanelData, prepare_panel, build_ecm_design
from pesaranpy.estimators.mg import MeanGroup, estimate_mean_group
from pesaranpy.estimators.pmg import PooledMeanGroup, estimate_pmg
from pesaranpy.diagnostics.hausman import HausmanResult, hausman_test

__version__ = "0.1.0"

__all__ = [
    # Preparação de dados
    "PanelData",
    "prepare_panel",
    "build_ecm_design",
    # Estimadores
    "MeanGroup",
    "estimate_mean_group",
    "PooledMeanGroup",
    "estimate_pmg",
    # Diagnóstico
    "HausmanResult",
    "hausman_test",
]
