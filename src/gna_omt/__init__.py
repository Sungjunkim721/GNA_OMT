"""Global Neural Adjoint for optical multilayer thin films."""
from .data import ThinFilmDataset
from .initialization import Population, initialize_population
from .inverse_design import GNAResult, run_gna
from .losses import GNALossTerms, gna_loss
from .model import LinearEmbedding, OMTMixer, material_normalization

__all__ = [
    "ThinFilmDataset",
    "Population",
    "initialize_population",
    "GNAResult",
    "run_gna",
    "GNALossTerms",
    "gna_loss",
    "LinearEmbedding",
    "OMTMixer",
    "material_normalization",
]
