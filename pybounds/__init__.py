
from .simulator import Simulator

from .observability import EmpiricalObservabilityMatrix
from .observability import SlidingEmpiricalObservabilityMatrix
from .observability import FisherObservability
from .observability import SlidingFisherObservability
from .observability import ObservabilityMatrixImage
from .observability import transform_states
from .observability import compute_observability

from .analysis import ObservabilityAnalysis, Linearization, SlidingO

from . import stochastic   # the stochastic recursions live in this namespace only (pybounds.stochastic.X)

from .jacobian import SymbolicJacobian

from .util import colorline, plot_heatmap_log_timeseries

try:
    from .jax_simulator import JaxSimulator, JaxEmpiricalObservabilityMatrix, JaxSlidingEmpiricalObservabilityMatrix
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

__all__ = [
    'Simulator',
    'EmpiricalObservabilityMatrix',
    'SlidingEmpiricalObservabilityMatrix',
    'FisherObservability',
    'SlidingFisherObservability',
    'ObservabilityMatrixImage',
    'transform_states',
    'compute_observability',
    'ObservabilityAnalysis',
    'Linearization',
    'SlidingO',
    'stochastic',
    'SymbolicJacobian',
    'colorline',
    'plot_heatmap_log_timeseries',
]

if _JAX_AVAILABLE:
    __all__ += [
        'JaxSimulator',
        'JaxEmpiricalObservabilityMatrix',
        'JaxSlidingEmpiricalObservabilityMatrix',
    ]
