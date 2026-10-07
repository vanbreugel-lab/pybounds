API Reference
=============

Simulation
----------

.. autoclass:: pybounds.Simulator
   :members:
   :show-inheritance:

Observability
-------------

.. autoclass:: pybounds.EmpiricalObservabilityMatrix
   :members:
   :show-inheritance:

.. autoclass:: pybounds.SlidingEmpiricalObservabilityMatrix
   :members:
   :show-inheritance:

.. autoclass:: pybounds.FisherObservability
   :members:
   :show-inheritance:

.. autoclass:: pybounds.SlidingFisherObservability
   :members:
   :show-inheritance:

.. autoclass:: pybounds.ObservabilityAnalysis
   :members:

.. autofunction:: pybounds.compute_observability

.. autoclass:: pybounds.Linearization

.. autoclass:: pybounds.SlidingO

Stochastic observability and constructability
---------------------------------------------

.. automodule:: pybounds.stochastic
   :members: stochastic_observability_gramian, stochastic_constructability_gramian,
             deterministic_observability_gramian, duality_check, process_covariance, linearize,
             sliding_gramians, window_observability_matrix

Visualisation
-------------

.. autoclass:: pybounds.ObservabilityMatrixImage
   :members:
   :show-inheritance:

.. autofunction:: pybounds.colorline

.. autofunction:: pybounds.plot_heatmap_log_timeseries

Utilities
---------

.. autoclass:: pybounds.SymbolicJacobian
   :members:
   :show-inheritance:

.. autofunction:: pybounds.transform_states

JAX Backend (optional)
----------------------

The JAX backend provides exact autodiff Jacobians via ``jax.jacfwd`` and
batched window computation via ``jax.vmap``.  Requires ``pip install
pybounds[jax]`` and dynamics/measurement functions written with
``jax.numpy``.

.. autoclass:: pybounds.JaxSimulator
   :members:
   :show-inheritance:

.. autoclass:: pybounds.JaxEmpiricalObservabilityMatrix
   :members:
   :show-inheritance:

.. autoclass:: pybounds.JaxSlidingEmpiricalObservabilityMatrix
   :members:
   :show-inheritance:
