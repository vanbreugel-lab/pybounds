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

.. autofunction:: pybounds.compute_observability

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
