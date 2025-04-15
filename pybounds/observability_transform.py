
# from .observability import EmpiricalObservabilityMatrix
#
#
# class DroneSimulator(EmpiricalObservabilityMatrix):
#     def __init__(self, dt=0.1, mpc_horizon=10, r_u=1e-2, input_mode='direct', control_mode='velocity_body_level'):
#         self.dynamics = DroneModel()
#         super().__init__(self.dynamics.f, self.dynamics.h, dt=dt, mpc_horizon=mpc_horizon,
#                          state_names=self.dynamics.state_names,
#                          input_names=self.dynamics.input_names,
#                          measurement_names=self.dynamics.measurement_names)