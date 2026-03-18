import numpy as np
import pytest
import pybounds


class TestFisherReturnTypes:

    def test_returns_three_tuple(self, fisher_obs):
        result = fisher_obs.get_fisher_information()
        assert len(result) == 3

    def test_F_is_dataframe(self, fisher_obs):
        import pandas as pd
        F, _, _ = fisher_obs.get_fisher_information()
        assert isinstance(F, pd.DataFrame)

    def test_F_inv_is_dataframe(self, fisher_obs):
        import pandas as pd
        _, F_inv, _ = fisher_obs.get_fisher_information()
        assert isinstance(F_inv, pd.DataFrame)

    def test_R_is_dataframe(self, fisher_obs):
        import pandas as pd
        _, _, R = fisher_obs.get_fisher_information()
        assert isinstance(R, pd.DataFrame)


class TestFisherShapes:

    def test_F_shape(self, fisher_obs):
        F, _, _ = fisher_obs.get_fisher_information()
        assert F.shape == (2, 2)

    def test_F_inv_shape(self, fisher_obs):
        _, F_inv, _ = fisher_obs.get_fisher_information()
        assert F_inv.shape == (2, 2)

    def test_F_columns(self, fisher_obs):
        F, _, _ = fisher_obs.get_fisher_information()
        assert list(F.columns) == ['g', 'd']

    def test_F_index(self, fisher_obs):
        F, _, _ = fisher_obs.get_fisher_information()
        assert list(F.index) == ['g', 'd']

    def test_F_inv_columns(self, fisher_obs):
        _, F_inv, _ = fisher_obs.get_fisher_information()
        assert list(F_inv.columns) == ['g', 'd']


class TestFisherMathematicalProperties:

    def test_F_is_symmetric(self, fisher_obs):
        F, _, _ = fisher_obs.get_fisher_information()
        assert np.allclose(F.values, F.values.T, atol=1e-10)

    def test_F_is_positive_semidefinite(self, fisher_obs):
        F, _, _ = fisher_obs.get_fisher_information()
        eigenvalues = np.linalg.eigvalsh(F.values)
        assert np.all(eigenvalues >= -1e-10), \
            f"F has negative eigenvalues: {eigenvalues}"

    def test_F_inv_is_symmetric(self, fisher_obs):
        _, F_inv, _ = fisher_obs.get_fisher_information()
        assert np.allclose(F_inv.values, F_inv.values.T, atol=1e-8)

    def test_regularized_F_times_F_inv_is_identity(self, fisher_obs):
        """(F + lam*I) @ F_inv should be approximately I."""
        F, F_inv, _ = fisher_obs.get_fisher_information()
        lam = 1e-8
        F_reg = F.values + lam * np.eye(2)
        product = F_reg @ F_inv.values
        assert np.allclose(product, np.eye(2), atol=1e-6)

    def test_error_variance_is_positive(self, fisher_obs):
        assert np.all(fisher_obs.error_variance.values > 0)

    def test_error_variance_has_state_columns(self, fisher_obs):
        assert 'g' in fisher_obs.error_variance.columns
        assert 'd' in fisher_obs.error_variance.columns


class TestFisherParameterEffects:

    def test_R_dict_sets_diagonal(self, eom):
        """R={'r': 0.1} should give diagonal entries of 0.1."""
        FO = pybounds.FisherObservability(eom.O_df, R={'r': 0.1}, lam=1e-8)
        _, _, R = FO.get_fisher_information()
        diag_vals = np.diag(R.values)
        assert np.allclose(diag_vals, 0.1, atol=1e-10)

    def test_larger_R_increases_error_variance(self, eom):
        """More sensor noise → larger minimum error variance (Cramér-Rao)."""
        FO_low = pybounds.FisherObservability(eom.O_df, R={'r': 0.01}, lam=1e-8)
        FO_high = pybounds.FisherObservability(eom.O_df, R={'r': 1.0}, lam=1e-8)
        ev_low = FO_low.error_variance.values
        ev_high = FO_high.error_variance.values
        assert np.all(ev_high > ev_low)

    def test_invalid_O_type_raises(self, eom):
        """Passing a plain list (no .shape) raises AttributeError before the isinstance check."""
        with pytest.raises((TypeError, AttributeError)):
            pybounds.FisherObservability([[1, 2], [3, 4]], R=0.1, lam=1e-8)
