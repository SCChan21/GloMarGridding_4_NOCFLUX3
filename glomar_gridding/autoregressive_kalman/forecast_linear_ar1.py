"""
Conduct local AR1 forecast of uncertain data
output new uncertainities
"""

import numpy as np


class Autoregressive1Forecast:
    """
    Class to compute AR1 forecast

    :param independent_var_t: 1D vector of data for t=t
    :type independent_var_t: np.ndarray

    :param errcov_independent_var_t: 2D error covariance for independent_var_t
    :type errcov_independent_var_t: np.ndarray

    :param lag_1_autocor_for_t_plus_1: 1D vector of lag-1 autocorrelation
    :type lag_1_autocor_for_t_plus_1: np.ndarray

    :param climatology_mean: climatological mean,
        shape == independent_var_t
    :type climatology_mean: np.ndarray

    :param climatology_variance: climatological variance,
        shape == independent_var_t
    :type climatology_mean: np.ndarray

    :param climatology_variance_is_sdev: Flag indicating if
        climatology_variance is variance or standard deviation
        default False, climatology_variance is variance (like SST**2)
    :type climatology_variance_is_sdev: bool
    """

    def __init__(
            self,
            independent_var_t: np.ndarray,
            errcov_independent_var_t: np.ndarray,
            lag_1_autocor: np.ndarray,
            climatology_mean: np.ndarray,
            climatology_variance: np.ndarray,
            climatology_variance_is_sdev: bool = False,
    ):
        #
        self.independent_var_t = independent_var_t
        self.errcov_independent_var_t = errcov_independent_var_t
        self.lag_1_autocor = lag_1_autocor
        self.climatology_mean = climatology_mean
        if climatology_variance_is_sdev:
            self.climatology_variance = np.square(climatology_variance)
        else:
            self.climatology_variance = climatology_variance
        self._check_args()

    def _check_args(self):
        """Check attributes set on init"""
        if len(self.independent_var_t.shape) != 1:
            raise ValueError("independent_var_t should be 1D")
        if len(self.errcov_independent_var_t.shape) != 2:
            raise ValueError("errcov_independent_var_t should be 2D")
        if self.errcov_independent_var_t.shape[0] != self.errcov_independent_var_t.shape[1]:  # noqa: E501
            raise ValueError("errcov_independent_var_t is not square")
        if len(self.lag_1_autocor.shape) != 1:
            raise ValueError("lag_1_autocor_for_t_plus_1 should be 1D")
        if len(self.climatology_mean.shape) != 1:
            raise ValueError("climatology_mean should be 1D")
        if len(self.climatology_variance.shape) != 1:
            raise ValueError("climatology_variance should be 1D")
        #
        if self.independent_var_t.shape[0] != self.errcov_independent_var_t.shape[0]:  # noqa: E501
            raise ValueError("independent_var_t shape inconsistent with errcov_independent_var_t")  # noqa: E501
        if self.independent_var_t.shape[0] != self.climatology_mean.shape[0]:
            raise ValueError("independent_var_t shape inconsistent with climatology_mean")  # noqa: E501
        if self.climatology_variance.shape[0] != self.climatology_mean.shape[0]:
            raise ValueError("climatology_variance shape inconsistent with climatology_mean")  # noqa: E501

    def compute_forecast(self):
        """Calls forecast_t_plus_1"""
        ans = forecast_t_plus_1(
            self.independent_var_t,
            self.errcov_independent_var_t,
            self.lag_1_autocor,
            self.climatology_mean,
            self.climatology_variance
        )
        self.forecast = ans[0]
        self.errcov = ans[1]


def forecast_t_plus_1(
    independent_var_t: np.ndarray,
    errcov_independent_var_t: np.ndarray,
    lag_1_autocor: np.ndarray,
    climatology_mean: np.ndarray,
    climatology_variance: np.ndarray,):
    """
    Compute AR1 forecast and estimate uncertainities

    :param independent_var_t: 1D vector of independent variables for t
    :type independent_var_t: np.ndarray

    :param errcov_independent_var_t: 2D errcov for independent_var_t
    :type errcov_independent_var_t: np.ndarray

    :param lag_1_autocor: 1D vector of lag correlation
    :type lag_1_autocor: np.ndarray

    :param climatology_mean: 1D climatological mean for independent_var
    :type climatology_mean: np.ndarray

    :param climatology_variance: 1D climatological variance for independent_var
    :type climatology_variance: np.ndarray

    :returns: list with AR1 forecast and error covariance
    :rtype: list
    """
    #
    ar1_matrix = np.diag(lag_1_autocor)
    #
    diff_with_climatology = independent_var_t - climatology_mean
    forecast_t_plus_1_anomaly = ar1_matrix @ diff_with_climatology
    forecast_t_plus_1_anomaly += climatology_mean
    #
    climvar_mult = np.eye(ar1_matrix.shape[0]) - ar1_matrix @ ar1_matrix
    errcov_clim = climvar_mult @ climatology_variance
    errcov_uncert_ind_var = ar1_matrix @ errcov_independent_var_t
    errcov = errcov_clim + errcov_uncert_ind_var
    #
    ans = [forecast_t_plus_1_anomaly, errcov]
    return ans


def main():
    """MAIN"""
    print("===MAIN===")


if __name__ == "__main__":
    main()
