"""Compute inverse variance weighted average"""

import numpy as np
from numpy import linalg


def compute_inverse_via_solve(square_matrix: np.ndarray) -> np.ndarray:
    """
    Solves the linear system for cov_inv
    cov_inv @ cov = np.eye
    """
    arr_shape = square_matrix.shape
    if len(arr_shape) != 2:
        raise ValueError("square_matrix is not a 2D matrix.")
    if arr_shape[0] != arr_shape[1]:
        raise ValueError("square_matrix is not square matrix")
    the_eye = np.eye(arr_shape[0])
    ans = linalg.solve(square_matrix, the_eye)
    return ans


def check_1d(a: np.ndarray):
    """Check if array a is 1D, 2D or something invalid"""
    if len(a.shape) == 1:
        return True
    elif len(a.shape) == 2:
        if a.shape[0] == a.shape[1]:
            return False
    raise ValueError('a is not 1 or 2D')


class KalmanOut:
    """
    class to compute blended forecast and observations
    variable names follows compute_inv_variance_wgt_mean_kalman

    :param forecast_vector: 1D vector of forecasts
    :type forecast_vector: np.ndarray
    :param obs_vector: 1D vector of (gridded) observations
    :type obs_vector: np.ndarray
    :param errcov_forecast: 2D matrix of error covariance for forecast_vector
    :type errcov_forecast: np.ndarray
    :param errcov_obs: 2D matrix of error covariance for obs_vector
    :type errcov_forecast: np.ndarray
    :param cov_forecast_and_obs: covariance between forecast & observations
    :type cov_forecast_and_obs: np.ndarray

    :param ez_covariances: ignore off-diagonals of errcov_forecast,
        errcov_obs and cov_forecast_and_obs if set to True, default True
    :type ez_covariances: bool

    """

    def __init__(
            self,
            forecast_vector: np.ndarray,
            obs_vector: np.ndarray,
            errcov_forecast: np.ndarray,
            errcov_obs: np.ndarray,
            cov_forecast_and_obs: np.ndarray,
            ez_covariances: bool = True,
            ):
        self.forecast_vector = forecast_vector
        self.obs_vector = obs_vector
        if ez_covariances:
            if check_1d(errcov_forecast):
                self.errcov_forecast = errcov_forecast
            else:
                self.errcov_forecast = np.diag(errcov_forecast)
            if check_1d(errcov_obs):
                self.errcov_obs = errcov_obs
            else:
                self.errcov_obs = np.diag(errcov_obs)
            if check_1d(cov_forecast_and_obs):
                self.cov_forecast_and_obs = cov_forecast_and_obs
            else:
                self.cov_forecast_and_obs = np.diag(cov_forecast_and_obs)
            self.inv_operator = np.reciprocal
            self.multiply_operator = np.multiply
            self.one_maker = np.ones
        else:
            self.errcov_forecast = errcov_forecast
            self.errcov_obs = errcov_obs
            self.cov_forecast_and_obs = cov_forecast_and_obs
            self.inv_operator = compute_inverse_via_solve
            self.multiply_operator = np.matmul
            self.one_maker = np.eye

    def compute_outputs(self):
        """Calls compute_inv_variance_wgt_mean_kalman"""
        ans = compute_inv_variance_wgt_mean_kalman(
            self.forecast_vector,
            self.obs_vector,
            self.errcov_forecast,
            self.errcov_obs,
            self.cov_forecast_and_obs,
            inv_operator=self.inv_operator,
            multiply_operator=self.multiply_operator,
            one_maker=self.one_maker,
        )
        self.wgt_mean = ans[0]
        self.errcov = ans[1]
        self.kalman_gain_from_new_obs = ans[2]
        self.wgts_from_ar_forecast = ans[3]


def compute_inv_variance_wgt_mean_kalman(
        forecast_vector: np.ndarray,
        obs_vector: np.ndarray,
        errcov_forecast: np.ndarray,
        errcov_obs: np.ndarray,
        cov_forecast_and_obs: np.ndarray,
        inv_operator: callable = compute_inverse_via_solve,
        multiply_operator: callable = np.matmul,
        one_maker: callable = np.eye,
    ):
    """
    Compute the inverse variance weighted average of
    forecast_vector & obs_vector using provided error covariances
    errcov_forecast & errcov_obs

    :param forecast_vector: 1D vector of forecasts
    :type forecast_vector: np.ndarray
    :param obs_vector: 1D vector of (gridded) observations
    :type obs_vector: np.ndarray
    :param errcov_forecast: 2D matrix of error covariance for forecast_vector
    :type errcov_forecast: np.ndarray
    :param errcov_obs: 2D matrix of error covariance for obs_vector
    :type errcov_forecast: np.ndarray
    :param cov_forecast_and_obs: 2D covariance between forecast & observations
    :type cov_forecast_and_obs: np.ndarray

    :returns: list with weighted avg and error covariance
    :rtype: list
    """
    #
    print('Computing inverse errcov_forecast')
    inv_errcov_forecast = inv_operator(errcov_forecast)
    print('Computing inverse errcov_obs')
    inv_errcov_obs = inv_operator(errcov_obs)
    #
    forecast_vector_shape = forecast_vector.shape
    if len(forecast_vector_shape) != 1:
        raise ValueError("forecast_vector is not 1D vector")
    if errcov_forecast.shape[0] != forecast_vector_shape[0]:
        raise ValueError("forecast_vector shape inconsistent with errcov_forecast")  # noqa: E501
    #
    obs_vector_shape = obs_vector.shape
    if len(obs_vector_shape) != 1:
        raise ValueError("obs_vector is not 1D vector")
    if errcov_obs.shape[0] != obs_vector_shape[0]:
        raise ValueError("obs_vector shape inconsistent with errcov_obs")
    #
    if forecast_vector_shape[0] != obs_vector_shape[0]:
        raise ValueError("obs_vector shape inconsistent with forecast_vector")
    #
    # Weights and error covariance if obs and forecast are uncorrelated
    print('Computing c_hat')
    c_hat = inv_operator(inv_errcov_forecast + inv_errcov_obs)
    print('Computing kalman_gain')
    kalman_gain = multiply_operator(c_hat, inv_errcov_obs)
    print('Computing forecast_wgt')
    forecast_wgt = multiply_operator(c_hat, inv_errcov_forecast)
    #
    # Output weighted mean
    print('Computing weighted mean')
    wgt_mean = multiply_operator(kalman_gain, obs_vector)
    wgt_mean += multiply_operator(forecast_wgt, forecast_vector)
    #
    # Output error covariance
    print('Computing updating uncertainities')
    w1w2cov = multiply_operator(
        multiply_operator(
            kalman_gain, forecast_wgt
        ),
        cov_forecast_and_obs)
    errcov = c_hat
    errcov += multiply_operator(
        2.0 * one_maker(obs_vector_shape[0]),
        w1w2cov
    )
    #
    return [wgt_mean, errcov, kalman_gain, forecast_wgt]


def main():
    """MAIN"""
    print("===MAIN===")


if __name__ == "__main__":
    main()
