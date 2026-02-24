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
    """

    def __init__(
            self,
            forecast_vector: np.ndarray,
            obs_vector: np.ndarray,
            errcov_forecast: np.ndarray,
            errcov_obs: np.ndarray,
            cov_forecast_and_obs: np.ndarray,
            ):
        self.forecast_vector = forecast_vector
        self.obs_vector = obs_vector
        self.errcov_forecast = errcov_forecast
        self.errcov_obs = errcov_obs
        self.cov_forecast_and_obs = cov_forecast_and_obs

    def compute_outputs(self):
        """Calls compute_inv_variance_wgt_mean_kalman"""
        ans = compute_inv_variance_wgt_mean_kalman(
            self.forecast_vector,
            self.obs_vector,
            self.errcov_forecast,
            self.errcov_obs,
            self.cov_forecast_and_obs
        )
        self.wgt_mean = ans[0]
        self.errcov = ans[1]


def compute_inv_variance_wgt_mean_kalman(
        forecast_vector: np.ndarray,
        obs_vector: np.ndarray,
        errcov_forecast: np.ndarray,
        errcov_obs: np.ndarray,
        cov_forecast_and_obs: np.ndarray,
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
    :param cov_forecast_and_obs: covariance between forecast & observations
    :type cov_forecast_and_obs: np.ndarray

    :returns: list with weighted avg and error covariance
    :rtype: list
    """
    #
    inv_errcov_forecast = compute_inverse_via_solve(errcov_forecast)
    inv_errcov_obs = compute_inverse_via_solve(errcov_obs)
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
    c_hat = compute_inverse_via_solve(inv_errcov_forecast + inv_errcov_obs)
    kalman_gain = c_hat @ inv_errcov_obs
    forecast_wgt = c_hat @ inv_errcov_forecast
    #
    # Output weighted mean
    wgt_mean = kalman_gain @ obs_vector + forecast_wgt @ forecast_vector
    #
    # Output error covariance
    w1w2cov = kalman_gain @ forecast_wgt @ cov_forecast_and_obs
    errcov = c_hat + (2.0 * np.eye(obs_vector_shape[0])) @ w1w2cov
    #
    return [wgt_mean, errcov]
