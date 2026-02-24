"""Compute inverse variance weighted average"""

import numpy as np
from numpy import linalg


def compute_inverse_via_solve(square_matrix: np.ndarray):
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


def compute_inv_variance_wgt_mean_kalman(
        forecast_vector: np.ndarray,
        obs_vector: np.ndarray,
        errcov_forecast: np.ndarray,
        errcov_obs: np.ndarray,
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
    :returns: weighted average of forecast_vector and obs_vector
    :rtype: np.ndarray
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
    c_hat = compute_inverse_via_solve(inv_errcov_forecast + inv_errcov_obs)
    kalman_gain = c_hat @ inv_errcov_obs
    forecast_wgt = c_hat @ inv_errcov_forecast
    ans = kalman_gain @ obs_vector + forecast_wgt @ forecast_vector
    return ans
