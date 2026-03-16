"""Compute inverse variance weighted average"""

import numpy as np
from numpy import linalg
import scipy as sp

from . import cov_diagonal as cd

# EFFECTIVELY_ZERO_VAR_DEFAULT = 0.1 ** 2
EFFECTIVELY_ZERO_VAR_DEFAULT = 1e-6


def compute_inverse_via_solve(square_matrix: np.ndarray) -> np.ndarray:
    """
    Solves the linear system for cov_inv
    cov @ cov_inv = np.eye
    """
    arr_shape = square_matrix.shape
    if len(arr_shape) != 2:
        raise ValueError("square_matrix is not a 2D matrix.")
    if arr_shape[0] != arr_shape[1]:
        raise ValueError("square_matrix is not square matrix")
    the_eye = np.eye(arr_shape[0])
    print(type(square_matrix))
    if isinstance(square_matrix, np.ndarray):
        print("square_matrix is np.ndarray")
        ans = linalg.solve(square_matrix, the_eye)
    elif isinstance(square_matrix, sp.sparse.sparray):
        print("sp.sparse.sparray detected, using toarray() method.")
        ans = linalg.solve(square_matrix.toarray(), the_eye)
    elif isinstance(square_matrix, sp.sparse.spmatrix):
        print("sp.sparse.spmatrix detected, using toarray() method.")
        ans = linalg.solve(square_matrix.toarray(), the_eye)
    else:
        raise ValueError(f"Unknown type {type(square_matrix)}")
    return ans


def check_1d(a: np.ndarray):
    """Check if array a is 1D, 2D or something invalid"""
    if len(a.shape) == 1:
        return True
    elif len(a.shape) == 2:
        if a.shape[0] == a.shape[1]:
            return False
    raise ValueError("a is not 1 or 2D")


def matmul(
    a: np.ndarray | sp.sparse.sparray,
    b: np.ndarray | sp.sparse.sparray,
):
    """
    Matrix multiplication
    - np.matmul does not work with sp.sparse.sparray
    - Can't use @ when it needs to mix with np.multiply
    """
    return a @ b


class KalmanOut:
    """
    class to compute blended forecast and observations
    variable names follows compute_inv_variance_wgt_mean_kalman
    """

    def __init__(
        self,
        forecast_vector: np.ndarray,
        obs_vector: np.ndarray,
        errcov_forecast: np.ndarray,
        errcov_obs: np.ndarray,
        cov_forecast_and_obs: np.ndarray | None,
        ez_covariances: bool = True,
    ):
        """
        __init__ for KalmanOut class

        :param forecast_vector: 1D vector of forecasts
        :type forecast_vector: np.ndarray
        :param obs_vector: 1D vector of (gridded) observations
        :type obs_vector: np.ndarray
        :param errcov_forecast: 2D matrix of errcov for forecast_vector
        :type errcov_forecast: np.ndarray
        :param errcov_obs: 2D matrix of errcov for obs_vector
        :type errcov_forecast: np.ndarray
        :param cov_forecast_and_obs: covariance between forecast & observations
        :type cov_forecast_and_obs: np.ndarray
        :param ez_covariances: ignore off-diagonals of errcov_forecast,
            errcov_obs and cov_forecast_and_obs if set to True, default True
        :type ez_covariances: bool
        """
        self.forecast_vector = forecast_vector
        self.obs_vector = obs_vector
        if ez_covariances:
            self.ez_covariances = True
            if check_1d(errcov_forecast):
                self.errcov_forecast = errcov_forecast
            else:
                self.errcov_forecast = np.diag(errcov_forecast)
            if check_1d(errcov_obs):
                self.errcov_obs = errcov_obs
            else:
                self.errcov_obs = np.diag(errcov_obs)
            if cov_forecast_and_obs is not None:
                if check_1d(cov_forecast_and_obs):
                    self.cov_forecast_and_obs = cov_forecast_and_obs
                else:
                    self.cov_forecast_and_obs = np.diag(cov_forecast_and_obs)
            else:
                self.cov_forecast_and_obs = None
            self.inv_operator = np.reciprocal
            self.multiply_operator = np.multiply
            self.one_maker = np.ones
        else:
            self.ez_covariances = False
            self.errcov_forecast = errcov_forecast
            self.errcov_obs = errcov_obs
            self.cov_forecast_and_obs = cov_forecast_and_obs
            self.inv_operator = compute_inverse_via_solve
            self.multiply_operator = matmul
            self.one_maker = np.eye

    def sparse_approx_4_errcov(
        self,
        sparse_threshold: float = EFFECTIVELY_ZERO_VAR_DEFAULT,
        convert2sparse: bool = True,
    ):
        """
        Setting small elements of self.errcov_obs and errcov_forecast to zero
        as defined by sparse_threshold value

        If there are small values in the diagonals

        This reduces memory footprint and may make block-splitting
        by cov_diagonal easier.

        It is not intended to be use for ez_covariance mode (?).
        This is only used to handle situations that the error covariances
        are big.

        :param sparse_threshold: off-diagonal values to set to zero
        :type sparse_threshold: float
        :param convert2sparse: convert array to scipy sparse array?
        :type convert2sparse: bool
        """
        if self.ez_covariances:
            err_msg = "This method is not intended to be use with "
            err_msg += "non-2D error covariances/ez_covariances."
            raise ValueError(err_msg)
        self.errcov_forecast = small_elements_2_zero_and_sparse(
            self.errcov_forecast,
            sparse_threshold=sparse_threshold,
            convert2sparse=convert2sparse,
        )
        self.errcov_obs = small_elements_2_zero_and_sparse(
            self.errcov_obs,
            sparse_threshold=sparse_threshold,
            convert2sparse=convert2sparse,
        )

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


def compute_inv_variance_wgt_mean_kalman_old(
    forecast_vector: np.ndarray,
    obs_vector: np.ndarray,
    errcov_forecast: np.ndarray,
    errcov_obs: np.ndarray,
    cov_forecast_and_obs: np.ndarray | None = None,
    inv_operator: callable = compute_inverse_via_solve,
    multiply_operator: callable = matmul,
    one_maker: callable = np.eye,
):
    """
    Compute the inverse variance weighted average of
    forecast_vector & obs_vector using provided error covariances
    errcov_forecast & errcov_obs

    This follows the classical form of general covariance and inverse
    variance weighting... but it requires THREE matrix inversion or
    reciporcal operations (bad)

    Why do THREE when you only need to do ONCE!?

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
    print("Computing inverse errcov_forecast")
    inv_errcov_forecast = inv_operator(errcov_forecast)  # Inverse 1
    print("Computing inverse errcov_obs")
    inv_errcov_obs = inv_operator(errcov_obs)  # Inverse 2
    #
    forecast_vector_shape = forecast_vector.shape
    if len(forecast_vector_shape) != 1:
        raise ValueError("forecast_vector is not 1D vector")
    if errcov_forecast.shape[0] != forecast_vector_shape[0]:
        raise ValueError(
            "forecast_vector shape inconsistent with errcov_forecast"
        )  # noqa: E501
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
    print("Computing c_hat")
    c_hat = inv_operator(inv_errcov_forecast + inv_errcov_obs)  # Inverse 3
    print("Computing kalman_gain")
    kalman_gain = multiply_operator(c_hat, inv_errcov_obs)
    print("Computing forecast_wgt")
    forecast_wgt = one_maker(kalman_gain.shape) - kalman_gain
    # The stupid way to compute it... even if they are the same
    # forecast_wgt = multiply_operator(c_hat, inv_errcov_forecast)
    #
    # Output weighted mean
    print("Computing weighted mean")
    wgt_mean = multiply_operator(kalman_gain, obs_vector)
    wgt_mean += multiply_operator(forecast_wgt, forecast_vector)
    #
    # Output error covariance
    print("Computing updating uncertainities")
    errcov = c_hat
    if cov_forecast_and_obs is not None:
        w1w2cov = multiply_operator(
            multiply_operator(kalman_gain, forecast_wgt), cov_forecast_and_obs
        )
        errcov += multiply_operator(
            2.0 * one_maker(obs_vector_shape[0]), w1w2cov
        )
    #
    return [wgt_mean, errcov, kalman_gain, forecast_wgt]


def compute_inv_variance_wgt_mean_kalman(
    forecast_vector: np.ndarray,
    obs_vector: np.ndarray,
    errcov_forecast: np.ndarray,
    errcov_obs: np.ndarray,
    cov_forecast_and_obs: np.ndarray | None = None,
    inv_operator: callable = compute_inverse_via_solve,
    multiply_operator: callable = matmul,
    one_maker: callable = np.eye,
):
    """
    Compute the inverse variance weighted average of
    forecast_vector & obs_vector using provided error covariances
    errcov_forecast & errcov_obs

    This uses a form that requires only ONE matrix inverses
    and reciporcals (good!) and is more commonly seen in
    Kalman Fiter guides (including the form uses in Wikipedia)
    https://en.wikipedia.org/wiki/Kalman_filter
    Probably everyone hate matrix inverses... (for good reason)

    compute_inv_variance_wgt_mean_kalman_old requires THREE
    matrix inversions/vector-element reciporcals (bad)

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
    # The one and only one inverse required! Wahoo!
    print("Computing inverse of sum of error covariances")
    inv_sum_of_errcovs = inv_operator(errcov_forecast + errcov_obs)
    #
    forecast_vector_shape = forecast_vector.shape
    if len(forecast_vector_shape) != 1:
        raise ValueError("forecast_vector is not 1D vector")
    if errcov_forecast.shape[0] != forecast_vector_shape[0]:
        raise ValueError(
            "forecast_vector shape inconsistent with errcov_forecast"
        )  # noqa: E501
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
    print("Computing kalman_gain")
    kalman_gain = multiply_operator(errcov_forecast, inv_sum_of_errcovs)
    print("Computing forecast_wgt")
    forecast_wgt = one_maker(kalman_gain.shape[0]) - kalman_gain
    #
    # Output weighted mean
    print("Computing weighted mean")
    wgt_mean = multiply_operator(kalman_gain, (obs_vector - forecast_vector))
    wgt_mean += forecast_vector
    #
    # Output error covariance
    print("Computing updating uncertainities")
    errcov = multiply_operator(
        one_maker(inv_sum_of_errcovs.shape[0]) - kalman_gain, errcov_forecast
    )
    if cov_forecast_and_obs is not None:
        w1w2cov = multiply_operator(
            multiply_operator(kalman_gain, forecast_wgt), cov_forecast_and_obs
        )
        errcov += multiply_operator(
            2.0 * one_maker(obs_vector_shape[0]), w1w2cov
        )
    #
    return [wgt_mean, errcov, kalman_gain, forecast_wgt]


def small_elements_2_zero_and_sparse(
    arr: np.ndarray,
    sparse_threshold: float = EFFECTIVELY_ZERO_VAR_DEFAULT,
    leave_diagonal_alone: bool = True,
    convert2sparse: bool = True,
) -> np.ndarray | sp.sparse.sparray:
    """
    :param arr: array to be manipulated
    :type arr: np.ndarray
    :param sparse_threshold: threshold in which values less than to be to set 0
    :type sparse_threshold: float
    :param leave_diagonal_alone: Should diagonals be modified as well?
    :type leave_diagonal_alone: bool
    :param convert2sparse: Should output be stored as sparse array instant
    :type convert2sparse: bool

    :returns: sparse array
    :rtype: np.ndarray | sp.sparse.sparray
    """
    if len(arr.shape) != 2:
        raise ValueError("This function accepts 2D arrays only.")
    if leave_diagonal_alone:
        gaid = np.diag(arr)
    arr[np.abs(arr) < sparse_threshold] = 0.0
    if leave_diagonal_alone:
        np.fill_diagonal(arr, gaid)
    if convert2sparse:
        arr = sp.sparse.csc_array(arr)
    return arr


class KalmanOutUncorrCorrSplit:
    """
    class to compute blended forecast and observations
    This splits the error covariances into diagonal and non-diagonal bits
    """

    def __init__(
        self,
        forecast_vector: np.ndarray,
        obs_vector: np.ndarray,
        errcov_forecast: np.ndarray,
        errcov_obs: np.ndarray,
        arr_2_decide_if_points_are_isolated: np.ndarray,
        cov_forecast_and_obs: np.ndarray | None,
        zero_threshold: float = cd.EFFECTIVELY_ZERO_DEFAULT,
    ):
        """
        __init__ for KalmanOut class

        :param forecast_vector: 1D vector of forecasts
        :type forecast_vector: np.ndarray
        :param obs_vector: 1D vector of (gridded) observations
        :type obs_vector: np.ndarray
        :param errcov_forecast: 2D matrix of errcov for forecast_vector
        :type errcov_forecast: np.ndarray
        :param errcov_obs: 2D matrix of errcov for obs_vector
        :type errcov_forecast: np.ndarray
        :param arr_2_decide_if_points_are_isolated:
            2D matrix that has diagonal only/and dense bit,
            used to decide if KF should diag-only
            or full correlated error covariance
            this should probably the spatial (variogram-based) covariance
        :type arr_2_decide_if_points_are_isolated: np.ndarray
        :param cov_forecast_and_obs: covariance between forecast & observations
        :type cov_forecast_and_obs: np.ndarray
        :param ez_covariances: ignore off-diagonals of errcov_forecast,
            errcov_obs and cov_forecast_and_obs if set to True, default True
        :type ez_covariances: bool
        """
        #
        _check_2d_and_square(errcov_forecast)
        _check_2d_and_square(errcov_obs)
        if cov_forecast_and_obs is not None:
            _check_2d_and_square(cov_forecast_and_obs)
        #
        ans = cd.diag_and_nondiag_rows_subsampler(
            # errcov_forecast + errcov_obs,
            arr_2_decide_if_points_are_isolated,
            zero_threshold=zero_threshold,
            return_subsampled_arr=False,
        )
        self.d_off_diagonal, _, self.d_diagonal_only, _ = ans
        print(f"{self.d_off_diagonal = }")
        print(f"{np.sum(self.d_off_diagonal) = }")
        print(f"{self.d_diagonal_only = }")
        print(f"{np.sum(self.d_diagonal_only) = }")
        #
        forecast_vector_d = self.d_diagonal_only @ forecast_vector
        obs_vector_d = self.d_diagonal_only @ obs_vector
        errcov_forecast_d = (
            self.d_diagonal_only @ errcov_forecast @ self.d_diagonal_only.T
        )  # noqa: E501
        errcov_obs_d = (
            self.d_diagonal_only @ errcov_obs @ self.d_diagonal_only.T
        )  # noqa: E501
        if cov_forecast_and_obs is not None:
            cov_forecast_and_obs_d = (
                self.d_diagonal_only
                @ cov_forecast_and_obs
                @ self.d_diagonal_only.T
            )  # noqa: E501
            cov_forecast_and_obs_d = np.diag(cov_forecast_and_obs_d)
        else:
            cov_forecast_and_obs_d = None
        #
        forecast_vector_c = self.d_off_diagonal @ forecast_vector
        obs_vector_c = self.d_off_diagonal @ obs_vector
        errcov_forecast_c = (
            self.d_off_diagonal @ errcov_forecast @ self.d_off_diagonal.T
        )  # noqa: E501
        errcov_obs_c = self.d_off_diagonal @ errcov_obs @ self.d_off_diagonal.T
        if cov_forecast_and_obs is not None:
            cov_forecast_and_obs_c = (
                self.d_off_diagonal
                @ cov_forecast_and_obs
                @ self.d_off_diagonal.T
            )  # noqa: E501
        else:
            cov_forecast_and_obs_c = None
        #
        self.uncorr_part = KalmanOut(
            forecast_vector_d,
            obs_vector_d,
            np.diag(errcov_forecast_d),
            np.diag(errcov_obs_d),
            cov_forecast_and_obs_d,
            ez_covariances=True,
        )
        self.corr_part = KalmanOut(
            forecast_vector_c,
            obs_vector_c,
            errcov_forecast_c,
            errcov_obs_c,
            cov_forecast_and_obs_c,
            ez_covariances=False,
        )

    def solve_uncorr(self):
        """Alias for instance_name.uncorr_part.compute_outputs()"""
        self.uncorr_part.compute_outputs()

    def solve_corr(self):
        """Alias for instance_name.corr_part.compute_outputs()"""
        self.corr_part.compute_outputs()

    def combine_results(self):
        """Blend the results of uncorrelated and correlated streams"""
        if not hasattr(self.uncorr_part, "kalman_gain_from_new_obs"):
            err_msg = "Output attributes missing for uncorr_part; "
            err_msg += "do instance_name.uncorr_part.compute_outputs() first."
            raise AttributeError(err_msg)
        if not hasattr(self.corr_part, "kalman_gain_from_new_obs"):
            err_msg = "Output attributes missing for corr_part; "
            err_msg += "do instance_name.corr_part.compute_outputs() first."
            raise AttributeError(err_msg)
        self.wgt_mean = np.array(
            np.matrix(self.uncorr_part.wgt_mean) @ self.d_diagonal_only
            + np.matrix(self.corr_part.wgt_mean) @ self.d_off_diagonal
        )[0]
        self.errcov = (
            self.d_diagonal_only.T
            @ np.diag(self.uncorr_part.errcov)
            @ self.d_diagonal_only  # noqa: E501
            + self.d_off_diagonal.T
            @ self.corr_part.errcov
            @ self.d_off_diagonal
        )
        self.kalman_gain_from_new_obs = (
            self.d_diagonal_only.T
            @ np.diag(self.uncorr_part.kalman_gain_from_new_obs)
            @ self.d_diagonal_only  # noqa: E501
            + self.d_off_diagonal.T
            @ self.corr_part.kalman_gain_from_new_obs
            @ self.d_off_diagonal  # noqa: E501
        )
        self.wgts_from_ar_forecast = (
            self.d_diagonal_only.T
            @ np.diag(self.uncorr_part.wgts_from_ar_forecast)
            @ self.d_diagonal_only  # noqa: E501
            + self.d_off_diagonal.T
            @ self.corr_part.wgts_from_ar_forecast
            @ self.d_off_diagonal  # noqa: E501
        )


def _check_2d_and_square(arr: np.ndarray):
    if len(arr.shape) != 2:
        raise ValueError("arr should be 2D")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("arr should be square")


def main():
    """MAIN"""
    print("===MAIN===")


if __name__ == "__main__":
    main()
