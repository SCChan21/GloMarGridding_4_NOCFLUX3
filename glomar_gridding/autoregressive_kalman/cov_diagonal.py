"""
Docstring for cov_diagonal

remove_diag_only_rows:
- Auto-detect diagonal only rows/columns from an input matrix
- Purge them to form a smaller matrix and saves the matrix

restore_diag_only_rows:
- reverses the process, but you need to give a filler val
"""

import numpy as np
import scipy as sp

EFFECTIVELY_ZERO_DEFAULT = 1e-6


def _more_than_one_element(
    row: np.ndarray, zero_threshold: float = EFFECTIVELY_ZERO_DEFAULT
):
    """Check if 1D vector more than one non-zero element"""
    return np.sum(row > zero_threshold) > 1


def remove_diag_only_rows(
    cov: np.ndarray,
    zero_threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Docstring for remove_diag_only_rows

    Parameters
    ----------
    cov: np.ndarray
        covariance matrix with possible diagonal only elements

    Returns
    -------
    ans: tuple[ndarray, ndarray]
        - a new covariance without those diagonal-only elements
        - the sampling matrix that generates it
    """
    n_rows = cov.shape[0]
    print(f"{cov.shape = }")
    n_validrows = 0
    has_off_diagonal_elements = np.apply_along_axis(
        lambda row: _more_than_one_element(
            row,
            zero_threshold=zero_threshold,
        ),
        0,
        cov,
    )
    n_validrows = int(np.sum(has_off_diagonal_elements))
    print(f"{n_validrows = }")
    if n_validrows < 1:
        raise ValueError(f"{n_validrows} must be at >= 1")
    #
    D = np.zeros((n_validrows, cov.shape[0]), dtype=np.uint8)
    row_count = 0
    for i in range(n_rows):
        if has_off_diagonal_elements[i] == 0:
            continue
        print(f"{row_count} {i}")
        D[row_count, i] = 1
        row_count += 1
    print(D)
    print(f"{D.shape = }")
    #
    new_cov_arr = np.matmul(D, cov)
    print(f"Progress update: {new_cov_arr.shape = }")
    new_cov_arr = np.matmul(new_cov_arr, D.T)
    print(new_cov_arr)
    print(f"{new_cov_arr.shape = }")
    #
    return new_cov_arr, D


def restore_diag_only_rows(
    trimmed_cov_arr: np.ndarray,
    D: np.ndarray,
    diag_fillvalue: float = 1.2,
    atol: float = 1e-6,
) -> np.ndarray:
    """
    Docstring for restore_diag_only_rows

    Parameters
    ----------
    trimmed_cov_arr: np.ndarray
        A trimmed numpy array that needs expaned
    D: np.ndarray
        The subsampling array that did the original purge (see the_purge)
    diag_fillvalue: float
        The diagonal fillvalue for restored rows and columns
    atol: float
        Instead of checking for exact 0s,
        this is the threshold to decide diag_fillvalue replacement will occur

    Returns
    -------
    ans: np.ndarray
        A larger (restored) covariance array
    """
    #
    print(f"{trimmed_cov_arr.shape = }")
    print(f"{D.shape = }")
    #
    new_cov_arr = np.matmul(D.T, trimmed_cov_arr)
    print(f"Progress update: {new_cov_arr.shape = }")
    new_cov_arr = np.matmul(new_cov_arr, D)
    print(f"{new_cov_arr.shape = }")
    #
    diag_vals = np.array(np.diag(new_cov_arr))
    old_diag_vals = diag_vals.copy()
    diag_vals[diag_vals <= atol] = diag_fillvalue
    np.fill_diagonal(new_cov_arr, diag_vals)
    #
    print(new_cov_arr)
    print(f"{np.sum(diag_vals) = }")
    print(f"{np.sum(old_diag_vals) = }")
    return new_cov_arr


def diag_and_nondiag_rows_subsampler(
    cov: np.ndarray,
    zero_threshold: float = EFFECTIVELY_ZERO_DEFAULT,
    return_subsampled_arr: bool = True,
) -> tuple[np.ndarray, None | np.ndarray, np.ndarray, None | np.ndarray]:
    """
    Docstring for diag_and_nondiag_rows_subsampler

    :param cov: covariance matrix with possible diagonal only elements
    :type cov: np.ndarray
    :return:
        a tuple with four matrices
        - d_off_diagonal: sampling matrix for the off-diagonal rows
        - the_denser_parts: a (somewhat denser) subsampled by that matrix
        - d_diagonal_only: the diagonal only rows
        - isolated_diag_vals: vector with diagonal values of those rows
    :rtype: tuple[ndarray, ndarray, ndarray, ndarray]
    """
    n_rows = cov.shape[0]
    print(f"{cov.shape = }")
    n_validrows = 0
    #
    # This returns True for rows that have off diagonal elements
    has_off_diagonal_elements = np.apply_along_axis(
        lambda row: _more_than_one_element(
            row,
            zero_threshold=zero_threshold,
        ),
        0,
        cov,
    )
    n_validrows = int(np.sum(has_off_diagonal_elements))
    n_diag_only = cov.shape[0] - n_validrows
    print(f"{n_validrows = }")
    print(f"{n_diag_only = }")
    if n_validrows < 1:
        raise ValueError(f"{n_validrows} must be at >= 1")
    #
    d_diagonal_only = np.zeros((n_diag_only, cov.shape[0]), dtype=bool)
    d_off_diagonal = np.zeros((n_validrows, cov.shape[0]), dtype=bool)
    row_count_off_diagonal = 0
    row_count_diagonal_only = 0
    for i in range(n_rows):
        if has_off_diagonal_elements[i] == 0:
            d_diagonal_only[row_count_diagonal_only, i] = True
            row_count_diagonal_only += 1
        else:
            d_off_diagonal[row_count_off_diagonal, i] = True
            row_count_off_diagonal += 1
    d_diagonal_only = sp.sparse.csr_matrix(d_diagonal_only)
    d_off_diagonal = sp.sparse.csr_matrix(d_off_diagonal)
    print(f"{type(d_off_diagonal) = }")
    print(f"{d_off_diagonal.shape = }")
    print(f"{type(d_diagonal_only) = }")
    print(f"{d_diagonal_only.shape = }")
    #
    diag_cov = np.diag(cov)
    diag_cov = np.array(diag_cov)
    if return_subsampled_arr:
        # isolated_diag_vals = np.matmul(
        #     d_diagonal_only.toarray(),
        #     diag_cov,
        # )
        # the_denser_parts = np.matmul(
        #     np.matmul(d_off_diagonal.toarray(), cov),
        #     d_off_diagonal.toarray().T,
        # )
        isolated_diag_vals = d_diagonal_only @ diag_cov
        the_denser_parts = d_off_diagonal @ cov @ d_off_diagonal.T
    else:
        isolated_diag_vals = None
        the_denser_parts = None
    #
    return (
        d_off_diagonal,
        the_denser_parts,
        d_diagonal_only,
        isolated_diag_vals,
    )
