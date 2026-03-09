"""
Docstring for cov_diagonal

remove_diag_only_rows:
- Auto-detect diagonal only rows/columns from an input matrix
- Purge them to form a smaller matrix and saves the matrix

restore_diag_only_rows:
- reverses the process, but you need to give a filler val
"""

import numpy as np


def _more_than_one_element(
        row: np.ndarray,
        zero_threshold: float = 1E-6):
    """Check if 1D vector more than one non-zero element"""
    return np.sum(row > zero_threshold) > 1


def remove_diag_only_rows(
        cov: np.ndarray,
        zero_threshold: float = 1E-6,
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
    ans = np.apply_along_axis(
        lambda row: _more_than_one_element(
            row,
            zero_threshold=zero_threshold,
        ),
        0,
        cov,
    )
    n_validrows = int(np.sum(ans))
    print(f"{n_validrows = }")
    if not n_validrows >= 1:
        raise ValueError(f"{n_validrows} must be at >= 1")
    #
    D = np.zeros((n_validrows, cov.shape[0]), dtype=np.uint8)
    row_count = 0
    for i in range(n_rows):
        if ans[i] == 0:
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
        atol: float = 1E-6,
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


def main():
    """MAIN"""
    print("===MAIN===")


if __name__ == "__main__":
    main()
