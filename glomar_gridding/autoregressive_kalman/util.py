"""util"""
import subprocess  # noqa: S404


def gpu_check():
    """
    Check if NVIDIA GPU(s) exist using standard libraries

    Catch-22: checks based on cuda, pytorch, cupy,
    tensorflow etc does not work! Can't check something
    using something that does not exist!

    All systems with NVIDIA GPUs should have nvidia-smi

    Parameters
    ----------
    None

    Returns
    -------
    bool:
        True - if GPU exists, False otherwise
    """
    try:
        subprocess.check_output("nvidia-smi")  # noqa: S607
    except Exception:
        return False
    return True


def main():
    """MAIN"""
    print("===MAIN===")


if __name__ == "__main__":
    main()
