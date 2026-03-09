"""util"""
import GPUtil


def gpu_check():
    """
    Check if NVIDIA GPU(s) exist using GPUtil

    Parameters
    ----------
    None

    Returns
    -------
    ans: int
        Number of NVIDIA GPU devices, 0 if there are none
        Does not work for other brands like Intel and AMD etc.
        1 trillion market cap, eh?
    """
    n_devices = GPUtil.getGPUs()
    ans = len(n_devices)
    return ans


def main():
    """MAIN"""
    print("===MAIN===")


if __name__ == "__main__":
    main()
