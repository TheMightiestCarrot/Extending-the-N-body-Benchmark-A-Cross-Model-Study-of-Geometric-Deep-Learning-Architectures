import re
import subprocess
import sys


def get_torch_version_from_requirements():
    """Find the torch version from requirements.txt."""
    try:
        with open("requirements.txt", "r") as f:
            for line in f:
                if "torch==" in line:
                    # Match torch==2.6.0, for example
                    match = re.search(r"torch==([0-9]+\.[0-9]+\.[0-9]+)", line)
                    if match:
                        return match.group(1)
    except FileNotFoundError:
        print("requirements.txt not found. Cannot determine torch version.")
        return None
    return None


def check_cuda_availability():
    """Check if CUDA is available, installing a temporary torch if needed."""
    try:
        # First, try to check without installing anything
        import torch

        return torch.cuda.is_available()
    except ImportError:
        # If torch is not installed, we need a temporary installation to check CUDA.
        # This is a bit of a hack but necessary if this script is the entry point.
        print("PyTorch not found. Installing a minimal version to check for CUDA...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "torch", "packaging"]
        )
        try:
            import torch

            return torch.cuda.is_available()
        except Exception as e:
            print(f"Failed to check for CUDA after installing torch: {e}")
            return False


def install_packages():
    """Install packages in the correct order to handle dependencies."""
    torch_version = get_torch_version_from_requirements()
    if not torch_version:
        sys.exit("Could not determine torch version from requirements.txt. Aborting.")

    print(f"Found torch version {torch_version} in requirements.txt")

    # 1. Install PyTorch first, by itself.
    print("\n--- Installing PyTorch ---")
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"torch=={torch_version}"])

    # 2. Install the PyG stack using the correct wheelhouse.
    print("\n--- Installing PyTorch Geometric stack ---")
    cuda_available = check_cuda_availability()
    # Base URL for PyTorch Geometric wheels, using the version from requirements.txt
    base_url = f"https://data.pyg.org/whl/torch-{torch_version}+"
    url_suffix = "cu124.html" if cuda_available else "cpu.html"
    full_url = base_url + url_suffix
    print(f"Using PyG wheelhouse: {full_url}")

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch_geometric",
            "torch_scatter",
            "torch_sparse",
            "torch_cluster",
            "torch_spline_conv",
            "-f",
            full_url,
        ]
    )

    # 3. Install the remaining packages from requirements.txt.
    # We can do this by reading the file and skipping already installed packages.
    print("\n--- Installing remaining packages from requirements.txt ---")

    # Define the exact packages that were already installed manually to avoid
    # accidentally skipping packages like 'torchmetrics'.
    installed_packages = {
        "torch",
        # Use hyphenated names to match PyPI convention; we will normalize anyway
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
    }

    def normalize(name: str) -> str:
        """Normalize package identifier: lowercase and unify '-'/'_' characters."""
        return re.sub(r"[-_]+", "-", name).lower()

    installed_normalized = {normalize(p) for p in installed_packages}

    def skip_on_cpu(pkg: str) -> bool:
        """Return True if the package should be skipped on CPU-only runners.

        We skip CUDA/NVIDIA/TRITON/OpenMM CUDA wheels and any '-cuNN' suffixed
        package when CUDA is not available. This dramatically reduces CI time
        and bandwidth while keeping functionality for CPU tests.
        """
        if cuda_available:
            return False
        p = normalize(pkg)
        return (
            p.startswith("nvidia-")
            or "cuda" in p
            or p.endswith("-cu12")
            or p == "openmm-cuda-12"
            or p == "triton"
        )

    with open("requirements.txt", "r") as f:
        remaining_packages = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Extract package name (e.g., from 'package==1.2.3' or 'package')
            package_name = re.split(r"[=<>!~]", line)[0].strip()
            package_norm = normalize(package_name)

            # Skip GPU-only packages on CPU runners
            if skip_on_cpu(package_name):
                print(f"Skipping GPU-only package on CPU runner: {package_name}")
                continue

            # Skip packages we already installed (normalize to avoid hyphen/underscore dupes)
            if package_norm in installed_normalized:
                continue

            remaining_packages.append(line)

    if remaining_packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + remaining_packages
        )

    print("\n--- Installation complete ---")


if __name__ == "__main__":
    install_packages()
