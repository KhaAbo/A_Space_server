#!/usr/bin/env python
"""
Setup script for MobileGaze gaze estimation project.
Installs dependencies from requirements.txt and additional packages.
"""

import subprocess
import sys
import os


def install_requirements():
    """Install dependencies from requirements.txt if it exists."""
    req_file = os.path.join("gaze-estimation", "requirements.txt")

    if not os.path.exists(req_file):
        print(f"Warning: {req_file} not found. Skipping requirements.txt installation.")
        return

    print(f"Installing dependencies from {req_file}...")

    # First, install numpy without strict version to get a pre-built wheel
    # This avoids compilation issues on Windows where C compilers may not be available
    print("Installing numpy (this may take a moment)...")
    try:
        # Install numpy without version constraint first to get a compatible pre-built wheel
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=1.20.0"])
        print("Numpy installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not install numpy: {e}")
        print("Trying to continue with other packages...")

    # Now install remaining packages from requirements.txt
    # We'll install them individually to handle errors gracefully
    print("Installing remaining packages from requirements.txt...")
    try:
        # Read requirements file and process each package
        with open(req_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Skip numpy since we installed it already
                if line.startswith("numpy"):
                    continue

                # Try to install each package
                package_name = (
                    line.split("==")[0]
                    if "==" in line
                    else line.split(">=")[0]
                    if ">=" in line
                    else line
                )
                print(f"Installing {line}...")
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", line]
                    )
                except subprocess.CalledProcessError as e:
                    print(
                        f"Warning: Failed to install {line}, trying with just package name..."
                    )
                    # Try installing without version constraint
                    try:
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", package_name]
                        )
                    except subprocess.CalledProcessError:
                        print(
                            f"Error: Could not install {package_name}. You may need to install it manually."
                        )

        print("Requirements installation completed.")
    except Exception as e:
        print(f"Warning: Error reading/processing requirements file: {e}")
        print("You may need to install packages manually.")


def install_additional_packages():
    """Install additional packages needed for the demo."""
    # Note: opencv-python is already in requirements.txt, so we skip it here
    additional_packages = ["matplotlib"]

    # torch and torchvision are also in requirements.txt, but we'll check if they need separate handling
    # Check if torch is already installed
    try:
        import torch

        print("torch is already installed.")
    except ImportError:
        print("Installing torch and torchvision...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "torch", "torchvision"]
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not install torch/torchvision: {e}")
            print("You may need to install them manually from pytorch.org")

    print("Installing additional packages...")
    for package in additional_packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not install {package}: {e}")

    print("Additional packages installation completed.")


def main():
    """Main setup function."""
    print("Setting up MobileGaze gaze estimation environment...")

    # Check if gaze-estimation directory exists
    if not os.path.exists("gaze-estimation"):
        print("Error: gaze-estimation directory not found.")
        print("Please clone the repository first:")
        print("  git clone https://github.com/yakhyo/gaze-estimation.git")
        sys.exit(1)

    try:
        install_requirements()
        install_additional_packages()
        print("\nSetup completed successfully!")
        print("\nNote: Don't forget to download model weights.")
        print("See the README.md for instructions on downloading weights.")
    except subprocess.CalledProcessError as e:
        print(f"\nError during installation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
