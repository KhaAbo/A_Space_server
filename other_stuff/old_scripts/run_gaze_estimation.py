#!/usr/bin/env python
"""
Script to run MobileGaze gaze estimation on video files.
Wrapper around inference.py with convenient defaults.
"""

import argparse
import sys
import os
import subprocess


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run gaze estimation on video files using MobileGaze",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults (input/ad_test.mp4 -> results/attention.mp4)
  python run_gaze_estimation.py

  # Specify custom video and model
  python run_gaze_estimation.py --source input/my_video.mp4 --model resnet34

  # Use CPU instead of CUDA
  python run_gaze_estimation.py --device cpu
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="input/in_train.mp4",
        help="Path to source video file (default: input/ad_test.mp4)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Model architecture: resnet18, resnet34, resnet50, mobilenetv2, mobileone_s0 (default: resnet50)",
    )

    parser.add_argument(
        "--weight",
        type=str,
        default=None,
        help="Path to model weights file (default: weights/{model}.pt)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/attention.mp4",
        help="Path to save output video (default: results/attention.mp4)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="gaze360",
        help="Dataset name for configuration (default: gaze360)",
    )

    parser.add_argument(
        "--view", action="store_true", help="Display the inference results in a window"
    )

    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Do not display the inference results (useful for headless execution)",
    )

    return parser.parse_args()


def get_default_weight(model_name):
    """Get default weight path for a model."""
    return os.path.join("gaze-estimation", "weights", f"{model_name}.pt")


def check_prerequisites():
    """Check if prerequisites are met."""
    if not os.path.exists("gaze-estimation"):
        print("Error: gaze-estimation directory not found.")
        print("Please clone the repository first:")
        print("  git clone https://github.com/yakhyo/gaze-estimation.git")
        sys.exit(1)

    inference_script = os.path.join("gaze-estimation", "inference.py")
    if not os.path.exists(inference_script):
        print(f"Error: {inference_script} not found.")
        sys.exit(1)


def main():
    """Main function to run gaze estimation."""
    args = parse_args()

    # Check prerequisites
    check_prerequisites()

    # Check if source video exists
    if not os.path.exists(args.source):
        print(f"Error: Source video not found: {args.source}")
        print("Please place your video file in the input/ folder.")
        sys.exit(1)

    # Determine weight path
    if args.weight is None:
        weight_path = get_default_weight(args.model)
    else:
        weight_path = args.weight

    # Convert weight path to relative path for when we're in gaze-estimation directory
    # Get absolute paths for proper comparison
    gaze_est_dir = os.path.abspath("gaze-estimation")

    # Normalize the weight path
    if os.path.isabs(weight_path):
        weight_path_abs = os.path.normpath(weight_path)
    else:
        weight_path_abs = os.path.abspath(weight_path)

    # Check if the weight path is inside gaze-estimation directory
    try:
        # Try to get relative path from gaze-estimation directory
        weight_path_relative = os.path.relpath(weight_path_abs, gaze_est_dir)

        # If the relative path starts with "..", it's outside gaze-estimation
        # So we should look for it in the standard weights/ location
        if weight_path_relative.startswith(".."):
            # Path is outside gaze-estimation, try standard location
            standard_weight = os.path.join(gaze_est_dir, "weights", f"{args.model}.pt")
            if os.path.exists(standard_weight):
                weight_path_relative = os.path.join("weights", f"{args.model}.pt")
            else:
                # Keep the relative path even if outside (user specified custom path)
                weight_path_relative = os.path.normpath(weight_path_relative)
        else:
            # Path is inside gaze-estimation, use the relative path
            weight_path_relative = os.path.normpath(weight_path_relative)
    except ValueError:
        # Can't compute relative path (different drives on Windows), keep original
        # But try to extract just the filename or relative part
        if "weights" in weight_path:
            # Try to extract weights/... part
            parts = weight_path.replace("\\", "/").split("/")
            if "weights" in parts:
                idx = parts.index("weights")
                weight_path_relative = "/".join(parts[idx:])
            else:
                weight_path_relative = weight_path
        else:
            weight_path_relative = weight_path

    # Check if weight file exists
    # First check the original path
    weight_exists = os.path.exists(weight_path)

    # Then check from current directory (before changing to gaze-estimation)
    if not weight_exists:
        weight_check_path = os.path.join("gaze-estimation", weight_path_relative)
        weight_exists = os.path.exists(weight_check_path)

    # Also try the standard location
    if not weight_exists:
        standard_path = os.path.join("gaze-estimation", "weights", f"{args.model}.pt")
        if os.path.exists(standard_path):
            weight_path_relative = os.path.join("weights", f"{args.model}.pt")
            weight_exists = True

    if not weight_exists:
        print("Error: Model weights not found!")
        print(f"  Looked for: {weight_path}")
        print(f"  And also: {os.path.join('gaze-estimation', weight_path_relative)}")
        print("  Please download the weights first. See README.md for instructions.")
        print("  You can download weights using:")
        print(f"    cd gaze-estimation && sh download.sh {args.model}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Build inference command
    inference_script = "inference.py"  # Will be run from gaze-estimation directory
    cmd = [
        sys.executable,
        inference_script,
        "--model",
        args.model,
        "--weight",
        weight_path_relative,
        "--source",
        args.source,
        "--output",
        args.output,
        "--dataset",
        args.dataset,
    ]

    # Handle view flag
    if args.view:
        cmd.append("--view")
    elif args.no_view:
        cmd.append("--no-view")
    else:
        # Default: don't show view for batch processing
        cmd.append("--no-view")

    # Print command for debugging
    print("Running gaze estimation...")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run the inference script
    try:
        # Change to gaze-estimation directory to ensure imports work
        original_dir = os.getcwd()
        os.chdir("gaze-estimation")

        # Adjust paths relative to gaze-estimation directory
        source_idx = cmd.index("--source") + 1
        output_idx = cmd.index("--output") + 1

        cmd[source_idx] = os.path.join("..", args.source)
        cmd[output_idx] = os.path.join("..", args.output)
        # Weight path is already adjusted to be relative to gaze-estimation directory

        subprocess.check_call(cmd)

        # Return to original directory
        os.chdir(original_dir)

        print(f"\nSuccess! Output video saved to: {args.output}")

    except subprocess.CalledProcessError as e:
        print(f"\nError running inference: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
