from pathlib import Path
import argparse

from src.utils import project_utils
from src.config import load_config
from src.pipeline import EyeContactPipeline


def main():
    parser = argparse.ArgumentParser(description="Eye Contact Detection Pipeline")
    parser.add_argument("input", help="Input video or image path")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--display", action="store_true", help="Display while processing")
    parser.add_argument("--skip-frames", type=int, default=None, help="Skip frames for faster video processing")
    
    args = parser.parse_args() # get args from command line
    config = load_config("config/config.yml") # load config from config/config.yml
    pipeline = EyeContactPipeline(config=config) # initialize pipeline
    input_path = Path(args.input) # get input path from command line

    if not input_path.exists(): # check if input path exists
        print(f"Error: Input not found: {input_path}")
        return 1
    
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"} # video extensions
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"} # image extensions
    
    video_cfg = config.section("video")
    skip_frames = args.skip_frames if args.skip_frames is not None else video_cfg.get("skip_frames", 0) # get skip frames from args or config
    display = args.display or video_cfg.get("display", False) # get display from args or config
    
    try:
        if input_path.suffix.lower() in video_exts: # check if input path is a video
            project_utils.process_video(
                str(input_path),
                pipeline.process_frame,
                dst=args.output,
                display=display,
                skip=skip_frames
            )
        elif input_path.suffix.lower() in image_exts: # check if input path is an image
            project_utils.process_image(
                str(input_path),
                pipeline.process_frame,
                dst=args.output
            )
        else:
            print(f"Error: Unsupported file type: {input_path.suffix}")
            return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
    
    return 0 # if successful


if __name__ == "__main__":
    exit(main())
