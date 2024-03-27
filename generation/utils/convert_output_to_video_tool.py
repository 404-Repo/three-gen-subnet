import os
import argparse
from utils import video_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="logs", type=str, help="Directory where obj files are stored")
    parser.add_argument("--out", default="videos", type=str, help="Directory where videos will be saved")
    args = parser.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)

    video_utils1 = video_utils.VideoUtils(512, 512, 4, 5, 10, -30, 10)
    video_utils1.render_gaussian_splatting_video(args.dir, out, 24)
