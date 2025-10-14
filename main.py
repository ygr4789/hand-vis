import argparse
import os
import subprocess
from pathlib import Path

from config import *
from preprocess.preprocess import preprocess_pkl_file

def render_sequence(script: str, data_path: str, video_path: str, camera_no: int, scene_no: int, high: bool) -> None:
    """Render a sequence using Blender."""
    cmd = [
        "blender",
        BLENDER_PATH,
        "--background",
        "--python", script,
        "--",
        "-i", str(data_path),
        "-o", str(video_path),
        "-c", str(camera_no),
        "-sc", str(scene_no),
    ]
    
    if high:
        cmd.append("-q")
        
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and render SMPL meshes")
    parser.add_argument("-i", "--input", type=str, required=True, help=".pkl file or path to data directory")
    parser.add_argument('-c', '--camera', type=int, help='Camera number, -1 for all cameras', default=0)
    parser.add_argument('-sc', '--scene', type=int, help='Scene number, default=0 for no furnitures', default=0)
    parser.add_argument('-q', '--high', action='store_true', help='Use high quality rendering settings')
    
    args = parser.parse_args()
    input_path = args.input
    camera_no = args.camera
    scene_no = args.scene
    high = args.high
    
    # Create necessary directories
    input_path = Path(input_path)
    if not input_path.is_file() or not input_path.suffix == '.pkl':
        raise ValueError(f"{input_path} is not a .pkl file")
        
    cache_dir = Path(CACHE_DIR)
    output_dir = Path(OUTPUT_DIR)
    cache_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    input_path = Path(input_path)
    quality = "cycles" if high else "eevee"
    video_path = output_dir / f"{input_path.stem}_{quality}_sc{scene_no}"
    
    intermediate_path = cache_dir / f"{input_path.stem}.pkl"
    preprocess_pkl_file(str(input_path), str(intermediate_path))
    render_sequence(RENDER_SCRIPT_PATH, str(intermediate_path), str(video_path), camera_no, scene_no, high)

if __name__ == "__main__":
    main() 