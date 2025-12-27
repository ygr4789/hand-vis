import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional, List

from config import *
from preprocess.preprocess import preprocess_pkl_file

def render_sequence(script: str, data_path: str, video_path: str, option_cmd: List[str]) -> None:
    """Render a sequence using Blender."""
    option_cmd.extend(["-i", str(data_path), "-o", str(video_path)])
    cmd = ["blender", BLENDER_PATH, "--background", "--python", script, "--", *option_cmd]
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and render SMPL meshes")
    parser.add_argument("-i", "--input", type=str, required=True, help=".pkl file or path to data directory")
    parser.add_argument('-c', '--camera', type=int, help='Camera number, -1 for all cameras', default=0)
    parser.add_argument('-sc', '--scene', type=int, help='Scene number, default=0 for no furnitures', default=0)
    parser.add_argument('-q', '--high', action='store_true', help='Use high quality rendering settings')
    parser.add_argument('-f', '--frame', type=int, help='Render only this frame (1-based). If omitted, render full animation', default=None)
    parser.add_argument('-ih', '--input_hand', action='store_true', help='Include input hand in the render')
    parser.add_argument('-cl', '--clothed', action='store_true', help='Render clothed scene')
    parser.add_argument('-fg', '--figure', action='store_true', help='Render figure scene with transparent background, only available for single frame image render')
    parser.add_argument('-ff', '--figure_floor', action='store_true', help='Render figure scene with transparent background and floor, only available for single frame image render')
    parser.add_argument('-z', '--zoom', type=str, choices=[None, '0', '1', '2', '1l', '1r', '2l', '2r'], default=None)
    
    args = parser.parse_args()
    input_path = args.input
    camera_no = args.camera
    scene_no = args.scene
    high = args.high
    frame_no = args.frame
    input_hand = args.input_hand
    figure = args.figure
    zoom = args.zoom
    clothed = args.clothed
    figure_floor = args.figure_floor
    # Create necessary directories
    input_path = Path(input_path)
    if not input_path.is_file() or not input_path.suffix == '.pkl':
        raise ValueError(f"{input_path} is not a .pkl file")
        
    cache_dir = Path(CACHE_DIR)
    output_dir = Path(OUTPUT_DIR)
    data_subdir = Path(f"{input_path.stem}")
    cache_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    input_path = Path(input_path)
    quality = "cycles" if high else "eevee"
    
    file_name = f"{quality}_sc{scene_no}"
        
    file_name_output = file_name + "_output"
    file_name_input = file_name + "_input"
    video_path_output = output_dir / data_subdir / file_name_output
    video_path_input = output_dir / data_subdir / file_name_input
    
    intermediate_path = cache_dir / f"{input_path.stem}.npz"
    preprocess_pkl_file(str(input_path), str(intermediate_path))
    
    option_cmd = [
        "-c", str(camera_no),
        "-sc", str(scene_no),
    ]
    if zoom:
        option_cmd.extend(["-z", str(zoom)])
    if high:
        option_cmd.append("-q")
    if input_hand:
        option_cmd.append("-ih")
    if frame_no:
        option_cmd.extend(["-f", str(frame_no)])
    if figure:
        option_cmd.append("-fg")
    if figure_floor:
        option_cmd.append("-ff")
    if clothed:
        option_cmd.append("-cl")
    
    render_sequence(RENDER_SCRIPT_PATH, str(intermediate_path), video_path_output, option_cmd + ["-m", "output"])
    render_sequence(RENDER_SCRIPT_PATH, str(intermediate_path), video_path_input, option_cmd + ["-m", "input"])

if __name__ == "__main__":
    main() 