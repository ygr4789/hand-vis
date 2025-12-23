import bpy
import os
import sys
import numpy as np
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.append(script_dir)

from render.utils import *
from render.camera import *
from render.prim import *

def parse_arguments():
    # Get all arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    # Create argument parser
    parser = argparse.ArgumentParser(description='Render SMPL visualization in Blender')
    parser.add_argument('-i', '--input', required=True, type=str)
    parser.add_argument('-o', '--output', required=True, type=str)
    parser.add_argument('-q', '--high', action='store_true')
    parser.add_argument('-c', '--camera', type=int, default=0)
    parser.add_argument('-sc', '--scene', type=int, default=0)
    parser.add_argument('-f', '--frame', type=int, default=None)
    parser.add_argument('-fg', '--figure', action='store_true')
    parser.add_argument('-ih', '--input_hand', action='store_true')
    parser.add_argument('-m', '--mode', type=str, choices=['output', 'input'], default='output')
    
    return parser.parse_args(argv)

def main():
    args = parse_arguments()
    data_path = args.input
    video_path = args.output
    render_high = args.high
    camera_no = args.camera
    scene_no = args.scene
    frame_no = args.frame
    render_mode = args.mode
    figure = args.figure
    input_hand = args.input_hand
    
    # Load scene and setup
    cleanup_existing_objects()
    setup_render_settings(render_high)
    setup_background_scene(scene_no)
    
    # Prepare render data
    data = np.load(data_path)
    
    obj_verts = data["obj_verts"]
    obj_faces = data["obj_faces"]
    num_frames = data["num_frames"]
    hand_left_faces = data["hand_left_faces"]
    hand_right_faces = data["hand_right_faces"]
    
    p1_joints = data[f"{render_mode}_p1_joints"]
    p2_joints = data[f"{render_mode}_p2_joints"]
    p1_hand_left_verts = data[f"{render_mode}_p1_hand_left_verts"]
    p1_hand_right_verts = data[f"{render_mode}_p1_hand_right_verts"]
    p2_hand_left_verts = data[f"{render_mode}_p2_hand_left_verts"]
    p2_hand_right_verts = data[f"{render_mode}_p2_hand_right_verts"]
    
    if render_mode == "input" and input_hand:
        p1_joints = p1_joints[:, :22]
        p2_joints = p2_joints[:, :22]
    
    p1_joints = convert_to_blender_coord(p1_joints)
    p2_joints = convert_to_blender_coord(p2_joints)
    obj_verts = convert_to_blender_coord(obj_verts)
    p1_hand_left_verts = convert_to_blender_coord(p1_hand_left_verts)
    p1_hand_right_verts = convert_to_blender_coord(p1_hand_right_verts)
    p2_hand_left_verts = convert_to_blender_coord(p2_hand_left_verts)
    p2_hand_right_verts = convert_to_blender_coord(p2_hand_right_verts)
    
    root_loc1 = p1_joints[:, 0]
    root_loc2 = p2_joints[:, 0]
    
    # # Create joints and bones
    anim_frames = num_frames*2-1
    setup_animation_settings(anim_frames)
    
    if frame_no is not None:
        frame_no = max(1, min(int(num_frames), int(frame_no)))
        idx = frame_no - 1

        p1_joints = p1_joints[idx:idx+1]
        p2_joints = p2_joints[idx:idx+1]
        p1_hand_left_verts = p1_hand_left_verts[idx:idx+1]
        p1_hand_right_verts = p1_hand_right_verts[idx:idx+1]
        p2_hand_left_verts = p2_hand_left_verts[idx:idx+1]
        p2_hand_right_verts = p2_hand_right_verts[idx:idx+1]
        obj_verts = obj_verts[idx:idx+1]
        num_frames = 1
    
    print("Preparing objects...")
    setup_joints_and_bones(p1_joints, "Red_soft")
    setup_joints_and_bones(p2_joints, "Blue_soft")
    setup_mesh_keyframes(obj_verts, obj_faces, "Gray")
    
    if render_mode == "output" or (render_mode == "input" and input_hand):
        setup_mesh_keyframes(p1_hand_left_verts, hand_left_faces, "Red")
        setup_mesh_keyframes(p1_hand_right_verts, hand_right_faces, "Red")
        setup_mesh_keyframes(p2_hand_left_verts, hand_left_faces, "Blue")
        setup_mesh_keyframes(p2_hand_right_verts, hand_right_faces, "Blue")
        
    print("Objects setup complete")
    
    # Render animation or a single frame
    camera_settings = prepare_camera_settings(root_loc1, root_loc2, camera_no)
    
    if frame_no is not None:
        render_single_frame(video_path, camera_settings, frame_no, figure)
    else:
        render_animation(video_path, camera_settings, anim_frames)
    
if __name__ == "__main__":
    main()