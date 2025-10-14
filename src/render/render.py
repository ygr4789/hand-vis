import bpy
import os
import sys

dir = os.path.dirname(__file__)
if not dir in sys.path:
    sys.path.append(dir)

import pickle
from utils import *
from camera import *
from prim import *

def parse_arguments():
    # Get all arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    # Create argument parser
    parser = argparse.ArgumentParser(description='Render SMPL visualization in Blender')
    parser.add_argument('-i', '--input', required=True, type=str, help='Path to obj output folder (input)')
    parser.add_argument('-o', '--output', required=True, type=str, help='Path to output video folder (output)')
    parser.add_argument('-q', '--high', action='store_true', help='Use high quality rendering settings')
    parser.add_argument('-c', '--camera', type=int, help='Camera number, default=-1 for all cameras', default=0)
    parser.add_argument('-sc', '--scene', type=int, help='Scene number, default=0 for no furnitures', default=0)
    
    return parser.parse_args(argv)

def main():
    args = parse_arguments()
    data_path = args.input
    video_path = args.output
    render_high = args.high
    camera_no = args.camera
    scene_no = args.scene
    
    # Load scene and setup
    cleanup_existing_objects()
    setup_render_settings(render_high)
    setup_background_scene(scene_no)
    
    # Prepare render data
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    p1_joints = data["p1_joints"]
    p2_joints = data["p2_joints"]
    
    p1_hand_left_verts = data["p1_hand_left_verts"]
    p1_hand_right_verts = data["p1_hand_right_verts"]
    p2_hand_left_verts = data["p2_hand_left_verts"]
    p2_hand_right_verts = data["p2_hand_right_verts"]
    
    p1_hand_left_faces = data["p1_hand_left_faces"]
    p1_hand_right_faces = data["p1_hand_right_faces"]
    p2_hand_left_faces = data["p2_hand_left_faces"]
    p2_hand_right_faces = data["p2_hand_right_faces"]
    
    obj_verts = data["obj_verts"]
    obj_faces = data["obj_faces"]
    num_frames = data["num_frames"]
    
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
    
    print("Preparing objects...")
    setup_joints_and_bones(p1_joints, "Red_soft")
    setup_joints_and_bones(p2_joints, "Blue_soft")
    setup_mesh_keyframes(p1_hand_left_verts, p1_hand_left_faces, "Red")
    setup_mesh_keyframes(p1_hand_right_verts, p1_hand_right_faces, "Red")
    setup_mesh_keyframes(p2_hand_left_verts, p2_hand_left_faces, "Blue")
    setup_mesh_keyframes(p2_hand_right_verts, p2_hand_right_faces, "Blue")
    setup_mesh_keyframes(obj_verts, obj_faces, "Yellow")
    print("Objects setup complete")
    # setup_mesh_keyframes(obj_verts, obj_faces, "Yellow_soft")
    
    # Render animation
    camera_settings = prepare_camera_settings(root_loc1, root_loc2, camera_no)
    render_animation(video_path, camera_settings, anim_frames)

if __name__ == "__main__":
    main()