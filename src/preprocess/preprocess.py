import pickle
import torch
import os
import numpy as np

from .hand_model import HandModel
from .close_surface import close_surface
from .safe_load import safe_load_pkl

def preprocess_pkl_file(pkl_path, save_path, mode):
    if os.path.exists(save_path):
        print(f"Preprocessed data already exists at {save_path}")
        return
        
    device = torch.device("cuda")

    with open(pkl_path, "rb") as f:
        data = safe_load_pkl(pkl_path)
        
        obj_faces_list = data["obj_faces_list"]
        original_obj_verts_list = data["original_obj_verts_list"]
        filtered_obj_verts_list = data["filtered_obj_verts_list"]
        filtered_obj_T = data["filtered_obj_T"]
        original_obj_T = data["original_obj_T"]
        
        stage1_result = data["stage1_result"]
        stage2_result = data["stage2_result"]
        stage3_result = data["stage3_result"]
        
        stage2_pseudo_gt_p1 = stage2_result[f"{mode}_p1"]
        stage2_pseudo_gt_p2 = stage2_result[f"{mode}_p2"]
        stage3_pseudo_gt_p1 = stage3_result[f"{mode}_p1"]
        stage3_pseudo_gt_p2 = stage3_result[f"{mode}_p2"]
        
        p1_joints = stage2_pseudo_gt_p1["jnts_list"].detach().cpu().numpy()
        p2_joints = stage2_pseudo_gt_p2["jnts_list"].detach().cpu().numpy()
        
        p1_hand_parmas_left = stage3_pseudo_gt_p1[0]["hand_params"]
        p1_hand_parmas_right = stage3_pseudo_gt_p1[1]["hand_params"]
        p1_wrist_T = stage3_pseudo_gt_p1[0]["wrist_T"]
        p1_wrist_T_right = stage3_pseudo_gt_p1[1]["wrist_T"]
        
        p2_hand_parmas_left = stage3_pseudo_gt_p2[0]["hand_params"]
        p2_hand_parmas_right = stage3_pseudo_gt_p2[1]["hand_params"]
        p2_wrist_T = stage3_pseudo_gt_p2[0]["wrist_T"]
        p2_wrist_T_right = stage3_pseudo_gt_p2[1]["wrist_T"]
        
        # import ipdb; ipdb.set_trace()
        input_data = data['input']
        input_p1_joints = input_data['gt_p1_jnts_list'].detach().cpu().numpy()
        input_p2_joints = input_data['gt_p2_jnts_list'].detach().cpu().numpy()
        input_p1_body_vertices = input_data['gt_p1_verts_list']
        input_p2_body_vertices = input_data['gt_p2_verts_list']

    obj_verts = filtered_obj_verts_list     
    obj_faces = obj_faces_list

    num_frames_p1_joints = p1_joints.shape[0]
    num_frames_p2_joints = p2_joints.shape[0]
    num_frames_input_p1_joints = input_p1_joints.shape[0]
    num_frames_input_p2_joints = input_p2_joints.shape[0]
    num_frames_obj = obj_verts.shape[0]

    num_frames = min(num_frames_p1_joints, num_frames_p2_joints, num_frames_input_p1_joints, num_frames_input_p2_joints, num_frames_obj)

    start, end = 0, num_frames
    if 'frame_start_end' in data:
        start, end = data['frame_start_end']
        num_frames = end - start

    # body
    p1_joints = p1_joints[start:end]
    p2_joints = p2_joints[start:end]
    input_p1_joints = input_p1_joints[start:end]
    input_p2_joints = input_p2_joints[start:end]

    # input hand 
    input_p1_body_vertices = input_p1_body_vertices[start:end]
    input_p2_body_vertices = input_p2_body_vertices[start:end]

    # output hand 
    p1_hand_parmas_left = p1_hand_parmas_left[start:end]
    p2_hand_parmas_left = p2_hand_parmas_left[start:end]
    p1_hand_parmas_right = p1_hand_parmas_right[start:end]
    p2_hand_parmas_right = p2_hand_parmas_right[start:end]
    
    # import ipdb; ipdb.set_trace()
    # already the same
    # p1_joints[:,20] = p1_hand_parmas_left[start:end,:3].cpu().numpy()
    # p1_joints[:,21] = p1_hand_parmas_right[start:end,:3].cpu().numpy()
    # p2_joints[:,20] = p2_hand_parmas_left[start:end,:3].cpu().numpy()
    # p2_joints[:,21] = p2_hand_parmas_right[start:end,:3].cpu().numpy()

    # obj
    obj_verts = obj_verts[start:end]

    hand_model_left = HandModel(left_hand=True, gender="female", device=device, batch_size=num_frames)
    hand_model_right = HandModel(left_hand=False, gender="female", device=device, batch_size=num_frames)

    with torch.no_grad():
        hand_model_left.set_parameters(p1_hand_parmas_left, skip_left_mirror=True)
        hand_model_right.set_parameters(p1_hand_parmas_right, skip_left_mirror=True)
        
        p1_hand_left_verts = hand_model_left.vertices.detach().cpu().numpy()
        p1_hand_right_verts = hand_model_right.vertices.detach().cpu().numpy()
        
        hand_model_left.set_parameters(p2_hand_parmas_left, skip_left_mirror=True)
        hand_model_right.set_parameters(p2_hand_parmas_right, skip_left_mirror=True)
        
        p2_hand_left_verts = hand_model_left.vertices.detach().cpu().numpy()
        p2_hand_right_verts = hand_model_right.vertices.detach().cpu().numpy()
        
    hand_verts_idx_left = hand_model_left.lhand_verts
    hand_verts_idx_right = hand_model_right.rhand_verts
    input_p1_hand_left_verts = input_p1_body_vertices[:,hand_verts_idx_left].detach().cpu().numpy()
    input_p2_hand_left_verts = input_p2_body_vertices[:,hand_verts_idx_left].detach().cpu().numpy()
    input_p1_hand_right_verts = input_p1_body_vertices[:,hand_verts_idx_right].detach().cpu().numpy()
    input_p2_hand_right_verts = input_p2_body_vertices[:,hand_verts_idx_right].detach().cpu().numpy()

    hand_left_faces = hand_model_left.hand_faces.detach().cpu().numpy()
    hand_right_faces = hand_model_right.hand_faces.detach().cpu().numpy()
    hand_left_faces = close_surface(hand_left_faces)
    hand_right_faces = close_surface(hand_right_faces)

    # Save data in numpy 1.23 compatibility format:
    np.savez_compressed(
        save_path,
        num_frames=num_frames,
        output_p1_hand_left_verts=p1_hand_left_verts,
        output_p1_hand_right_verts=p1_hand_right_verts,
        output_p2_hand_left_verts=p2_hand_left_verts,
        output_p2_hand_right_verts=p2_hand_right_verts,
        output_p1_joints=p1_joints,
        output_p2_joints=p2_joints,
        input_p1_hand_left_verts=input_p1_hand_left_verts,
        input_p2_hand_left_verts=input_p2_hand_left_verts,
        input_p1_hand_right_verts=input_p1_hand_right_verts,
        input_p2_hand_right_verts=input_p2_hand_right_verts,
        input_p1_joints=input_p1_joints,
        input_p2_joints=input_p2_joints,
        hand_left_faces=hand_left_faces,
        hand_right_faces=hand_right_faces,
        obj_verts=obj_verts,
        obj_faces=obj_faces,
        allow_pickle=True  # for potential lists/objects; adjust as required
    )