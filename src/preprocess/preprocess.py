import pickle
import torch
import os
import numpy as np

from .hand_model import HandModel

def preprocess_pkl_file(pkl_path, save_path):
    if os.path.exists(save_path):
        print(f"Preprocessed data already exists at {save_path}")
        return
        
    device = torch.device("cuda")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        
        obj_faces_list = data["obj_faces_list"]
        original_obj_verts_list = data["original_obj_verts_list"]
        filtered_obj_verts_list = data["filtered_obj_verts_list"]
        filtered_obj_T = data["filtered_obj_T"]
        original_obj_T = data["original_obj_T"]
        
        stage1_result = data["stage1_result"]
        stage2_result = data["stage2_result"]
        stage3_result = data["stage3_result"]
        
        stage2_pseudo_gt_p1 = stage2_result["pseudo_gt_p1"]
        stage2_pseudo_gt_p2 = stage2_result["pseudo_gt_p2"]
        stage3_pseudo_gt_p1 = stage3_result["pseudo_gt_p1"]
        stage3_pseudo_gt_p2 = stage3_result["pseudo_gt_p2"]
        
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

    obj_verts = filtered_obj_verts_list     
    obj_faces = obj_faces_list

    num_frames_p1_joints = p1_joints.shape[0]
    num_frames_p2_joints = p2_joints.shape[0]
    num_frames_obj = obj_verts.shape[0]

    num_frames = min(num_frames_p1_joints, num_frames_p2_joints, num_frames_obj)

    p1_joints = p1_joints[:num_frames]
    p2_joints = p2_joints[:num_frames]
    obj_verts = obj_verts[:num_frames]

    hand_model_left = HandModel(left_hand=True, gender="female", device=device, batch_size=num_frames)
    hand_model_right = HandModel(left_hand=False, gender="female", device=device, batch_size=num_frames)

    p1_hand_left_verts = []
    p2_hand_left_verts = []
    p1_hand_right_verts = []
    p2_hand_right_verts = []

    with torch.no_grad():
        hand_model_left.set_parameters(p1_hand_parmas_left, skip_left_mirror=True)
        hand_model_right.set_parameters(p1_hand_parmas_right, skip_left_mirror=True)
        
        p1_hand_left_verts = hand_model_left.vertices.detach().cpu().numpy()
        p1_hand_right_verts = hand_model_right.vertices.detach().cpu().numpy()
        
        hand_model_left.set_parameters(p2_hand_parmas_left, skip_left_mirror=True)
        hand_model_right.set_parameters(p2_hand_parmas_right, skip_left_mirror=True)
        
        p2_hand_left_verts = hand_model_left.vertices.detach().cpu().numpy()
        p2_hand_right_verts = hand_model_right.vertices.detach().cpu().numpy()

    p1_hand_left_faces = hand_model_left.hand_faces.detach().cpu().numpy()
    p1_hand_right_faces = hand_model_right.hand_faces.detach().cpu().numpy()
    p2_hand_left_faces = hand_model_left.hand_faces.detach().cpu().numpy()
    p2_hand_right_faces = hand_model_right.hand_faces.detach().cpu().numpy()

    # Save data in numpy 1.23 compatibility format:
    np.savez_compressed(
        save_path,
        num_frames=num_frames,
        p1_hand_left_verts=p1_hand_left_verts,
        p1_hand_right_verts=p1_hand_right_verts,
        p1_hand_left_faces=p1_hand_left_faces,
        p1_hand_right_faces=p1_hand_right_faces,
        p1_joints=p1_joints,
        p2_hand_left_verts=p2_hand_left_verts,
        p2_hand_right_verts=p2_hand_right_verts,
        p2_hand_left_faces=p2_hand_left_faces,
        p2_hand_right_faces=p2_hand_right_faces,
        p2_joints=p2_joints,
        obj_verts=obj_verts,
        obj_faces=obj_faces,
        allow_pickle=True  # for potential lists/objects; adjust as required
    )