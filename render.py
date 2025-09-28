import pickle
import numpy as np
import torch

from vedo import Mesh, Plotter, Box, Sphere, Cylinder
from vedo.applications import AnimationPlayer

from hand_model import HandModel
from prim import *

device = torch.device("cuda")

with open("data/125_0_pseudo_gt_handsel2_all_stage_result.pkl", "rb") as f:
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

bone_pair_values = np.array(list(bone_pair.values()))

p1_bones_pos_tail = p1_joints[:, bone_pair_values[:, 0], :]
p1_bones_pos_head = p1_joints[:, bone_pair_values[:, 1], :]
p1_bones_pos = (p1_bones_pos_tail + p1_bones_pos_head) / 2
p1_bones_dir = p1_bones_pos_head - p1_bones_pos_tail
p1_bones_len = np.linalg.norm(p1_bones_dir, axis=2)
p1_bones_dir = p1_bones_dir / p1_bones_len[:, :, None]

p2_bones_pos_tail = p2_joints[:, bone_pair_values[:, 0], :]
p2_bones_pos_head = p2_joints[:, bone_pair_values[:, 1], :]
p2_bones_pos = (p2_bones_pos_tail + p2_bones_pos_head) / 2
p2_bones_dir = p2_bones_pos_head - p2_bones_pos_tail
p2_bones_len = np.linalg.norm(p2_bones_dir, axis=2)
p2_bones_dir = p2_bones_dir / p2_bones_len[:, :, None]

num_frames = 90
filtered_obj_verts_list = filtered_obj_verts_list[:num_frames]

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

# Create vedo mesh objects (initialize with first frame)
floot_color = 'white'
obj_color = 'lightgreen'
p1_body_color = 'lightblue'
p1_hand_color = 'blue'
p2_body_color = 'pink'
p2_hand_color = 'red'

obj_mesh = Mesh([filtered_obj_verts_list[0], obj_faces_list])
obj_mesh.c(obj_color).alpha(0.7)

p1_left_mesh = Mesh([p1_hand_left_verts[0], p1_hand_left_faces])
p1_left_mesh.c(p1_hand_color).alpha(0.8)

p1_right_mesh = Mesh([p1_hand_right_verts[0], p1_hand_right_faces])
p1_right_mesh.c(p1_hand_color).alpha(0.8)

p2_left_mesh = Mesh([p2_hand_left_verts[0], p2_hand_left_faces])
p2_left_mesh.c(p2_hand_color).alpha(0.8)

p2_right_mesh = Mesh([p2_hand_right_verts[0], p2_hand_right_faces])
p2_right_mesh.c(p2_hand_color).alpha(0.8)

floor = Box(size=(5, 0.01, 5)).c(floot_color)

p1_joint_spheres = []
for i in joint_radii:
    p1_joint_spheres.append(Sphere(p1_joints[0][i], r=joint_radii[i]).c(p1_body_color))

p2_joint_spheres = []
for i in joint_radii:
    p2_joint_spheres.append(Sphere(p2_joints[0][i], r=joint_radii[i]).c(p2_body_color))

p1_bone_cylinders = []
for i in bone_pair:
    p1_bone_cylinders.append(Cylinder(p1_bones_pos[0][i], r=bone_radii[i], height=p1_bones_len[0][i], axis=p1_bones_dir[0][i]).c(p1_body_color))

p2_bone_cylinders = []
for i in bone_pair:
    p2_bone_cylinders.append(Cylinder(p2_bones_pos[0][i], r=bone_radii[i], height=p2_bones_len[0][i], axis=p2_bones_dir[0][i]).c(p2_body_color))

def update_meshes(frame):
    obj_mesh.points = filtered_obj_verts_list[frame]
    p1_left_mesh.points = p1_hand_left_verts[frame]
    p1_right_mesh.points = p1_hand_right_verts[frame]
    p2_left_mesh.points = p2_hand_left_verts[frame]
    p2_right_mesh.points = p2_hand_right_verts[frame]
    
    for i in range(len(p1_joint_spheres)):
        p1_joint_spheres[i].pos(p1_joints[frame][i])
    for i in range(len(p2_joint_spheres)):
        p2_joint_spheres[i].pos(p2_joints[frame][i])
    for i in range(len(p1_bone_cylinders)):
        plt.remove(p1_bone_cylinders[i])
        p1_bone_cylinders[i] = Cylinder(p1_bones_pos[frame][i], r=bone_radii[i], height=p1_bones_len[frame][i], axis=p1_bones_dir[frame][i]).c(p1_body_color)
        plt.add(p1_bone_cylinders[i])
        
    for i in range(len(p2_bone_cylinders)):
        plt.remove(p2_bone_cylinders[i])
        p2_bone_cylinders[i] = Cylinder(p2_bones_pos[frame][i], r=bone_radii[i], height=p2_bones_len[frame][i], axis=p2_bones_dir[frame][i]).c(p2_body_color)
        plt.add(p2_bone_cylinders[i])
        
    plt.render()

plt = AnimationPlayer(update_meshes, irange=(0, num_frames-1), loop=True, dt = 33)

plt.add(obj_mesh, p1_left_mesh, p1_right_mesh, p2_left_mesh, p2_right_mesh, floor, *p1_joint_spheres, *p2_joint_spheres, *p1_bone_cylinders, *p2_bone_cylinders)
plt.show()
