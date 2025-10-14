import numpy as np

from render.index import *

def interpolate(a, b, t):
  return a + (b - a) * t

def dir(a, b):
  dir = b - a
  dir = dir / np.linalg.norm(dir)
  return dir

class Bones:
  def __init__(self, joints: np.ndarray):
    """
    joints: (frames, 22, 3)
    """
    self.frames = joints.shape[0]
    self.spheres: list[Sphere] = []
    self.cylinders: list[Cylinder] = []
    self.set_bones(joints)
    
  def set_bones(self, joints):
    pelvis = joints[:,JOINT_GLOBAL]
    left_hip = joints[:,JOINT_LEFT_HIP]
    right_hip = joints[:,JOINT_RIGHT_HIP]
    spine1 = joints[:,JOINT_SPINE1]
    left_knee = joints[:,JOINT_LEFT_KNEE]
    right_knee = joints[:,JOINT_RIGHT_KNEE]
    spine2 = joints[:,JOINT_SPINE2]
    left_ankle = joints[:,JOINT_LEFT_ANKLE]
    right_ankle = joints[:,JOINT_RIGHT_ANKLE]
    spine3 = joints[:,JOINT_SPINE3]
    left_toe = joints[:,JOINT_LEFT_TOE]
    right_toe = joints[:,JOINT_RIGHT_TOE]
    neck = joints[:,JOINT_NECK]
    head = joints[:,JOINT_HEAD]
    left_collar = joints[:,JOINT_LEFT_COLLAR]
    right_collar = joints[:,JOINT_RIGHT_COLLAR]
    left_shoulder = joints[:,JOINT_LEFT_SHOULDER]
    right_shoulder = joints[:,JOINT_RIGHT_SHOULDER]
    left_elbow = joints[:,JOINT_LEFT_ELBOW]
    right_elbow = joints[:,JOINT_RIGHT_ELBOW]
    left_wrist = joints[:,JOINT_LEFT_WRIST]
    right_wrist = joints[:,JOINT_RIGHT_WRIST]
    
    left_shin = interpolate(left_knee, left_ankle, 0.4)
    right_shin = interpolate(right_knee, right_ankle, 0.4)
    
    left_arm = interpolate(left_elbow, left_wrist, 0.6)
    right_arm = interpolate(right_elbow, right_wrist, 0.6)
    
    chin = interpolate(neck, head, 0.5)
    left_collar_end = interpolate(left_collar, left_shoulder, 0.7)
    right_collar_end = interpolate(right_collar, right_shoulder, 0.7)
    
    shoulder_dir = dir(left_shoulder, right_shoulder)
    hip_dir = dir(left_hip, right_hip)
    
    lower_spine = interpolate(pelvis, spine1, 0.2)
    lower_spine_length = 0.5
    lower_spine_size = 0.07
    lower_spine_start = lower_spine - hip_dir * lower_spine_length
    lower_spine_end = lower_spine + hip_dir * lower_spine_length
    
    upper_spine = interpolate(spine1, spine2, 0.7)
    upper_spine_length = 0.5
    upper_spine_size = 0.07
    upper_spine_start = upper_spine - shoulder_dir * upper_spine_length
    upper_spine_end = upper_spine + shoulder_dir * upper_spine_length
    
    middle_spine = interpolate(lower_spine, upper_spine, 0.5)
    middle_spine_length = 0.4
    middle_spine_size = 0.07
    middle_spine_start = middle_spine - shoulder_dir * middle_spine_length
    middle_spine_end = middle_spine + shoulder_dir * middle_spine_length
    
    chest = interpolate(spine3, neck, 0.1)
    chest_length = 0.6
    chest_size = 0.075
    chest_start = chest - shoulder_dir * chest_length
    chest_end = chest + shoulder_dir * chest_length
    
    self.add_bone(lower_spine_start, lower_spine_end, lower_spine_size)
    self.add_bone(upper_spine_start, upper_spine_end, upper_spine_size)
    self.add_bone(middle_spine_start, middle_spine_end, middle_spine_size)
    self.add_bone(chest_start, chest_end, chest_size)
    
    hip_size = 0.07
    knee_size = 0.055
    upper_ankle_size = 0.045
    lower_ankle_size = 0.03
    toe_size = 0.03
    head_size = 0.08
    shoulder_size = 0.07
    elbow_size = 0.04
    upper_arm_size = 0.035
    lower_arm_size = 0.02
    neck_size = 0.04
    
    self.add_bone(left_hip, right_hip, hip_size)
    self.add_bone(left_hip, left_knee, knee_size)
    self.add_bone(right_hip, right_knee, knee_size)
    self.add_bone(left_knee, left_shin, upper_ankle_size)
    self.add_bone(right_knee, right_shin, upper_ankle_size)
    self.add_bone(left_shin, left_ankle, lower_ankle_size)
    self.add_bone(right_shin, right_ankle, lower_ankle_size)
    
    self.add_bone(left_ankle, left_toe, toe_size)
    self.add_bone(right_ankle, right_toe, toe_size)
    
    self.add_bone(chin, head, head_size)
    self.add_bone(left_collar, left_collar_end, shoulder_size)
    self.add_bone(right_collar, right_collar_end, shoulder_size)
    self.add_bone(left_shoulder, left_elbow, elbow_size)
    self.add_bone(right_shoulder, right_elbow, elbow_size)
    self.add_bone(left_elbow, left_arm, upper_arm_size)
    self.add_bone(right_elbow, right_arm, upper_arm_size)
    self.add_bone(left_arm, left_wrist, lower_arm_size)
    self.add_bone(right_arm, right_wrist, lower_arm_size)
    
    self.add_bone(chest, chin, neck_size)
    
  def add_bone(self, tails, heads, radius=0.05):
    bones_pos = (tails + heads) / 2
    bones_dir = heads - tails
    bones_len = np.linalg.norm(bones_dir, axis=1)
    bones_dir = bones_dir / bones_len[:, None]
    
    self.spheres.append(Sphere(tails, r=radius))
    self.spheres.append(Sphere(heads, r=radius))
    self.cylinders.append(Cylinder(bones_pos, direction=bones_dir, height=bones_len, r=radius))

class Sphere:
  def __init__(self, pos: np.ndarray, r: float = 0.05):
    """
    pos: (frames, 3)
    r: float
    """
    self.pos = pos
    self.r = r

class Cylinder:
  def __init__(self, pos: np.ndarray, direction: np.ndarray, height: np.ndarray, r: float = 0.05):
    """
    pos: (frames, 3)
    direction: (frames, 3)
    height: (frames, )
    r: float
    """
    self.pos = pos
    self.direction = direction
    self.r = r
    self.height = height