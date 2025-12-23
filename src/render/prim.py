import bpy
import numpy as np

from render.bones import Bones

def create_mesh_for_frame(verts, faces, frame_num, material):
    """Create mesh object for a specific frame"""
    # Create new mesh datablock
    mesh = bpy.data.meshes.new(f"Frame_{frame_num}_mesh")
    obj = bpy.data.objects.new(f"Frame_{frame_num}", mesh)

    # Create mesh from vertices and faces
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    # Link object to scene
    bpy.context.scene.collection.objects.link(obj)
    
    # Add material
    obj.data.materials.append(bpy.data.materials[material])
    
    # Smooth shading
    with bpy.context.temp_override(selected_editable_objects=[obj]):
        bpy.ops.object.shade_smooth()
        
    return obj

def create_sphere(material, radius=0.05):
    """Create sphere object for joint visualization"""
    # Set different radius for each joint
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius)
    sphere = bpy.context.active_object
    sphere.data.materials.append(bpy.data.materials[material])
    with bpy.context.temp_override(selected_editable_objects=[sphere]):
        bpy.ops.object.shade_smooth()
        
    return sphere

def create_cylinder(material, radius=0.05):
    """Create a cylinder object connecting two joints to represent a bone (side surface only, no end caps)"""
    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=radius)
    cylinder = bpy.context.active_object
    
    # Remove end cap faces, keep only side surface
    # Select end cap faces by checking their normals (they point along Z axis)
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = cylinder.data
    
    # Deselect all faces first
    for face in mesh.polygons:
        face.select = False
    
    # Select faces with normals aligned with Z axis (end caps)
    for face in mesh.polygons:
        if abs(face.normal.z) > 0.9:  # End cap (normal mostly along Z axis)
            face.select = True
    
    # Switch to edit mode and delete selected end cap faces
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.delete(type='FACE')
    
    # Add loop cuts to sides for better subdivision
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Add material
    cylinder.data.materials.append(bpy.data.materials[material])
    
    # Smooth shading
    with bpy.context.temp_override(selected_editable_objects=[cylinder]):
        bpy.ops.object.shade_smooth()
    
    return cylinder

def setup_keyframe(obj, frame_num):
    """Set up keyframes for visibility of object"""
    # Hide at start
    obj.hide_render = True
    obj.hide_viewport = True
    obj.keyframe_insert(data_path="hide_render", frame=1)
    obj.keyframe_insert(data_path="hide_viewport", frame=1)
    
    # Show at current frame
    obj.hide_render = False
    obj.hide_viewport = False
    obj.keyframe_insert(data_path="hide_render", frame=frame_num)
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
    
    # Hide at next frame
    obj.hide_render = True
    obj.hide_viewport = True
    obj.keyframe_insert(data_path="hide_render", frame=frame_num + 1)
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_num + 1)

def setup_mesh_keyframes(verts_list, obj_faces_list, material):
    """Create mesh objects for each frame"""
    num_frames = len(verts_list)*2-1
    for frame_num in range(1, num_frames+1):
        # Create mesh for the frame
        if frame_num == num_frames:
            verts = verts_list[-1]
        elif frame_num % 2 == 0:
            verts = verts_list[frame_num//2]
        else:
            verts = (verts_list[frame_num//2] + verts_list[frame_num//2+1]) / 2
        obj = create_mesh_for_frame(verts, obj_faces_list, frame_num, material)
        setup_keyframe(obj, frame_num)

def setup_sphere_keyframes(sphere, pos):
    frame_num = pos.shape[0]
    
    for frame in range(frame_num):
        anim_frame = frame * 2 + 1
        sphere.location = pos[frame]
        sphere.keyframe_insert(data_path="location", frame=anim_frame)

def setup_cylinder_keyframes(cylinder, pos, direction, height):
    frame_num = direction.shape[0]
    
    for frame in range(frame_num):
        anim_frame = frame * 2 + 1
        
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction[frame])
        rotation_angle = np.arccos(np.dot(z_axis, direction[frame]))
        cylinder.scale[2] = height[frame] / 2
        cylinder.keyframe_insert(data_path="scale", frame=anim_frame)
        
        if np.any(rotation_axis):
            cylinder.rotation_mode = 'AXIS_ANGLE'
            cylinder.rotation_axis_angle = [rotation_angle] + list(rotation_axis.tolist())
        
        cylinder.location = pos[frame]
        cylinder.keyframe_insert(data_path="location", frame=anim_frame)
        cylinder.keyframe_insert(data_path="rotation_axis_angle", frame=anim_frame)

def setup_joints_and_bones(joints, material):
    bones = Bones(joints)
    
    for sphere in bones.spheres:
        s = create_sphere(material, sphere.r)
        setup_sphere_keyframes(s, sphere.pos)
        
    for cylinder in bones.cylinders:
        c = create_cylinder(material, cylinder.r)
        setup_cylinder_keyframes(c, cylinder.pos, cylinder.direction, cylinder.height)