import bpy
import mathutils
import os
import sys
import threading
import argparse
from contextlib import contextmanager
import numpy as np
import math

@contextmanager
def stdout_redirected(keyword=None, on_match=None):
    """
    Redirect stdout to a pipe and scan it live for a keyword.
    If found, call `on_match(line)` or print it.
    """
    original_fd = sys.stdout.fileno()
    saved_fd = os.dup(original_fd)

    read_fd, write_fd = os.pipe()

    def reader():
        with os.fdopen(read_fd) as read_pipe:
            string = ""
            for line in iter(read_pipe.readline, ''):
                if keyword is not None and keyword in line and on_match:
                    os.write(saved_fd, b'\r')
                    os.write(saved_fd, b' ' * len(string))
                    os.write(saved_fd, b'\r')
                    string = on_match(line)
                    os.write(saved_fd, string)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    os.dup2(write_fd, original_fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, original_fd)
        os.close(write_fd)
        thread.join()
        os.close(saved_fd)

def convert_to_blender_coord(x):
    new_x = x.copy()
    new_x[..., 1], new_x[..., 2] = x[..., 2], x[..., 1]
    return new_x

def cleanup_existing_objects():
    """Hide existing mesh objects except Plane"""
    sample_collection = bpy.data.collections.get('Sample')
    if sample_collection:
        for obj in sample_collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

def setup_background_scene(scene_no):
    """Setup background scene"""
    scenes_collection = bpy.data.collections.get('Scenes')
    if not scenes_collection:
        print("Warning: 'Scenes' collection not found")
        return
    if f'Scene{scene_no}' not in [c.name for c in scenes_collection.children]:
        if scene_no != 0:
            print(f"Warning: 'Scene{scene_no}' not found in 'Scenes' collection")
        return
    
    background_objects = []
    for scene_collection in scenes_collection.children:
        scene_collection.hide_render = scene_collection.name != f'Scene{scene_no}'
        background_objects.extend([obj for obj in scene_collection.objects if obj.type == 'MESH'])
    
    floor_obj = bpy.data.objects.get('Floor')
    if floor_obj:
        background_objects.append(floor_obj)

    for obj in background_objects:
        bpy.context.scene.cursor.location = (0, 0, 0)
        with bpy.context.temp_override(selected_editable_objects=[obj]):
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    
    return background_objects

def setup_render_settings(render_high):
    """Configure render settings based on quality mode"""
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'

    if render_high:
        setup_high_quality_settings()
    else:
        setup_low_quality_settings()

def setup_low_quality_settings():
    """Configure settings for fast, low-quality rendering"""
    if bpy.app.version >= (4, 2, 0):
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
        # bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.resolution_percentage = 50

    # Use hasattr to avoid attribute errors
    eevee = getattr(bpy.context.scene, 'eevee', None)
    if eevee:
        if hasattr(eevee, 'taa_render_samples'):
            eevee.taa_render_samples = 16
        for attr in ['use_soft_shadows', 'use_bloom', 'use_ssr', 'use_ssr_refraction']:
            if hasattr(eevee, attr):
                setattr(eevee, attr, False)

    # bpy.context.scene.use_nodes = False
    # bpy.context.scene.render.use_compositing = False
    bpy.context.scene.render.use_sequencer = False
    bpy.context.scene.render.film_transparent = False

def setup_high_quality_settings():
    """Configure settings for high-quality rendering"""
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 1024
    bpy.context.scene.cycles.use_denoising = False # device issue
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.adaptive_threshold = 0.02
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100
    # bpy.context.scene.use_nodes = False
    # bpy.context.scene.render.use_compositing = True
    bpy.context.scene.render.use_sequencer = True
    bpy.context.scene.render.film_transparent = False

def setup_animation_settings(num_frames):
    """Configure animation and frame settings"""
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames

def setup_camera_setting(camera_setting):
    camera = bpy.context.scene.camera
    camera.location = camera_setting['cam_location']
    
    look_at = camera_setting.get('look_at')
    
    if look_at is not None:
        camera.data.angle = 0.07
        
        # Animate camera to look at look_at point for each frame
        # look_at is (T, 3) location sequence
        num_frames = look_at.shape[0]
        cam_location_vec = mathutils.Vector(camera_setting['cam_location'])
        
        for frame in range(num_frames):
            anim_frame = frame * 2 + 1
            target_point = mathutils.Vector(look_at[frame])
            
            # Calculate direction from camera to target
            direction = target_point - cam_location_vec
            direction.normalize()
            
            # Calculate rotation to look at target
            # Use track_quat to get rotation that points -Z axis at target
            cam_rotation = direction.to_track_quat('-Z', 'Y').to_euler()
            
            camera.rotation_euler = cam_rotation
            camera.keyframe_insert(data_path="rotation_euler", frame=anim_frame)
    else:
        # Static camera rotation
        camera.rotation_euler = camera_setting['cam_rotation']
    
    center = camera_setting['center']
    angle = camera_setting['angle']
    
    scenes_collection = bpy.data.collections.get('Scenes')
    background_objects = []
    for scene_collection in scenes_collection.children:
        background_objects.extend([obj for obj in scene_collection.objects if obj.type == 'MESH'])
    
    floor_obj = bpy.data.objects.get('Floor')
    if floor_obj:
        background_objects.append(floor_obj)

    for obj in background_objects:
        obj.location = center
        obj.rotation_euler = (0, 0, angle)
        
    sun = bpy.data.objects.get('Sun')
    light_rotation = (math.radians(30), 0, angle + math.radians(20))
    if sun:
        sun.rotation_euler = light_rotation

def setup_floor_render(figure, figure_floor, checkerboard):
    floor_obj = bpy.data.objects.get('Floor')
    
    if figure:
        floor_obj.hide_render = True
        floor_obj.hide_viewport = True
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        
    if figure_floor:
        floor_obj = bpy.data.objects.get('Floor')
        if hasattr(floor_obj, 'is_shadow_catcher'):
            floor_obj.is_shadow_catcher = True
        if hasattr(floor_obj, 'use_shadow_catcher'):
            floor_obj.use_shadow_catcher = True
        if floor_obj.data.materials:
            floor_mat = floor_obj.data.materials[0]
            if floor_mat.use_nodes:
                for node in floor_mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        node.inputs['Roughness'].default_value = 1.0
                        break
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        
    if checkerboard:
        checkerboard_material = bpy.data.materials.get('CheckerBoard')
        if checkerboard_material and floor_obj:
            floor_obj.data.materials.clear()
            floor_obj.data.materials.append(checkerboard_material)
        
def render_animation(video_path, camera_settings):
    """Render animation from different camera angles"""
    for camera_setting in camera_settings:
        cam_text = camera_setting['text']
        video_path = video_path + f"_{cam_text}"
        bpy.context.scene.render.filepath = video_path
        setup_camera_setting(camera_setting)
        
        print(f"Rendering animation for {cam_text}...")
        with stdout_redirected(keyword="Fra:", on_match=lambda line: line[:-1].encode()):
            bpy.ops.render.render(animation=True)
        print()
        print(f"Saved to {video_path}")

def render_single_frame(output_path, camera_settings, frame_no):
    """Render a single frame (still image) from different camera angles"""
    # Use image output for stills
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.frame_current = 1
        
    for camera_setting in camera_settings:
        cam_text = camera_setting['text']
        filepath = f"{output_path}_{cam_text}_f{frame_no:04d}.png"
        bpy.context.scene.render.filepath = filepath
        setup_camera_setting(camera_setting)
        
        print(f"Rendering frame {frame_no} for {cam_text}...")
        with stdout_redirected(keyword="Fra:", on_match=lambda line: line[:-1].encode()):
            bpy.ops.render.render(animation=False, write_still=True)
        print(f"Saved to {filepath}")