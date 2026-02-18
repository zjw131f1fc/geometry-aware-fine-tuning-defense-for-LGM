"""
Blender script to render Objaverse objects in OmniObject3D format.
Fixed version with proper object loading, rendering, and multi-layer sampling.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--object_uid", type=str, required=True)
parser.add_argument("--num_views", type=int, default=100)
parser.add_argument("--resolution", type=int, default=800)
parser.add_argument("--camera_dist", type=float, default=2.0)
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--fast", action="store_true", help="Use EEVEE engine for faster rendering (lower quality)")

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# Override engine if --fast is specified
if args.fast:
    args.engine = "BLENDER_EEVEE"

print(f"Rendering {args.object_uid}")
print(f"Object path: {args.object_path}")
print(f"Num views: {args.num_views}")
print(f"Engine: {args.engine}")


def reset_scene():
    """Reset scene to clean state."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)


def setup_scene():
    """Setup basic scene."""
    scene = bpy.context.scene
    render = scene.render

    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = args.resolution
    render.resolution_y = args.resolution
    render.resolution_percentage = 100

    if args.engine == "CYCLES":
        # Enable GPU rendering
        scene.cycles.device = "GPU"

        # Configure GPU devices
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences

        # Enable all available GPU devices
        cprefs.compute_device_type = 'CUDA'  # or 'OPTIX' for newer GPUs

        # Get available devices
        cprefs.get_devices()

        # Enable all CUDA devices
        for device in cprefs.devices:
            if device.type == 'CUDA':
                device.use = True
                print(f"[DEBUG] Enabled GPU: {device.name}")

        # Optimize Cycles settings for speed
        scene.cycles.samples = 64  # Reduced from 128 for faster rendering
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.cycles.transparent_max_bounces = 3
        scene.cycles.transmission_bounces = 3
        scene.cycles.filter_width = 0.01
        scene.cycles.use_denoising = True

        # Additional speed optimizations
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01

    # Transparent background (like OmniObject3D)
    scene.render.film_transparent = True


def add_camera():
    """Add camera to scene."""
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Camera"
    bpy.context.scene.camera = camera
    camera.data.lens = 35
    camera.data.sensor_width = 32
    return camera


def add_lighting():
    """Add lighting to scene."""
    bpy.ops.object.light_add(type='SUN')
    sun = bpy.context.object
    sun.data.energy = 3.0
    sun.location = (0, 0, 10)
    sun.rotation_euler = (0.785, 0, 0)

    bpy.ops.object.light_add(type='AREA')
    area = bpy.context.object
    area.data.energy = 500
    area.data.size = 10
    area.location = (5, 5, 5)

    bpy.ops.object.light_add(type='AREA')
    area2 = bpy.context.object
    area2.data.energy = 300
    area2.data.size = 10
    area2.location = (-5, -5, 3)


def load_object(filepath):
    """Load object from file."""
    print(f"[DEBUG] Loading object: {filepath}")
    print(f"[DEBUG] File exists: {os.path.exists(filepath)}")
    print(f"[DEBUG] File size: {os.path.getsize(filepath) / 1024:.2f} KB")

    try:
        if filepath.endswith('.glb') or filepath.endswith('.gltf'):
            bpy.ops.import_scene.gltf(filepath=filepath)
        elif filepath.endswith('.obj'):
            bpy.ops.import_scene.obj(filepath=filepath)
        elif filepath.endswith('.fbx'):
            bpy.ops.import_scene.fbx(filepath=filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to load object: {e}")
        return []

    imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    print(f"[DEBUG] Loaded {len(imported_objects)} mesh objects")

    for obj in imported_objects:
        print(f"[DEBUG]   - Object: {obj.name}, vertices: {len(obj.data.vertices)}, faces: {len(obj.data.polygons)}")

    return imported_objects


def ensure_materials(objects):
    """Ensure all objects have materials (add default if missing)."""
    for obj in objects:
        if not obj.data.materials:
            # Only add material if object has none
            mat = bpy.data.materials.new(name="DefaultMaterial")
            mat.use_nodes = True

            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
                bsdf.inputs['Roughness'].default_value = 0.5

            obj.data.materials.append(mat)
            print(f"[DEBUG] Added default material to {obj.name}")
        else:
            print(f"[DEBUG] Keeping original materials for {obj.name}")


def normalize_scene(objects):
    """Normalize objects to fit in unit sphere centered at origin."""
    if not objects:
        print("[ERROR] No objects to normalize")
        return 1.0

    print(f"[DEBUG] Normalizing {len(objects)} objects")

    def scene_bbox():
        """Calculate bounding box of all mesh objects."""
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        for obj in bpy.context.scene.objects.values():
            if isinstance(obj.data, bpy.types.Mesh):
                for coord in obj.bound_box:
                    coord = Vector(coord)
                    coord = obj.matrix_world @ coord
                    bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                    bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        return Vector(bbox_min), Vector(bbox_max)

    def scene_root_objects():
        """Get all root objects (objects without parents)."""
        for obj in bpy.context.scene.objects.values():
            if not obj.parent:
                yield obj

    # Calculate initial bounding box
    bbox_min, bbox_max = scene_bbox()
    size_x = bbox_max.x - bbox_min.x
    size_y = bbox_max.y - bbox_min.y
    size_z = bbox_max.z - bbox_min.z
    size = max(size_x, size_y, size_z)

    print(f"[DEBUG] Bounding box: min={bbox_min}, max={bbox_max}")
    print(f"[DEBUG] Object dimensions: X={size_x:.4f}, Y={size_y:.4f}, Z={size_z:.4f}")
    print(f"[DEBUG] Max dimension: {size:.4f}")

    if size < 0.0001:
        print(f"[ERROR] Object is too small (size={size})")
        return 1.0

    # Scale all root objects to fit in unit sphere
    scale_factor = 1.0 / size
    print(f"[DEBUG] Scale factor: {scale_factor}")

    for obj in scene_root_objects():
        obj.scale = obj.scale * scale_factor

    # Update scene
    bpy.context.view_layer.update()

    # Center objects at origin
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2

    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.context.view_layer.update()

    # Verify final state
    final_bbox_min, final_bbox_max = scene_bbox()
    final_size = max(
        final_bbox_max.x - final_bbox_min.x,
        final_bbox_max.y - final_bbox_min.y,
        final_bbox_max.z - final_bbox_min.z
    )

    print(f"[DEBUG] Final object size: {final_size:.4f}")
    print(f"[DEBUG] Final center: {(final_bbox_min + final_bbox_max) / 2}")

    if final_size < 0.1 or final_size > 10:
        print(f"[WARNING] Final object size ({final_size:.4f}) is outside normal range [0.1, 10]")

    return scale_factor


def setup_camera_at_position(camera, azimuth, elevation, distance):
    """Position camera at spherical coordinates."""
    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)

    camera.location = (x, y, z)

    direction = Vector((0, 0, 0)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    return camera.matrix_world.copy()


def render_views():
    """Render all views."""
    output_path = Path(args.output_dir) / args.object_uid / "render"
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] Output directory: {output_path}")

    reset_scene()
    setup_scene()

    camera = add_camera()
    add_lighting()

    objects = load_object(args.object_path)
    if not objects:
        print("[ERROR] No objects loaded! Cannot render.")
        # Save error info
        error_file = output_path / "error.txt"
        with open(error_file, 'w') as f:
            f.write(f"Failed to load object: {args.object_path}\n")
        return

    ensure_materials(objects)
    scale_factor = normalize_scene(objects)

    # Verify scene is ready
    print(f"[DEBUG] Scene setup complete:")
    print(f"[DEBUG]   - Camera: {camera.name} at {camera.location}")
    print(f"[DEBUG]   - Camera FOV: {2 * math.atan(camera.data.sensor_width / (2 * camera.data.lens)) * 180 / math.pi:.2f}°")
    print(f"[DEBUG]   - Render engine: {bpy.context.scene.render.engine}")
    print(f"[DEBUG]   - Resolution: {args.resolution}x{args.resolution}")
    print(f"[DEBUG]   - Camera distance: {args.camera_dist}")

    transforms_data = {
        "camera_angle_x": 2 * math.atan(camera.data.sensor_width / (2 * camera.data.lens)),
        "frames": []
    }

    # Multi-group orthogonal view sampling for LGM
    # Strategy: Multiple sets of 4 orthogonal views (0°, 90°, 180°, 270°) with different offsets
    # All views at elevation=0° (horizontal) for LGM compatibility

    if args.num_views <= 20:
        num_groups = 4  # 4 groups × 4 views = 16 views
        angle_step = 22.5  # Groups at 0°, 22.5°, 45°, 67.5°
    elif args.num_views <= 50:
        num_groups = 8  # 8 groups × 4 views = 32 views
        angle_step = 11.25  # More groups for better coverage
    else:
        num_groups = 16  # 16 groups × 4 views = 64 views
        angle_step = 5.625

    # Fixed elevation at 0° (horizontal view, LGM standard)
    elevation_deg = 0.0
    elevation = elevation_deg * math.pi / 180.0

    view_idx = 0
    empty_renders = []

    for group_idx in range(num_groups):
        if view_idx >= args.num_views:
            break

        # Starting angle for this group
        start_angle = group_idx * angle_step

        # Generate 4 orthogonal views with this starting angle
        for i in range(4):
            if view_idx >= args.num_views:
                break

            azimuth_deg = (start_angle + i * 90) % 360
            azimuth = azimuth_deg * math.pi / 180.0

            # Setup camera
            transform_matrix = setup_camera_at_position(camera, azimuth, elevation, args.camera_dist)

            # Render
            output_file = str(images_dir / f"r_{view_idx}.png")
            bpy.context.scene.render.filepath = output_file

            try:
                bpy.ops.render.render(write_still=True)

                # Check if file was created and has reasonable size
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    if file_size < 1000:  # Less than 1KB is suspicious
                        print(f"[WARNING] View {view_idx} rendered but file is very small ({file_size} bytes)")
                        empty_renders.append(view_idx)
                else:
                    print(f"[ERROR] View {view_idx} render file not created!")
                    empty_renders.append(view_idx)

            except Exception as e:
                print(f"[ERROR] Failed to render view {view_idx}: {e}")
                empty_renders.append(view_idx)

            # Save transform
            transforms_data["frames"].append({
                "file_path": f"r_{view_idx}",
                "rotation": azimuth,
                "transform_matrix": [[float(x) for x in row] for row in transform_matrix],
                "scale": scale_factor
            })

            print(f"[INFO] Rendered view {view_idx+1}/{args.num_views} (group {group_idx+1}/{num_groups}, azimuth {azimuth_deg:.1f}°, elevation {elevation_deg:.1f}°)")
            view_idx += 1

    # Save transforms.json
    with open(output_path / "transforms.json", 'w') as f:
        json.dump(transforms_data, f, indent=4)

    # Save debug summary
    debug_file = output_path / "debug_info.txt"
    with open(debug_file, 'w') as f:
        f.write(f"Object: {args.object_uid}\n")
        f.write(f"Object path: {args.object_path}\n")
        f.write(f"Scale factor: {scale_factor}\n")
        f.write(f"Total views: {view_idx}\n")
        f.write(f"Empty/problematic renders: {len(empty_renders)}\n")
        if empty_renders:
            f.write(f"Empty render indices: {empty_renders}\n")
        f.write(f"Engine: {args.engine}\n")
        f.write(f"Resolution: {args.resolution}x{args.resolution}\n")

    if empty_renders:
        print(f"[WARNING] {len(empty_renders)} views had rendering issues: {empty_renders}")

    print(f"[INFO] Rendering complete: {output_path}")
    print(f"[INFO] Debug info saved to: {debug_file}")


if __name__ == "__main__":
    render_views()
