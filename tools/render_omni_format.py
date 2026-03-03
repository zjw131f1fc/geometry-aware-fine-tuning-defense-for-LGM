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
parser.add_argument("--num_views", type=int, default=48)
parser.add_argument("--resolution", type=int, default=800)
parser.add_argument("--camera_dist", type=float, default=2.0)
parser.add_argument(
    "--normalize_max_dim",
    type=float,
    default=1.0,
    help="Normalize imported asset so its max bbox dimension becomes this value (default: 1.0).",
)
parser.add_argument(
    "--auto_orient",
    type=str,
    default="none",
    choices=["none", "flat", "upright", "omni5"],
    help=(
        "Auto-orient imported asset before normalization. "
        "flat=minimize Z extent; upright=maximize Z extent; "
        "omni5=category-aware (plant upright, others flat) based on object_uid prefix."
    ),
)
parser.add_argument(
    "--auto_yaw",
    type=str,
    default="none",
    choices=["none", "camera_right", "camera_up", "world_x"],
    help=(
        "After auto_orient, rotate the asset around world-Z to align its major axis. "
        "camera_* requires --pose_template_json (uses view0 camera basis)."
    ),
)
parser.add_argument(
    "--yaw_flip",
    action="store_true",
    help="Flip the final auto_yaw by 180 degrees (useful if object looks upside-down in image).",
)
parser.add_argument(
    "--yaw_offset_deg",
    type=float,
    default=0.0,
    help="Extra yaw rotation (degrees) around world-Z after auto_yaw/yaw_flip (e.g. 90 for image-plane quarter turn).",
)
parser.add_argument(
    "--extra_pitch_deg",
    type=float,
    default=0.0,
    help=(
        "Extra rotation (degrees) around view0 camera right axis to fine-tune 'toe down/up'. "
        "Requires --pose_template_json."
    ),
)
parser.add_argument(
    "--extra_roll_deg",
    type=float,
    default=0.0,
    help=(
        "Extra rotation (degrees) around view0 camera forward axis (image-plane rotation). "
        "Use ±90 to rotate the rendered object in the image plane. Requires --pose_template_json."
    ),
)
parser.add_argument("--elevations", type=str, default="-20,0,20",
                    help="Comma-separated elevation angles in degrees (default: -20,0,20)")
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--fast", action="store_true", help="Use EEVEE engine for faster rendering (lower quality)")
parser.add_argument(
    "--cycles_device",
    type=str,
    default="AUTO",
    choices=["AUTO", "GPU", "CPU"],
    help="Cycles device selection. AUTO picks GPU when available, otherwise CPU.",
)
parser.add_argument(
    "--pose_template_json",
    type=str,
    default="",
    help="Optional transforms.json template. If provided, render with exactly the same camera poses."
)

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# Override engine if --fast is specified
if args.fast:
    args.engine = "BLENDER_EEVEE"

print(f"Rendering {args.object_uid}")
print(f"Object path: {args.object_path}")
print(f"Num views: {args.num_views}")
print(f"Elevations: {args.elevations}")
print(f"Engine: {args.engine}")
if args.pose_template_json:
    print(f"Pose template: {args.pose_template_json}")


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
        # Configure Cycles device. Prefer GPU, but fall back to CPU on headless / no-GPU environments.
        prefs = bpy.context.preferences
        cprefs = prefs.addons["cycles"].preferences

        want = (args.cycles_device or "AUTO").upper()
        use_gpu = want in ("AUTO", "GPU")
        if want == "CPU":
            use_gpu = False

        if use_gpu:
            try:
                cprefs.compute_device_type = "CUDA"
                cprefs.get_devices()

                enabled = 0
                for device in cprefs.devices:
                    if getattr(device, "type", None) == "CUDA":
                        device.use = True
                        enabled += 1
                        print(f"[DEBUG] Enabled CUDA GPU: {device.name}")
                if enabled > 0:
                    scene.cycles.device = "GPU"
                else:
                    print("[WARN] No CUDA devices found; falling back to CPU for Cycles")
                    scene.cycles.device = "CPU"
            except Exception as e:
                print(f"[WARN] Failed to enable GPU for Cycles ({e}); falling back to CPU")
                scene.cycles.device = "CPU"
        else:
            scene.cycles.device = "CPU"

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
    """Load object from file.

    Returns:
        (mesh_objects, imported_objects)
        - mesh_objects: imported mesh objects (type == 'MESH')
        - imported_objects: all imported objects created by the importer (meshes/empties/etc.)
    """
    print(f"[DEBUG] Loading object: {filepath}")
    print(f"[DEBUG] File exists: {os.path.exists(filepath)}")
    print(f"[DEBUG] File size: {os.path.getsize(filepath) / 1024:.2f} KB")

    before = set(bpy.context.scene.objects)
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
        return [], []

    after = set(bpy.context.scene.objects)
    imported_objects = list(after - before)
    imported_meshes = [obj for obj in imported_objects if obj.type == "MESH"]

    print(f"[DEBUG] Imported objects: total={len(imported_objects)}, meshes={len(imported_meshes)}")

    for obj in imported_meshes:
        print(f"[DEBUG]   - Mesh: {obj.name}, vertices: {len(obj.data.vertices)}, faces: {len(obj.data.polygons)}")

    return imported_meshes, imported_objects


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


def relink_texture_if_needed(objects, object_path):
    """Fix common GSO texture path issue for OBJ+MTL imports."""
    if not object_path.lower().endswith(".obj"):
        return

    obj_path = Path(object_path)
    candidate_paths = [
        obj_path.parent / "texture.png",
        obj_path.parent.parent / "materials" / "textures" / "texture.png",
        obj_path.parent / "materials" / "textures" / "texture.png",
    ]
    texture_path = next((p for p in candidate_paths if p.exists()), None)
    if texture_path is None:
        return

    try:
        image = bpy.data.images.load(str(texture_path), check_existing=True)
    except Exception as e:
        print(f"[WARNING] Failed to load texture {texture_path}: {e}")
        return

    relink_count = 0
    for obj in objects:
        if obj.type != "MESH":
            continue
        for mat in obj.data.materials:
            if mat is None:
                continue
            if not mat.use_nodes:
                mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            bsdf = nodes.get("Principled BSDF")
            if bsdf is None:
                continue

            tex_nodes = [n for n in nodes if n.type == "TEX_IMAGE"]
            if tex_nodes:
                tex_node = tex_nodes[0]
            else:
                tex_node = nodes.new("ShaderNodeTexImage")
                tex_node.location = (bsdf.location.x - 350, bsdf.location.y + 120)
            tex_node.image = image

            for link in list(bsdf.inputs["Base Color"].links):
                links.remove(link)
            links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
            relink_count += 1

    if relink_count > 0:
        print(f"[DEBUG] Relinked texture to {relink_count} materials: {texture_path}")


def _infer_category_from_uid(object_uid: str):
    # GSO render scripts use "{cls}_{model}".
    if not object_uid or "_" not in object_uid:
        return None
    return object_uid.split("_", 1)[0]


def _collect_asset_points(mesh_objects):
    points = []
    for obj in mesh_objects:
        if obj.type != "MESH":
            continue
        mw = obj.matrix_world
        for v in obj.data.vertices:
            points.append(mw @ v.co)
    return points


def _collect_asset_faces(mesh_objects):
    """Collect face centers/normals/areas in world space (approx, polygon-level)."""
    faces = []
    for obj in mesh_objects:
        if obj.type != "MESH":
            continue
        mw = obj.matrix_world
        rot = mw.to_3x3()
        me = obj.data
        verts = me.vertices
        for poly in me.polygons:
            idx = poly.vertices
            if not idx:
                continue
            center = Vector((0.0, 0.0, 0.0))
            for vi in idx:
                center += mw @ verts[vi].co
            center /= float(len(idx))
            n = rot @ poly.normal
            if n.length > 0:
                n.normalize()
            faces.append((center, n, float(poly.area)))
    return faces


def auto_orient_asset(asset_root, mesh_objects, mode: str, object_uid: str = ""):
    """Rotate asset_root by 90° increments to match a simple 'flat/upright' heuristic."""
    mode = (mode or "none").lower()
    if mode == "none":
        return

    if mode == "omni5":
        cat = (_infer_category_from_uid(object_uid) or "").lower()
        mode = "upright" if cat == "plant" else "flat"
        print(f"[DEBUG] auto_orient=omni5 -> category={cat!r} -> mode={mode}")

    if mode not in ("flat", "upright"):
        print(f"[WARN] Unknown auto_orient mode: {mode}; skip")
        return

    points = _collect_asset_points(mesh_objects)
    if not points:
        print("[WARN] No vertices found for auto-orient; skip")
        return

    faces = _collect_asset_faces(mesh_objects)

    # Make rotation evaluation translation-invariant.
    centroid = Vector((0.0, 0.0, 0.0))
    for p in points:
        centroid += p
    centroid /= float(len(points))
    points = [p - centroid for p in points]

    if faces:
        faces = [(c - centroid, n, a) for (c, n, a) in faces]
        # Keep evaluation cheap for large meshes.
        if len(faces) > 5000:
            stride = (len(faces) // 5000) + 1
            faces = faces[::stride]

    angles = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    best = None

    def eval_rotation(r3):
        min_v = Vector((math.inf, math.inf, math.inf))
        max_v = Vector((-math.inf, -math.inf, -math.inf))
        zs = []
        xs = []
        ys = []
        for p in points:
            q = r3 @ p
            min_v.x = min(min_v.x, q.x)
            min_v.y = min(min_v.y, q.y)
            min_v.z = min(min_v.z, q.z)
            max_v.x = max(max_v.x, q.x)
            max_v.y = max(max_v.y, q.y)
            max_v.z = max(max_v.z, q.z)
            zs.append(float(q.z))
            xs.append(float(q.x))
            ys.append(float(q.y))
        ext = max_v - min_v

        # "Which side is down" heuristic:
        # if many vertices lie close to the minimum-Z plane AND cover a large XY footprint,
        # it's likely a stable "ground contact" side (works well for shoes/bowls/boxes).
        z_min = min(zs)
        z_max = max(zs)
        z_range = max(z_max - z_min, 1e-8)
        eps = max(0.02 * z_range, 1e-4)
        bottom_idx = [i for i, z in enumerate(zs) if (z - z_min) <= eps]
        top_idx = [i for i, z in enumerate(zs) if (z_max - z) <= eps]

        bottom_cnt = len(bottom_idx)
        top_cnt = len(top_idx)

        bottom_area = 0.0
        if bottom_cnt > 0:
            bx = [xs[i] for i in bottom_idx]
            by = [ys[i] for i in bottom_idx]
            bottom_area = (max(bx) - min(bx)) * (max(by) - min(by))

        top_area = 0.0
        if top_cnt > 0:
            tx = [xs[i] for i in top_idx]
            ty = [ys[i] for i in top_idx]
            top_area = (max(tx) - min(tx)) * (max(ty) - min(ty))

        xy_area = float(ext.x * ext.y)

        # Face-based signal (robust to inverted normals):
        # large "planar" area near the extreme planes usually corresponds to the supporting surface.
        thr = 0.7  # cos threshold for "mostly up/down"
        bottom_planar_area = 0.0
        top_planar_area = 0.0
        if faces:
            for (fc, fn, fa) in faces:
                c = r3 @ fc
                nz = float((r3 @ fn).z)
                if c.z <= z_min + eps and abs(nz) > thr:
                    bottom_planar_area += fa
                if c.z >= z_max - eps and abs(nz) > thr:
                    top_planar_area += fa

        return ext, bottom_cnt, top_cnt, bottom_area, top_area, xy_area, bottom_planar_area, top_planar_area

    for rx in angles:
        mx = Matrix.Rotation(rx, 4, "X")
        for ry in angles:
            my = Matrix.Rotation(ry, 4, "Y")
            for rz in angles:
                mz = Matrix.Rotation(rz, 4, "Z")
                r4 = mz @ my @ mx
                ext, bottom_cnt, top_cnt, bottom_area, top_area, xy_area, bottom_planar_area, top_planar_area = eval_rotation(
                    r4.to_3x3()
                )
                z = float(ext.z)

                # Primary objective:
                # - flat: minimize Z thickness
                # - upright: maximize Z height
                #
                # Secondary objective:
                # - for flat objects, prefer the side with large bottom footprint and many near-bottom vertices
                #   (encourages "sole down" for shoes).
                if mode == "flat":
                    score = (
                        z,
                        -bottom_planar_area,
                        top_planar_area,
                        -bottom_area,
                        -bottom_cnt,
                        top_area,
                        top_cnt,
                        -xy_area,
                    )
                else:
                    score = (
                        -z,
                        -xy_area,
                        -top_planar_area,
                        -bottom_planar_area,
                        -bottom_area,
                        -bottom_cnt,
                        top_area,
                        top_cnt,
                    )

                cand = (
                    score,
                    (rx, ry, rz),
                    ext,
                    bottom_cnt,
                    top_cnt,
                    bottom_area,
                    top_area,
                    xy_area,
                    bottom_planar_area,
                    top_planar_area,
                )
                if best is None or cand[0] < best[0]:
                    best = cand

    if best is None:
        return

    (
        _,
        (rx, ry, rz),
        ext,
        bottom_cnt,
        top_cnt,
        bottom_area,
        top_area,
        xy_area,
        bottom_planar_area,
        top_planar_area,
    ) = best
    print(
        f"[DEBUG] Auto-orient selected: rx={rx:.3f}, ry={ry:.3f}, rz={rz:.3f}, "
        f"ext={ext}, xy_area={xy_area:.6f}, "
        f"bottom_cnt={bottom_cnt}, bottom_area={bottom_area:.6f}, bottom_planar_area={bottom_planar_area:.6f}, "
        f"top_cnt={top_cnt}, top_area={top_area:.6f}, top_planar_area={top_planar_area:.6f}"
    )

    r4 = Matrix.Rotation(rz, 4, "Z") @ Matrix.Rotation(ry, 4, "Y") @ Matrix.Rotation(rx, 4, "X")
    asset_root.matrix_world = r4
    bpy.context.view_layer.update()

def _pca_major_axis_xy(points_world):
    """Return a unit Vector in XY plane representing the major axis (largest variance)."""
    xs = [float(p.x) for p in points_world]
    ys = [float(p.y) for p in points_world]
    if not xs:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    a = sum(d * d for d in dx) / len(dx)
    c = sum(d * d for d in dy) / len(dy)
    b = sum(dx[i] * dy[i] for i in range(len(dx))) / len(dx)
    # Largest eigenvector of [[a,b],[b,c]]
    if abs(b) < 1e-12:
        if a >= c:
            v = Vector((1.0, 0.0, 0.0))
        else:
            v = Vector((0.0, 1.0, 0.0))
    else:
        tr = a + c
        det = a * c - b * b
        disc = max(tr * tr - 4.0 * det, 0.0)
        lam1 = 0.5 * (tr + math.sqrt(disc))
        # eigenvector (lam1 - c, b)
        v = Vector((lam1 - c, b, 0.0))
    if v.length < 1e-12:
        return None
    v.normalize()
    return v


def auto_yaw_to_reference(
    asset_root,
    mesh_objects,
    yaw_ref: Vector | None,
    yaw_flip: bool = False,
    yaw_offset_deg: float = 0.0,
):
    if yaw_ref is None:
        return
    yaw_ref_xy = Vector((float(yaw_ref.x), float(yaw_ref.y), 0.0))
    if yaw_ref_xy.length < 1e-8:
        return
    yaw_ref_xy.normalize()

    points = _collect_asset_points(mesh_objects)
    if not points:
        return
    major = _pca_major_axis_xy(points)
    if major is None or major.length < 1e-8:
        return
    major_xy = Vector((major.x, major.y, 0.0))
    major_xy.normalize()

    # Choose between major and -major (180° ambiguity) to minimize rotation.
    def angle_to(src, dst):
        dot = max(min(float(src.x * dst.x + src.y * dst.y), 1.0), -1.0)
        cross_z = float(src.x * dst.y - src.y * dst.x)
        return math.atan2(cross_z, dot)

    a1 = angle_to(major_xy, yaw_ref_xy)
    a2 = angle_to(Vector((-major_xy.x, -major_xy.y, 0.0)), yaw_ref_xy)
    angle = a1 if abs(a1) <= abs(a2) else a2

    if yaw_flip:
        angle += math.pi
    if abs(float(yaw_offset_deg)) > 1e-8:
        angle += float(yaw_offset_deg) * math.pi / 180.0

    r = Matrix.Rotation(angle, 4, "Z")
    asset_root.matrix_world = r @ asset_root.matrix_world
    bpy.context.view_layer.update()

def apply_extra_pitch(asset_root, pitch_axis: Vector | None, extra_pitch_deg: float):
    if pitch_axis is None:
        return
    if abs(float(extra_pitch_deg)) < 1e-8:
        return
    axis = Vector((float(pitch_axis.x), float(pitch_axis.y), float(pitch_axis.z)))
    if axis.length < 1e-8:
        return
    axis.normalize()
    theta = float(extra_pitch_deg) * math.pi / 180.0
    r = Matrix.Rotation(theta, 4, axis)
    asset_root.matrix_world = r @ asset_root.matrix_world
    bpy.context.view_layer.update()

def apply_extra_roll(asset_root, roll_axis: Vector | None, extra_roll_deg: float):
    if roll_axis is None:
        return
    if abs(float(extra_roll_deg)) < 1e-8:
        return
    axis = Vector((float(roll_axis.x), float(roll_axis.y), float(roll_axis.z)))
    if axis.length < 1e-8:
        return
    axis.normalize()
    theta = float(extra_roll_deg) * math.pi / 180.0
    r = Matrix.Rotation(theta, 4, axis)
    asset_root.matrix_world = r @ asset_root.matrix_world
    bpy.context.view_layer.update()


def normalize_scene(
    objects,
    yaw_ref: Vector | None = None,
    pitch_axis: Vector | None = None,
    roll_axis: Vector | None = None,
):
    """Normalize imported asset to a unit box centered at origin."""
    if not objects:
        print("[ERROR] No asset objects to normalize")
        return 1.0

    mesh_objects = [o for o in objects if o.type == "MESH"]
    if not mesh_objects:
        print("[ERROR] No mesh objects found in imported asset; skip normalization")
        return 1.0

    print(f"[DEBUG] Normalizing asset: objects={len(objects)}, meshes={len(mesh_objects)}")

    # Create a dedicated root empty so normalization only affects the imported asset
    # (and doesn't accidentally touch camera/lights).
    asset_root = bpy.data.objects.new("AssetRoot", None)
    bpy.context.scene.collection.objects.link(asset_root)
    asset_root.matrix_world = Matrix.Identity(4)

    asset_set = set(objects)
    top_level = [o for o in objects if (o.parent is None) or (o.parent not in asset_set)]
    print(f"[DEBUG] Parenting {len(top_level)} top-level imported objects under AssetRoot")
    for obj in top_level:
        world = obj.matrix_world.copy()
        obj.parent = asset_root
        obj.matrix_world = world

    bpy.context.view_layer.update()

    auto_orient_asset(asset_root, mesh_objects, args.auto_orient, object_uid=args.object_uid)
    auto_yaw_to_reference(
        asset_root,
        mesh_objects,
        yaw_ref=yaw_ref,
        yaw_flip=bool(args.yaw_flip),
        yaw_offset_deg=float(args.yaw_offset_deg),
    )
    apply_extra_pitch(asset_root, pitch_axis=pitch_axis, extra_pitch_deg=float(args.extra_pitch_deg))
    apply_extra_roll(asset_root, roll_axis=roll_axis, extra_roll_deg=float(args.extra_roll_deg))

    def meshes_bbox(meshes):
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        for obj in meshes:
            if obj.type != "MESH":
                continue
            for coord in obj.bound_box:
                coord = Vector(coord)
                coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        return Vector(bbox_min), Vector(bbox_max)

    # Calculate initial bounding box (asset only)
    bbox_min, bbox_max = meshes_bbox(mesh_objects)
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

    scale_factor = float(args.normalize_max_dim) / size
    print(f"[DEBUG] Scale factor: {scale_factor}")

    asset_root.scale = asset_root.scale * scale_factor
    bpy.context.view_layer.update()

    # Center objects at origin
    bbox_min, bbox_max = meshes_bbox(mesh_objects)
    offset = -(bbox_min + bbox_max) / 2

    asset_root.matrix_world.translation += offset
    bpy.context.view_layer.update()

    # Verify final state
    final_bbox_min, final_bbox_max = meshes_bbox(mesh_objects)
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


def load_pose_template():
    """Load camera poses from a template transforms.json."""
    if not args.pose_template_json:
        return None, None

    template_path = Path(args.pose_template_json)
    if not template_path.exists():
        raise FileNotFoundError(f"Pose template not found: {template_path}")

    with open(template_path, "r") as f:
        template = json.load(f)

    frames = template.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in pose template: {template_path}")

    # Respect --num_views upper bound.
    if args.num_views > 0:
        frames = frames[:args.num_views]

    return frames, template.get("camera_angle_x", None)


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

    mesh_objects, imported_objects = load_object(args.object_path)
    if not mesh_objects:
        print("[ERROR] No objects loaded! Cannot render.")
        # Save error info
        error_file = output_path / "error.txt"
        with open(error_file, 'w') as f:
            f.write(f"Failed to load object: {args.object_path}\n")
        return

    ensure_materials(mesh_objects)
    relink_texture_if_needed(mesh_objects, args.object_path)

    template_frames, template_camera_angle_x = load_pose_template()

    # If requested, use view0 camera basis as yaw reference.
    yaw_ref = None
    pitch_axis = None
    roll_axis = None
    auto_yaw = (args.auto_yaw or "none").lower()
    if auto_yaw != "none":
        if auto_yaw == "world_x":
            yaw_ref = Vector((1.0, 0.0, 0.0))
        elif template_frames is None:
            print(f"[WARN] --auto_yaw={args.auto_yaw} requires --pose_template_json; skip auto_yaw")
        else:
            m = template_frames[0].get("transform_matrix", None)
            if m is not None:
                R = Matrix(m).to_3x3()
                if auto_yaw == "camera_right":
                    yaw_ref = R @ Vector((1.0, 0.0, 0.0))
                elif auto_yaw == "camera_up":
                    yaw_ref = R @ Vector((0.0, 1.0, 0.0))
                else:
                    print(f"[WARN] Unknown --auto_yaw={args.auto_yaw}; skip")

    # pitch axis is always view0 camera right (for image-vertical adjustment), if we have a template.
    if template_frames is not None:
        m = template_frames[0].get("transform_matrix", None)
        if m is not None:
            R0 = Matrix(m).to_3x3()
            pitch_axis = R0 @ Vector((1.0, 0.0, 0.0))  # camera right
            # camera forward is local -Z
            roll_axis = R0 @ Vector((0.0, 0.0, -1.0))
    else:
        if abs(float(args.extra_pitch_deg)) > 1e-8:
            print("[WARN] --extra_pitch_deg requires --pose_template_json; skip extra_pitch")
        if abs(float(args.extra_roll_deg)) > 1e-8:
            print("[WARN] --extra_roll_deg requires --pose_template_json; skip extra_roll")

    scale_factor = normalize_scene(imported_objects, yaw_ref=yaw_ref, pitch_axis=pitch_axis, roll_axis=roll_axis)

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

    if template_camera_angle_x is not None:
        transforms_data["camera_angle_x"] = float(template_camera_angle_x)

    view_idx = 0
    empty_renders = []

    if template_frames is not None:
        print(f"[INFO] Using template poses: {len(template_frames)} views")
        for template_frame in template_frames:
            transform_matrix = template_frame.get("transform_matrix", None)
            if transform_matrix is None:
                print(f"[WARNING] Skip view {view_idx}: missing transform_matrix in template")
                continue

            camera.matrix_world = Matrix(transform_matrix)
            bpy.context.view_layer.update()
            transform_matrix = camera.matrix_world.copy()

            output_file = str(images_dir / f"r_{view_idx}.png")
            bpy.context.scene.render.filepath = output_file

            try:
                bpy.ops.render.render(write_still=True)

                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    if file_size < 1000:
                        print(f"[WARNING] View {view_idx} rendered but file is very small ({file_size} bytes)")
                        empty_renders.append(view_idx)
                else:
                    print(f"[ERROR] View {view_idx} render file not created!")
                    empty_renders.append(view_idx)

            except Exception as e:
                print(f"[ERROR] Failed to render view {view_idx}: {e}")
                empty_renders.append(view_idx)

            frame = {
                "file_path": f"r_{view_idx}",
                "transform_matrix": [[float(x) for x in row] for row in transform_matrix],
                "scale": scale_factor
            }
            if "rotation" in template_frame:
                frame["rotation"] = float(template_frame["rotation"])
            if "elevation" in template_frame:
                frame["elevation"] = float(template_frame["elevation"])
            transforms_data["frames"].append(frame)

            print(f"[INFO] Rendered template view {view_idx + 1}/{len(template_frames)}")
            view_idx += 1
    else:
        # Multi-elevation orthogonal view sampling for LGM
        # Strategy: At each elevation layer, generate groups of 4 orthogonal views
        # (0°, 90°, 180°, 270°) with different azimuth offsets.
        elevations_deg = [float(e) for e in args.elevations.split(',')]
        num_elevations = len(elevations_deg)

        views_per_elevation = args.num_views // num_elevations
        remainder = args.num_views % num_elevations

        print(f"[INFO] Elevation layers: {elevations_deg}")
        print(f"[INFO] Views per elevation: ~{views_per_elevation}")

        for elev_idx, elev_deg in enumerate(elevations_deg):
            if view_idx >= args.num_views:
                break

            elevation = elev_deg * math.pi / 180.0

            n_views_this_elev = views_per_elevation + (1 if elev_idx < remainder else 0)

            if n_views_this_elev <= 20:
                num_groups = 4
                angle_step = 22.5
            elif n_views_this_elev <= 50:
                num_groups = 8
                angle_step = 11.25
            else:
                num_groups = 16
                angle_step = 5.625

            views_rendered_this_elev = 0

            for group_idx in range(num_groups):
                if view_idx >= args.num_views or views_rendered_this_elev >= n_views_this_elev:
                    break

                start_angle = group_idx * angle_step

                for i in range(4):
                    if view_idx >= args.num_views or views_rendered_this_elev >= n_views_this_elev:
                        break

                    azimuth_deg = (start_angle + i * 90) % 360
                    azimuth = azimuth_deg * math.pi / 180.0

                    transform_matrix = setup_camera_at_position(camera, azimuth, elevation, args.camera_dist)

                    output_file = str(images_dir / f"r_{view_idx}.png")
                    bpy.context.scene.render.filepath = output_file

                    try:
                        bpy.ops.render.render(write_still=True)

                        if os.path.exists(output_file):
                            file_size = os.path.getsize(output_file)
                            if file_size < 1000:
                                print(f"[WARNING] View {view_idx} rendered but file is very small ({file_size} bytes)")
                                empty_renders.append(view_idx)
                        else:
                            print(f"[ERROR] View {view_idx} render file not created!")
                            empty_renders.append(view_idx)

                    except Exception as e:
                        print(f"[ERROR] Failed to render view {view_idx}: {e}")
                        empty_renders.append(view_idx)

                    transforms_data["frames"].append({
                        "file_path": f"r_{view_idx}",
                        "rotation": azimuth,
                        "elevation": elev_deg,
                        "transform_matrix": [[float(x) for x in row] for row in transform_matrix],
                        "scale": scale_factor
                    })

                    print(f"[INFO] Rendered view {view_idx + 1}/{args.num_views} (elev {elev_deg:.1f}°, group {group_idx + 1}/{num_groups}, azimuth {azimuth_deg:.1f}°)")
                    view_idx += 1
                    views_rendered_this_elev += 1

    # Save transforms.json
    with open(output_path / "transforms.json", 'w') as f:
        json.dump(transforms_data, f, indent=4)

    # Save debug summary
    debug_file = output_path / "debug_info.txt"
    with open(debug_file, 'w') as f:
        f.write(f"Object: {args.object_uid}\n")
        f.write(f"Object path: {args.object_path}\n")
        f.write(f"Scale factor: {scale_factor}\n")
        f.write(f"Pose template: {args.pose_template_json if args.pose_template_json else 'None'}\n")
        f.write(f"Elevations: {args.elevations}\n")
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
