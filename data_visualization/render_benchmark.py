"""
Multi4D benchmark renderer (public release version).

Renders a single 4D-capture frame mesh with its diffuse texture from a set
of pre-calibrated camera views. Mesh/UV geometry is read directly out of a binary PLY (per-vertex
x,y,z,s,t, exported from Blender) with a small from-scratch parser, and
camera calibration is read from a JSON file provided in data.
Dependencies:
    pip install numpy Pillow moderngl

Usage:
    python render_benchmark.py \\
        --ply Frame027087.ply \\
        --texture Frame027087_u1_v1_diffuse.jpg \\
        --cameras-json Frame027087_cameras.json \\
        --out renders

    python render_benchmark.py --ply F.ply --texture T.jpg --cameras-json C.json --camera-names MV-002,MV-004
    python render_benchmark.py --ply F.ply --texture T.jpg --cameras-json C.json --max-size 900
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Generic binary PLY parser (mesh geometry only: per-vertex x,y,z,s,t + faces)
# ---------------------------------------------------------------------------

_PLY_TYPE_MAP = {
    'char': 'i1', 'uchar': 'u1', 'int8': 'i1', 'uint8': 'u1',
    'short': 'i2', 'ushort': 'u2', 'int16': 'i2', 'uint16': 'u2',
    'int': 'i4', 'uint': 'u4', 'int32': 'i4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4',
    'double': 'f8', 'float64': 'f8',
}


def load_mesh(path):
    """
    Load a binary-little-endian PLY (as exported by Blender's PLY exporter,
    one UV per vertex via the `s`/`t` properties -- vertices are duplicated
    at UV seams so the UV index buffer is just the face index buffer).
    Returns dict with:
        vertices   (N,3) float64
        faces      (M,3) int64   -- vertex indices per triangle
        uv         (N,2) float64 or None
        uv_indices (M,3) int64 or None  -- per-triangle-corner index into uv

    Blender's PLY export is Z-up; this project's camera/world convention
    (inherited from the original FBX captures) is Y-up, so vertices are
    remapped on load: fbx_x = ply_x, fbx_y = ply_z, fbx_z = -ply_y.
    """
    with open(path, 'rb') as f:
        line = f.readline().strip()
        if line != b'ply':
            raise ValueError(f"not a PLY file: {path}")

        elements = []
        cur = None
        fmt = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected EOF in PLY header")
            parts = line.strip().split()
            if not parts:
                continue
            tag = parts[0].decode('ascii')
            if tag in ('comment', 'obj_info'):
                continue
            if tag == 'format':
                fmt = parts[1].decode('ascii')
                if fmt != 'binary_little_endian':
                    raise ValueError(f"unsupported PLY format {fmt!r}")
            elif tag == 'element':
                cur = {'name': parts[1].decode('ascii'), 'count': int(parts[2]), 'properties': []}
                elements.append(cur)
            elif tag == 'property':
                if parts[1] == b'list':
                    count_type = parts[2].decode('ascii')
                    elem_type = parts[3].decode('ascii')
                    pname = parts[4].decode('ascii')
                    cur['properties'].append(('list', pname, count_type, elem_type))
                else:
                    ptype = parts[1].decode('ascii')
                    pname = parts[2].decode('ascii')
                    cur['properties'].append(('scalar', pname, ptype))
            elif tag == 'end_header':
                break

        data = {}
        for el in elements:
            scalar_props = [p for p in el['properties'] if p[0] == 'scalar']
            list_props = [p for p in el['properties'] if p[0] == 'list']

            if list_props:
                assert len(list_props) == 1 and not scalar_props
                _, pname, count_type, elem_type = list_props[0]
                count_dtype = np.dtype(_PLY_TYPE_MAP[count_type])
                elem_dtype = np.dtype(_PLY_TYPE_MAP[elem_type])

                start = f.tell()
                first_count = np.frombuffer(f.read(count_dtype.itemsize), dtype=count_dtype)[0]
                f.seek(start)
                stride = count_dtype.itemsize + int(first_count) * elem_dtype.itemsize
                raw = f.read(stride * el['count'])
                arr = np.frombuffer(raw, dtype=np.dtype([('count', count_dtype), ('idx', elem_dtype, int(first_count))]))
                if not (arr['count'] == first_count).all():
                    raise ValueError(f"variable-length list property {pname!r} not supported")
                data[(el['name'], pname)] = arr['idx'].astype(np.int64)
            else:
                dtype = np.dtype([(p[1], _PLY_TYPE_MAP[p[2]]) for p in scalar_props])
                raw = f.read(dtype.itemsize * el['count'])
                arr = np.frombuffer(raw, dtype=dtype)
                for p in scalar_props:
                    data[(el['name'], p[1])] = arr[p[1]].copy()

    px, py, pz = data[('vertex', 'x')], data[('vertex', 'y')], data[('vertex', 'z')]
    vertices = np.stack([px, pz, -py], axis=1).astype(np.float64)
    faces = data[('face', 'vertex_indices')]

    uv = None
    uv_indices = None
    if ('vertex', 's') in data and ('vertex', 't') in data:
        uv = np.stack([data[('vertex', 's')], data[('vertex', 't')]], axis=1).astype(np.float64)
        uv_indices = faces
    elif ('vertex', 'u') in data and ('vertex', 'v') in data:
        uv = np.stack([data[('vertex', 'u')], data[('vertex', 'v')]], axis=1).astype(np.float64)
        uv_indices = faces

    return {"vertices": vertices, "faces": faces, "uv": uv, "uv_indices": uv_indices}



def load_camera_json(path, name_filter=None):
    """
    Returns a list of dicts: {"name", "R" (3,3), "t" (3,), "K" (3,3),
    "width", "height"}, built from the JSON provided in the dataset.
    """
    with open(path, "r") as f:
        data = json.load(f)

    cams = []
    for c in data["cameras"]:
        if name_filter is not None and not any(nf in c["name"] for nf in name_filter):
            continue

        focal_px = c["focal_px"]
        cx, cy = c["principal_point"]
        K = np.array([[focal_px, 0, cx],
                       [0, focal_px, cy],
                       [0, 0, 1]], dtype=np.float64)

        cams.append({
            "name": c["name"],
            "R": np.array(c["rotation"], dtype=np.float64),
            "t": np.array(c["translation"], dtype=np.float64),
            "K": K,
            "width": int(c["width"]),
            "height": int(c["height"]),
        })
    return cams


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_VERTEX_SHADER = """
#version 330
in vec4 in_clip_pos;
in vec2 in_uv;
in vec3 in_shade;
out vec2 v_uv;
out vec3 v_shade;
void main() {
    gl_Position = in_clip_pos;
    v_uv = in_uv;
    v_shade = in_shade;
}
"""

_FRAGMENT_SHADER = """
#version 330
uniform sampler2D tex;
uniform float ambient;
uniform bool use_texture;
in vec2 v_uv;
in vec3 v_shade;
out vec4 f_color;
void main() {
    vec3 color = use_texture ? texture(tex, v_uv).rgb : v_shade;
    f_color = vec4(color * ambient, 1.0);
}
"""


def compute_vertex_normals(vertices, faces):
    """Area-weighted vertex normals from triangle faces, used for the
    --no-texture flat-shaded mesh view (no UV/diffuse texture needed)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normals / norms


def project_to_clip(vertices, R, t, K, width, height, near_pad=0.5, far_pad=1.5):
    """
    Project world-space vertices with a calibrated pinhole camera
    (cam_space = R @ world + t, pixel = K @ cam_space) into OpenGL clip-space
    coordinates, so the GPU only has to do the perspective divide + rasterize.
    """
    cam_space = (R @ vertices.T).T + t  # (N,3), camera-space XYZ
    depth = cam_space[:, 2]

    pixels = (K @ cam_space.T).T
    uv_px = pixels[:, :2] / pixels[:, 2:3]

    near = max(float(depth.min()) * near_pad, 1e-3)
    far = float(depth.max()) * far_pad

    ndc_x = 2.0 * uv_px[:, 0] / width - 1.0
    ndc_y = 1.0 - 2.0 * uv_px[:, 1] / height  # image-space v grows down, NDC y grows up
    ndc_z = (2.0 * depth - (far + near)) / (far - near)

    w_clip = depth
    clip = np.empty((len(vertices), 4), dtype=np.float32)
    clip[:, 0] = ndc_x * w_clip
    clip[:, 1] = ndc_y * w_clip
    clip[:, 2] = ndc_z * w_clip
    clip[:, 3] = w_clip
    return clip


_MESH_COLOR = np.array([0.75, 0.75, 0.75], dtype=np.float32)
_MESH_LIGHT_DIR_CAM = np.array([0.0, 0.0, -1.0])  # camera-facing headlight, in camera space


def render_views(mesh_path, texture_path, out_dir, cameras, mesh, max_size=None, ambient=1.0, use_texture=True):
    import moderngl

    os.makedirs(out_dir, exist_ok=True)

    vertices = mesh["vertices"]
    faces = mesh["faces"]
    uv = mesh["uv"]
    uv_indices = mesh["uv_indices"]

    pos_idx = faces.flatten()

    if use_texture:
        if uv is None or uv_indices is None:
            raise ValueError("mesh has no UV layer; cannot apply a diffuse texture (use --no-texture instead)")
        uv_idx = uv_indices.flatten()
        uv_expanded = uv[uv_idx].astype('f4').copy()
        # PLY V=0 is the bottom row, same as OpenGL's texture convention (row 0 of
        # the uploaded buffer = V=0). The image is flipped top/bottom below so
        # PIL's top-origin row order matches that GL convention -- so the PLY UVs
        # are used as-is here, with no additional V flip.
        vertex_normals = None
    else:
        uv_expanded = np.zeros((len(pos_idx), 2), dtype='f4')
        vertex_normals = compute_vertex_normals(vertices, faces)

    ctx = moderngl.create_standalone_context()
    prog = ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER)
    prog["ambient"].value = ambient
    prog["use_texture"].value = use_texture

    if use_texture:
        img = Image.open(texture_path).convert("RGB")
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        tex = ctx.texture(img.size, 3, img.tobytes())
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.use(location=0)
        prog["tex"].value = 0
    else:
        tex = ctx.texture((1, 1), 3, bytes(3))  # dummy, unused by the shader when use_texture is false

    uv_vbo = ctx.buffer(uv_expanded.tobytes())
    clip_vbo = ctx.buffer(reserve=len(pos_idx) * 4 * 4)  # vec4 f4 per corner
    shade_vbo = ctx.buffer(reserve=len(pos_idx) * 3 * 4)  # vec3 f4 per corner

    vao = ctx.vertex_array(prog, [
        (clip_vbo, "4f", "in_clip_pos"),
        (uv_vbo, "2f", "in_uv"),
        (shade_vbo, "3f", "in_shade"),
    ])

    base_name = os.path.splitext(os.path.basename(mesh_path))[0]
    name_suffix = "" if use_texture else "_mesh"

    for cam in cameras:
        cam_w, cam_h = cam["width"], cam["height"]
        if max_size is not None and max(cam_w, cam_h) > max_size:
            scale = max_size / max(cam_w, cam_h)
        else:
            scale = 1.0
        render_w, render_h = max(1, round(cam_w * scale)), max(1, round(cam_h * scale))

        clip = project_to_clip(vertices, cam["R"], cam["t"], cam["K"], cam_w, cam_h)
        clip_expanded = clip[pos_idx]
        clip_vbo.write(clip_expanded.tobytes())

        if not use_texture:
            normals_cam = vertex_normals @ cam["R"].T
            diffuse = np.clip(normals_cam @ _MESH_LIGHT_DIR_CAM, 0.0, 1.0)
            shade = (0.25 + 0.75 * diffuse)[:, None] * _MESH_COLOR[None, :]
            shade_vbo.write(shade[pos_idx].astype('f4').tobytes())

        fbo = ctx.framebuffer(
            color_attachments=[ctx.texture((render_w, render_h), 3)],
            depth_attachment=ctx.depth_renderbuffer((render_w, render_h)),
        )
        fbo.use()
        ctx.viewport = (0, 0, render_w, render_h)
        ctx.clear(0.08, 0.08, 0.08)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        vao.render(moderngl.TRIANGLES, vertices=len(pos_idx))

        data = fbo.color_attachments[0].read()
        out_img = Image.frombytes("RGB", (render_w, render_h), data).transpose(Image.FLIP_TOP_BOTTOM)
        out_path = os.path.join(out_dir, f"{base_name}_{cam['name']}{name_suffix}.png")
        out_img.save(out_path)
        print(f"  saved {out_path} ({render_w}x{render_h})")

        fbo.color_attachments[0].release()
        fbo.depth_attachment.release()
        fbo.release()

    vao.release()
    clip_vbo.release()
    uv_vbo.release()
    shade_vbo.release()
    tex.release()
    prog.release()
    ctx.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True, help="Path to a per-frame capture PLY with per-vertex UV (mesh geometry only)")
    ap.add_argument("--texture", default=None, help="Path to the diffuse texture (e.g. FrameXXXXXX_u1_v1_diffuse.jpg); required unless --no-texture")
    ap.add_argument("--no-texture", action="store_true", help="Render flat-shaded geometry only (no diffuse texture/UV needed)")
    ap.add_argument("--cameras-json", required=True, help="Path to the camera JSON.")
    ap.add_argument("--out", default=None, help="Output directory (default: <ply_dir>/renders)")
    ap.add_argument("--camera-names", default=None, help="Comma-separated substrings to filter camera names (default: all)")
    ap.add_argument("--max-size", type=int, default=1024, help="Cap longest output image side in pixels (default 1024, 0 = native camera resolution)")
    ap.add_argument("--ambient", type=float, default=1.0, help="Flat brightness multiplier on the sampled texture (default 1.0 = unlit raw texture)")
    args = ap.parse_args()

    if not args.no_texture and not args.texture:
        ap.error("--texture is required unless --no-texture is set")

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ply)), "renders")
    name_filter = [s.strip() for s in args.camera_names.split(",")] if args.camera_names else None
    max_size = None if args.max_size == 0 else args.max_size

    print(f"Loading mesh from {args.ply} ...")
    mesh = load_mesh(args.ply)
    print(f"  {len(mesh['vertices'])} vertices, {len(mesh['faces'])} triangles, "
          f"UV: {'yes' if mesh['uv'] is not None else 'no'}")

    print(f"Loading cameras from {args.cameras_json} ...")
    cameras = load_camera_json(args.cameras_json, name_filter=name_filter)
    print(f"  {len(cameras)} camera(s): {', '.join(c['name'] for c in cameras)}")

    print(f"Rendering to {out_dir} ...")
    render_views(args.ply, args.texture, out_dir, cameras, mesh, max_size=max_size,
                 ambient=args.ambient, use_texture=not args.no_texture)
    print("Done.")


if __name__ == "__main__":
    main()
