#!/usr/bin/env python3
"""Bake a travelling sine wave into a glTF water surface as two morph targets.

    h(s, t) = a * sin(k*s - w*t)
            = a*sin(k*s) * cos(w*t)  +  a*cos(k*s) * (-sin(w*t))

so target A carries a*sin(k*s), target B carries a*cos(k*s), and a looping
`weights` animation drives them with (cos wt, -sin wt). The travelling wave is
reproduced exactly (up to the linear interpolation of the weight keys) by the
renderer's existing morph pipeline; no renderer code is involved.

  s    world position along the wave direction
  h    offset along world up, added to the surface's existing shape

Normal deltas are the first-order tilt -(n . up) * grad(h): an up-facing
surface tilts against the slope, a down-facing one mirrors it, and vertical
edges are left alone. The renderer normalizes morphed normals.

Standard library only.

  tools/make_water_wave.py data/water_caustic.glb data/water_caustic_wave.glb
"""

import argparse
import json
import math
import struct
import sys
from array import array

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
FLOAT = 5126
UP = (0.0, 1.0, 0.0)


def read_glb(path):
    data = open(path, "rb").read()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2:
        sys.exit(f"{path}: not a glTF 2.0 binary")
    offset, gltf, binary = 12, None, b""
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8: offset + 8 + length]
        if kind == CHUNK_JSON:
            gltf = json.loads(chunk)
        elif kind == CHUNK_BIN:
            binary = bytes(chunk)
        offset += 8 + length
    return gltf, bytearray(binary)


def write_glb(path, gltf, binary):
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    gltf_len = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, 2, gltf_len))
        f.write(struct.pack("<II", len(js), CHUNK_JSON) + js)
        f.write(struct.pack("<II", len(binary), CHUNK_BIN) + binary)


def read_vec3(gltf, binary, index):
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    if acc["componentType"] != FLOAT or acc["type"] != "VEC3":
        sys.exit(f"accessor {index}: expected float VEC3")
    stride = view.get("byteStride", 12)
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    return [struct.unpack_from("<3f", binary, base + i * stride) for i in range(acc["count"])]


def append_accessor(gltf, binary, values, kind, count, with_bounds):
    """Append float data as its own bufferView + accessor; returns the accessor index."""
    binary += b"\0" * (-len(binary) % 4)
    offset = len(binary)
    binary += array("f", values).tobytes()
    gltf["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset})
    width = {"SCALAR": 1, "VEC3": 3}[kind]
    acc = {"bufferView": len(gltf["bufferViews"]) - 1, "componentType": FLOAT, "count": count, "type": kind}
    if with_bounds:  # Required for POSITION targets and animation inputs.
        acc["min"] = [min(values[c::width]) for c in range(width)]
        acc["max"] = [max(values[c::width]) for c in range(width)]
    gltf["accessors"].append(acc)
    return len(gltf["accessors"]) - 1


# Quaternion helpers, (x, y, z, w) as glTF stores them.
def rotate(q, v):
    x, y, z, w = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + (y * tz - z * ty),
            v[1] + w * ty + (z * tx - x * tz),
            v[2] + w * tz + (x * ty - y * tx))


def conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--node", type=int, default=0, help="node holding the water mesh (default 0)")
    ap.add_argument("--amplitude", type=float, default=0.02, help="wave amplitude, world units (0.02)")
    ap.add_argument("--wavelength", type=float, default=0.5, help="world units (0.5)")
    ap.add_argument("--period", type=float, default=2.0, help="seconds per wavelength travelled (2)")
    ap.add_argument("--direction", type=float, default=0.0,
                    help="travel direction in degrees in the world XZ plane, 0 = +x, 90 = +z (0)")
    ap.add_argument("--cycles", type=int, default=1, help="periods per animation loop (1)")
    ap.add_argument("--keys", type=int, default=120, help="weight keys per period (120)")
    args = ap.parse_args()
    if args.output == args.input:
        sys.exit("refusing to overwrite the input scene")

    gltf, binary = read_glb(args.input)
    node = gltf["nodes"][args.node]
    if any(args.node in n.get("children", []) for n in gltf["nodes"]):
        sys.exit(f"node {args.node} has a parent; only root nodes are supported")
    if "matrix" in node:
        sys.exit(f"node {args.node} uses a matrix; only TRS nodes are supported")
    mesh = gltf["meshes"][node["mesh"]]
    q = tuple(node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    t = node.get("translation", (0.0, 0.0, 0.0))
    scale = node.get("scale", (1.0, 1.0, 1.0))
    if any(abs(c - scale[0]) > 1e-6 for c in scale):
        sys.exit(f"node {args.node} has non-uniform scale; not supported")
    sc = scale[0]

    a = args.amplitude
    k = 2.0 * math.pi / args.wavelength
    ang = math.radians(args.direction)
    direction = (math.cos(ang), 0.0, math.sin(ang))

    # World-space vectors into mesh-local space: inverse rotation, inverse scale
    # for offsets. Normals only rotate.
    qi = conjugate(q)
    up_local = tuple(c / sc for c in rotate(qi, UP))

    for prim in mesh["primitives"]:
        if "targets" in prim:
            sys.exit("the mesh already has morph targets")
        positions = read_vec3(gltf, binary, prim["attributes"]["POSITION"])
        normals = read_vec3(gltf, binary, prim["attributes"]["NORMAL"])

        dpos = ([], [])   # target A (sin), target B (cos)
        dnrm = ([], [])
        for p, n in zip(positions, normals):
            pw = rotate(q, (p[0] * sc, p[1] * sc, p[2] * sc))
            pw = (pw[0] + t[0], pw[1] + t[1], pw[2] + t[2])
            phase = k * dot(pw, direction)
            s, c = math.sin(phase), math.cos(phase)
            nw = rotate(q, n)
            facing = dot(nw, UP)
            for target, h, slope in ((0, s, c), (1, c, -s)):
                dpos[target].extend(u * a * h for u in up_local)
                # grad(a*sin) = a*k*cos along direction; grad(a*cos) = -a*k*sin.
                g = -facing * a * k * slope
                dnrm[target].extend(rotate(qi, (direction[0] * g, 0.0, direction[2] * g)))

        count = len(positions)
        prim["targets"] = [
            {"POSITION": append_accessor(gltf, binary, dpos[i], "VEC3", count, True),
             "NORMAL": append_accessor(gltf, binary, dnrm[i], "VEC3", count, False)}
            for i in range(2)
        ]
    mesh["weights"] = [1.0, 0.0]  # The t = 0 pose, so the unplayed scene already shows the wave.
    mesh.setdefault("extras", {})["targetNames"] = ["wave_sin", "wave_cos"]

    # Looping weights animation. The last key equals the first, so the loop seam is exact.
    keys = args.keys * args.cycles
    duration = args.period * args.cycles
    omega = 2.0 * math.pi / args.period
    times, weights = [], []
    for i in range(keys + 1):
        tk = duration * i / keys
        times.append(tk)
        weights.extend((math.cos(omega * tk), -math.sin(omega * tk)))
    sampler = {"input": append_accessor(gltf, binary, times, "SCALAR", len(times), True),
               "output": append_accessor(gltf, binary, weights, "SCALAR", len(weights), False),
               "interpolation": "LINEAR"}
    gltf.setdefault("animations", []).append({
        "name": "water_wave",
        "samplers": [sampler],
        "channels": [{"sampler": 0, "target": {"node": args.node, "path": "weights"}}],
    })

    gltf["buffers"][0]["byteLength"] = len(binary) + (-len(binary) % 4)
    write_glb(args.output, gltf, binary)
    print(f"{args.output}: wave a={a} wavelength={args.wavelength} period={args.period}s "
          f"dir={args.direction}deg, {len(mesh['primitives'])} primitive(s), {keys + 1} keys")


if __name__ == "__main__":
    main()
