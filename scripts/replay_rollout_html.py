#!/usr/bin/env python3
"""
Create a self-contained HTML replay from a recorded rollout (.npz).

No matplotlib, no extra deps. Just numpy + stdlib.
Open the resulting .html in your browser to scrub/play the trajectory.

Features:
- Pan (drag), zoom (wheel), reset (double-click)
- Follow camera checkbox
- Triangle car icon
- Optional embedded map PNG background (auto from track_centerline_csv)
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_simple_yaml_map(yaml_text: str) -> dict:
    """
    Minimal parser for typical ROS map yaml:
      image: Sakhir_map.png
      resolution: 0.050000
      origin: [-12.345, -67.890, 0.0]
    We keep it intentionally simple/robust for this use-case.
    """
    out = {}
    for line in yaml_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        out[k] = v

    # parse resolution
    if "resolution" in out:
        try:
            out["resolution"] = float(out["resolution"])
        except Exception:
            pass

    # parse origin
    if "origin" in out:
        v = out["origin"].strip()
        # expect like: [-12.3, -45.6, 0.0]
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1]
            parts = [p.strip() for p in inner.split(",")]
            try:
                out["origin"] = [float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0]
            except Exception:
                pass

    return out


def _guess_map_files_from_centerline(centerline_csv: Path):
    """
    Given:
      assets/f1tenth_racetracks/Sakhir/Sakhir_centerline.csv
    Guess:
      assets/f1tenth_racetracks/Sakhir/Sakhir_map.png
      assets/f1tenth_racetracks/Sakhir/Sakhir_map.yaml (or .yml)
    """
    track_dir = centerline_csv.parent
    track_name = centerline_csv.stem.replace("_centerline", "")  # "Sakhir"
    png = track_dir / f"{track_name}_map.png"
    yaml1 = track_dir / f"{track_name}_map.yaml"
    yaml2 = track_dir / f"{track_name}_map.yml"
    yaml_path = yaml1 if yaml1.exists() else (yaml2 if yaml2.exists() else None)
    png_path = png if png.exists() else None
    return png_path, yaml_path, track_name, track_dir


def _embed_png_data_uri(png_path: Path) -> str:
    b = png_path.read_bytes()
    enc = base64.b64encode(b).decode("ascii")
    return "data:image/png;base64," + enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True, help="Path to .npz recorded rollout")
    ap.add_argument("--out", default="rollouts/replay.html", help="Output HTML path")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth frame to reduce size")
    ap.add_argument("--max_frames", type=int, default=20000, help="Hard cap frames to keep HTML reasonable")
    ap.add_argument("--lidar", action="store_true", help="Include lidar sector plot")
    ap.add_argument("--no_map", action="store_true", help="Disable drawing the map PNG background")
    args = ap.parse_args()

    data = np.load(args.rollout, allow_pickle=True)

    pose = np.asarray(data["pose"], dtype=float)          # [T,3]
    action = np.asarray(data["action"], dtype=float)      # [T,2]
    e_head = np.asarray(data["e_head"], dtype=float)      # [T]
    e_lat = np.asarray(data["e_lat"], dtype=float)        # [T]
    centerline_csv = Path(str(data["track_centerline_csv"]))

    # Ensure centerline is Nx2 even if CSV has extra columns
    centerline = np.loadtxt(centerline_csv, delimiter=",", ndmin=2)[:, :2]

    # Stride + cap
    idx = np.arange(0, pose.shape[0], max(1, args.stride), dtype=int)
    if idx.size > args.max_frames:
        idx = idx[: args.max_frames]

    pose = pose[idx]
    action = action[idx]
    e_head = e_head[idx]
    e_lat = e_lat[idx]

    lidar = None
    if args.lidar and "lidar_sectors" in data:
        lidar = np.asarray(data["lidar_sectors"], dtype=float)[idx]

    # --- map background (PNG + YAML) ---
    map_payload = None
    if not args.no_map:
        png_path, yaml_path, track_name, track_dir = _guess_map_files_from_centerline(centerline_csv)
        if png_path is not None and yaml_path is not None:
            y = _parse_simple_yaml_map(_read_text(yaml_path))
            # require origin + resolution to place image in world
            if isinstance(y.get("origin", None), list) and isinstance(y.get("resolution", None), float):
                data_uri = _embed_png_data_uri(png_path)
                map_payload = {
                    "data_uri": data_uri,
                    "resolution": float(y["resolution"]),
                    "origin": [float(y["origin"][0]), float(y["origin"][1]), float(y["origin"][2] if len(y["origin"]) > 2 else 0.0)],
                    # We don't know image dims without PIL; we can read them in JS from Image() once loaded.
                    "track_name": track_name,
                    "png_name": png_path.name,
                    "yaml_name": yaml_path.name,
                }

    payload = {
        "centerline": centerline.tolist(),
        "pose": pose.tolist(),
        "action": action.tolist(),
        "e_head": e_head.tolist(),
        "e_lat": e_lat.tolist(),
        "lidar": None if lidar is None else lidar.tolist(),
        "map": map_payload,  # may be null
    }

    payload_json = json.dumps(payload)
    lidar_display = "block" if args.lidar else "none"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>F1TENTH Rollout Replay</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    .row {{ display: flex; gap: 16px; align-items: flex-start; }}
    canvas {{ border: 1px solid #ccc; background: #fff; }}
    .panel {{ min-width: 360px; }}
    .mono {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; white-space: pre; }}
    input[type="range"] {{ width: 100%; }}
    button {{ margin-right: 8px; }}
    .hint {{ color: #444; font-size: 12px; margin-top: 6px; }}
    label {{ margin-right: 10px; }}
  </style>
</head>
<body>
  <h2>F1TENTH Rollout Replay</h2>

  <div class="row">
    <div>
      <canvas id="map" width="900" height="900"></canvas>

      <div class="hint">
        Drag to pan • Mouse wheel to zoom • Double-click to reset
      </div>

      <div style="margin-top:8px;">
        <input id="slider" type="range" min="0" max="0" value="0"/>
      </div>

      <div style="margin-top:8px;">
        <button id="play">Play</button>
        <button id="pause">Pause</button>

        <label>FPS <input id="fps" type="number" value="30" min="1" max="240" style="width:70px;"></label>
        <label>Trail <input id="trail" type="number" value="300" min="0" max="20000" style="width:80px;"></label>
        <label>Follow <input id="follow" type="checkbox" checked></label>
        <label>Show map <input id="showmap" type="checkbox" checked></label>
      </div>
    </div>

    <div class="panel">
      <div class="mono" id="info"></div>
      <canvas id="lidar" width="360" height="260" style="margin-top:12px; display:{lidar_display};"></canvas>
    </div>
  </div>

<script>
"use strict";

const DATA = {payload_json};

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");

const info = document.getElementById("info");
const slider = document.getElementById("slider");
const playBtn = document.getElementById("play");
const pauseBtn = document.getElementById("pause");
const fpsInput = document.getElementById("fps");
const trailInput = document.getElementById("trail");
const followInput = document.getElementById("follow");
const showMapInput = document.getElementById("showmap");

const lidarCanvas = document.getElementById("lidar");
const lctx = lidarCanvas.getContext("2d");

const centerline = DATA.centerline;
const pose = DATA.pose;
const action = DATA.action;
const e_head = DATA.e_head;
const e_lat = DATA.e_lat;
const lidar = DATA.lidar;
const mapInfo = DATA.map;

slider.max = String(pose.length - 1);

// ---- geometry helpers ----
function bounds(points) {{
  let xmin=Infinity, xmax=-Infinity, ymin=Infinity, ymax=-Infinity;
  for (const p of points) {{
    const x=p[0], y=p[1];
    if (x<xmin) xmin=x;
    if (x>xmax) xmax=x;
    if (y<ymin) ymin=y;
    if (y>ymax) ymax=y;
  }}
  return {{xmin,xmax,ymin,ymax}};
}}

const b = bounds(pose.map(p => [p[0], p[1]]));

// Add padding around extents
const pad = 0.25;
const w = (b.xmax - b.xmin) * (1 + pad);
const h = (b.ymax - b.ymin) * (1 + pad);

// baseScale so whole rollout fits
const baseScale = Math.min(canvas.width / w, canvas.height / h);

// camera state
const cam = {{
  scale: baseScale,
  tx: 0, // pixels
  ty: 0, // pixels
}};

// world center we start at; "follow" will override each frame
const worldCenter = {{
  x: (b.xmin + b.xmax) / 2,
  y: (b.ymin + b.ymax) / 2,
}};

function worldToScreen(x, y) {{
  const X = (x - worldCenter.x) * cam.scale + canvas.width/2 + cam.tx;
  const Y = canvas.height/2 - (y - worldCenter.y) * cam.scale + cam.ty;
  return [X, Y];
}}

function screenToWorld(X, Y) {{
  const x = (X - canvas.width/2 - cam.tx) / cam.scale + worldCenter.x;
  const y = (canvas.height/2 + cam.ty - Y) / cam.scale + worldCenter.y;
  return [x, y];
}}

// ---- pan + zoom controls ----
let dragging = false;
let lastX = 0;
let lastY = 0;

canvas.addEventListener("mousedown", (e) => {{
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
}});

window.addEventListener("mouseup", () => {{
  dragging = false;
}});

window.addEventListener("mousemove", (e) => {{
  if (!dragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;

  cam.tx += dx;
  cam.ty += dy;

  renderFrame(Number(slider.value));
}});

// zoom around cursor
canvas.addEventListener("wheel", (e) => {{
  e.preventDefault();

  const zoomFactor = Math.exp(-e.deltaY * 0.001);
  const mouseX = e.offsetX;
  const mouseY = e.offsetY;

  // world under cursor before zoom
  const w0 = screenToWorld(mouseX, mouseY);

  cam.scale *= zoomFactor;
  cam.scale = Math.max(baseScale * 0.15, Math.min(baseScale * 40.0, cam.scale));

  // keep world under cursor stable
  const s1 = worldToScreen(w0[0], w0[1]);
  cam.tx += (mouseX - s1[0]);
  cam.ty += (mouseY - s1[1]);

  renderFrame(Number(slider.value));
}}, {{ passive: false }});

canvas.addEventListener("dblclick", () => {{
  cam.scale = baseScale;
  cam.tx = 0;
  cam.ty = 0;
  renderFrame(Number(slider.value));
}});

// ---- map background image (optional) ----
let mapImg = null;
let mapImgLoaded = false;

if (mapInfo && mapInfo.data_uri) {{
  mapImg = new Image();
  mapImg.onload = () => {{
    mapImgLoaded = true;
    renderFrame(Number(slider.value));
  }};
  mapImg.src = mapInfo.data_uri;
}} else {{
  showMapInput.checked = false;
  showMapInput.disabled = true;
}}

function drawMapBackground() {{
  if (!mapInfo || !mapImgLoaded || !showMapInput.checked) return;

  // ROS map convention:
  //   world_x = origin_x + pixel_x * resolution
  //   world_y = origin_y + (image_height - 1 - pixel_y) * resolution
  //
  // So:
  //   pixel_x = (world_x - origin_x) / resolution
  //   pixel_y = (image_height - 1) - (world_y - origin_y) / resolution
  //
  // To draw image, we compute screen coords for world corners of image and draw scaled.
  const res = mapInfo.resolution;
  const ox = mapInfo.origin[0];
  const oy = mapInfo.origin[1];

  const iw = mapImg.width;
  const ih = mapImg.height;

  // world corners of the map image
  const world_min_x = ox;
  const world_min_y = oy;
  const world_max_x = ox + iw * res;
  const world_max_y = oy + ih * res;

  // In ROS map convention, the image "up" corresponds to +y in world,
  // but PNG pixel y increases downward, so we need to flip vertically when drawing.

  // screen corners (we'll draw using a transform)
  const p00 = worldToScreen(world_min_x, world_min_y); // bottom-left in world
  const p10 = worldToScreen(world_max_x, world_min_y);
  const p01 = worldToScreen(world_min_x, world_max_y);

  // Compute screen-space basis vectors
  const vx = [p10[0] - p00[0], p10[1] - p00[1]]; // corresponds to +x in world
  const vy = [p01[0] - p00[0], p01[1] - p00[1]]; // corresponds to +y in world

  // We want to map image pixel coords to world coords:
  // pixel (0, ih) corresponds to (world_min_x, world_min_y)
  // pixel (iw, ih) corresponds to (world_max_x, world_min_y)
  // pixel (0, 0) corresponds to (world_min_x, world_max_y)
  //
  // That implies drawing the image with a vertical flip.
  ctx.save();

  // Place origin at p00 (world_min_x, world_min_y)
  ctx.translate(p00[0], p00[1]);

  // Build transform mapping image pixels -> screen
  // x-axis: vx / iw
  // y-axis (image down): -vy / ih   (flip)
  ctx.transform(
    vx[0] / iw, vx[1] / iw,
    -vy[0] / ih, -vy[1] / ih,
    0, 0
  );

  // draw image such that its bottom-left aligns with p00:
  // because we flipped y, we draw at (0, -ih)
  ctx.globalAlpha = 0.90;
  ctx.drawImage(mapImg, 0, -ih, iw, ih);

  ctx.restore();
}}

// ---- drawing ----
function clear() {{
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}}

function drawCenterline() {{
  if (!centerline || centerline.length === 0) return;
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "#777";
  ctx.beginPath();
  for (let i=0;i<centerline.length;i++) {{
    const Xy = worldToScreen(centerline[i][0], centerline[i][1]);
    if (i===0) ctx.moveTo(Xy[0], Xy[1]);
    else ctx.lineTo(Xy[0], Xy[1]);
  }}
  ctx.stroke();
}}

function drawTrail(k, trailLen) {{
  const start = Math.max(0, k - trailLen);
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = "#1f77b4";
  ctx.beginPath();
  for (let i=start;i<=k;i++) {{
    const x = pose[i][0], y = pose[i][1];
    const Xy = worldToScreen(x,y);
    if (i===start) ctx.moveTo(Xy[0], Xy[1]);
    else ctx.lineTo(Xy[0], Xy[1]);
  }}
  ctx.stroke();
}}

function drawCarTriangle(k) {{
  const x = pose[k][0], y = pose[k][1], yaw = pose[k][2];
  const P = worldToScreen(x,y);

  // triangle size in SCREEN pixels (constant size regardless of zoom)
  const tipLen = 22;
  const baseLen = 14;
  const baseHalfWidth = 11;

  // yaw in screen coords (+y down => flip sin)
  const dx = Math.cos(yaw);
  const dy = -Math.sin(yaw);

  // perpendicular (left) in screen coords
  const px = -dy;
  const py = dx;

  const tipX = P[0] + dx * tipLen;
  const tipY = P[1] + dy * tipLen;

  const baseCX = P[0] - dx * baseLen;
  const baseCY = P[1] - dy * baseLen;

  const leftX  = baseCX + px * baseHalfWidth;
  const leftY  = baseCY + py * baseHalfWidth;
  const rightX = baseCX - px * baseHalfWidth;
  const rightY = baseCY - py * baseHalfWidth;

  ctx.fillStyle = "#d62728";
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(leftX, leftY);
  ctx.lineTo(rightX, rightY);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = "#8c1b1b";
  ctx.lineWidth = 2;
  ctx.stroke();
}}

function drawLidar(k) {{
  if (!lidar) return;

  const vals = lidar[k];
  const W = lidarCanvas.width, H = lidarCanvas.height;

  lctx.fillStyle = "#fff";
  lctx.fillRect(0,0,W,H);

  lctx.fillStyle = "#000";
  lctx.font = "12px Arial";
  lctx.fillText("LiDAR sectors (normalized)", 8, 14);

  const n = vals.length;
  const barW = (W - 16) / n;
  const y0 = H - 16;

  lctx.strokeStyle = "#ccc";
  lctx.beginPath();
  lctx.moveTo(8, y0);
  lctx.lineTo(W-8, y0);
  lctx.stroke();

  for (let i=0;i<n;i++) {{
    const v = Math.max(0, Math.min(1, vals[i]));
    const hh = v * (H - 40);
    const x = 8 + i*barW;
    lctx.fillStyle = "#2ca02c";
    lctx.fillRect(x, y0 - hh, barW-1, hh);
  }}
}}

function renderFrame(k) {{
  if (followInput && followInput.checked) {{
    worldCenter.x = pose[k][0];
    worldCenter.y = pose[k][1];
  }}

  clear();
  drawMapBackground();
  drawCenterline();
  drawTrail(k, Number(trailInput.value));
  drawCarTriangle(k);
  drawLidar(k);

  const a0 = action[k][0].toFixed(3);
  const a1 = action[k][1].toFixed(3);
  const eh = e_head[k].toFixed(3);
  const el = e_lat[k].toFixed(3);
  const x = pose[k][0], y = pose[k][1], yaw = pose[k][2];

  let mapLine = "map: (none)\\n";
  if (mapInfo && mapInfo.png_name) {{
    mapLine = "map: " + mapInfo.png_name + " (" + mapInfo.yaml_name + ")\\n";
  }}

  info.textContent =
    "frame: " + k + "/" + (pose.length - 1) + "\\n" +
    "pose: x=" + x.toFixed(3) + " y=" + y.toFixed(3) + " yaw=" + yaw.toFixed(3) + "\\n" +
    "action: speed_norm=" + a0 + " steer_norm=" + a1 + "\\n" +
    "errors: e_head=" + eh + " e_lat=" + el + "\\n" +
    mapLine +
    "view: scale=" + cam.scale.toFixed(3) + " pan=(" + cam.tx.toFixed(1) + "," + cam.ty.toFixed(1) + ")\\n";
}}

// ---- playback controls ----
let timer = null;

function play() {{
  if (timer) return;
  const fps = Math.max(1, Number(fpsInput.value));
  timer = setInterval(() => {{
    let k = Number(slider.value);
    k = Math.min(pose.length-1, k+1);
    slider.value = String(k);
    renderFrame(k);
    if (k >= pose.length-1) pause();
  }}, 1000/fps);
}}

function pause() {{
  if (timer) clearInterval(timer);
  timer = null;
}}

slider.addEventListener("input", () => renderFrame(Number(slider.value)));
playBtn.addEventListener("click", play);
pauseBtn.addEventListener("click", pause);
followInput.addEventListener("change", () => renderFrame(Number(slider.value)));
showMapInput.addEventListener("change", () => renderFrame(Number(slider.value)));

renderFrame(0);

</script>
</body>
</html>
"""

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print(f"Wrote HTML replay: {out_path}")
    if map_payload is None and not args.no_map:
        print("Note: map PNG/YAML not embedded (could not auto-find or missing origin/resolution).")
        print("Expected next to centerline: <Track>_map.png and <Track>_map.yaml")
    print("Open via local server:")
    print("  python3 -m http.server 8000")
    print("  then in Windows: http://localhost:8000/" + out_path.as_posix())


if __name__ == "__main__":
    main()
