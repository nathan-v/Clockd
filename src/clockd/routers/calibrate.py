from __future__ import annotations

import logging
import os

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

from clockd.config import CameraConfig, save_camera
from clockd.services.pipeline import VEHICLE_CLASS_IDS, process_video
from clockd.utils.video import cleanup, read_upload_with_limit, stream_upload_to_disk

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50MB limit for image uploads

router = APIRouter(prefix="/calibrate", tags=["calibrate"])


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode PNG")
    return buf.tobytes()


def _get_camera(request: Request, camera_id: str) -> CameraConfig:
    cam = request.app.state.cameras.get(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")
    return cam


def _draw_roi(img: np.ndarray, cam: CameraConfig) -> np.ndarray:
    out = img.copy()
    roi_pts = cam.roi_polygon or cam.calibration.source_points
    polygon = np.array(roi_pts, dtype=np.int32)

    # Semi-transparent fill
    overlay = out.copy()
    cv2.fillPoly(overlay, [polygon], (0, 180, 0))
    cv2.addWeighted(overlay, 0.2, out, 0.8, 0, out)

    # Polygon outline
    cv2.polylines(out, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    # Source points with labels
    src_pts = np.array(cam.calibration.source_points, dtype=np.int32)
    labels = ["TL", "TR", "BR", "BL"]
    for pt, label in zip(src_pts, labels):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(out, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
            out,
            f"{label} ({x},{y})",
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # Dimensions label
    w = cam.calibration.target_width_m
    h = cam.calibration.target_height_m
    cv2.putText(
        out,
        f"Real-world: {w}m x {h}m",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return out


@router.post("/preview")
async def calibrate_preview(
    request: Request,
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    detect: bool = Form(True),
) -> Response:
    """Return the uploaded frame with ROI polygon, source points, and optionally vehicle detections drawn."""
    cam = _get_camera(request, camera_id)
    data = await read_upload_with_limit(file, MAX_IMAGE_BYTES)
    img = _decode_image(data)
    out = _draw_roi(img, cam)

    if detect:
        from clockd.services.detector import create_detector

        server_cfg = request.app.state.server_cfg
        model_name = cam.model_override or server_cfg.model
        confidence = cam.confidence_override or server_cfg.confidence
        detector = create_detector(
            backend=server_cfg.detection_backend,
            model_name=model_name,
            confidence=confidence,
            codeproject_url=server_cfg.codeproject_ai.url,
            codeproject_timeout=server_cfg.codeproject_ai.timeout,
            roboflow_url=server_cfg.roboflow.url,
            roboflow_model_id=server_cfg.roboflow.model_id,
            roboflow_timeout=server_cfg.roboflow.timeout,
            localai_url=server_cfg.localai.url,
            localai_model=server_cfg.localai.model,
            localai_timeout=server_cfg.localai.timeout,
        )
        detections = detector.detect(img)

        # Filter to vehicles
        if detections.class_id is not None and len(detections) > 0:
            mask = np.isin(detections.class_id, VEHICLE_CLASS_IDS)
            detections = detections[mask]

        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            label = f"vehicle {conf:.2f}"

            # Check if bottom-center is inside ROI
            bc_x, bc_y = (x1 + x2) / 2, float(y2)
            roi_pts = cam.roi_polygon or cam.calibration.source_points
            polygon = np.array(roi_pts, dtype=np.float32)
            inside = cv2.pointPolygonTest(polygon, (bc_x, bc_y), False)

            color = (0, 255, 0) if inside >= 0 else (128, 128, 128)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return Response(content=_encode_png(out), media_type="image/png")


@router.post("/warp")
async def calibrate_warp(
    request: Request,
    file: UploadFile = File(...),
    camera_id: str = Form(...),
) -> Response:
    """Return the bird's-eye perspective-transformed view of the ROI region."""
    cam = _get_camera(request, camera_id)
    data = await read_upload_with_limit(file, MAX_IMAGE_BYTES)
    img = _decode_image(data)

    tw = cam.calibration.target_width_m
    th = cam.calibration.target_height_m

    # Scale to reasonable pixel size (50 pixels per meter), capped to prevent
    # excessive memory allocation from large calibration dimensions
    MAX_WARP_PX = 4000
    scale = 50
    w_px = int(tw * scale)
    h_px = int(th * scale)
    if w_px > MAX_WARP_PX or h_px > MAX_WARP_PX:
        scale = int(min(MAX_WARP_PX / tw, MAX_WARP_PX / th))
        scale = max(scale, 1)
        w_px = int(tw * scale)
        h_px = int(th * scale)

    source = np.array(cam.calibration.source_points, dtype=np.float32)
    target = np.array([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(img, m, (w_px, h_px))

    # Add grid lines every meter
    for x in range(0, w_px, scale):
        cv2.line(warped, (x, 0), (x, h_px), (255, 255, 255), 1)
    for y in range(0, h_px, scale):
        cv2.line(warped, (0, y), (w_px, y), (255, 255, 255), 1)

    # Label axes
    cv2.putText(warped, f"{tw}m", (w_px - 60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(warped, f"{th}m", (5, h_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return Response(content=_encode_png(warped), media_type="image/png")


@router.post("/extract-frame")
async def extract_frame(
    request: Request,
    file: UploadFile = File(...),
    frame_number: int = Form(0),
) -> Response:
    """Extract a single frame from an uploaded video. Defaults to first frame."""
    server_cfg = request.app.state.server_cfg
    max_bytes = server_cfg.max_upload_mb * 1024 * 1024
    tmp_path = await stream_upload_to_disk(file, server_cfg.upload_dir, max_bytes)

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_number < 0 or frame_number >= total:
            cap.release()
            raise HTTPException(
                status_code=400,
                detail=f"frame_number must be 0-{total - 1}, got {frame_number}",
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise HTTPException(status_code=400, detail="Could not read frame")

        return Response(content=_encode_png(frame), media_type="image/png")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@router.post("/speed-test")
async def speed_test(
    request: Request,
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    known_speed: float = Form(...),
    unit: str = Form("mph"),
    apply: bool = Form(False),
):
    """Upload a video recorded at a known speed to compute the calibration factor.

    The endpoint processes the video, picks the highest-confidence vehicle track,
    compares its measured speed to ``known_speed``, and returns the recommended
    ``speed_calibration_factor``.  Pass ``apply=true`` to auto-save it to the
    camera config.
    """
    cam = _get_camera(request, camera_id)
    server_cfg = request.app.state.server_cfg

    if unit not in ("mph", "kmh"):
        raise HTTPException(status_code=400, detail="unit must be 'mph' or 'kmh'")
    if known_speed <= 0:
        raise HTTPException(status_code=400, detail="known_speed must be positive")

    max_bytes = server_cfg.max_upload_mb * 1024 * 1024
    video_path = await stream_upload_to_disk(file, server_cfg.upload_dir, max_bytes)

    try:
        result = process_video(video_path, cam, server_cfg, unit)
    finally:
        cleanup(video_path)

    if not result.vehicles:
        raise HTTPException(
            status_code=422,
            detail="No vehicles detected in the video. Ensure a vehicle is visible within the calibration polygon.",
        )

    # Pick the track with the most detections (most reliable measurement)
    best = max(result.vehicles, key=lambda v: v.num_detections)
    measured_speed = best.speed_avg

    if measured_speed <= 0:
        raise HTTPException(
            status_code=422, detail="Measured speed was zero — check calibration polygon"
        )

    # Factor accounts for the current calibration factor already baked in
    recommended_factor = round(known_speed / measured_speed * cam.speed_calibration_factor, 4)

    response = {
        "camera_id": camera_id,
        "known_speed": known_speed,
        "measured_speed": round(measured_speed, 1),
        "current_factor": cam.speed_calibration_factor,
        "recommended_factor": recommended_factor,
        "unit": unit,
        "track_id": best.track_id,
        "num_detections": best.num_detections,
        "applied": False,
    }

    if apply:
        cam = cam.model_copy(update={"speed_calibration_factor": recommended_factor})
        cameras_dir = server_cfg.cameras_dir
        save_camera(cameras_dir, cam)
        request.app.state.cameras[camera_id] = cam
        response["applied"] = True
        logger.info(
            "Speed calibration applied: camera=%s factor=%.4f (known=%.1f measured=%.1f %s)",
            camera_id,
            recommended_factor,
            known_speed,
            measured_speed,
            unit,
        )

    return response


@router.get("/ui", response_class=HTMLResponse)
async def calibration_ui() -> HTMLResponse:
    """Interactive calibration UI for clicking source points on a camera frame."""
    return HTMLResponse(content=_CALIBRATION_UI_HTML)


_CALIBRATION_UI_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clockd Calibration Tool</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #1a1a2e; color: #e0e0e0; padding: 20px; }
  h1 { margin-bottom: 8px; color: #fff; }
  .subtitle { color: #888; margin-bottom: 20px; }
  .layout { display: flex; gap: 20px; flex-wrap: wrap; }
  .canvas-wrap { position: relative; border: 2px solid #333; border-radius: 8px;
                 overflow: hidden; background: #000; flex-shrink: 0; }
  canvas { display: block; cursor: crosshair; }
  .sidebar { min-width: 320px; flex: 1; }
  .section { background: #16213e; border-radius: 8px; padding: 16px;
             margin-bottom: 16px; }
  .section h3 { margin-bottom: 12px; color: #4cc9f0; }
  label { display: block; margin-bottom: 4px; font-size: 14px; color: #aaa; }
  input, select { width: 100%; padding: 8px; border-radius: 4px; border: 1px solid #333;
         background: #0f3460; color: #fff; margin-bottom: 12px; font-size: 14px; }
  button { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;
           font-size: 14px; margin-right: 8px; margin-bottom: 8px; }
  .btn-primary { background: #4cc9f0; color: #000; }
  .btn-danger { background: #e63946; color: #fff; }
  .btn-secondary { background: #333; color: #fff; }
  .point-list { list-style: none; }
  .point-list li { padding: 6px 8px; margin-bottom: 4px; background: #0f3460;
                   border-radius: 4px; display: flex; justify-content: space-between;
                   align-items: center; font-family: monospace; }
  .point-label { color: #4cc9f0; font-weight: bold; margin-right: 8px; }
  .remove-btn { background: none; border: none; color: #e63946; cursor: pointer;
                font-size: 16px; padding: 0 4px; margin: 0; }
  pre { background: #0a0a1a; padding: 12px; border-radius: 4px; overflow-x: auto;
        font-size: 13px; white-space: pre-wrap; user-select: all; }
  .drop-zone { border: 2px dashed #333; border-radius: 8px; padding: 40px;
               text-align: center; cursor: pointer; transition: border-color 0.2s; }
  .drop-zone:hover, .drop-zone.drag-over { border-color: #4cc9f0; }
  .instructions { font-size: 13px; color: #888; line-height: 1.6; }
</style>
</head>
<body>
<h1>Clockd Calibration Tool</h1>
<p class="subtitle">Click 4 points on your camera frame to define the road region</p>

<div class="layout">
  <div>
    <div class="canvas-wrap" id="canvasWrap" style="display:none">
      <canvas id="canvas"></canvas>
    </div>
    <div class="drop-zone" id="dropZone">
      <p>Drop a camera frame image here, or click to select</p>
      <p style="margin-top:8px; font-size:13px; color:#666">
        PNG / JPG from your camera, or use<br>
        <code>POST /calibrate/extract-frame</code> to grab one from a video
      </p>
      <input type="file" id="fileInput" accept="image/*" style="display:none">
    </div>
  </div>

  <div class="sidebar">
    <div class="section">
      <h3>Source Points</h3>
      <div class="instructions">
        <p>Click corners of the road in this order:</p>
        <p><strong>1. TL</strong> (top-left, far side) &rarr; <strong>2. TR</strong> (top-right, far side)</p>
        <p><strong>3. BR</strong> (bottom-right, near side) &rarr; <strong>4. BL</strong> (bottom-left, near side)</p>
      </div>
      <ul class="point-list" id="pointList"></ul>
      <button class="btn-danger" onclick="clearPoints()">Clear All</button>
      <button class="btn-secondary" onclick="undoPoint()">Undo Last</button>
    </div>

    <div class="section">
      <h3>Real-World Dimensions</h3>
      <label>Road width (meters)</label>
      <input type="number" id="targetWidth" value="8.0" step="0.1" min="0.1">
      <label>Road depth/length (meters)</label>
      <input type="number" id="targetHeight" value="40.0" step="0.1" min="0.1">
    </div>

    <div class="section">
      <h3>Camera Config</h3>
      <label>Camera ID</label>
      <input type="text" id="cameraId" value="my_camera">
      <label>Description</label>
      <input type="text" id="description" value="">
      <label>Resolution (auto-detected)</label>
      <input type="text" id="resolution" readonly>
    </div>

    <div class="section">
      <h3>YAML Config</h3>
      <p style="font-size:13px;color:#888;margin-bottom:8px">Edit directly or click points on the image. Changes sync both ways.</p>
      <textarea id="yamlOutput" rows="20" spellcheck="false"
        style="width:100%;font-family:monospace;font-size:13px;background:#0a0a1a;color:#e0e0e0;
               border:1px solid #333;border-radius:4px;padding:10px;resize:vertical;tab-size:2">Click 4 points on the image to generate config...</textarea>
      <button class="btn-primary" onclick="copyYaml()">Copy to Clipboard</button>
      <button class="btn-primary" onclick="submitConfig()">Create via API</button>
      <button class="btn-secondary" onclick="applyYaml()">Apply Edits to Preview</button>
    </div>
  </div>
</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const labels = ['TL', 'TR', 'BR', 'BL'];
const colors = ['#e63946', '#f4a261', '#2a9d8f', '#264653'];
let points = [];
let img = null;
let imgW = 0, imgH = 0, scale = 1;

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const canvasWrap = document.getElementById('canvasWrap');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => { if (e.target.files.length) loadFile(e.target.files[0]); });

function loadFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    img = new Image();
    img.onload = () => {
      imgW = img.naturalWidth;
      imgH = img.naturalHeight;
      document.getElementById('resolution').value = imgW + 'x' + imgH;
      const maxW = Math.min(960, window.innerWidth - 400);
      scale = Math.min(maxW / imgW, 1);
      canvas.width = imgW * scale;
      canvas.height = imgH * scale;
      dropZone.style.display = 'none';
      canvasWrap.style.display = 'block';
      draw();
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

canvas.addEventListener('click', e => {
  if (!img || points.length >= 4) return;
  const rect = canvas.getBoundingClientRect();
  const x = Math.round((e.clientX - rect.left) / scale);
  const y = Math.round((e.clientY - rect.top) / scale);
  points.push([x, y]);
  draw();
  updateUI();
});

function draw() {
  if (!img) return;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  // Draw polygon
  if (points.length >= 2) {
    ctx.beginPath();
    ctx.moveTo(points[0][0]*scale, points[0][1]*scale);
    for (let i = 1; i < points.length; i++)
      ctx.lineTo(points[i][0]*scale, points[i][1]*scale);
    if (points.length === 4) ctx.closePath();
    ctx.strokeStyle = '#4cc9f0';
    ctx.lineWidth = 2;
    ctx.stroke();
    if (points.length === 4) {
      ctx.fillStyle = 'rgba(76,201,240,0.15)';
      ctx.fill();
    }
  }
  // Draw points
  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p[0]*scale, p[1]*scale, 6, 0, Math.PI*2);
    ctx.fillStyle = colors[i];
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 14px monospace';
    ctx.fillText(labels[i]+' ('+p[0]+','+p[1]+')', p[0]*scale+10, p[1]*scale-10);
  });
}

function updateUI() {
  const list = document.getElementById('pointList');
  list.innerHTML = points.map((p, i) =>
    '<li><span><span class="point-label">'+labels[i]+'</span>'+p[0]+', '+p[1]+'</span>' +
    '<button class="remove-btn" onclick="removePoint('+i+')">&times;</button></li>'
  ).join('');
  generateYaml();
}

function removePoint(i) { points.splice(i, 1); draw(); updateUI(); }
function clearPoints() { points = []; draw(); updateUI(); }
function undoPoint() { points.pop(); draw(); updateUI(); }

let yamlManuallyEdited = false;

function generateYaml() {
  if (yamlManuallyEdited) return;  // don't overwrite manual edits
  const out = document.getElementById('yamlOutput');
  if (points.length < 4) { out.value = 'Click '+(4-points.length)+' more point(s)...'; return; }
  const id = document.getElementById('cameraId').value || 'my_camera';
  const desc = document.getElementById('description').value;
  const tw = document.getElementById('targetWidth').value;
  const th = document.getElementById('targetHeight').value;
  const res = document.getElementById('resolution').value;
  let yaml = 'camera_id: "'+id+'"\\n';
  if (desc) yaml += 'description: "'+desc+'"\\n';
  if (res) yaml += 'resolution: ['+res.replace('x',', ')+']\\n';
  yaml += '\\ncalibration:\\n  source_points:\\n';
  points.forEach(p => { yaml += '    - ['+p[0]+', '+p[1]+']\\n'; });
  yaml += '  target_width_m: '+tw+'\\n';
  yaml += '  target_height_m: '+th+'\\n';
  yaml += '\\nmin_detections: 10\\nsmoothing_window: 5\\nspeed_calibration_factor: 1.0\\nspeed_range:\\n  min_mph: 3.0\\n  max_mph: 150.0\\n';
  out.value = yaml.replace(/\\\\n/g, '\\n');
}

function forceGenerateYaml() {
  yamlManuallyEdited = false;
  generateYaml();
}

// Mark yaml as manually edited when user types in the textarea
document.getElementById('yamlOutput').addEventListener('input', () => { yamlManuallyEdited = true; });

function parseYamlConfig() {
  // Simple YAML parser for our known format
  const text = document.getElementById('yamlOutput').value;
  const config = {};
  // camera_id
  const idMatch = text.match(/camera_id:\\s*"?([^"\\n]+)"?/);
  if (idMatch) config.camera_id = idMatch[1].trim();
  // description
  const descMatch = text.match(/description:\\s*"?([^"\\n]*)"?/);
  if (descMatch) config.description = descMatch[1].trim();
  // resolution
  const resMatch = text.match(/resolution:\\s*\\[\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\]/);
  if (resMatch) config.resolution = [parseInt(resMatch[1]), parseInt(resMatch[2])];
  // source_points
  const ptMatches = [...text.matchAll(/- \\[\\s*(-?[\\d.]+)\\s*,\\s*(-?[\\d.]+)\\s*\\]/g)];
  if (ptMatches.length >= 4) {
    config.source_points = ptMatches.slice(0, 4).map(m => [parseFloat(m[1]), parseFloat(m[2])]);
  }
  // target dims
  const twMatch = text.match(/target_width_m:\\s*([\\d.]+)/);
  if (twMatch) config.target_width_m = parseFloat(twMatch[1]);
  const thMatch = text.match(/target_height_m:\\s*([\\d.]+)/);
  if (thMatch) config.target_height_m = parseFloat(thMatch[1]);
  // min_detections
  const mdMatch = text.match(/min_detections:\\s*(\\d+)/);
  if (mdMatch) config.min_detections = parseInt(mdMatch[1]);
  // smoothing_window
  const swMatch = text.match(/smoothing_window:\\s*(\\d+)/);
  if (swMatch) config.smoothing_window = parseInt(swMatch[1]);
  // speed_calibration_factor
  const scfMatch = text.match(/speed_calibration_factor:\\s*([\\d.]+)/);
  if (scfMatch) config.speed_calibration_factor = parseFloat(scfMatch[1]);
  // speed_range
  const minSpd = text.match(/min_mph:\\s*([\\d.]+)/);
  const maxSpd = text.match(/max_mph:\\s*([\\d.]+)/);
  if (minSpd || maxSpd) {
    config.speed_range = {};
    if (minSpd) config.speed_range.min_mph = parseFloat(minSpd[1]);
    if (maxSpd) config.speed_range.max_mph = parseFloat(maxSpd[1]);
  }
  return config;
}

function applyYaml() {
  const config = parseYamlConfig();
  if (config.camera_id) document.getElementById('cameraId').value = config.camera_id;
  if (config.description !== undefined) document.getElementById('description').value = config.description;
  if (config.target_width_m) document.getElementById('targetWidth').value = config.target_width_m;
  if (config.target_height_m) document.getElementById('targetHeight').value = config.target_height_m;
  if (config.resolution) document.getElementById('resolution').value = config.resolution.join('x');
  if (config.source_points && config.source_points.length === 4) {
    points = config.source_points.map(p => [Math.round(p[0]), Math.round(p[1])]);
    draw();
    updateUI();
  }
  yamlManuallyEdited = false;
}

function copyYaml() {
  const text = document.getElementById('yamlOutput').value;
  navigator.clipboard.writeText(text);
}

function submitConfig() {
  const config = parseYamlConfig();
  if (!config.source_points || config.source_points.length < 4) {
    alert('Need 4 source points in the YAML config');
    return;
  }
  const body = {
    camera_id: config.camera_id || document.getElementById('cameraId').value,
    description: config.description || document.getElementById('description').value || '',
    calibration: {
      source_points: config.source_points,
      target_width_m: config.target_width_m || parseFloat(document.getElementById('targetWidth').value),
      target_height_m: config.target_height_m || parseFloat(document.getElementById('targetHeight').value),
    },
  };
  if (config.resolution) body.resolution = config.resolution;
  if (config.min_detections) body.min_detections = config.min_detections;
  if (config.smoothing_window) body.smoothing_window = config.smoothing_window;
  if (config.speed_calibration_factor) body.speed_calibration_factor = config.speed_calibration_factor;
  if (config.speed_range) body.speed_range = config.speed_range;
  fetch('/cameras', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) })
    .then(r => { if (!r.ok) return r.json().then(e => { throw new Error(e.detail); }); return r.json(); })
    .then(d => alert('Camera "'+d.camera_id+'" created!'))
    .catch(e => alert('Error: '+e.message));
}

// Re-generate on input changes (resets manual edit flag)
['cameraId','description','targetWidth','targetHeight'].forEach(id =>
  document.getElementById(id).addEventListener('input', forceGenerateYaml));
// Clicking on canvas also resets manual edit flag
canvas.addEventListener('mouseup', () => { yamlManuallyEdited = false; });
</script>
</body>
</html>
"""
