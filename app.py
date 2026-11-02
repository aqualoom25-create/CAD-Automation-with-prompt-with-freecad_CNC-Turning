"""
Local Web Hosting Studio — CNC Turning & Hex Mill-Turn Image-to-STEP Converter
===============================================================================
Provides an interactive localhost web dashboard with:
  - Dedicated "Input & Upload Turning Part" Tab (Drag & Drop ANY CNC turning part image:
    PNG, JPG, WEBP, BMP) + live conversion into a watertight 3D B-Rep STEP solid
  - Interactive 3D WebGL CAD Viewer (Three.js STL/B-Rep viewer with PBR metallic shading & wireframe)
  - Custom Bilateral Tolerance (+/- 0.50 mm) & Calibration Scale Re-Calculator
  - One-click downloads for Nominal STEP, +0.50 mm MMC STEP, -0.50 mm LMC STEP, STL, and JSON Report
  - Full Feature-by-Feature Bilateral Tolerance Schedule & Engineering Visual Gallery
"""

import os
import json
import time
import shutil
from werkzeug.utils import secure_filename
from flask import Flask, render_template_string, send_from_directory, request, jsonify

from cnc_turning_part_converter import convert_any_turning_image_to_step

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

CURRENT_IMG_TRACKER = os.path.join(OUT_DIR, "active_image_path.txt")

app = Flask(__name__)


def get_active_image_path() -> str:
    if os.path.exists(CURRENT_IMG_TRACKER):
        with open(CURRENT_IMG_TRACKER, "r", encoding="utf-8") as f:
            p = f.read().strip()
        if p and os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, "reference_turning_part.png")


def set_active_image_path(path: str) -> None:
    with open(CURRENT_IMG_TRACKER, "w", encoding="utf-8") as f:
        f.write(path)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CNC Turning Part to 3D STEP Studio (±0.50 mm Tolerance)</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --green: #22c55e;
      --amber: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding-bottom: 60px;
    }
    header {
      background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 100%);
      border-bottom: 1px solid var(--border);
      padding: 18px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    .title-group h1 {
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }
    .title-group p {
      color: var(--muted);
      font-size: 0.88rem;
    }
    .badge-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .badge {
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid rgba(56, 189, 248, 0.4);
      color: var(--accent);
    }
    .badge.green {
      background: rgba(34, 197, 94, 0.15);
      border-color: rgba(34, 197, 94, 0.4);
      color: #4ade80;
    }
    /* Navigation Tabs */
    .tab-bar {
      background: #131c31;
      border-bottom: 1px solid var(--border);
      padding: 0 32px;
      display: flex;
      gap: 8px;
      overflow-x: auto;
    }
    .tab-btn {
      padding: 13px 20px;
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--muted);
      background: transparent;
      border: none;
      border-bottom: 3px solid transparent;
      cursor: pointer;
      transition: all 0.15s;
      white-space: nowrap;
    }
    .tab-btn:hover {
      color: var(--text);
      background: rgba(255, 255, 255, 0.03);
    }
    .tab-btn.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
      background: rgba(56, 189, 248, 0.08);
    }
    .container {
      max-width: 1480px;
      margin: 24px auto;
      padding: 0 24px;
    }
    .input-grid {
      display: grid;
      grid-template-columns: 1.05fr 1.35fr;
      gap: 22px;
      margin-bottom: 24px;
    }
    .top-grid {
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 22px;
      margin-bottom: 24px;
    }
    @media (max-width: 1024px) {
      .input-grid, .top-grid { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
      flex-wrap: wrap;
      gap: 8px;
    }
    .card-header h2 {
      font-size: 1.08rem;
      font-weight: 600;
      color: #e2e8f0;
    }
    /* Drag and drop upload zone */
    .dropzone {
      border: 2px dashed #38bdf8;
      background: rgba(15, 23, 42, 0.65);
      border-radius: 10px;
      padding: 22px 18px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
      position: relative;
    }
    .dropzone:hover, .dropzone.dragover {
      background: rgba(56, 189, 248, 0.12);
      border-color: #7dd3fc;
    }
    .dropzone input[type="file"] {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
    }
    .dropzone-icon {
      font-size: 2.1rem;
      margin-bottom: 6px;
      display: block;
    }
    .dropzone-title {
      font-size: 0.98rem;
      font-weight: 700;
      color: #e2e8f0;
    }
    .dropzone-sub {
      font-size: 0.8rem;
      color: var(--muted);
      margin-top: 4px;
    }
    .preview-compare-box {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      align-items: center;
    }
    .preview-pane {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      text-align: center;
    }
    .preview-pane img {
      width: 100%;
      height: 295px;
      object-fit: contain;
      border-radius: 6px;
      background: #f8fafc;
      cursor: zoom-in;
    }
    .preview-pane .pane-label {
      font-size: 0.82rem;
      font-weight: 600;
      color: #cbd5e1;
      margin-bottom: 8px;
    }
    #viewer3d {
      width: 100%;
      height: 490px;
      background: radial-gradient(circle at center, #1e293b 0%, #090d16 100%);
      border-radius: 8px;
      border: 1px solid var(--border);
      position: relative;
      overflow: hidden;
    }
    .viewer-controls {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 0.84rem;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--border);
      background: #334155;
      color: var(--text);
      text-decoration: none;
      transition: all 0.15s;
    }
    .btn:hover { background: #475569; }
    .btn.primary {
      background: #0284c7;
      border-color: #38bdf8;
    }
    .btn.primary:hover { background: #0369a1; }
    .btn.success {
      background: #15803d;
      border-color: #4ade80;
    }
    .btn.success:hover { background: #166534; }
    .btn.amber {
      background: #d97706;
      border-color: #fbbf24;
    }
    .btn.amber:hover { background: #b45309; }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }
    .kpi {
      background: rgba(15, 23, 42, 0.65);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
    }
    .kpi .label {
      font-size: 0.76rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .kpi .val {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--accent);
      margin-top: 2px;
    }
    .download-stack {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-top: 12px;
    }
    .form-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-top: 12px;
    }
    .form-group label {
      display: block;
      font-size: 0.78rem;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .form-group input, .form-group select {
      width: 100%;
      padding: 8px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--text);
      font-size: 0.86rem;
    }
    .status-banner {
      display: none;
      margin-top: 12px;
      padding: 10px 14px;
      border-radius: 8px;
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid var(--accent);
      color: #bae6fd;
      font-size: 0.85rem;
      font-weight: 600;
    }
    .gallery-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 18px;
      margin-bottom: 24px;
    }
    @media (max-width: 900px) {
      .gallery-grid { grid-template-columns: 1fr; }
    }
    .gallery-item {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    .gallery-item img {
      width: 100%;
      height: auto;
      display: block;
      cursor: zoom-in;
    }
    .gallery-caption {
      padding: 10px 14px;
      font-size: 0.85rem;
      font-weight: 600;
      color: #cbd5e1;
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.83rem;
    }
    th, td {
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    th {
      background: rgba(15, 23, 42, 0.8);
      color: var(--accent);
      font-weight: 600;
      position: sticky;
      top: 0;
    }
    tr:hover { background: rgba(255, 255, 255, 0.03); }
    .pass-pill {
      background: rgba(34, 197, 94, 0.2);
      color: #4ade80;
      padding: 2px 8px;
      border-radius: 99px;
      font-weight: 700;
      font-size: 0.75rem;
    }
    #lightbox {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.88);
      z-index: 9999;
      justify-content: center;
      align-items: center;
      padding: 24px;
    }
    #lightbox img {
      max-width: 95vw;
      max-height: 92vh;
      border-radius: 8px;
      border: 2px solid var(--accent);
    }
  </style>
</head>
<body>
  <header>
    <div class="title-group">
      <h1>CNC Turning &amp; Hex Mill-Turn Part — 3D STEP &amp; ±0.50 mm Tolerance Studio</h1>
      <p>{{ summary.part_name }} • {{ summary.thread_specification }}</p>
    </div>
    <div class="badge-row">
      <span class="badge">B-Rep Topology: {{ summary.step_brep_validation.watertight_solids_count }} Solid / {{ summary.step_brep_validation.brep_faces_count }} Faces / {{ summary.step_brep_validation.brep_edges_count }} Edges</span>
      <span class="badge green">Max Profile Error: {{ "%.4f"|format(summary.profile_accuracy_verification.max_deviation_mm) }} mm (&le; {{ "%.2f"|format(summary.specified_tolerance_mm) }} mm PASS)</span>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <nav class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('all', this)">&#128256; All-in-One Studio (Input + 3D STEP + Tolerance)</button>
    <button class="tab-btn" onclick="switchTab('input', this)">&#128228; Input Tab: Upload Any Turning Part Image</button>
    <button class="tab-btn" onclick="switchTab('viewer', this)">&#127760; Interactive 3D STEP Viewer &amp; Downloads</button>
    <button class="tab-btn" onclick="switchTab('gallery', this)">&#128444; Visual Verification &amp; 4-View CAD Sheets</button>
    <button class="tab-btn" onclick="switchTab('tolerance', this)">&#128208; &plusmn;{{ "%.2f"|format(summary.specified_tolerance_mm) }} mm Tolerance Schedule ({{ features|length }} Features)</button>
  </nav>

  <div class="container">
    <!-- SECTION 1: INPUT TAB (UPLOAD ANY TURNING PART IMAGE & CONVERT TO 3D STEP) -->
    <div id="sec-input" class="tab-section">
      <div class="input-grid">
        <div class="card">
          <div class="card-header">
            <h2>&#128228; Input Tab — Upload Any CNC Turning Part Image</h2>
            <button type="button" class="btn" onclick="resetToReference()" title="Restore default threaded hex sensor probe">
              &#8635; Load Reference Hex Probe
            </button>
          </div>

          <form id="uploadConvertForm" onsubmit="handleUploadConvert(event)">
            <div class="dropzone" id="dropzone">
              <input type="file" id="partImageInput" name="image" accept=".png,.jpg,.jpeg,.webp,.bmp" onchange="previewSelectedFile(this)" />
              <span class="dropzone-icon">&#128247;</span>
              <div class="dropzone-title" id="dropzoneTitle">Drag &amp; Drop Any Turning Part Image Here or Click to Browse</div>
              <div class="dropzone-sub">Supports PNG, JPG, JPEG, WEBP, BMP • Auto-segments steps, tapers, grooves, threads &amp; bores</div>
            </div>

            <div class="form-row">
              <div class="form-group">
                <label>Bilateral Tolerance (&plusmn; mm)</label>
                <input type="number" step="0.05" min="0.05" max="5.0" id="tolInput" name="tol_mm" value="{{ "%.2f"|format(summary.specified_tolerance_mm) }}" required />
              </div>
              <div class="form-group">
                <label>Calibration Scale (mm/px)</label>
                <input type="number" step="0.005" min="0.01" max="5.0" id="scaleInput" name="scale_mm_per_px" value="{{ "%.3f"|format(summary.calibration_scale_mm_per_px) }}" required />
              </div>
              <div class="form-group">
                <label>Spindle Axis Orientation</label>
                <select id="orientInput" name="orientation">
                  <option value="auto" selected>Auto-Detect Axis</option>
                  <option value="vertical">Vertical Turning Axis</option>
                  <option value="horizontal">Horizontal Turning Axis</option>
                </select>
              </div>
            </div>

            <div style="display:flex; gap:10px; margin-top:14px;">
              <button type="submit" class="btn success" id="convertBtn" style="flex:1; justify-content:center; padding:11px 16px; font-size:0.93rem;">
                &#128640; Convert Uploaded Turning Part to 3D STEP &amp; Calculate &plusmn;0.50 mm Tolerance
              </button>
            </div>
            <div class="status-banner" id="statusBanner">
              &#9881; Segmenting turning part silhouette, extracting axial features, building OpenCASCADE 3D B-Rep STEP solid, and rendering studio views...
            </div>
          </form>
        </div>

        <!-- Live Input vs Converted 3D STEP Preview Card -->
        <div class="card">
          <div class="card-header">
            <h2>Live Input Part vs. Converted 3D B-Rep STEP Solid</h2>
            <a class="btn success" href="/download/cnc_turning_part.step">&#11015; Download Final STEP File</a>
          </div>
          <div class="preview-compare-box">
            <div class="preview-pane">
              <div class="pane-label">1. Input Turning Part Image</div>
              <img id="inputPreviewImg" src="/outputs/current_input_part.png?t={{ ts }}" onclick="openLightbox(this.src)" alt="Input Turning Part" />
            </div>
            <div class="preview-pane">
              <div class="pane-label">2. Converted Final 3D STEP Render</div>
              <img id="outputStepImg" src="/outputs/step_3d_studio_render.png?t={{ ts }}" onclick="openLightbox(this.src)" alt="Converted 3D STEP Model" />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- SECTION 2: INTERACTIVE 3D WEBGL VIEWER & STEP DOWNLOADS -->
    <div id="sec-viewer" class="tab-section">
      <div class="top-grid">
        <div class="card">
          <div class="card-header">
            <h2>Interactive 3D B-Rep STEP Model Viewer (Drag to Rotate, Scroll to Zoom)</h2>
            <span style="font-size:0.8rem; color:var(--muted);">
              BBox: {{ summary.step_brep_validation.bounding_box_mm.x_span_mm }} &times; {{ summary.step_brep_validation.bounding_box_mm.y_span_mm }} &times; {{ summary.step_brep_validation.bounding_box_mm.z_span_mm }} mm
            </span>
          </div>
          <div id="viewer3d"></div>
          <div class="viewer-controls">
            <button class="btn primary" onclick="resetCamera()">Reset Isometric View</button>
            <button class="btn" onclick="setFrontView()">Front Elevation View</button>
            <button class="btn" onclick="setTopView()">Top Plan View</button>
            <button class="btn" onclick="toggleWireframe()">Toggle B-Rep Mesh Wireframe</button>
            <button class="btn" onclick="toggleAutoRotate()">Toggle Auto-Rotate</button>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h2>STEP File Downloads &amp; Engineering Summary</h2>
          </div>
          <div class="kpi-grid">
            <div class="kpi">
              <div class="label">Overall Height (Z)</div>
              <div class="val">{{ summary.overall_height_nominal_mm }} &plusmn; {{ summary.specified_tolerance_mm }} mm</div>
            </div>
            <div class="kpi">
              <div class="label">Max Outer Diameter</div>
              <div class="val">&Oslash;{{ summary.max_outer_diameter_across_corners_mm }} &plusmn; {{ summary.specified_tolerance_mm }} mm</div>
            </div>
            <div class="kpi">
              <div class="label">Recognized Features</div>
              <div class="val">{{ features|length }} Turning Zones</div>
            </div>
            <div class="kpi">
              <div class="label">Smallest Pin / Tip OD</div>
              <div class="val">&Oslash;{{ summary.top_probe_pin_diameter_mm.nominal }} &plusmn; {{ summary.specified_tolerance_mm }} mm</div>
            </div>
            <div class="kpi">
              <div class="label">Worst-Case / RSS Stack-Up</div>
              <div class="val">&plusmn;{{ summary.axial_tolerance_stackup.worst_case_cumulative_stackup_mm }} / &plusmn;{{ summary.axial_tolerance_stackup.rss_statistical_stackup_3sigma_mm }} mm</div>
            </div>
            <div class="kpi">
              <div class="label">Volume / Mass (316 SS)</div>
              <div class="val">{{ summary.step_brep_validation.volume_mm3 }} mm&sup3; ({{ summary.step_brep_validation.estimated_mass_ss316_grams }} g)</div>
            </div>
          </div>

          <div class="download-stack">
            <a class="btn success" href="/download/cnc_turning_part.step" style="justify-content:center; padding:11px;">
              &#11015; Download Nominal Exact 3D STEP File (cnc_turning_part.step)
            </a>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
              <a class="btn primary" href="/download/cnc_turning_part_mmc_plus_0_5mm.step" style="justify-content:center;">
                &#11015; +{{ "%.2f"|format(summary.specified_tolerance_mm) }} mm MMC STEP
              </a>
              <a class="btn primary" href="/download/cnc_turning_part_lmc_minus_0_5mm.step" style="justify-content:center;">
                &#11015; -{{ "%.2f"|format(summary.specified_tolerance_mm) }} mm LMC STEP
              </a>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
              <a class="btn" href="/download/cnc_turning_part_mesh.stl" style="justify-content:center;">
                &#11015; Download 3D STL Mesh
              </a>
              <a class="btn" href="/download/cnc_turning_part_report.json" target="_blank" style="justify-content:center;">
                &#128196; View Engineering JSON Report
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- SECTION 3: VISUAL VERIFICATION GALLERY -->
    <div id="sec-gallery" class="tab-section">
      <div class="card" style="margin-bottom:24px;">
        <div class="card-header">
          <h2>Final STEP File Renders, Multi-View CAD Sheet &amp; ±{{ "%.2f"|format(summary.specified_tolerance_mm) }} mm Tolerance Plots (Click Any Image to Zoom)</h2>
        </div>
        <div class="gallery-grid">
          <div class="gallery-item" style="grid-column: span 2;">
            <img src="/outputs/side_by_side_comparison.png?t={{ ts }}" onclick="openLightbox(this.src)" alt="Side by Side Comparison" />
            <div class="gallery-caption">
              <span>1. Uploaded Turning Part Photo vs. Converted 3D STEP Model (PBR Shaded &amp; B-Rep Mesh)</span>
              <a class="btn" href="/outputs/side_by_side_comparison.png" download>Download PNG</a>
            </div>
          </div>
          <div class="gallery-item">
            <img src="/outputs/step_3d_studio_render.png?t={{ ts }}" onclick="openLightbox(this.src)" alt="3D STEP Studio Render" />
            <div class="gallery-caption">
              <span>2. Final 3D STEP File — Studio PBR Metallic Render</span>
              <a class="btn" href="/outputs/step_3d_studio_render.png" download>Download PNG</a>
            </div>
          </div>
          <div class="gallery-item">
            <img src="/outputs/multi_view_step_sheet.png?t={{ ts }}" onclick="openLightbox(this.src)" alt="Multi-View CAD Sheet" />
            <div class="gallery-caption">
              <span>3. 4-View Engineering Sheet (Isometric, Front, Top, B-Rep Edges)</span>
              <a class="btn" href="/outputs/multi_view_step_sheet.png" download>Download PNG</a>
            </div>
          </div>
          <div class="gallery-item">
            <img src="/outputs/feature_recognition_overlay.png?t={{ ts }}" onclick="openLightbox(this.src)" alt="Feature Recognition Map" />
            <div class="gallery-caption">
              <span>4. Recognized Turning Features &amp; Axial Diameter Profile</span>
              <a class="btn" href="/outputs/feature_recognition_overlay.png" download>Download PNG</a>
            </div>
          </div>
          <div class="gallery-item">
            <img src="/outputs/tolerance_0_5mm_analysis.png?t={{ ts }}" onclick="openLightbox(this.src)" alt="0.5mm Tolerance Analysis" />
            <div class="gallery-caption">
              <span>5. &plusmn;{{ "%.2f"|format(summary.specified_tolerance_mm) }} mm Bilateral Tolerance Deviation &amp; MMC/LMC Envelope</span>
              <a class="btn" href="/outputs/tolerance_0_5mm_analysis.png" download>Download PNG</a>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- SECTION 4: TOLERANCE SCHEDULE TABLE -->
    <div id="sec-tolerance" class="tab-section">
      <div class="card">
        <div class="card-header">
          <h2>Complete CNC Turning &amp; Mill-Turn Feature Schedule with &plusmn;{{ "%.2f"|format(summary.specified_tolerance_mm) }} mm Bilateral Tolerances</h2>
        </div>
        <div style="overflow-x:auto;">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Feature Name</th>
                <th>Machining Operation</th>
                <th>Axial Span Z (mm)</th>
                <th>Length L &plusmn; Tol (mm)</th>
                <th>Nominal &Oslash; D_nom (mm)</th>
                <th>Min &Oslash; D_min (LMC)</th>
                <th>Max &Oslash; D_max (MMC)</th>
                <th>Image Meas. (mm)</th>
                <th>Error |&Delta;D| (mm)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {% for f in features %}
              <tr>
                <td><strong>{{ f.feature_id }}</strong></td>
                <td>{{ f.name }}</td>
                <td style="color:var(--muted);">{{ f.operation_type }}</td>
                <td>{{ "%.2f"|format(f.z_start_mm) }} – {{ "%.2f"|format(f.z_end_mm) }}</td>
                <td>{{ "%.2f"|format(f.length_nom_mm) }} [{{ "%.2f"|format(f.length_min_mm) }}, {{ "%.2f"|format(f.length_max_mm) }}]</td>
                <td style="color:var(--accent); font-weight:700;">&Oslash;{{ "%.2f"|format(f.diameter_nom_mm) }}</td>
                <td>{{ "%.2f"|format(f.diameter_min_mm) }}</td>
                <td>{{ "%.2f"|format(f.diameter_max_mm) }}</td>
                <td>{{ "%.3f"|format(f.reconstructed_mm) }}</td>
                <td>{{ "%.4f"|format(f.deviation_from_image_mm) }}</td>
                <td><span class="pass-pill">PASS (&le; {{ "%.2f"|format(f.tolerance_bilateral_mm) }} mm)</span></td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <div id="lightbox" onclick="this.style.display='none'">
    <img id="lightboxImg" src="" alt="Zoomed View" />
  </div>

  <script>
    function switchTab(tabId, btnEl) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      if (btnEl) btnEl.classList.add('active');

      const map = {
        input: ['sec-input'],
        viewer: ['sec-viewer'],
        gallery: ['sec-gallery'],
        tolerance: ['sec-tolerance'],
        all: ['sec-input', 'sec-viewer', 'sec-gallery', 'sec-tolerance'],
      };
      const visibleIds = map[tabId] || map.all;
      ['sec-input', 'sec-viewer', 'sec-gallery', 'sec-tolerance'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = visibleIds.includes(id) ? 'block' : 'none';
      });
      window.dispatchEvent(new Event('resize'));
    }

    function openLightbox(src) {
      document.getElementById('lightboxImg').src = src;
      document.getElementById('lightbox').style.display = 'flex';
    }

    function previewSelectedFile(input) {
      if (input.files && input.files[0]) {
        const file = input.files[0];
        document.getElementById('dropzoneTitle').textContent = 'Selected Image: ' + file.name + ' (Ready to Convert)';
        const reader = new FileReader();
        reader.onload = function(e) {
          document.getElementById('inputPreviewImg').src = e.target.result;
        };
        reader.readAsDataURL(file);
      }
    }

    async function handleUploadConvert(e) {
      e.preventDefault();
      const btn = document.getElementById('convertBtn');
      const banner = document.getElementById('statusBanner');
      btn.disabled = true;
      btn.textContent = '⏳ Converting Turning Part Image to 3D STEP & Calculating Tolerances...';
      banner.style.display = 'block';

      const formData = new FormData();
      const fileInput = document.getElementById('partImageInput');
      if (fileInput.files && fileInput.files[0]) {
        formData.append('image', fileInput.files[0]);
      }
      formData.append('tol_mm', document.getElementById('tolInput').value);
      formData.append('scale_mm_per_px', document.getElementById('scaleInput').value);
      formData.append('orientation', document.getElementById('orientInput').value);

      try {
        const res = await fetch('/api/upload_convert', {
          method: 'POST',
          body: formData,
        });
        const data = await res.json();
        if (res.ok && data.status === 'ok') {
          window.location.reload();
        } else {
          alert('Conversion failed: ' + (data.error || 'Unknown error'));
        }
      } catch (err) {
        alert('Error communicating with server: ' + err);
      } finally {
        btn.disabled = false;
        btn.textContent = '🚀 Convert Uploaded Turning Part to 3D STEP & Calculate ±0.50 mm Tolerance';
        banner.style.display = 'none';
      }
    }

    async function resetToReference() {
      const btn = document.getElementById('convertBtn');
      const banner = document.getElementById('statusBanner');
      btn.disabled = true;
      banner.style.display = 'block';
      banner.textContent = 'Restoring Reference Hex + 3D Helical Thread Probe Part...';
      try {
        const res = await fetch('/api/reset_reference', { method: 'POST' });
        if (res.ok) window.location.reload();
      } finally {
        btn.disabled = false;
        banner.style.display = 'none';
      }
    }

    // Three.js Interactive 3D Viewer loading /outputs/cnc_turning_part_mesh.stl
    const container = document.getElementById('viewer3d');
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(32, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.2;

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const keyLight = new THREE.DirectionalLight(0xfff8f0, 0.95);
    keyLight.position.set(60, -80, 90);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xdbeafe, 0.55);
    fillLight.position.set(-70, 60, 40);
    scene.add(fillLight);
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.45);
    rimLight.position.set(0, 80, -60);
    scene.add(rimLight);

    let partMesh = null;
    let wireMesh = null;
    let showWire = false;

    const loader = new THREE.STLLoader();
    loader.load('/outputs/cnc_turning_part_mesh.stl?t={{ ts }}', function (geometry) {
      geometry.computeVertexNormals();
      geometry.center();

      const material = new THREE.MeshStandardMaterial({
        color: 0xd6d3cd,
        metalness: 0.72,
        roughness: 0.26,
      });
      partMesh = new THREE.Mesh(geometry, material);
      partMesh.rotation.x = -Math.PI / 2;
      scene.add(partMesh);

      const wireMat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        wireframe: true,
        transparent: true,
        opacity: 0.28,
      });
      wireMesh = new THREE.Mesh(geometry, wireMat);
      wireMesh.rotation.x = -Math.PI / 2;
      wireMesh.visible = false;
      scene.add(wireMesh);

      resetCamera();
    });

    function resetCamera() {
      camera.position.set(52, 42, 68);
      controls.target.set(0, 0, 0);
      controls.update();
    }
    function setFrontView() {
      controls.autoRotate = false;
      camera.position.set(0, 0, 92);
      controls.target.set(0, 0, 0);
      controls.update();
    }
    function setTopView() {
      controls.autoRotate = false;
      camera.position.set(0, 92, 0.01);
      controls.target.set(0, 0, 0);
      controls.update();
    }
    function toggleWireframe() {
      showWire = !showWire;
      if (wireMesh) wireMesh.visible = showWire;
    }
    function toggleAutoRotate() {
      controls.autoRotate = !controls.autoRotate;
    }

    window.addEventListener('resize', () => {
      if (!container.clientWidth || !container.clientHeight) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    });

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    json_path = os.path.join(OUT_DIR, "cnc_turning_part_report.json")
    input_preview = os.path.join(OUT_DIR, "current_input_part.png")
    if not os.path.exists(json_path) or not os.path.exists(input_preview):
        convert_any_turning_image_to_step(
            get_active_image_path(),
            out_dir=OUT_DIR,
            repo_dir=BASE_DIR,
            tol_mm=0.50,
            scale_mm_per_px=0.1500,
        )
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return render_template_string(
        HTML_TEMPLATE,
        summary=data["summary"],
        features=data["features_0_5mm_tolerance_schedule"],
        ts=int(time.time() * 1000),
    )


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    return send_from_directory(OUT_DIR, filename)


@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(OUT_DIR, filename, as_attachment=True)


@app.route("/api/upload_convert", methods=["POST"])
def api_upload_convert():
    try:
        tol_mm = float(request.form.get("tol_mm", 0.50))
        scale_mm = float(request.form.get("scale_mm_per_px", 0.1500))
        orientation = request.form.get("orientation", "auto")

        file_obj = request.files.get("image")
        if file_obj and file_obj.filename:
            fname = secure_filename(file_obj.filename)
            saved_path = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{fname}")
            file_obj.save(saved_path)
            set_active_image_path(saved_path)
        else:
            saved_path = get_active_image_path()

        report = convert_any_turning_image_to_step(
            saved_path,
            out_dir=OUT_DIR,
            repo_dir=BASE_DIR,
            tol_mm=tol_mm,
            scale_mm_per_px=scale_mm,
            orientation=orientation,
            helical_threads=True,
        )
        return jsonify({
            "status": "ok",
            "part_name": report["summary"]["part_name"],
            "feature_count": len(report["features_0_5mm_tolerance_schedule"]),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 400


@app.route("/api/reset_reference", methods=["POST"])
def api_reset_reference():
    ref_path = os.path.join(BASE_DIR, "reference_turning_part.png")
    set_active_image_path(ref_path)
    convert_any_turning_image_to_step(
        ref_path,
        out_dir=OUT_DIR,
        repo_dir=BASE_DIR,
        tol_mm=0.50,
        scale_mm_per_px=0.1500,
        orientation="auto",
        helical_threads=True,
    )
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting CNC Turning Part 3D STEP Studio on http://127.0.0.1:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
