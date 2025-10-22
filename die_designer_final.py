# die_design_tool_v121f_plus.py — v1.2.1f foundation + Saved Designs table upgrades
# Preserves: Excel suggestions, Add Rows (+3), DXF generation + previews, Save workflow, metadata,
#            inline Die Manager charts and file export.
# Adds:     Multi-select filters (Pellet Size, Species), per-row checkbox column, Select All,
#           Delete Selected / Load Selected using checkbox, hide blank/zero rows, compact half-height table.

import os
import math
import uuid
import logging
import datetime
import traceback
from io import BytesIO
from math import cos, sin, radians

import json
import glob
import base64
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ezdxf
import plotly.express as px

# -------- Constants ------------------------
MAX_ROWS = 18
MAX_BOLT_HOLES = 20
ROWS_STEP = 3

# -------- Paths & Logging ------------------
BASE_DIR = "/volume1/docker/Diedesign"
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOGFILE = os.path.join(LOG_DIR, "die_design_tool.log")

def log_error(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

SAVE_DIR = os.path.join(BASE_DIR, "saved_designs")
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "save_button.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

print("Starting Die Design Tool (v1.2.1f+table-upgrades)…")
log_error("Startup: v1.2.1f+table-upgrades begin.")

# -------- Caches for inset zoom -----------
cached_doc = None
cached_inset_center = None

# -------- Excel Suggestions (optional) ----
try:
    excel_data = pd.read_excel("Die_design_data.xlsx", engine="openpyxl")
except Exception as e:
    excel_data = pd.DataFrame()
    log_error(f"Failed to load Excel data: {e}")

def get_suggestions(feed_type, pellet_size, bulk_density, fat, throughput):
    if excel_data.empty or 'Type' not in excel_data.columns:
        return {}
    df = excel_data.copy()
    df = df[df['Type'].astype(str).str.lower() == str(feed_type).lower()]
    if df.empty:
        return {}
    df['score'] = (
        (df['Size'] - pellet_size).abs() +
        (df['Target Bulk Density'] - bulk_density).abs() / 100 +
        (df['Fat'] - fat).abs()
    )
    best = df.loc[df['score'].idxmin()]

    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    return {
        "Die Opening": safe_float(best.get('Die Opening')),
        "Land /Channel Length": safe_float(best.get('Land / Channel Length')),
        "Open Area per tonne": safe_float(best.get('Open Area per Tonne')),
        "Expansion": safe_float(best.get('Expansion %')),
        "L/D Ratio": safe_float(best.get('L/D Ratio')),
        "Total Plate Open Area": (
            safe_float(best.get('Open Area per Tonne')) * throughput
            if safe_float(best.get('Open Area per Tonne')) else None
        ),
    }

# -------- Core calculations + image/table --------------
def calculate_die_design(pellet_size_mm, dry_meal_throughput_tph, bulk_density_g_per_l, final_fat_percent,
                         feed_type, channel_length_mm, cone_length_mm, opening_dia,
                         cone_width_mm, number_of_holes):
    die_total_length_mm = channel_length_mm + cone_length_mm
    cone_angle_deg = math.degrees(math.atan(cone_width_mm / (2 * cone_length_mm))) if cone_length_mm > 0 else 0
    open_area_one_hole_mm2 = math.pi * (opening_dia / 2) ** 2
    total_plate_open_area_mm2 = open_area_one_hole_mm2 * number_of_holes
    open_area_per_tonne_mm2_per_tph = total_plate_open_area_mm2 / max(dry_meal_throughput_tph, 1e-6)
    expansion_percent = (1 - (opening_dia / max(pellet_size_mm, 1e-6))) * 100

    # Cross-section figure
    fig, ax = plt.subplots(figsize=(4, 4.2))
    ax.set_facecolor('#ffffff')
    ax.set_xlabel("Width (mm)", fontsize=9)
    ax.set_ylabel("Length (mm)", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Steel background
    steel_x = [-cone_width_mm * 1.2, cone_width_mm * 1.2, cone_width_mm * 1.2, -cone_width_mm * 1.2, -cone_width_mm * 1.2]
    steel_y = [0, 0, die_total_length_mm, die_total_length_mm, 0]
    ax.fill(steel_x, steel_y, color='#cccccc', alpha=1.0)

    # Cone
    if cone_length_mm > 0:
        cone_top = cone_width_mm / 2
        cone_bottom = opening_dia / 2
        cone_x = [-cone_top, -cone_bottom, cone_bottom, cone_top, -cone_top]
        cone_y = [0, cone_length_mm, cone_length_mm, 0, 0]
        ax.fill(cone_x, cone_y, color='#3366cc', alpha=1.0)

    # Channel
    rect_x = [-opening_dia / 2, opening_dia / 2, opening_dia / 2, -opening_dia / 2, -opening_dia / 2]
    rect_y = [cone_length_mm, cone_length_mm, die_total_length_mm, die_total_length_mm, cone_length_mm]
    ax.plot(rect_x, rect_y, color='#3366cc')
    ax.fill(rect_x, rect_y, color='#3366cc', alpha=1.0)

    x_margin, y_margin = 2, 2
    ax.set_xlim(-cone_width_mm / 2 - x_margin, cone_width_mm / 2 + x_margin)
    ax.set_ylim(die_total_length_mm + y_margin, -y_margin)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    image_html = f"<img src='data:image/png;base64,{image_base64}' style='width: 400px; height: 420px; margin-right: 20px;'/>"

    # Suggestions
    suggestions = get_suggestions(feed_type, pellet_size_mm, bulk_density_g_per_l, final_fat_percent, dry_meal_throughput_tph)

    parameters = [
        "Die Opening", "Land /Channel Length", "Total Plate Open Area",
        "Open Area per tonne", "Expansion", "L/D Ratio", "Die Total Length", "Cone Angle"
    ]
    calculated = [
        round(opening_dia, 2),
        round(channel_length_mm, 2),
        int(round(total_plate_open_area_mm2, 0)),
        int(round(open_area_per_tonne_mm2_per_tph, 0)),
        round(expansion_percent, 1),
        round(channel_length_mm / opening_dia, 2) if opening_dia > 0 else 0,
        round(die_total_length_mm, 2),
        round(cone_angle_deg, 1),
    ]
    suggested = [
        round(suggestions.get("Die Opening", 0), 2) if suggestions.get("Die Opening") is not None else "—",
        round(suggestions.get("Land /Channel Length", 0), 2) if suggestions.get("Land /Channel Length") is not None else "—",
        round(suggestions.get("Total Plate Open Area", 0), 0) if suggestions.get("Total Plate Open Area") is not None else "—",
        round(suggestions.get("Open Area per tonne", 0), 0) if suggestions.get("Open Area per tonne") is not None else "—",
        round(suggestions.get("Expansion", 0), 1) if suggestions.get("Expansion") is not None else "—",
        round(suggestions.get("L/D Ratio", 0), 2) if suggestions.get("L/D Ratio") is not None else "—",
        "—",
        "—",
    ]

    # Table HTML with simple color coding
    table_html_wrapped = ["<div class='panel-table' style='padding:10px; text-align:center;'>"
                          "<table style='width:100%; max-width:700px; margin:auto; border-collapse:collapse;'>"
                          "<thead><tr>"
                          "<th style='border-bottom:1px solid #ccc;'>Parameter</th>"
                          "<th style='border-bottom:1px solid #ccc;'>Calculated</th>"
                          "<th style='border-bottom:1px solid #ccc;'>Suggested</th>"
                          "</tr></thead><tbody>"]
    for p, c, s in zip(parameters, calculated, suggested):
        try:
            s_val = float(s); c_val = float(c)
            diff = abs(c_val - s_val)
            avg = (abs(c_val) + abs(s_val)) / 2 if (abs(c_val) + abs(s_val)) else 1
            percent_diff = diff / avg
            if percent_diff < 0.05: color = "#dff5e1"  # pale green
            elif percent_diff < 0.15: color = "#f3f7d9" # pale yellow-green
            else: color = "#fde2e1"                    # pale red
        except (ValueError, TypeError):
            color = "#ffffff"
        table_html_wrapped.append(
            f"<tr><td style='padding:6px 10px; text-align:left;'>{p}</td>"
            f"<td style='padding:6px 10px; text-align:center; background-color:{color};'>{c}</td>"
            f"<td style='padding:6px 10px; text-align:center; background-color:{color};'>{s}</td></tr>"
        )
    table_html_wrapped.append("</tbody></table></div>")
    image_html_wrapped = f"<div class='panel-image' style='text-align:center; padding:10px;'>{image_html}</div>"

    calc_values = {
        "die_total_length": die_total_length_mm,
        "cone_angle": cone_angle_deg,
        "open_area_one_hole": open_area_one_hole_mm2,
        "total_plate_open_area": total_plate_open_area_mm2,
        "open_area_per_tonne": open_area_per_tonne_mm2_per_tph,
        "expansion_percent": expansion_percent
    }

    return image_html_wrapped, "".join(table_html_wrapped), calc_values

# -------- DXF helpers -------------------------------------------
def apply_chamfer(msp, diameter, chamfer, direction):
    if chamfer > 0:
        chamfered_diameter = diameter - 2 * chamfer if direction == "inward" else diameter + 2 * chamfer
        msp.add_circle((0, 0), chamfered_diameter / 2)

def add_detailed_inset(msp, opening_dia, cone_width, rows_data, inset_scale=2.0):
    scale = inset_scale
    origin_x, origin_y = 300, 0
    valid_rows = [row for row in rows_data if row.get('num_holes', 0) > 0 and row.get('pcd', 0) > 0]
    if not valid_rows:
        return None
    inset_center = None
    middle_row_index = len(valid_rows) // 2
    for idx, row in enumerate(valid_rows):
        pcd = row['pcd']; n = row['num_holes']
        radius = (pcd / 2) * scale
        spacing = 360 / n if n >= 3 else 120
        offset = (spacing / 2) if idx % 2 == 1 else 0
        start_angle = -spacing + offset
        for i in range(3):
            ang = start_angle + i * spacing
            x = origin_x + radius * cos(radians(ang))
            y = origin_y + radius * sin(radians(ang))
            msp.add_circle((x, y), opening_dia / 2 * scale)
            if cone_width > opening_dia:
                msp.add_circle((x, y), cone_width / 2 * scale)
            if idx == middle_row_index and i == 1:
                inset_center = (x, y)
    return inset_center

def create_dxf_doc(
    plate_diameter, opening_dia, cone_width, outer_bolt_count,
    outer_bolt_pcd, outer_bolt_diameter, inner_bolt_count, inner_bolt_pcd,
    inner_bolt_diameter, outer_opening_pcd, inner_opening_pcd, die_center_hole_diameter,
    chamfer_plate, chamfer_outer_opening, chamfer_inner_opening, chamfer_center_hole,
    rows_data, pellet_size, inset_scale
):
    try:
        doc = ezdxf.new()
        msp = doc.modelspace()

        # Plate outline
        msp.add_circle((0, 0), plate_diameter / 2)
        apply_chamfer(msp, plate_diameter, chamfer_plate, "inward")

        # Center hole
        if die_center_hole_diameter > 0:
            msp.add_circle((0, 0), die_center_hole_diameter / 2)
            apply_chamfer(msp, die_center_hole_diameter, chamfer_center_hole, "outward")

        # Outer/Inner openings
        if outer_opening_pcd > 0:
            msp.add_circle((0, 0), outer_opening_pcd / 2)
            apply_chamfer(msp, outer_opening_pcd, chamfer_outer_opening, "outward")
        if inner_opening_pcd > 0:
            msp.add_circle((0, 0), inner_opening_pcd / 2)
            apply_chamfer(msp, inner_opening_pcd, chamfer_inner_opening, "inward")

        # Bolts
        def bolt_pattern(count, pcd, dia):
            if count and pcd and dia:
                for i in range(int(count)):
                    ang = 2 * math.pi * i / int(count)
                    x = (pcd / 2) * math.cos(ang)
                    y = (pcd / 2) * math.sin(ang)
                    msp.add_circle((x, y), dia / 2)
        bolt_pattern(outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter)
        bolt_pattern(inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter)

        # Rows of holes
        for idx, row in enumerate(rows_data):
            n = int(row.get("num_holes", 0) or 0)
            pcd = float(row.get("pcd", 0) or 0)
            if n <= 0 or pcd <= 0: continue
            radius = pcd / 2
            angle_offset = (360 / n) / 2 if idx % 2 == 1 else 0
            for i in range(n):
                ang = 360 / n * i + angle_offset
                x = radius * math.cos(math.radians(ang))
                y = radius * math.sin(math.radians(ang))
                msp.add_circle((x, y), opening_dia / 2)
                if cone_width > opening_dia:
                    msp.add_circle((x, y), cone_width / 2)

        inset_center = add_detailed_inset(msp, opening_dia, cone_width, rows_data, inset_scale=inset_scale)
        return doc, inset_center
    except Exception:
        log_error(traceback.format_exc())
        return ezdxf.new(), None

def create_preview_image(doc, filename, mode="plate", plate_diameter=250, center=None, zoom=1.0):
    try:
        fig, ax = plt.subplots()
        msp = doc.modelspace()
        for entity in msp:
            if entity.dxftype() == 'CIRCLE':
                c = entity.dxf.center; r = entity.dxf.radius
                if mode == "plate" and c[0] < 250:
                    ax.add_patch(plt.Circle(c, r, fill=False, edgecolor='black'))
                elif mode == "inset":
                    ax.add_patch(plt.Circle(c, r, fill=False, edgecolor='blue'))
        ax.set_aspect('equal')
        if mode == "plate":
            margin = 0.1 * plate_diameter
            ax.set_xlim(-plate_diameter / 2 - margin, plate_diameter / 2 + margin)
            ax.set_ylim(-plate_diameter / 2 - margin, plate_diameter / 2 + margin)
        else:
            if center is not None:
                cx, cy = center
                inset_radius = 150 / (zoom ** 1.5)
                ax.set_xlim(cx - inset_radius, cx + inset_radius)
                ax.set_ylim(cy - inset_radius, cy + inset_radius)
            else:
                ax.autoscale()
        plt.axis('off')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return Image.open(buf)
    except Exception:
        log_error(traceback.format_exc())
        raise

def generate_die_plate_dxf(
    pellet_size, plate_diameter, opening_dia, cone_width,
    chamfer_plate, chamfer_outer_opening, chamfer_inner_opening, chamfer_center_hole,
    outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
    inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
    outer_opening_pcd, inner_opening_pcd, die_center_hole_diameter,
    inset_zoom, *row_inputs
):
    try:
        rows = []
        for i in range(0, len(row_inputs), 2):
            rows.append({"num_holes": row_inputs[i], "pcd": row_inputs[i+1]})
        filename = f"die_plate_{uuid.uuid4().hex}"
        doc, inset_center = create_dxf_doc(
            plate_diameter, opening_dia, cone_width,
            outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
            inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
            outer_opening_pcd, inner_opening_pcd, die_center_hole_diameter,
            chamfer_plate, chamfer_outer_opening, chamfer_inner_opening, chamfer_center_hole,
            rows, pellet_size, inset_zoom
        )
        dxf_path = f"/tmp/{filename}.dxf"
        doc.saveas(dxf_path)
        plate_img = create_preview_image(doc, filename, mode="plate", plate_diameter=plate_diameter)
        inset_img = create_preview_image(doc, filename, mode="inset", center=inset_center, zoom=inset_zoom)

        # Cache for live zoom
        global cached_doc, cached_inset_center
        cached_doc = doc
        cached_inset_center = inset_center

        return plate_img, inset_img, dxf_path, gr.update(value=dxf_path, visible=True)
    except Exception:
        log_error(traceback.format_exc())
        raise

def update_inset_zoom(inset_zoom):
    if cached_doc is None or cached_inset_center is None:
        log_error("Zoom update failed: No cached DXF or inset center.")
        return None
    return create_preview_image(cached_doc, filename="cached", mode="inset", center=cached_inset_center, zoom=inset_zoom)

# -------- Persistence -------------------------------------------
def list_saved_designs(save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    files = sorted(glob.glob(f"{save_dir}/*.json"), reverse=True)
    return [os.path.basename(f) for f in files]

def load_die_design(filename, save_dir=SAVE_DIR):
    path = os.path.join(save_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_die_design(data, save_dir=SAVE_DIR):
    try:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"die_design_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log_error(f"Design saved to {filename}")
        return filename
    except Exception as e:
        log_error(f"Error saving design: {e}")
        raise

# --------- UI --------------------------------------
def build_ui():
    with gr.Blocks(
        css="""
        .card-panel { background-color:#FCFCFC; border:2px solid #5aadab; border-radius:10px;
                      padding:16px; margin:12px 0; width:100%; display:flex; flex-direction:column; gap:10px;
                      height: 520px; }
        .button-panel { background-color:#FCFCFC; border:2px solid #5aadab; border-radius:10px;
                        padding:16px; margin:12px 0; width:100%; display:flex; flex-direction:column; gap:10px;
                        height: 520px; }
        .icon-save-button::before { content: "ߒ; }
        .panel-table table { display: table; width: 100%; border-collapse: collapse; }
        .panel-spacer { height: 12px; }
        @media (min-width: 768px) {
            .responsive-container { flex-direction: row !important; align-items: stretch !important; }
        }
        .soft-bg { background: #f4faf6; border: 1px solid #d7eee2; border-radius: 10px; padding: 10px; }
        /* Compact dataframe rows (half-height) */
        .gr-dataframe table thead th { text-align:center !important; padding:4px 6px !important; }
        .gr-dataframe table tbody td { text-align:center !important; padding:2px 4px !important; line-height: 1.0 !important; }
        .gr-dataframe table tbody tr { height: 16px !important; }
        .gr-dataframe { font-size: 12px; }
        """,
        title="Die Design Calculator"
    ) as demo:
        gr.Markdown("## Die Design Calculator")
        gr.Markdown("*Version: 1.2.1f + Saved Designs table upgrades*", elem_id="version-label")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Product Specifications")
                pellet_size = gr.Slider(0.3, 20, value=9.0, step=0.1, label="Pellet Size (mm)")
                throughput = gr.Slider(0.5, 30, value=9.0, step=0.1, label="Throughput drymeal (t/h)")
                density = gr.Slider(320, 750, value=500, step=0.1, label="Bulk Density Target (g/l)")
                fat = gr.Slider(8, 43, value=24, step=0.1, label="Final Fat Target (%)")
                with gr.Row():
                    floating = gr.Checkbox(label="Floating")
                    shrimp = gr.Checkbox(label="Shrimp_Prawn")
                    salmonid = gr.Checkbox(label="Salmonid")
                    other_sinking = gr.Checkbox(label="Other Sinking")

            with gr.Column():
                gr.Markdown("### Die Design Parameters")
                opening_dia = gr.Slider(0.3, 22, value=6.3, step=0.01, label="Die Diameter (mm)")
                channel_length = gr.Slider(0, 30, value=14.0, step=0.1, label="Channel/Land Length (mm)")
                cone_length = gr.Slider(0, 30, value=4.0, step=0.1, label="Cone Length (mm)")
                cone_width = gr.Slider(0, 22, value=13.0, step=0.1, label="Cone Width (mm)")
                holes = gr.Slider(5, 7000, step=1, value=500, label="Number of Holes")

        with gr.Row(elem_classes=["responsive-container"]):
            with gr.Column(elem_classes=["card-panel"]):
                gr.Markdown("<h4>Hole Cross Section</h4>")
                output_image = gr.HTML()

            with gr.Column(elem_classes=["card-panel"]):
                gr.Markdown("<h4>Your Design vs Suggestions</h4>")
                gr.HTML("<div class='panel-spacer'></div>")
                output_table = gr.HTML()

            with gr.Column(elem_classes=["button-panel"]):
                gr.Markdown("<h4>Design Actions</h4>")

                with gr.Accordion("Optional metadata (recommended)", open=False):
                    extruder_type   = gr.Textbox(label="Extruder type", placeholder="e.g., Wenger X-165", lines=1)
                    production_line = gr.Textbox(label="Production line", placeholder="e.g., Line 3", lines=1)
                    plant_name      = gr.Textbox(label="Plant name", placeholder="e.g., Ho Chi Minh", lines=1)
                    product_type    = gr.Textbox(label="Product type", placeholder="e.g., Floating grower", lines=1)
                    die_performance = gr.Textbox(label="Die performance", placeholder="e.g., 7.8 tph, 1200 hrs life", lines=1)
                    comments        = gr.Textbox(label="Comments", placeholder="Notes…", lines=2)
                quick_save_button = gr.Button("Quick Save (no metadata)", elem_classes=["icon-save-button"], interactive=True)
                save_with_meta_button = gr.Button("Save + metadata", interactive=True)

                gr.HTML("<div class='panel-spacer'></div>")
                with gr.Row():
                    calculate_btn = gr.Button("Calculate")
                    save_button = gr.Button("Save this Design", elem_classes=["icon-save-button"], interactive=True)
                with gr.Row():
                    reset_button = gr.Button("Reset Design")
                quick_dxf = gr.File(label="Latest DXF", interactive=False, visible=False)
                save_status = gr.HTML(value="<div></div>", visible=True)

        with gr.Tab("all measurements in mm"):
            with gr.Group(elem_classes=["soft-bg"]):
                gr.Markdown("#### Plate inside & outside")
                with gr.Row():
                    with gr.Column():
                        plate_diameter = gr.Slider(label="Plate Diameter", minimum=100, maximum=600, step=0.1, value=250)
                        chamfer_plate = gr.Slider(label="Chamfer 45°, if any", minimum=0, maximum=10, step=0.1, value=2)
                    with gr.Column():
                        die_center_hole_diameter = gr.Slider(label="Die Center Hole Diameter", minimum=0, maximum=200, step=0.1, value=48.5)
                        chamfer_center_hole = gr.Slider(label="Chamfer 45°, if any", minimum=0, maximum=10, step=0.1, value=0)

            with gr.Group(elem_classes=["soft-bg"]):
                gr.Markdown("#### Cutting Track")
                with gr.Row():
                    with gr.Column():
                        outer_opening_pcd = gr.Slider(label="Outer cutting track - PCD", minimum=0, maximum=350, step=0.1, value=189)
                        chamfer_outer_opening = gr.Slider(label="Chamfer 45°, if any", minimum=0, maximum=10, step=0.1, value=0)
                    with gr.Column():
                        inner_opening_pcd = gr.Slider(label="Inner cuttting track - PCD", minimum=90, maximum=250, step=0.1, value=128)
                        chamfer_inner_opening = gr.Slider(label="Chamfer 45°, if any", minimum=0, maximum=10, step=0.1, value=0)

            with gr.Group(elem_classes=["soft-bg"]):
                gr.Markdown("#### Bolt Holes")
                with gr.Row():
                    with gr.Column():
                        outer_bolt_count = gr.Slider(label="Outer Bolts - Number of holes", minimum=0, maximum=MAX_BOLT_HOLES, value=12, step=1)
                        outer_bolt_pcd = gr.Slider(label="Outer Bolts - PCD", minimum=0, maximum=400, step=0.1, value=230)
                        outer_bolt_diameter = gr.Slider(label="Outer Bolts - Hole Diameter", minimum=0, maximum=20, step=0.1, value=10.5)
                    with gr.Column():
                        inner_bolt_count = gr.Slider(label="Inner Bolts - Number of holes", minimum=0, maximum=MAX_BOLT_HOLES, value=4, step=1)
                        inner_bolt_pcd = gr.Slider(label="Inner Bolts - PCD", minimum=0, maximum=200, step=0.1, value=75)
                        inner_bolt_diameter = gr.Slider(label="Inner Bolts - Hole Diameter", minimum=0, maximum=20, step=0.1, value=8.2)

            # Holes and Rows (3-column compact + Add Rows)
            gr.Markdown("#### Holes and Rows")
            visible_rows = gr.State(ROWS_STEP)
            row_components = []
            with gr.Accordion("Rows (add in groups of 3)", open=True):
                current_row = None
                for i in range(MAX_ROWS):
                    if i % 3 == 0:
                        current_row = gr.Row()
                    with current_row:
                        with gr.Column():
                            num_holes = gr.Slider(
                                label=f"Row {i+1} - Holes", minimum=0, maximum=150, step=1, value=0,
                                visible=True if i < ROWS_STEP else False
                            )
                            pcd = gr.Slider(
                                label=f"Row {i+1} - PCD (mm)", minimum=90, maximum=350, step=0.1, value=0,
                                visible=True if i < ROWS_STEP else False
                            )
                            row_components.append((num_holes, pcd))
                add_rows_btn = gr.Button("Add Rows")

            with gr.Row():
                with gr.Column():
                    plate_preview = gr.Image(label="Die Plate DXF Preview", type="pil")
                    generate_btn = gr.Button("Generate DXF/Autocad file")
                with gr.Column():
                    inset_preview = gr.Image(label="Inset View Preview", type="pil")
                    inset_zoom = gr.Slider(
                        label="Zoom Inset View", minimum=1.0, maximum=10.0, step=0.1, value=2.5,
                        info="Zooms the inset view without recalculating geometry"
                    )

        # ---- States ----
        calc_ok_state = gr.State(False)
        calc_values_state = gr.State({})
        dxf_path_state = gr.State(None)

        # ---- Helpers ----
        def get_selected_feed_type(floating, shrimp, salmonid, other_sinking):
            if floating: return "Floating"
            if shrimp: return "Shrimp_Prawn"
            if salmonid: return "Salmonid"
            if other_sinking: return "Other sinking"
            return "Salmonid"

        def on_calculate(ps, th, bd, ft, fl, sh, sa, osk, cl, col, od, cw, nh):
            image_html, table_html, calc_values = calculate_die_design(
                ps, th, bd, ft, get_selected_feed_type(fl, sh, sa, osk), cl, col, od, cw, nh
            )
            return image_html, table_html, True, calc_values

        def clear_save_message():
            return gr.update(value="")

        def on_any_input_change():
            return False  # forces recalc before save

        # ------- Save helpers -------
        def build_payload_inputs(ps, th, bd, ft, fl, sh, sa, osk,
                                 cl, col, od, cw, nh,
                                 plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                                 outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                                 outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                                 inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                                 *row_inputs):
            hole_rows = []
            pairs = min(len(row_inputs) // 2, MAX_ROWS)
            for i in range(pairs):
                hole_rows.append({"row": i + 1, "num_holes": row_inputs[2*i], "pcd": row_inputs[2*i + 1]})
            payload_inputs = {
                "pellet_size": ps, "throughput": th, "bulk_density": bd, "fat": ft,
                "feed_type": get_selected_feed_type(fl, sh, sa, osk),
                "channel_length": cl, "cone_length": col, "opening_dia": od, "cone_width": cw, "number_of_holes": nh,
                "plate_diameter": plate_diameter, "chamfer_plate": chamfer_plate,
                "die_center_hole_diameter": die_center_hole_diameter, "chamfer_center_hole": chamfer_center_hole,
                "outer_opening_pcd": outer_opening_pcd, "chamfer_outer_opening": chamfer_outer_opening,
                "inner_opening_pcd": inner_opening_pcd, "chamfer_inner_opening": chamfer_inner_opening,
                "outer_bolt_count": outer_bolt_count, "outer_bolt_pcd": outer_bolt_pcd, "outer_bolt_diameter": outer_bolt_diameter,
                "inner_bolt_count": inner_bolt_count, "inner_bolt_pcd": inner_bolt_pcd, "inner_bolt_diameter": inner_bolt_diameter,
                "hole_rows": hole_rows
            }
            return payload_inputs

        def do_save(calc_ok, calc_values, payload_inputs, metadata=None):
            if not calc_ok:
                return gr.update(value="<div style='color:red; font-weight:bold;'>Please click 'Calculate' before saving.</div>", visible=True)
            payload = {
                "timestamp": datetime.datetime.now().isoformat(),
                "inputs": payload_inputs,
                "calculated": calc_values,
                "metadata": (metadata or {})
            }
            try:
                filename = save_die_design(payload, SAVE_DIR)
                return gr.update(value=f"<div style='color:green; font-weight:bold;'>Saved! ({os.path.basename(filename)})</div>", visible=True)
            except Exception as e:
                return gr.update(value=f"<div style='color:red; font-weight:bold;'>Save failed: {e}</div>", visible=True)

        def on_save_with_meta(calc_ok, calc_values,
                              ps, th, bd, ft, fl, sh, sa, osk,
                              cl, col, od, cw, nh,
                              plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                              outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                              outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                              inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                              *row_inputs_and_meta):
            *row_inputs, extruder_type, production_line, plant_name, product_type, die_performance, comments = row_inputs_and_meta
            payload_inputs = build_payload_inputs(
                ps, th, bd, ft, fl, sh, sa, osk, cl, col, od, cw, nh,
                plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter, *row_inputs
            )
            metadata = {
                "extruder_type": extruder_type or "",
                "production_line": production_line or "",
                "plant_name": plant_name or "",
                "product_type": product_type or "",
                "die_performance": die_performance or "",
                "comments": comments or ""
            }
            return do_save(calc_ok, calc_values, payload_inputs, metadata)

        def on_quick_save(calc_ok, calc_values,
                          ps, th, bd, ft, fl, sh, sa, osk,
                          cl, col, od, cw, nh,
                          plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                          outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                          outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                          inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                          *row_inputs):
            payload_inputs = build_payload_inputs(
                ps, th, bd, ft, fl, sh, sa, osk, cl, col, od, cw, nh,
                plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter, *row_inputs
            )
            return do_save(calc_ok, calc_values, payload_inputs, metadata=None)

        def on_save(calc_ok, calc_values,
                    ps, th, bd, ft, fl, sh, sa, osk,
                    cl, col, od, cw, nh,
                    plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                    outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                    outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                    inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                    *row_inputs):
            payload_inputs = build_payload_inputs(
                ps, th, bd, ft, fl, sh, sa, osk, cl, col, od, cw, nh,
                plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter, *row_inputs
            )
            return do_save(calc_ok, calc_values, payload_inputs)

        # ---- Bindings (Designer) ----
        calculate_btn.click(
            fn=on_calculate,
            inputs=[pellet_size, throughput, density, fat, floating, shrimp, salmonid, other_sinking,
                    channel_length, cone_length, opening_dia, cone_width, holes],
            outputs=[output_image, output_table, calc_ok_state, calc_values_state]
        )
        calculate_btn.click(fn=clear_save_message, outputs=[save_status])

        save_button.click(
            fn=on_save,
            inputs=[calc_ok_state, calc_values_state,
                    pellet_size, throughput, density, fat, floating, shrimp, salmonid, other_sinking,
                    channel_length, cone_length, opening_dia, cone_width, holes,
                    plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                    outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                    outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                    inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                    ] + [comp for pair in row_components for comp in pair],
            outputs=[save_status]
        )

        for comp in [
            pellet_size, throughput, density, fat, floating, shrimp, salmonid, other_sinking,
            channel_length, cone_length, opening_dia, cone_width, holes,
            plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
            outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
            outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
            inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
        ] + [comp for pair in row_components for comp in pair]:
            comp.change(fn=on_any_input_change, inputs=[], outputs=[calc_ok_state], show_progress=False)

        def on_add_rows(n_visible):
            new_visible = min(n_visible + ROWS_STEP, MAX_ROWS)
            updates = []
            for idx, (nh, p) in enumerate(row_components):
                show = idx < new_visible
                updates.extend([gr.update(visible=show), gr.update(visible=show)])
            return [new_visible] + updates

        add_rows_outputs = [visible_rows] + [comp for pair in row_components for comp in pair]
        add_rows_btn.click(fn=on_add_rows, inputs=[visible_rows], outputs=add_rows_outputs)

        generate_btn.click(
            fn=generate_die_plate_dxf,
            inputs=[
                pellet_size, plate_diameter, opening_dia, cone_width,
                chamfer_plate, chamfer_outer_opening, chamfer_inner_opening, chamfer_center_hole,
                outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                outer_opening_pcd, inner_opening_pcd, die_center_hole_diameter,
                inset_zoom, *[comp for pair in row_components for comp in pair]
            ],
            outputs=[plate_preview, inset_preview, dxf_path_state, quick_dxf]
        )
        inset_zoom.change(fn=update_inset_zoom, inputs=[inset_zoom], outputs=[inset_preview])

        def on_reset_click():
            updates = []
            updates += [
                gr.update(value=9.0), gr.update(value=9.0), gr.update(value=500), gr.update(value=24),
                gr.update(value=False), gr.update(value=False), gr.update(value=False), gr.update(value=False),
            ]
            updates += [
                gr.update(value=6.3), gr.update(value=14.0), gr.update(value=4.0), gr.update(value=13.0), gr.update(value=500),
            ]
            updates += [
                gr.update(value=250), gr.update(value=2),
                gr.update(value=48.5), gr.update(value=0),
                gr.update(value=189), gr.update(value=0),
                gr.update(value=128), gr.update(value=0),
                gr.update(value=12), gr.update(value=230), gr.update(value=10.5),
                gr.update(value=4), gr.update(value=75), gr.update(value=8.2),
            ]
            for idx, (nh, p) in enumerate(row_components):
                show = idx < ROWS_STEP
                updates.append(gr.update(value=0, visible=show))
                updates.append(gr.update(value=0, visible=show))
            updates += [
                gr.update(value=None, visible=False),
                gr.update(value="<div></div>"),
                gr.update(value=None),
                gr.update(value=None),
                None,
                False,
                ROWS_STEP
            ]
            return updates

        reset_outputs = [
            pellet_size, throughput, density, fat, floating, shrimp, salmonid, other_sinking,
            opening_dia, channel_length, cone_length, cone_width, holes,
            plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
            outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
            outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
            inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
        ] + [comp for pair in row_components for comp in pair] + [
            quick_dxf, save_status, plate_preview, inset_preview, dxf_path_state, calc_ok_state, visible_rows
        ]
        reset_button.click(fn=on_reset_click, inputs=[], outputs=reset_outputs)

        # ============= Inline Die Manager (UPGRADED TABLE) =============
        def _r1(x):
            try: return round(float(x), 1)
            except: return None
        def _r2(x):
            try: return round(float(x), 2)
            except: return None
        def _r0(x):
            try: return int(round(float(x)))
            except: return None
        def _num(x):
            try: return float(x)
            except: return None

        def derive_metrics(inputs, calc):
            od = _num(inputs.get("opening_dia"))
            cl = _num(inputs.get("channel_length"))
            cw = _num(inputs.get("cone_width"))
            nh = _num(inputs.get("number_of_holes"))
            th = _num(inputs.get("throughput"))
            ps = _num(inputs.get("pellet_size"))
            bd = _num(inputs.get("bulk_density"))
            o1 = _num(calc.get("open_area_one_hole"))
            tpo = _num(calc.get("total_plate_open_area"))
            oapt = _num(calc.get("open_area_per_tonne"))
            expc = _num(calc.get("expansion_percent"))
            dl = _num(calc.get("die_total_length"))
            ca = _num(calc.get("cone_angle"))
            if o1 is None and od is not None:
                o1 = 3.141592653589793 * (od/2.0)**2
            if tpo is None and (o1 is not None and nh is not None):
                tpo = o1 * nh
            if oapt is None and (tpo is not None and th not in (None, 0)):
                oapt = tpo / max(th, 1e-6)
            if expc is None and (od is not None and ps not in (None, 0)):
                expc = (1 - (od / max(ps, 1e-6))) * 100.0
            if dl is None and (cl is not None and _num(inputs.get("cone_length")) is not None):
                dl = cl + _num(inputs.get("cone_length"))
            if ca is None and (cw is not None and _num(inputs.get("cone_length"))):
                try:
                    import math
                    L = _num(inputs.get("cone_length")) or 0
                    ca = math.degrees(math.atan(cw / (2*L))) if L>0 else 0.0
                except:
                    ca = None
            ld = None
            if od not in (None, 0) and cl is not None:
                ld = cl/od
            rows = inputs.get("hole_rows", []) or []
            rows_count = sum(1 for r in rows if _num(r.get("num_holes")) not in (None, 0) and _num(r.get("pcd")) not in (None, 0))
            return {
                "bulk_density": bd,
                "open_area_one_hole": o1,
                "total_plate_open_area": tpo,
                "open_area_per_tonne": oapt,
                "expansion_percent": expc,
                "die_total_length": dl,
                "cone_angle": ca,
                "ld_ratio": ld,
                "rows_count": rows_count,
            }

        def collect_manager_rows():
            rows = []
            for fname in list_saved_designs():
                try:
                    data = load_die_design(fname)
                    inputs = data.get("inputs", {}) or {}
                    calc = data.get("calculated", {}) or {}
                    meta = data.get("metadata", {}) or {}
                    d = derive_metrics(inputs, calc)

                    def fmt_time(ts):
                        try:
                            import datetime as _dt
                            dt = _dt.datetime.fromisoformat(ts.replace("Z",""))
                            return dt.strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            return ts

                    species_label = meta.get("product_type") or inputs.get("feed_type","")
                    rows.append({
                        "Select": False,
                        "File": fname,
                        "Saved": fmt_time(data.get("timestamp","")),
                        "Species": species_label,
                        "Pellet Size (mm)": _r1(inputs.get("pellet_size")),
                        "# Rows": _r0(d["rows_count"]),
                        "# Holes": _r0(inputs.get("number_of_holes")),
                        "Die Ø (mm)": _r2(inputs.get("opening_dia")),
                        "Land L (mm)": _r1(inputs.get("channel_length")),
                        "Cone L (mm)": _r1(inputs.get("cone_length")),
                        "Cone W (mm)": _r1(inputs.get("cone_width")),
                        "OA 1 hole (mm²)": _r2(d["open_area_one_hole"]),
                        "OA total (mm²)": _r0(d["total_plate_open_area"]),
                        "OA/t (mm²/t/h)": _r0(d["open_area_per_tonne"]),
                        "L/D": _r2(d["ld_ratio"]),
                        "Die L (mm)": _r1(d["die_total_length"]),
                        "Cone °": _r1(d["cone_angle"]),
                    })
                except Exception as e:
                    log_error(f"collect_manager_rows() failed for {fname}: {e}")
            return pd.DataFrame(rows)

        def clean_table_view(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame(columns=["Select","File","Saved","Species","Pellet Size (mm)","# Rows","# Holes","Die Ø (mm)","Land L (mm)","Cone L (mm)","Cone W (mm)","OA 1 hole (mm²)","OA total (mm²)","OA/t (mm²/t/h)","L/D","Die L (mm)","Cone °"])
            out = df.copy()
            out["Saved"] = out["Saved"].fillna("").astype(str)
            out["# Rows"] = pd.to_numeric(out["# Rows"], errors="coerce").fillna(0).astype(int)
            out = out[(out["Saved"].str.len() > 0) & (out["# Rows"] > 0)]
            if "Select" not in out.columns:
                out.insert(0, "Select", False)
            else:
                out["Select"] = out["Select"].astype(bool)
            return out

        def uniq(series):
            vals = series.dropna().astype(str).unique().tolist()
            try:
                vals = sorted(vals, key=lambda x: float(x))
            except Exception:
                vals = sorted(vals)
            return vals

        def apply_filters(df, pellet_choices, species_choices):
            out = df.copy()
            if pellet_choices:
                out = out[out["Pellet Size (mm)"].astype(str).isin([str(x) for x in pellet_choices])]
            if species_choices:
                out = out[out["Species"].astype(str).isin([str(x) for x in species_choices])]
            return out

        def refresh_table(pellet_choices, species_choices, select_all):
            base = clean_table_view(collect_manager_rows())
            view = apply_filters(base, pellet_choices or [], species_choices or [])
            if not view.empty:
                view["Select"] = bool(select_all)
            return view

        def toggle_select(df, select_all):
            import pandas as pd
            if isinstance(df, list):
                df = pd.DataFrame(df[1:], columns=df[0])
            if "Select" in df.columns:
                df["Select"] = bool(select_all)
            return df

        def delete_selected(df, pellet_choices, species_choices):
            import pandas as pd
            if isinstance(df, list):
                df = pd.DataFrame(df[1:], columns=df[0])
            if df is None or df.empty or "Select" not in df.columns or "File" not in df.columns:
                return df, "No rows to delete."
            sel = df[df["Select"] == True]
            if sel.empty:
                return df, "No rows selected."
            count = 0
            for _, r in sel.iterrows():
                fname = r.get("File")
                if fname:
                    try:
                        os.remove(os.path.join(SAVE_DIR, fname))
                        count += 1
                    except Exception as e:
                        log_error(f"Delete failed for {fname}: {e}")
            refreshed = refresh_table(pellet_choices, species_choices, select_all=False)
            return refreshed, f"Deleted {count} row(s)."

        def load_selected_to_designer(df):
            import pandas as pd
            if isinstance(df, list):
                df = pd.DataFrame(df[1:], columns=df[0])
            if df is None or df.empty or "Select" not in df.columns or "File" not in df.columns:
                return [gr.update()] * (8 + 5 + 14 + (MAX_ROWS*2) + 7) + ["No selection."]
            chosen = df[df["Select"] == True]
            if len(chosen) == 0:
                return [gr.update()] * (8 + 5 + 14 + (MAX_ROWS*2) + 7) + ["Select exactly one row."]
            if len(chosen) > 1:
                return [gr.update()] * (8 + 5 + 14 + (MAX_ROWS*2) + 7) + ["Please select only one row."]
            filename = chosen.iloc[0]["File"]
            try:
                data = load_die_design(filename)
                inp = data.get("inputs", {}) or {}
                ps = inp.get("pellet_size", 9.0)
                th = inp.get("throughput", 9.0)
                bd = inp.get("bulk_density", 500.0)
                fatv = inp.get("fat", 24.0)
                ft_type = inp.get("feed_type","Salmonid")
                fl = (ft_type == "Floating")
                sh = (ft_type == "Shrimp_Prawn")
                sa = (ft_type == "Salmonid")
                osk = (ft_type == "Other sinking")
                od = inp.get("opening_dia", 6.3)
                cl = inp.get("channel_length", 14.0)
                col = inp.get("cone_length", 4.0)
                cw = inp.get("cone_width", 13.0)
                nh = inp.get("number_of_holes", 500)
                pdm = inp.get("plate_diameter", 250.0)
                chp = inp.get("chamfer_plate", 2.0)
                dch = inp.get("die_center_hole_diameter", 48.5)
                cch = inp.get("chamfer_center_hole", 0.0)
                oop = inp.get("outer_opening_pcd", 189.0)
                coo = inp.get("chamfer_outer_opening", 0.0)
                iop = inp.get("inner_opening_pcd", 128.0)
                cio = inp.get("chamfer_inner_opening", 0.0)
                obc = inp.get("outer_bolt_count", 12)
                obp = inp.get("outer_bolt_pcd", 230.0)
                obd = inp.get("outer_bolt_diameter", 10.5)
                ibc = inp.get("inner_bolt_count", 4)
                ibp = inp.get("inner_bolt_pcd", 75.0)
                ibd = inp.get("inner_bolt_diameter", 8.2)
                rows = inp.get("hole_rows", []) or []
                updates = [
                    gr.update(value=ps), gr.update(value=th), gr.update(value=bd), gr.update(value=fatv),
                    gr.update(value=fl), gr.update(value=sh), gr.update(value=sa), gr.update(value=osk),
                    gr.update(value=od), gr.update(value=cl), gr.update(value=col), gr.update(value=cw), gr.update(value=nh),
                    gr.update(value=pdm), gr.update(value=chp),
                    gr.update(value=dch), gr.update(value=cch),
                    gr.update(value=oop), gr.update(value=coo),
                    gr.update(value=iop), gr.update(value=cio),
                    gr.update(value=obc), gr.update(value=obp), gr.update(value=obd),
                    gr.update(value=ibc), gr.update(value=ibp), gr.update(value=ibd),
                ]
                for idx in range(MAX_ROWS):
                    if idx < len(rows):
                        updates.extend([gr.update(value=rows[idx].get("num_holes",0), visible=True),
                                        gr.update(value=rows[idx].get("pcd",0), visible=True)])
                    else:
                        show = idx < ROWS_STEP
                        updates.extend([gr.update(value=0, visible=show), gr.update(value=0, visible=show)])
                updates += [gr.update(value=None, visible=False),
                            gr.update(value="<div></div>"),
                            gr.update(value=None),
                            gr.update(value=None),
                            None, False, ROWS_STEP, "Loaded."]
                return updates
            except Exception as e:
                log_error(f"load_selected_to_designer() failed: {e}")
                return [gr.update()] * (8 + 5 + 14 + (MAX_ROWS*2) + 7) + ["Load failed."]

        gr.Markdown("### Die Manager")
        with gr.Accordion("Saved designs (compact table, filters, actions, charts)", open=True):
            base_df = clean_table_view(collect_manager_rows())
            # Build filter option sets
            pellet_options = uniq(base_df["Pellet Size (mm)"]) if not base_df.empty else []
            species_options = uniq(base_df["Species"]) if not base_df.empty else []

            with gr.Row():
                pellet_filter = gr.CheckboxGroup(choices=pellet_options, label="Filter: Pellet Size (mm)", value=[])
                species_filter = gr.CheckboxGroup(choices=species_options, label="Filter: Species", value=[])
                select_all = gr.Checkbox(value=False, label="Select All / None")

            # Dataframe with per-row checkbox
            saved_table = gr.Dataframe(
                value=base_df,
                interactive=True,
                headers=list(base_df.columns) if not base_df.empty else None,
                label="Saved Die Designs (checkbox to select rows)",
                wrap=True
            )

            with gr.Row():
                refresh_btn = gr.Button("↻ Refresh")
                delete_btn = gr.Button("ߗ᯸elete Selected")
                load_btn = gr.Button("ߓ堌oad Selected into Designer")

            manager_status = gr.Markdown("")

            # Charts preserved
            chart1 = gr.Plot(label="OA/t (Y) vs Bulk Density (X)")
            chart2 = gr.Plot(label="OA/t (Y) vs Expansion (%)")

            def update_charts_multi(pellets, specieses):
                df = refresh_table(pellets, specieses, select_all=False)
                if df is None or df.empty:
                    return None, None
                for col in ["OA/t (mm²/t/h)", "Bulk Density", "Expansion (%)", "# Holes"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                fx = df.dropna(subset=["Bulk Density", "OA/t (mm²/t/h)"]) if "Bulk Density" in df.columns else pd.DataFrame()
                gx = df.dropna(subset=["Expansion (%)", "OA/t (mm²/t/h)"]) if "Expansion (%)" in df.columns else pd.DataFrame()
                if fx.empty:
                    fig1 = px.scatter(pd.DataFrame({"Bulk Density":[],"OA/t (mm²/t/h)":[]}), x="Bulk Density", y="OA/t (mm²/t/h)")
                else:
                    fig1 = px.scatter(fx, x="Bulk Density", y="OA/t (mm²/t/h)",
                                      hover_name="Species", size="# Holes", color="Species")
                if gx.empty:
                    fig2 = px.scatter(pd.DataFrame({"Expansion (%)":[],"OA/t (mm²/t/h)":[]}), x="Expansion (%)", y="OA/t (mm²/t/h)")
                else:
                    fig2 = px.scatter(gx, x="Expansion (%)", y="OA/t (mm²/t/h)",
                                      hover_name="Species", size="# Holes", color="Species")
                fig1.update_layout(height=360, margin=dict(l=10,r=10,t=40,b=10))
                fig2.update_layout(height=360, margin=dict(l=10,r=10,t=40,b=10))
                return fig1, fig2

            # Wiring
            refresh_btn.click(fn=refresh_table, inputs=[pellet_filter, species_filter, select_all], outputs=[saved_table])
            pellet_filter.change(fn=refresh_table, inputs=[pellet_filter, species_filter, select_all], outputs=[saved_table])
            species_filter.change(fn=refresh_table, inputs=[pellet_filter, species_filter, select_all], outputs=[saved_table])
            select_all.change(fn=toggle_select, inputs=[saved_table, select_all], outputs=[saved_table])

            delete_btn.click(fn=delete_selected, inputs=[saved_table, pellet_filter, species_filter], outputs=[saved_table, manager_status])
            load_btn.click(
                fn=load_selected_to_designer,
                inputs=[saved_table],
                outputs=[
                    # Top specs
                    pellet_size, throughput, density, fat, floating, shrimp, salmonid, other_sinking,
                    # Design
                    opening_dia, channel_length, cone_length, cone_width, holes,
                    # Plate/track/bolts
                    plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                    outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                    outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                    inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                ] + [comp for pair in row_components for comp in pair] + [
                    quick_dxf, save_status, plate_preview, inset_preview, dxf_path_state, calc_ok_state, visible_rows, manager_status
                ]
            )

            # Charts follow filter changes
            pellet_filter.change(fn=update_charts_multi, inputs=[pellet_filter, species_filter], outputs=[chart1, chart2])
            species_filter.change(fn=update_charts_multi, inputs=[pellet_filter, species_filter], outputs=[chart1, chart2])

        # Bind new save buttons
        quick_save_button.click(
            fn=on_quick_save,
            inputs=[calc_ok_state, calc_values_state,
                    pellet_size, throughput, density, fat, floating, shrimp, salmonid, other_sinking,
                    channel_length, cone_length, opening_dia, cone_width, holes,
                    plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                    outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                    outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                    inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter] +
                   [comp for pair in row_components for comp in pair],
            outputs=[save_status]
        )
        save_with_meta_button.click(
            fn=on_save_with_meta,
            inputs=[calc_ok_state, calc_values_state,
                    pellet_size, throughput, density, fat, floating, shrimp, salmonid, other_sinking,
                    channel_length, cone_length, opening_dia, cone_width, holes,
                    plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                    outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                    outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                    inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter] +
                   [comp for pair in row_components for comp in pair] +
                   [extruder_type, production_line, plant_name, product_type, die_performance, comments],
            outputs=[save_status]
        )

        return demo

# -------- App Startup ------------------------------------------
try:
    log_error("Startup: Calling build_ui()")
    demo = build_ui()
    log_error("Startup: build_ui() returned successfully.")
except Exception as e:
    log_error(f"Startup: build_ui() failed with error: {e}")
    raise

if __name__ == "__main__":
    try:
        log_error("Startup: Launching Gradio app...")
        demo.queue().launch(server_name="0.0.0.0", server_port=7866)
        log_error("Startup: Gradio app launched successfully.")
    except Exception as e:
        log_error(f"Startup: Gradio launch failed with error: {e}")
        raise
