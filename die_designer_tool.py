# die_design_tool.py

# -------- Imports (core) -------------------
# Standard library
import base64
import datetime
import glob
import json
import logging
import os
import traceback
import uuid
import math
import time
from html import escape as html_escape
from io import BytesIO
from typing import List, Optional, Tuple  # Py3.8: use typing.Tuple, not tuple[...]
# Third-party
import ezdxf
import gradio as gr
import numpy as np  # remove if unused
import pandas as pd
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict

# Optional: only if you actually use transforms; otherwise remove or keep with noqa
# from ezdxf.math import Matrix44  # noqa: F401

# --- ezdxf font + logging setup (previews won't depend on system fonts) ---
try:
    # Tell ezdxf to prefer Matplotlib's built-in fonts (DejaVu) if available
    try:
        ezdxf.options.use_matplotlib_fonts = True      # safe on modern ezdxf
    except Exception:
        pass
    try:
        ezdxf.options.default_font = "DejaVuSans.ttf"  # use mpl DejaVu
    except Exception:
        pass

    # Silence ezdxf font discovery warnings in logs
    logging.getLogger("ezdxf").setLevel(logging.ERROR)
except Exception:
    pass

#-- Favicon helper-----------------
def _favicon_data_url():
    fav = Path(BASE_DIR) / "favicon.ico"
    if not fav.exists():
        alt = Path(BASE_DIR) / "favicon-512.png"
        fav = alt if alt.exists() else None
    if not fav:
        return None, None, None
    mime = "image/x-icon" if fav.suffix.lower() == ".ico" else "image/png"
    b64 = base64.b64encode(fav.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}", mime, str(fav)

# -------- Version / Build tag --------------
VERSION = "2.0.1"
# Optional: provide an env suffix like BUILD_TAG="staging" → banner shows “… — staging”
ENV_BUILD_TAG = os.environ.get("BUILD_TAG", "").strip()

def version_banner_md() -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    tag = f" — {ENV_BUILD_TAG}" if ENV_BUILD_TAG else ""
    return f"### Die Design Tool v{VERSION}{tag}\n<small>{ts}</small>"

# -------- Constants ------------------------
MAX_ROWS = 18
MAX_BOLT_HOLES = 20
ROWS_STEP = 3
DELETE_COL = "DEL"   # ASCII-safe delete column header & cell text

# ---- Table schema (GLOBAL; used by ensure_manager_columns, _df_datatypes, etc.) ----
MANAGER_COLUMNS = [
    "Saved","Feed Type","Pellet Size (mm)","# Rows","# Holes",
    "Die Ø (mm)","Land L (mm)","Cone L (mm)","Cone W (mm)",
    "OA 1 hole (mm²)","OA total (mm²)","OA/t (mm²/t/h)",
    "L/D","Die L (mm)","Cone °",
    "Bulk Density","Expansion (%)",
    "Extruder","Line","Plant","Type","Performance","Comments",
    "File",
]

MANAGER_NUMBER_COLS = {
    "# Rows","# Holes","Pellet Size (mm)","Die Ø (mm)","Land L (mm)","Cone L (mm)","Cone W (mm)",
    "OA 1 hole (mm²)","OA total (mm²)","OA/t (mm²/t/h)","L/D","Die L (mm)","Cone °",
    "Bulk Density","Expansion (%)",
}

NO_CHANGE = gr.update()
def SET(v=""):
    return gr.update(value=v)

# -------- Paths & Logging ------------------
# Base dir priority: env → Synology path → script dir
_DEFAULT_BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_ENV_BASE    = os.environ.get("DIE_TOOL_BASE_DIR")
_SYNO_BASE   = "/volume1/docker/Diedesign"
BASE_DIR     = _ENV_BASE or (_SYNO_BASE if os.path.isdir(_SYNO_BASE) else _DEFAULT_BASE)

APP_BASE = Path(BASE_DIR)

# FAVICON = os.path.join(BASE_DIR, "favicon.ico")
FAVICON = APP_BASE / "favicon.ico"   # <- preferred (Path)

SAVE_DIR = APP_BASE / "saved_designs"
LOG_DIR  = APP_BASE / "logs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGFILE = str(LOG_DIR / "die_design_tool.log")  # RotatingFileHandler wants a string path

# -------- Logger (rotating) -------------
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("die_design_tool")
logger.setLevel(logging.INFO)  # INFO is enough for normal runs

# Avoid duplicate handlers if reloaded
if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == LOGFILE
           for h in logger.handlers):
    os.makedirs(LOG_DIR, exist_ok=True)
    rh = RotatingFileHandler(
        LOGFILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB per file
        backupCount=3,              # keep 3 old files: .1, .2, .3
        encoding="utf-8",
        delay=True,                 # open file lazily
    )
    rh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(rh)

# Keep our logs self-contained (don’t bubble to root)
logger.propagate = False

# Optional: tame the root logger to avoid duplicate console spam from deps
root = logging.getLogger()
root.handlers.clear()
root.setLevel(logging.WARNING)

def log_info(message: str):
    try:
        logger.info(message)
    except Exception:
        pass

def log_error(message: str):
    """Safe logger that won’t crash if logging breaks."""
    try:
        logger.error(message)
    except Exception:
        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(LOGFILE, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {message}\n")
        except Exception:
            pass

# Friendly startup info
log_info(f"GRADIO VERSION: {getattr(gr, '__version__', 'unknown')}")
log_info(f"HAS gr.JS COMPONENT: {hasattr(gr, 'JS')}")

import sys, faulthandler
faulthandler.enable()  # dumps tracebacks on fatal errors

def _install_global_excepthook():
    def _hook(exc_type, exc, tb):
        try:
            logger.critical("UNCAUGHT import-time exception", exc_info=(exc_type, exc, tb))
        finally:
            # still print to stderr so Docker logs show it too
            sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = _hook

_install_global_excepthook()

def __runtime_probe():
    import hashlib
    p = os.path.abspath(__file__)
    try:
        with open(p, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
    except Exception:
        md5 = "md5-err"
    msg = (
        f"*** BUILD: v{VERSION}{(' — ' + ENV_BUILD_TAG) if ENV_BUILD_TAG else ''} *** "
        f"file={p} size={os.path.getsize(p) if os.path.exists(p) else 'NA'} md5={md5}"
    )
    log_info(msg)

# Emit the runtime probe now that logger helpers exist
__runtime_probe()  

# -------- Matplotlib bootstrap (headless + fonts) ------------
# Writable cache (important on NAS/Docker)
MPLCFG = os.path.join(LOG_DIR, ".mplconfig")
os.makedirs(MPLCFG, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCFG)

import matplotlib
matplotlib.use("Agg")  # headless everywhere
from matplotlib import font_manager
from pathlib import Path

def _ensure_mpl_font():
    try:
        dv = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        if dv.exists():
            font_manager.fontManager.addfont(str(dv))
            try:
                font_manager._load_fontmanager(try_read_cache=False)  # newer mpl
            except Exception:
                try:
                    font_manager._rebuild()  # older mpl
                except Exception:
                    pass
            matplotlib.rcParams["font.family"] = "DejaVu Sans"
            matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
            try:
                path = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
                log_info(f"Font ready: {path}")       # <- info
            except Exception as e:
                log_error(f"Font resolve failed: {e}")
        else:
            log_error("Font bootstrap: DejaVuSans.ttf not found in mpl-data")
    except Exception as e:
        log_error(f"Font bootstrap failed: {e}")

_ensure_mpl_font()
import matplotlib.pyplot as plt

# -------- Global DXF Cache -----------------
cached_doc = None
cached_filename = ""
cached_inset_center = None
cached_export_path = None

# -------- Rendering Flags ------------------
FORCE_ENTITY_COLOR = False  # keep True until ByLayer colors are solid

# --- Die styles Helper---------------------------------------------- ---
DIE_STYLE_CIRCULAR = "Circular"
DIE_STYLE_STAGGERED = "Staggered"

# ---------- Unified color policy for Matplotlib renders ----------
from ezdxf import colors as _dxfcolors

def _aci_to_rgb01(aci: int) -> tuple:
    """
    Convert AutoCAD ACI index to (r,g,b) in 0..1.
    Special-case ACI 7 (white) to black so it shows on white backgrounds.
    """
    try:
        idx = int(aci)
    except Exception:
        idx = 7
    if idx == 7:
        return (0.0, 0.0, 0.0)  # treat "white" as black for readability
    try:
        r, g, b = _dxfcolors.aci2rgb(idx if 1 <= idx <= 255 else 7)
        return (r/255.0, g/255.0, b/255.0)
    except Exception:
        return (0.0, 0.0, 0.0)

COLOR_POLICY = {
    "preview": {
        "BOLTS": (0.0, 0.0, 0.0),
        "BACK_BOLTS": (0.0, 0.0, 0.0),
        "INSET_OPENINGS": _aci_to_rgb01(5),
        "INSET_CONE":     _aci_to_rgb01(4),
        "INSET_CS":       _aci_to_rgb01(6),
    },
    "pdf": {
        "BOLTS": (0.0, 0.0, 0.0),
        "BACK_BOLTS": (0.0, 0.0, 0.0),
        "INSET_OPENINGS": _aci_to_rgb01(5),
        "INSET_CONE":     _aci_to_rgb01(4),
        "INSET_CS":       _aci_to_rgb01(6),

        # >>> add these <<<
        "BACK_SEGMENTS":  (0.2, 0.2, 0.2),   # walls
        "BACK_SEG_RINGS": (0.5, 0.5, 0.5),   # inner/outer segment PCD
        "BACK_INLET_PCD": (0.0, 0.0, 0.0),
    },
}

def resolve_edgecolor(ent, doc, *, context="preview", mode=None):
    """
    Unified color resolver:
      1) entity true_color (if present)
      2) per-context COLOR_POLICY override
      3) friendly layer shortcuts (your explicit mappings)
      4) ACI fallback via _aci_to_rgb01 (ACI 7 -> black)
    """
    from ezdxf import colors as dxfcolors

    def _rgb01(r, g, b):  # 0..255 -> 0..1
        return (r/255.0, g/255.0, b/255.0)

    ly = (getattr(ent.dxf, "layer", "") or "").upper()

    # (1) entity true_color wins
    tc = getattr(ent.dxf, "true_color", None)
    if tc:
        r = (tc >> 16) & 0xFF; g = (tc >> 8) & 0xFF; b = tc & 0xFF
        return (r/255.0, g/255.0, b/255.0)

    # (2) per-context policy table
    try:
        ctx_map = COLOR_POLICY.get(context, {})
        if ly in ctx_map:
            return ctx_map[ly]
    except Exception:
        pass

    # (3) explicit friendly overrides
    BLUE_OPEN = _rgb01(0, 102, 204)
    BLUE_CONE = _rgb01(40, 140, 230)
    BLUE_CS   = _rgb01(90, 180, 255)
    GREY_PCD  = _rgb01(80, 80, 80)
    GREY_CHAM = _rgb01(60, 60, 60)
    GREY_SEG  = _rgb01(51, 51, 51)
    GREY_SEGR = _rgb01(128, 128, 128)
    BLACK     = (0.0, 0.0, 0.0)

    if context in {"preview", "pdf"}:
        if ly in {"OUTLINE", "BACK_OUTLINE"}: return BLACK
        if ly in {"BOLTS", "BACK_BOLTS"}:     return BLACK
        if ly in {"BOLTS_CS"}:                return BLACK
        if ly in {"HOLES", "BACK_HOLES", "INSET_OPENINGS"}: return BLUE_OPEN
        if ly in {"CONE", "BACK_CONE", "INSET_CONE"}:       return BLUE_CONE
        if ly in {"COUNTERSINK", "BACK_CS", "INSET_CS"}:    return BLUE_CS
        if ly == "PCD_GUIDE":         return GREY_PCD
        if ly == "CHAMFERS":          return GREY_CHAM
        if ly == "BACK_SEGMENTS":     return GREY_SEG
        if ly == "BACK_SEG_RINGS":    return GREY_SEGR
        if ly == "BACK_INLET_PCD":    return BLACK

    # (4) ACI fallback (ByEntity, then ByLayer), using our ACI→RGB01 that maps 7→black
    try:
        col_idx = int(getattr(ent.dxf, "color", 256) or 256)
    except Exception:
        col_idx = 256
    if col_idx <= 0 or col_idx == 256:  # BYLAYER / BYBLOCK
        try:
            col_idx = int(doc.layers.get(ly).dxf.color or 7)
        except Exception:
            col_idx = 7
    return _aci_to_rgb01(col_idx)

# --- Unified hole count (for charts/tables) ---
def total_holes(rows_data=None, holes_xy=None) -> int:
    if holes_xy:
        return len(holes_xy)
    return sum(int(r.get("num_holes") or 0) for r in (rows_data or []))

# -------- Excel suggestions (optional) ----
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
def calculate_die_design(
    pellet_size_mm, dry_meal_throughput_tph, bulk_density_g_per_l, final_fat_percent,
    feed_type, channel_length_mm, cone_length_mm, opening_dia,
    cone_width_mm, number_of_holes, countersink_plus_radius_mm=0.0, countersink_depth_mm=0.0,
    *,
    # NEW: annotation controls (optional; safe defaults)
    show_diameter_labels: bool = True,
    top_label_offset_mm: float = 4.0,     # distance above top surface (y<0)
    bottom_label_offset_mm: float = 4.0,  # distance below channel end
    label_fontsize: int = 10
):
    """
    Returns (image_html, table_html, calc_values)

    Adds two UI annotations to the cross-section figure:
      - Top label: ⌀ of top opening (cone_width [+ 2*cs_plus] if cone, else opening_dia)
      - Bottom label: ⌀ of die opening (opening_dia)
    Offsets are in mm in data coords and are robust to hole height/width changes.
    """
    die_total_length_mm = channel_length_mm + cone_length_mm
    cone_angle_deg = math.degrees(math.atan(cone_width_mm / (2 * cone_length_mm))) if cone_length_mm > 0 else 0
    open_area_one_hole_mm2 = math.pi * (opening_dia / 2) ** 2
    total_plate_open_area_mm2 = open_area_one_hole_mm2 * number_of_holes
    open_area_per_tonne_mm2_per_tph = total_plate_open_area_mm2 / max(dry_meal_throughput_tph, 1e-6)
    expansion_percent = (1 - (opening_dia / max(pellet_size_mm, 1e-6))) * 100

    # Normalize countersink (+radius semantics) against cone presence
    cs_dia, cs_dep = _validate_countersink_plus(
        opening_dia=opening_dia,
        cone_width=cone_width_mm,
        countersink_plus_radius=countersink_plus_radius_mm,
        countersink_depth=countersink_depth_mm,
        cone_length=cone_length_mm,
    )

    # Build values used in both success/failure paths
    calc_values = {
        "die_total_length": die_total_length_mm,
        "cone_angle": cone_angle_deg,
        "open_area_one_hole": open_area_one_hole_mm2,
        "total_plate_open_area": total_plate_open_area_mm2,
        "open_area_per_tonne": open_area_per_tonne_mm2_per_tph,
        "expansion_percent": expansion_percent,
    }

    # Suggestions / table (do this outside the plotting try so we always have it)
    suggestions = get_suggestions(
        feed_type, pellet_size_mm, bulk_density_g_per_l, final_fat_percent, dry_meal_throughput_tph
    )
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
    em_dash = "\u2014"
    suggested = [
        round(suggestions.get("Die Opening", 0), 2) if suggestions.get("Die Opening") is not None else em_dash,
        round(suggestions.get("Land /Channel Length", 0), 2) if suggestions.get("Land /Channel Length") is not None else em_dash,
        round(suggestions.get("Total Plate Open Area", 0), 0) if suggestions.get("Total Plate Open Area") is not None else em_dash,
        round(suggestions.get("Open Area per tonne", 0), 0) if suggestions.get("Open Area per tonne") is not None else em_dash,
        round(suggestions.get("Expansion", 0), 1) if suggestions.get("Expansion") is not None else em_dash,
        round(suggestions.get("L/D Ratio", 0), 2) if suggestions.get("L/D Ratio") is not None else em_dash,
        em_dash,
        em_dash,
    ]

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
            if percent_diff < 0.05: color = "#dff5e1"
            elif percent_diff < 0.15: color = "#f3f7d9"
            else: color = "#fde2e1"
        except (ValueError, TypeError):
            color = "#ffffff"
        table_html_wrapped.append(
            f"<tr><td style='padding:9px 12px; text-align:left;'>{p}</td>"
            f"<td style='padding:9px 12px; text-align:center; background-color:{color};'>{c}</td>"
            f"<td style='padding:9px 12px; text-align:center; background-color:{color};'>{s}</td></tr>"
        )
    table_html_wrapped.append("</tbody></table></div>")
    table_html_str = "".join(table_html_wrapped)

    # ----- Plotting in a safety wrapper -----
    try:
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64

        HAS_CONE = (float(cone_length_mm) > 0.0) and (float(cone_width_mm) > 0.0)
        if not HAS_CONE:
            cs_dia, cs_dep = 0.0, 0.0  # kill countersink if cone is absent

        fig, ax = plt.subplots(figsize=(4, 4.2))
        ax.set_facecolor('#ffffff')
        ax.set_xlabel("Width (mm)", fontsize=9)
        ax.set_ylabel("Length (mm)", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        od    = float(opening_dia)
        Lk    = float(cone_length_mm) if HAS_CONE else 0.0
        w_ent = max(od, float(cone_width_mm)) if HAS_CONE else od
        y_cs  = (min(max(0.0, float(cs_dep)), Lk) if HAS_CONE else 0.0)

        # steel background (robust across Matplotlib versions)
        STEEL = "#cccccc"
        ax.axhspan(0.0, die_total_length_mm, xmin=0.0, xmax=1.0, facecolor=STEEL, zorder=0)

        # cone/countersink polygons
        polys = []
        if HAS_CONE and y_cs > 0.0:
            if float(cs_dia) > w_ent:
                polys.append([(-float(cs_dia)/2, 0.0), ( float(cs_dia)/2, 0.0),
                              ( w_ent/2, y_cs), (-w_ent/2, y_cs)])
            else:
                polys.append([(-w_ent/2, 0.0), ( w_ent/2, 0.0),
                              ( w_ent/2, y_cs), (-w_ent/2, y_cs)])
        if HAS_CONE and (Lk > y_cs):
            polys.append([(-w_ent/2, y_cs), (w_ent/2, y_cs), (od/2, Lk), (-od/2, Lk)])
        for poly in polys:
            ax.fill([p[0] for p in poly], [p[1] for p in poly], color="#3366cc", alpha=1.0)

        # land/channel rectangle
        chan_top = Lk if HAS_CONE else 0.0
        rect_x = [-od/2, od/2, od/2, -od/2, -od/2]
        rect_y = [chan_top, chan_top, die_total_length_mm, die_total_length_mm, chan_top]
        ax.fill(rect_x, rect_y, color="#3366cc", alpha=1.0)

        # framing
        x_margin, base_y_margin = 2.0, 2.0
        effective_top_width = max(od, float(cs_dia), float(cone_width_mm)) if HAS_CONE else od
        half_span = 0.5 * effective_top_width

        # Ensure labels fit if enabled
        y_margin = max(
            base_y_margin,
            (top_label_offset_mm + 1.0) if show_diameter_labels else base_y_margin,
            (bottom_label_offset_mm + 1.0) if show_diameter_labels else base_y_margin,
        )
        ax.set_xlim(-half_span - x_margin, half_span + x_margin)
        ax.set_ylim(die_total_length_mm + y_margin, -y_margin)  # invert Y

        # diameter labels
        if show_diameter_labels:
            if HAS_CONE:
                if float(countersink_plus_radius_mm) > 0.0:
                    top_dia_val = float(cone_width_mm) + 2.0 * float(countersink_plus_radius_mm)
                else:
                    top_dia_val = float(cone_width_mm)
            else:
                top_dia_val = float(opening_dia)
            bottom_dia_val = float(opening_dia)

            ax.text(0.0, -abs(float(top_label_offset_mm or 0.0)),
                    f"⌀ {top_dia_val:.2f}", ha="center", va="top",
                    fontsize=int(label_fontsize or 10), fontweight="bold",
                    color="#111111", zorder=5, clip_on=False)
            ax.text(0.0, die_total_length_mm + abs(float(bottom_label_offset_mm or 0.0)),
                    f"⌀ {bottom_dia_val:.2f}", ha="center", va="bottom",
                    fontsize=int(label_fontsize or 10), fontweight="bold",
                    color="#111111", zorder=5, clip_on=False)

        # encode figure → base64
        from io import BytesIO
        import base64
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_html = (
            f"<img src='data:image/png;base64,{image_base64}' "
            f"style='width: 400px; height: 420px; margin-right: 20px;'/>"
        )
    except Exception:
        import traceback
        try:
            log_error("CROSS-SECTION RENDER FAILED:\n" + traceback.format_exc())
        except Exception:
            pass
        image_html = "<div class='chip error'>Cross-section render failed — see log.</div>"

    image_html_wrapped = f"<div class='panel-image' style='text-align:center; padding:10px;'>{image_html}</div>"
    return image_html_wrapped, table_html_str, calc_values

# ----- Datamodel + culling for "Circular on Segmented" plates (Py 3.8-safe) -----
from dataclasses import dataclass
from typing import TypedDict, List, Tuple, Optional
import math


class RowSpec(TypedDict):
    """Spec for a single PCD row in a circular pattern."""
    pcd: float        # Pitch Circle Diameter (mm)
    count: int        # Number of holes on this PCD
    start_deg: float  # Starting angle, degrees


@dataclass
class CircularPattern:
    """A set of circular PCD rows (reuse your existing circular inputs)."""
    rows: List[RowSpec]


@dataclass
class SegmentedBase:
    """Segmented plate parameters (radial walls)."""
    n_segments: int                 # e.g. 6, 8, 12
    wall_thickness: float           # mm (radial wall "width")
    inner_radius: float = 0.0       # hub radius (mm)
    outer_radius: Optional[float] = None  # default = plate radius if None
    rotation_deg: float = 0.0       # rotate wall boundaries (deg)
    clearance: float = 0.0          # extra keep-out beyond hole radius (mm)

def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi

def _min_ang_to_boundaries(theta: float, n_seg: int, rot: float) -> float:
    """
    Smallest absolute angular distance (in radians) from angle `theta`
    to any segment boundary ray, given `n_seg` radial walls and rotation `rot` (radians).
    """
    seg_angle = 2 * math.pi / n_seg
    # Reduce to a canonical segment with boundary at 0 after rotation.
    local = _wrap_angle(theta - rot)
    k = round(local / seg_angle)  # nearest boundary index
    return abs(_wrap_angle(local - k * seg_angle))

def cull_by_segments(
    holes: List[Tuple[float, float, float]],  # (x_mm, y_mm, r_hole_mm)
    base: SegmentedBase,
) -> List[Tuple[float, float, float]]:
    """
    Remove holes that intersect radial walls or violate inner/outer keep-outs.
    """
    # Basic sanity
    assert base.n_segments >= 2, "n_segments must be ≥ 2"
    assert base.wall_thickness >= 0.0, "wall_thickness must be ≥ 0"
    assert base.inner_radius >= 0.0, "inner_radius must be ≥ 0"
    if base.outer_radius is not None:
        assert base.outer_radius >= 0.0, "outer_radius must be ≥ 0"

    kept: List[Tuple[float, float, float]] = []
    rot = math.radians(base.rotation_deg)

    for x, y, r_h in holes:
        r = math.hypot(x, y)
        theta = math.atan2(y, x)

        # Radial keep-outs (hub + optional outer radius)
        if r < base.inner_radius + r_h + base.clearance:
            continue
        if base.outer_radius is not None and r > base.outer_radius - r_h - base.clearance:
            continue

        # Distance to the nearest segment boundary ray at this radius
        dtheta = _min_ang_to_boundaries(theta, base.n_segments, rot)
        d_perp = r * math.sin(dtheta)  # perpendicular distance to boundary ray

        keepout = (base.wall_thickness / 2.0) + r_h + base.clearance
        if d_perp < keepout:
            continue

        kept.append((x, y, r_h))

    return kept

# ----- Helpers to build Tab 3 circular segmenteds dies -----

def _t3_capture_rows_to_cv(cv: dict, n_rows, *vals):
    """
    Persist Tab 3 rows into calc-values state so generate_die_plate_dxf can honor them.
    vals is flat: [holes0, pcd0, holes1, pcd1, ...]
    """
    try:
        N = int(n_rows or 0)
    except Exception:
        N = 0

    flat = list(vals or [])
    # Truncate / pad to exactly 2*N items
    flat = (flat + [0] * max(0, 2 * N - len(flat)))[: 2 * N]

    cv["t3_n_rows"] = N
    cv["t3_rows_flat"] = flat

    try:
        log_error(f"T3 CAPTURE -> n_rows={N}, flat={flat}")
    except Exception:
        pass
    return cv


def _snap_to_rows_pcd(pts_xy, rows):
    import math
    rings = [0.5 * float(r.get("pcd", 0.0) or 0.0) for r in (rows or []) if float(r.get("pcd", 0.0) or 0.0) > 0.0]
    if not rings:
        return pts_xy or []
    out = []
    for (x, y) in (pts_xy or []):
        r = math.hypot(x, y)
        if r <= 1e-9:
            out.append((x, y))
            continue
        Rtarget = min(rings, key=lambda R: abs(R - r))
        s = Rtarget / r
        out.append((x * s, y * s))
    return out

def _t3_rows_from_components(n_rows, *vals,
                             pcd_min=90.0, pcd_max=350.0,
                             holes_max=250, pcd_step=0.1,
                             even_start_deg=0.0, odd_start_deg=0.0):
    """
    Build rows for Tab 3 from flat slider values:
      vals = [holes0, pcd0, holes1, pcd1, ...]
    Returns: [{"pcd": float, "count": int, "start_deg": float}, ...]
    """
    rows = []
    try:
        n = int(n_rows or 0)
        if n < 0:
            n = 0
    except Exception:
        n = 0

    # Ensure we don't index past vals if it's shorter than expected
    needed = 2 * n
    if len(vals) < needed:
        vals = list(vals) + [0] * (needed - len(vals))

    for i in range(n):
        # Robust parsing
        try:
            holes = int(float(vals[2*i] or 0))
        except Exception:
            holes = 0
        try:
            pcd = float(vals[2*i + 1] or 0.0)
        except Exception:
            pcd = 0.0

        # Sanity + clamp to UI ranges
        if holes < 0:
            holes = 0
        holes = min(holes, int(holes_max))
        if pcd <= 0.0:
            continue
        pcd = max(pcd_min, min(pcd_max, pcd))

        # Match slider step
        if pcd_step and pcd_step > 0:
            pcd = round(pcd / pcd_step) * pcd_step
            # Guard against tiny FP drift (e.g., 119.899999 -> 119.9)
            pcd = round(pcd, 3)

        if holes > 0:
            start_deg = (odd_start_deg if (i % 2 == 1) else even_start_deg)
            rows.append({"pcd": float(pcd), "num_holes": int(holes), "start_deg": float(start_deg)})

    return _ensure_row_start_angles(rows, base_start_deg=even_start_deg)


def _distribute_holes_across_segments(total: int, segments: int) -> list:
    """Return a per-segment count list that sums to ``total`` holes."""
    try:
        total = int(total or 0)
    except Exception:
        total = 0
    try:
        segments = int(segments or 0)
    except Exception:
        segments = 0

    if total <= 0 or segments <= 0:
        return [0] * max(0, segments)

    base = total // segments
    rem = total % segments
    dist = [base] * segments
    for idx in range(rem):
        dist[idx] += 1
    return dist


def _segmentwise_xy_from_rows(
    rows,
    *,
    segments: int,
    wall_width: float,
    padding: float,
    corner_radius: float = 0.0,  # kept for parity / future tweaks
):
    """Place holes per segment so each segment has its own even spacing."""
    import math

    S = int(segments or 0)
    if S < 3:
        return _circular_xy_from_rows(rows)

    theta = 2.0 * math.pi / S
    pad = max(0.0, float(padding or 0.0))
    wall = max(0.0, float(wall_width or 0.0))
    req_wall = pad + 0.5 * wall

    pts = []
    safe_rows = _ensure_row_start_angles(rows or [])

    for row in safe_rows:
        try:
            total = int(row.get("num_holes") or row.get("count") or 0)
        except Exception:
            total = 0
        if total <= 0:
            continue

        try:
            pcd = float(row.get("pcd") or 0.0)
        except Exception:
            pcd = 0.0
        if pcd <= 0.0:
            continue

        R = 0.5 * pcd
        if R <= 0.0:
            continue

        # Angular clearance required so the hole center stays clear of the wall.
        if req_wall > 0.0:
            ratio = req_wall / max(R, 1e-9)
            if ratio >= 1.0:
                # No usable angular room at this radius inside a segment.
                continue
            ratio = max(0.0, min(1.0, ratio))
            margin_ang = math.asin(ratio)
        else:
            margin_ang = 0.0

        if margin_ang >= 0.5 * theta:
            # Walls occupy the whole segment.
            continue

        span = theta - 2.0 * margin_ang
        if span <= 0.0:
            continue

        start_deg = float(row.get("start_deg") or 0.0)
        seg_offset_float = (start_deg / 360.0) * S
        seg_offset_int = int(math.floor(seg_offset_float)) % S if S > 0 else 0
        offset_norm = seg_offset_float - math.floor(seg_offset_float)

        per_segment = _distribute_holes_across_segments(total, S)
        if seg_offset_int:
            shift = seg_offset_int % S
            per_segment = per_segment[-shift:] + per_segment[:-shift]

        for seg_idx in range(S):
            count = per_segment[seg_idx]
            if count <= 0:
                continue

            seg_start = seg_idx * theta + margin_ang
            seg_end = (seg_idx + 1) * theta - margin_ang
            seg_span = seg_end - seg_start
            if seg_span <= 0:
                continue

            if count == 1:
                # Keep solo hole centered regardless of offsets.
                angle = seg_start + 0.5 * seg_span
                pts.append((R * math.cos(angle), R * math.sin(angle)))
                continue

            base_positions = [((k + 0.5) / count) for k in range(count)]
            base_angles = [seg_start + pos * seg_span for pos in base_positions]

            delta = (offset_norm % 1.0) * seg_span
            adjusted = []
            for ang in base_angles:
                shifted = ang + delta
                if seg_span > 0.0:
                    while shifted > seg_end:
                        shifted -= seg_span
                    while shifted < seg_start:
                        shifted += seg_span
                eps = min(1e-4, 0.25 * seg_span)
                if eps > 0.0:
                    shifted = min(max(shifted, seg_start + eps), seg_end - eps)
                adjusted.append(shifted)

            adjusted.sort()
            for ang in adjusted:
                pts.append((R * math.cos(ang), R * math.sin(ang)))

    return pts


def _t3_build_kept_points(cv_in, n_rows, *row_vals_and_seg_params):
    """
    Tab 3 'Hybrid' builder:
      1) Build circular rows from Tab 3 sliders (like Tab 1)
      2) Generate circular hole centers (Tab 1 routine)
      3) Cull against segmented walls/annulus
    Returns: kept_points (list of (x,y) or (x,y,r)), counts_md, msg_md
    """
    import gradio as gr

    cv = dict(cv_in or {})

    # --- Parse how many pairs we actually have in the flat payload ---
    try:
        n = int(n_rows or 0)
    except Exception:
        n = 0

    flat = list(row_vals_and_seg_params)

    # Segmented params are always the last 6 values in the payload
    seg_defaults = (8, 8.0, 120.0, 198.0, 5.0, 2.0)
    if len(flat) >= 6:
        seg_tail = flat[-6:]
        row_vals = flat[:-6]
    else:
        seg_tail = list(seg_defaults)
        row_vals = flat

    have_pairs = max(0, min(n, len(row_vals) // 2))
    row_vals = row_vals[: 2 * have_pairs]

    (seg_N, seg_t, seg_rin_pcd, seg_rout_pcd, seg_fillet, seg_pad) = (
        (seg_tail + list(seg_defaults))[:6]
    )

    # Persist current Tab 3 rows into calc-values for later hydration / summaries
    try:
        cv = _t3_capture_rows_to_cv(cv, have_pairs, *row_vals)
    except Exception:
        pass

    # --- Build rows exactly like Tab 1 uses ---
    rows = _t3_rows_from_components(have_pairs, *row_vals)
    if not rows:
        return (
            cv,
            [],
            "**Requested:** 0  \n**Dropped by walls:** 0  \n**Placed:** 0",
            "No rows set on Tab 3.",
        )

    # --- Generate circular hole centers (Tab 1 maths) ---
    seg_count = int(seg_N)
    if seg_count >= 3:
        pts_xy = _segmentwise_xy_from_rows(
            rows,
            segments=seg_count,
            wall_width=float(seg_t),
            padding=float(seg_pad),
            corner_radius=float(seg_fillet),
        )
    else:
        pts_xy = _circular_xy_from_rows(rows)

    # Approximate hole radius from calculated values when available
    opening_mm = 0.0
    for key in ("opening_dia", "die_opening_diameter", "opening_diameter"):
        try:
            opening_mm = float(cv.get(key) or 0.0)
        except Exception:
            opening_mm = 0.0
        if opening_mm > 0:
            break
    hole_r = max(0.01, 0.5 * float(opening_mm or 0.0))

    holes_norm = [(float(x), float(y), hole_r) for (x, y) in pts_xy]

    # --- Segmented base (PCDs are converted to radii inside our model) ---
    seg_base = SegmentedBase(
        n_segments=int(seg_N),
        wall_thickness=float(seg_t),
        inner_radius=float(seg_rin_pcd) / 2.0 if float(seg_rin_pcd or 0.0) > 0 else 0.0,
        outer_radius=float(seg_rout_pcd) / 2.0 if float(seg_rout_pcd or 0.0) > 0 else None,
        rotation_deg=0.0,
        clearance=float(seg_pad),
    )

    # --- Cull circular holes by the segment envelope ---
    kept = cull_by_segments(holes_norm, seg_base)

    # You likely store just (x,y) in state; strip radii for the state if desired:
    kept_xy = [(x, y) for (x, y, _r) in kept]
    kept_xy = _snap_to_rows_pcd(kept_xy, rows)   # ensure perfect PCD placement

    total_requested = len(holes_norm)
    total_kept = len(kept)
    total_dropped = total_requested - total_kept

    counts_md = (
        f"**Requested:** {total_requested}  \n"
        f"**Dropped by walls:** {total_dropped}  \n"
        f"**Placed:** {total_kept}"
    )
    msg_md = "Hybrid generated (circular rows culled by segment walls)."

    return cv, kept_xy, counts_md, msg_md

def _draw_front_holes(doc, *, holes_xy, rows):
    """
    Ensure the front/knife side shows the product holes even if the lower-level
    builder didn’t add them. When `holes_xy` is provided, use it; otherwise
    generate the same circular array from `rows`.
    """
    try:
        msp = doc.modelspace()
        # Prefer explicit (x,y) centers
        pts = list(holes_xy or [])
        if not pts and rows:
            # Recreate Tab-1 circular ring centers
            pts = _circular_xy_from_rows(rows)

        if not pts:
            return 0

        # Pick a layer that’s already used for front/knife holes in your file
        # (adjust if yours is different):
        LAYER = "HOLES"  # or your front holes layer name
        try:
            if LAYER not in doc.layers:
                doc.layers.add(LAYER, dxfattribs={"color": 7})
        except Exception:
            pass

        # Hole radius = opening_dia/2 on the *knife/front* (no countersink here)
        # If your project uses a different radius, adapt as needed.
        # We’ll fetch opening_dia from the doc’s XDATA if you store it there; otherwise
        # callers can draw the ring later again if they need a different radius.
        # To keep it robust, just draw center dots with a minimal radius if unknown.
        r = 0.25  # harmless marker if you prefer just centers; change if you want real hole sizes
        # If you want real hole sizes here, pass `opening_dia` into this helper and set r=opening_dia/2.

        added = 0
        for (x, y) in pts:
            msp.add_circle((float(x), float(y)), float(r),
                           dxfattribs={"layer": LAYER, "lineweight": 25})
            added += 1
        try:
            log_error(f"FRONT: holes drawn={added} on layer {LAYER}")
        except Exception:
            pass
        return added
    except Exception:
        # never break drawing if this fallback fails
        return 0

#------------------Button behaviour for Generate DXF, Autocad and PDF buttons ------------------------------------
import os

def _get_writable_tmp_dir():
    """
    Return a writable directory for transient files.
    Prefers APP_DIR, then /tmp, then CWD. Honors DXF_OUT_DIR if set.
    """
    candidates = [
        os.getenv("DXF_OUT_DIR"),
        os.getenv("APP_DIR"),
        "/tmp",
        os.getcwd(),
    ]
    for d in candidates:
        if not d:
            continue
        try:
            os.makedirs(d, exist_ok=True)
            test = os.path.join(d, ".wtest")
            with open(test, "wb") as f:
                f.write(b"ok")
            os.remove(test)
            return d
        except Exception:
            continue
    return os.getcwd()

def _gval(x, default=0.0):
    """
    Return a float from either a raw numeric/string or a Gradio component.
    Allows running before the new sliders are wired into the .click inputs.
    """
    try:
        return float(x)
    except Exception:
        try:
            return float(getattr(x, "value"))
        except Exception:
            return float(default)

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0

def _gen_pdf_link_path(calc_vals):
    """Generate PDF via cached exporter; return an update for gr.File."""
    import traceback
    try:
        upd = _export_pdf_from_cached(calc_vals=calc_vals)
        # Be defensive: if exporter returned a plain path/string, normalize to update
        if isinstance(upd, str):
            return gr.update(value=upd, visible=True)
        return upd
    except Exception:
        log_error("export_pdf_from_cached failed:\n" + traceback.format_exc())
        return gr.update(visible=False, value=None)

def _pdf_busy():
    # status HTML + disable the button while exporting
    return gr.update(value="<em>Generating PDF…</em>"), gr.update(interactive=False)

def _pdf_done_from_component(file_like):
    import os, gradio as gr
    path = None
    if isinstance(file_like, dict) and "name" in file_like:
        path = file_like["name"]
    elif isinstance(file_like, str):
        path = file_like
    ok = bool(path and os.path.exists(path))
    msg = "PDF ready" if ok else "PDF failed — check logs."
    return gr.update(value=msg), gr.update(interactive=ok)

def _pdf_clear_status():
    return gr.update(value="")

def _hide_pdf_link():
    import gradio as gr
    return gr.update(visible=False, value=None)

def _mark_ready_and_show_dxf(dxf_path: str):
    import gradio as gr
    ok = bool(dxf_path)  # fine to keep; you could also write `if dxf_path:`
    if ok:
        return (
            True,  # dxf_ready_state
            gr.update(value=dxf_path, visible=True, interactive=True),  # dxf_dl
            gr.update(visible=True, interactive=True),  # open_pdf_btn
            gr.update(visible=False),  # stale_banner
        )
    else:
        return (
            False,
            gr.update(visible=False, interactive=False),
            gr.update(visible=False, interactive=False),
            gr.update(visible=True, value="Need to click Generate"),
        )

def _mark_stale():

    return (
        False,  # dxf_ready_state
        gr.update(visible=False, interactive=False),  # dxf_dl
        gr.update(visible=False, interactive=False),  # open_pdf_btn
        gr.update(visible=True, value="Drawings need to be Generated again"),
    )

def on_make_drawings_hybrid(*args):
    """
    Works with binding where circseg_pts_state is the *last* input.
    Returns the same 8 outputs as on_make_drawings.
    """
    import gradio as gr, traceback

    # Split inputs: everything except last -> params, last -> kept_points
    if not args:
        params, raw_points = (), None
    else:
        params, raw_points = args[:-1], args[-1]

    def _normalize_points(v):
        # unwrap gr.State-like
        v = getattr(v, "value", v)

        # common dict shapes
        if isinstance(v, dict):
            v = v.get("points") or v.get("kept") or v.get("holes") or v.get("data")

        # pandas DataFrame
        try:
            import pandas as _pd
            if isinstance(v, _pd.DataFrame):
                cols = {c.lower(): c for c in v.columns}
                if "x" in cols and "y" in cols:
                    return [(float(x), float(y)) for x, y in zip(v[cols["x"]], v[cols["y"]])]
        except Exception:
            pass

        # numpy array
        try:
            import numpy as _np
            if isinstance(v, _np.ndarray) and v.ndim == 2 and v.shape[1] >= 2:
                return [(float(x), float(y)) for x, y in v[:, :2]]
        except Exception:
            pass

        # list/tuple of items
        out = []
        if isinstance(v, (list, tuple)):
            for item in v:
                # allow (x,y) / (x,y,r) / {"x":..,"y":..}
                try:
                    if isinstance(item, dict) and "x" in item and "y" in item:
                        out.append((float(item["x"]), float(item["y"])))
                    else:
                        x, y = item[0], item[1]
                        out.append((float(x), float(y)))
                except Exception:
                    continue
        return out or None

    kept_points = _normalize_points(raw_points)

    try:

        if kept_points:
            log_error(f"HYBRID: forwarding {len(kept_points)} kept points to holes_override.")
            plate_img, inset_img, dxf_path = generate_die_plate_dxf(*params, holes_override=kept_points, force_hybrid_tab3=True)
        else:
            log_error("HYBRID: no kept points provided; calling generator without holes_override.")
            plate_img, inset_img, dxf_path = generate_die_plate_dxf(*params, force_hybrid_tab3=True)

        if dxf_path:
            return (
                True,
                gr.update(value=dxf_path, visible=True, interactive=True),
                gr.update(visible=True, interactive=True),
                gr.update(visible=False),
                dxf_path,
                gr.update(value=plate_img, visible=True),
                gr.update(value=inset_img, visible=True),
                gr.update(value="DXF ready.", visible=True),
            )
        else:
            return (
                False,
                gr.update(visible=False, interactive=False),
                gr.update(visible=False, interactive=False),
                gr.update(visible=True, value="Drawing failed. Fix inputs then click **Generate**."),
                "",
                gr.update(), gr.update(),
                gr.update(value="DXF was not generated.", visible=True),
            )
    except Exception:
        log_error("on_make_drawings_hybrid failed:\n" + traceback.format_exc())
        return (
            False,
            gr.update(visible=False, interactive=False),
            gr.update(visible=False, interactive=False),
            gr.update(visible=True, value="Error generating hybrid design."),
            "",
            gr.update(), gr.update(),
            gr.update(value="DXF error.", visible=True),
        )

def on_make_drawings(*params):
    try:
        plate_img, inset_img, dxf_path = generate_die_plate_dxf(*params)
        if dxf_path:
            return (
                True,                           # dxf_ready_state
                gr.update(value=dxf_path, visible=True, interactive=True),  # dxf_dl (DownloadButton)
                gr.update(visible=True, interactive=True),    # open_pdf_btn
                gr.update(visible=False),       # stale_banner
                dxf_path,                       # last_dxf_path_st
                gr.update(value=plate_img, visible=True),  # plate_preview
                gr.update(value=inset_img, visible=True),  # inset_preview
                gr.update(value="DXF ready.", visible=True)  # pdf_status (or status_html)
            )
        else:
            return (
                False,
                gr.update(visible=False, interactive=False),
                gr.update(visible=False, interactive=False),
                gr.update(visible=True, value="Drawing failed. Fix inputs then click **Generate Drawings**."),
                "",
                gr.update(), gr.update(),
                gr.update(value="DXF was not generated.", visible=True),
            )
    except Exception:
        log_error("on_make_drawings failed:\n" + traceback.format_exc())
        return (
            False,
            gr.update(visible=False, interactive=False),
            gr.update(visible=False, interactive=False),
            gr.update(visible=True, value="Error while generating drawings. Check inputs and try again."),
            "",
            gr.update(), gr.update(),
            gr.update(value="DXF error.", visible=True),
        )

def on_generate_pdf(dxf_ready, vals_dict):
    """
    Generate the PDF from the cached DXF/doc.
    Guarded by dxf_ready so users can't click out of order.
    """
    import gradio as gr
    if not dxf_ready:
        return gr.update(value=None, visible=False), gr.update(value="DXF not ready.", visible=True)
    try:
        # use your cached exporter; it returns a gr.update(...) for the File
        pdf_update = _export_pdf_from_cached(calc_vals=vals_dict)
        # friendly status
        return pdf_update, gr.update(value="PDF ready.", visible=True)
    except Exception:
        log_error("on_generate_pdf failed:\n" + traceback.format_exc())
        return gr.update(value=None, visible=False), gr.update(value="PDF error.", visible=True)

# ---------- Safe coercers (for grid / UI values) ----------
def _g(v, default=None, *, unwrap_singleton=True, treat_none_strings=True):
    """
    Gentle getter:
      - Unwrap single-item [list]/(tuple) if enabled
      - Trim strings; treat "", "none", "nan", "null" as empty if enabled
      - Treat float/NumPy/pandas NaN as empty
    """
    # 1) unwrap [x] / (x,)
    if unwrap_singleton and isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]

    # 2) None
    if v is None:
        return default

    # 3) strings
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return default
        if treat_none_strings and s.lower() in {"none", "nan", "null"}:
            return default
        return s

    # 4) NaN (float / numpy / pandas)
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return default
    except Exception:
        pass
    try:
        import numpy as _np  # if available
        if isinstance(v, _np.floating) and _np.isnan(v):
            return default
    except Exception:
        pass

    return v

def _to_float(v, default=0.0, **kw):
    v = _g(v, default=None, **kw)          # unwrap, trim, handle "none"/"nan"
    try:
        return float(str(v).strip())
    except Exception:
        return default

def _to_int(v, default=0, *, mode="truncate", **kw):
    """
    mode: "truncate" (int(float(x))), "nearest" (round), "floor", "ceil"
    """
    v = _g(v, default=None, **kw)
    try:
        f = float(str(v).strip())
    except Exception:
        return default

    # Guard NaN / infinities without math
    if f != f or f == float("inf") or f == float("-inf"):
        return default

    if mode == "nearest":
        try:
            return int(round(f))  # ties-to-even
        except Exception:
            return default

    # int() truncates toward 0; adjust for floor/ceil manually
    try:
        i = int(f)
    except Exception:
        return default

    if mode == "floor":
        return i if (f == i or f >= 0.0) else (i - 1)
    elif mode == "ceil":
        return i if (f == i or f <= 0.0) else (i + 1)
    else:  # "truncate"
        return i

def _is_nan(x):
    try:
        import math
        if isinstance(x, float) and math.isnan(x):
            return True
    except Exception:
        pass
    try:
        import numpy as np
        return isinstance(x, np.floating) and np.isnan(x)
    except Exception:
        return False

# ===== Helpers: paths & tiny download buttons (single source of truth) =====
import os
import gradio as gr

_LAST_DXF_PATH = None

def _set_last_dxf_path(p):
    global _LAST_DXF_PATH
    _LAST_DXF_PATH = p

def _get_last_dxf_path():
    return _LAST_DXF_PATH

def _path_from_any(x):
    """
    Normalize exporter outputs:
      - string/PathLike -> string
      - gr.File dict    -> dict['name']
      - gr.update(...)  -> .value
    """
    if isinstance(x, (str, os.PathLike)):
        return str(x)
    if isinstance(x, dict):
        return x.get("name") or x.get("value")
    v = getattr(x, "value", None)  # handle gr.update(...)
    if isinstance(v, (str, os.PathLike)):
        return str(v)
    return None

def _show_button_for_path(x):
    p = _path_from_any(x)  # handles str, PathLike, dict{'name'|'value'}, gr.update
    ok = bool(p and os.path.exists(p))
    return gr.update(value=(p if ok else None), visible=ok)  # set value AND visibility

def _show_dxf_download(x):
    return _show_button_for_path(x)

def _status_from_path(x, ok_msg="Ready", fail_msg="Failed — check logs."):
    p = _path_from_any(x)
    ok = bool(p and os.path.exists(p))
    return gr.update(value=(ok_msg if ok else fail_msg)), gr.update(interactive=True)

def _dxf_status_from_path(x):
    return _status_from_path(x, "DXF ready", "DXF failed — check logs.")

def find_backside_inset_center(doc, plate_diameter):
    """
    Pick a hole on the *middle* BACK_HOLES ring, near angle 0° (right side).
    Robust against the backside being offset vertically.
    """
    import math
    msp = doc.modelspace()

    # Recompute the same backside offset used in add_backside_plate
    R = max(0.0, float(plate_diameter or 0.0)) / 2.0
    offset_y = -(2.0 * R + max(30.0, 0.15 * R))
    cx0, cy0 = 0.0, offset_y

    # Collect centers on BACK_HOLES
    pts = []
    for c in msp.query("CIRCLE[layer=='BACK_HOLES']"):
        x = float(c.dxf.center[0]); y = float(c.dxf.center[1])
        dx, dy = x - cx0, y - cy0
        r = math.hypot(dx, dy)
        ang = (math.degrees(math.atan2(dy, dx)) % 360.0)
        pts.append((r, ang, x, y))

    if not pts:
        return None

    # Cluster by radius to get rings
    pts.sort(key=lambda t: t[0])
    rings = []
    cur = []
    tol = 0.25  # mm tolerance for grouping
    for (r, ang, x, y) in pts:
        if not cur or abs(r - cur[-1][0]) <= tol:
            cur.append((r, ang, x, y))
        else:
            rings.append(cur); cur = [(r, ang, x, y)]
    if cur:
        rings.append(cur)

    # Choose the middle ring
    ring = rings[len(rings)//2]

    # Pick the hole closest to 0° (rightmost)
    ring.sort(key=lambda t: min(t[1], 360.0 - t[1]))
    _, _, x0, y0 = ring[0]
    return (float(x0), float(y0))
#-----Helper to ensure cone is absent if EITHER length or width is zero--
EPS_LEN = 1e-6   # you can make this 0.0 if you only want exact zeros
EPS_DIA = 1e-6

def _link_cone_len_and_width(cone_len: float, cone_top_dia: float):
    try:
        L = float(cone_len or 0.0)
        W = float(cone_top_dia or 0.0)
    except Exception:
        return 0.0, 0.0
    if L <= EPS_LEN or W <= EPS_DIA:
        return 0.0, 0.0
    return L, W

def _render_cross_section_png(
    opening_dia: float,
    channel_length: float,
    cone_length: float,
    cone_width: float,
    countersink_depth: float = 0.0,
    countersink_plus_radius: float = 0.0,
    countersink_diameter: float = 0.0,   # fallback if +radius not provided
    size_px=(360, 280),
) -> str:
    """
    Render a compact cross-section PNG as a data: URI.

    Countersink semantics:
      - If countersink_plus_radius > 0: draw an *angled countersink* (a wider cone)
        that starts at y=0 with width = cone_width + 2*plus and blends back into the
        main cone exactly at y=countersink_depth.
      - If countersink_plus_radius == 0 and countersink_depth > 0: draw a *straight
        counterbore* (vertical walls) down to that depth, then the main cone.
      - If countersink_depth == 0: just the main cone.

    Y increases downward.
    """
    import io, base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Normalize inputs
    od = max(0.0, float(opening_dia or 0.0))
    Lc = max(0.0, float(channel_length or 0.0))
    Lk = max(0.0, float(cone_length or 0.0))
    Wk = max(od,  float(cone_width or 0.0))

    # Prefer +radius; otherwise derive it from an absolute diameter if provided
    cs_plus = max(0.0, float(countersink_plus_radius or 0.0))
    if cs_plus <= 0.0 and countersink_diameter:
        cs_plus = max(0.0, 0.5 * (float(countersink_diameter) - Wk))

    cs_h = max(0.0, float(countersink_depth or 0.0))
    if Lk > 0.0:
        cs_h = min(cs_h, Lk)  # depth cannot exceed cone length
    else:
        cs_h = 0.0

    # Helper: width of the *main cone* at depth y (linear from Wk at 0 to od at Lk)
    def w_main(y: float) -> float:
        if Lk <= 0:
            return od
        t = max(0.0, min(1.0, y / Lk))
        return Wk + (od - Wk) * t

    # Top width of angled countersink (if any)
    w_top_cs = Wk + 2.0 * cs_plus

    total_len = Lk + Lc
    # Envelope to size the steel background
    half_W_env = 0.5 * max(Wk, w_top_cs, od)

    # Figure/axes
    dpi = 100.0
    fig_w = max(1, size_px[0]) / dpi
    fig_h = max(1, size_px[1]) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Inset-style axes filling most of the small figure
    box = ax.inset_axes([0.06, 0.10, 0.88, 0.80])
    box.set_aspect("equal")
    box.axis("off")

    # Steel background
    steel = mpatches.Rectangle(
        (-1.2 * half_W_env, 0.0),
        2.4 * half_W_env,
        total_len if total_len > 0 else 1.0,
        facecolor="#cccccc", edgecolor="none", zorder=1
    )
    box.add_patch(steel)

    # Build the polygon for the flow passage
    poly_x, poly_y = [], []

    if Lk > 0:
        if cs_h > 0 and cs_plus > 0:
            # --- ANGLED countersink: wider cone from y=0 to y=cs_h, then main cone to Lk
            w_join = w_main(cs_h)
            # left wall top -> join -> bottom; then right wall back to top
            poly_x = [-w_top_cs/2, -w_join/2, -od/2,  od/2,  w_join/2,  w_top_cs/2, -w_top_cs/2]
            poly_y = [0.0,         cs_h,      Lk,     Lk,    cs_h,      0.0,        0.0]
        elif cs_h > 0 and cs_plus == 0:
            # --- STRAIGHT counterbore: vertical walls down to cs_h, then main cone
            poly_x = [-Wk/2, -Wk/2, -od/2,  od/2,  Wk/2,  Wk/2, -Wk/2]
            poly_y = [0.0,   cs_h,  Lk,     Lk,    cs_h,  0.0,  0.0]
        else:
            # --- No countersink: a single liner taper from Wk at top to od at bottom
            poly_x = [-Wk/2, -od/2,  od/2,  Wk/2, -Wk/2]
            poly_y = [0.0,    Lk,     Lk,    0.0,  0.0]

        box.add_patch(
            mpatches.Polygon(
                list(zip(poly_x, poly_y)),
                closed=True, facecolor="#3366cc", edgecolor="#3366cc", zorder=2
            )
        )

    # Land/channel
    if Lc > 0 and od > 0:
        box.add_patch(
            mpatches.Rectangle(
                (-od/2, Lk), od, Lc,
                facecolor="#3366cc", edgecolor="#3366cc", zorder=2
            )
        )

    # Frame & caption
    pad = max(2.0, 0.10 * (total_len if total_len > 0 else 10.0))
    box.set_xlim(-half_W_env * 1.25, half_W_env * 1.25)
    box.set_ylim(total_len + pad, -pad)  # downward Y
    box.text(
        0.0, -0.06 * (total_len if total_len > 0 else 10.0),
        "Die cross-section",
        ha="center", va="top", fontsize=8, transform=box.transData, zorder=3,
    )

    # Encode PNG -> data URI
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"

#----Zoom Inset Preview Progress Spinner and Tunables-------------------

def _purge_back_center_circle(
    doc,
    *,
    plate_diameter,
    back_gap_min,
    back_gap_ratio,
    opening_dia=None,      # NEW: helps size the “small ghost” threshold
    tol_xy=0.6             # center tolerance in mm
):
    """
    Remove only stray small circles centered on the BACK plate center.
    Protects bolt/guide layers and any large PCD rings.
    """

    try:
        msp = doc.modelspace()
    except Exception:
        return

    # Back plate center Y (must match your backside offset math)
    R = max(0.0, float(plate_diameter or 0.0)) / 2.0
    y0 = -(2.0 * R + max(float(back_gap_min), float(back_gap_ratio) * R))

    # Radius threshold: treat “ghost” as ~opening-sized (with a little slack).
    if opening_dia is not None:
        max_small_r = max(0.5 * float(opening_dia) + 0.75, 6.0)   # e.g., Ø15 → r<=8.25 kept as ghost
    else:
        # Fallback: still small enough to never touch bolt PCDs (50–70 mm dia)
        max_small_r = 12.0  # radius in mm (≈Ø24)
    
    # Layers we will never purge, even if they’re at center
    PROTECT = {
        "BACK_INLET_PCD", "BACK_BOLTS", "BOLTS", "BOLTS_CS",
        "PCD_GUIDE", "BACK_OUTLINE"
    }

    removed = 0
    # iterate on a list() snapshot so we can destroy while iterating
    for e in list(msp.query("CIRCLE")):
        try:
            lyr = (e.dxf.layer or "").upper()
            r = float(e.dxf.radius)
            if lyr in PROTECT:
                if not (lyr == "BACK_INLET_PCD" and r <= max_small_r):
                    continue

            (cx, cy) = e.dxf.center[0:2]

            # Must be centered AND small to be considered a ghost
            if abs(float(cx)) <= tol_xy and abs(float(cy) - y0) <= tol_xy and r <= max_small_r:
                e.destroy()
                removed += 1
        except Exception:
            continue

    try:
        log_error(
            f"PURGE(back center): removed={removed}, y0={y0:.2f}, "
            f"tol={tol_xy:.2f}, max_small_r={max_small_r:.2f}"
        )
    except Exception:
        pass

# ---------- DXF constants & layer helpers (shared)  ----------

# Back/inlet layers (one source of truth)
L_BACK_OUTLINE    = "BACK_OUTLINE"
L_BACK_HOLES      = "BACK_HOLES"
L_BACK_CONE       = "BACK_CONE"
L_BACK_CS         = "BACK_CS"
L_BACK_BOLTS      = "BACK_BOLTS"
L_BACK_INLET_PCD  = "BACK_INLET_PCD"
L_BACK_TAPER      = "BACK_TAPER_PCD"   # grey, thin
L_PCD_GUIDE       = "PCD_GUIDE"
L_CHAMFER_PLATE   = "CHAMFER_PLATE"
L_CHAMFER_OUTER   = "CHAMFER_OUTER_OPEN"
L_CHAMFER_INNER   = "CHAMFER_INNER_OPEN"
L_CHAMFER_CENTER  = "CHAMFER_CENTER_HOLE"

CHAMFER_GREY_LAYERS = {L_CHAMFER_OUTER, L_CHAMFER_INNER, L_CHAMFER_CENTER}

def ensure_chamfer_layers(doc) -> None:
    """Create chamfer layers if missing; plate=black, others=grey."""
    def _ensure(name: str, color: int):
        try:
            doc.layers.get(name)  # exists? leave it
        except Exception:
            try:
                doc.layers.add(name, dxfattribs={"color": color})
            except Exception:
                pass

    # Plate must be black
    _ensure(L_CHAMFER_PLATE, 7)

    # Others grey
    for name in (L_CHAMFER_OUTER, L_CHAMFER_INNER, L_CHAMFER_CENTER):
        _ensure(name, 8)

    # If plate layer already existed but had the wrong color, fix it:
    try:
        doc.layers.get(L_CHAMFER_PLATE).dxf.color = 7
    except Exception:
        pass

def ensure_core_layers(doc) -> None:
    """Create core layers if missing. Tolerant across ezdxf versions."""
    try:
        # .get() exists on newer ezdxf; if it fails, we'll .add()
        _ = doc.layers.get
    except Exception:
        pass

    def _ensure(name: str, color: int) -> None:
        try:
            doc.layers.get(name)   # newer ezdxf
            return
        except Exception:
            pass
        try:
            doc.layers.add(name, dxfattribs={"color": color})
        except Exception:
            # very old ezdxf: try again defensively
            try:
                if name not in doc.layers:
                    doc.layers.add(name, dxfattribs={"color": color})
            except Exception:
                pass

    for name, color in [
        (L_BACK_OUTLINE,    7),
        (L_BACK_HOLES,      5),
        (L_BACK_CONE,       4),
        (L_BACK_CS,         6),
        (L_BACK_BOLTS,      7),
        (L_BACK_INLET_PCD,  7),
        (L_BACK_TAPER,      8),  # slightly grey
        (L_PCD_GUIDE,       8),
        (L_CHAMFER_PLATE,   7),
        
    ]:
        _ensure(name, color)

def ensure_ttf_text_style(doc) -> None:
    """Make the 'Standard' text style use a TTF Matplotlib can render."""
    try:
        try:
            st = doc.styles.get("Standard")
        except KeyError:
            st = doc.styles.add("Standard")
        try:
            st.dxf.font = "DejaVuSans.ttf"  # shipped with Matplotlib
            st.dxf.bigfont = ""             # no SHX bigfont
        except Exception:
            pass
    except Exception:
        # Never fail drawing because of a text style hiccup
        pass

def ent_attrs(layer_name: str, lw: int = 25, *, color: int = None) -> dict:
    """
    Standard DXF entity attributes:
      - layer + lineweight always
      - color forced to 1 (red) if FORCE_ENTITY_COLOR is True, unless an explicit color is passed
    """
    attrs = {"layer": layer_name, "lineweight": int(lw)}
    if color is None and globals().get("FORCE_ENTITY_COLOR", False):
        color = 1
    if color is not None:
        attrs["color"] = int(color)
    return attrs

def _circular_xy_from_rows(rows, *, start_deg_odd=0.0, start_deg_even=0.0):
    """
    Build (x,y) centers for circular rows, exactly like Tab 1:
      rows = [{"num_holes": N, "pcd": P}, ...]
    start_deg_* lets you mimic your row-wise angular offsets if you want.
    """
    import math
    pts = []
    rows_with_starts = _ensure_row_start_angles(rows or [], base_start_deg=start_deg_even)
    for idx, r in enumerate(rows_with_starts):
        n = int(r.get("num_holes", 0) or 0)
        p = float(r.get("pcd", 0.0) or 0.0)
        if n <= 0 or p <= 0.0:
            continue
        R = 0.5 * p
        start_deg = r.get("start_deg")
        if start_deg is None:
            start_deg = start_deg_odd if (idx % 2 == 1) else start_deg_even
        try:
            angle_offset_deg = float(start_deg)
        except Exception:
            angle_offset_deg = 0.0
        angle_offset = math.radians(angle_offset_deg)
        for k in range(n):
            a = 2.0 * math.pi * (k / n) + angle_offset
            pts.append((R * math.cos(a), R * math.sin(a)))
    return pts

#-----Tab 3 layout of holes for circular / segmented dies--------#

def _cull_points_by_segments_xy(
    pts_xy,
    *,
    segments: int,
    inner_pcd_mm: float,
    outer_pcd_mm: float,
    wall_width_mm: float,
    corner_radius_mm: float,
    padding_mm: float,
    rotation_deg: float = 0.0,
):
    """
    Point-only cull against segmented walls + annulus using pure math
    (no SegmentedBase dependency).
    """
    import math

    kept = []
    S = max(3, int(segments or 0))

    r_in_raw  = max(0.0, float(inner_pcd_mm or 0.0) / 2.0)
    r_out_raw = max(0.0, float(outer_pcd_mm or 0.0) / 2.0)

    pad = float(padding_mm or 0.0)
    fil = float(corner_radius_mm or 0.0)
    w   = float(wall_width_mm or 0.0)

    Rin  = r_in_raw  + pad + fil
    Rout = r_out_raw - pad - fil
    if Rout <= Rin:
        return []

    theta = 2.0 * math.pi / S
    req_wall = pad + 0.5 * w
    TOL = 1e-6

    rot = math.radians(float(rotation_deg or 0.0))

    for (x, y) in pts_xy or []:
        # apply segment-rotation if any
        xr =  x*math.cos(rot) - y*math.sin(rot)
        yr =  x*math.sin(rot) + y*math.cos(rot)

        r = math.hypot(xr, yr)
        if r < Rin + TOL or r > Rout - TOL:
            continue

        ang = math.atan2(yr, xr)
        ang = (ang + math.pi) % (2.0*math.pi) - math.pi

        # nearest segment midline
        s_float = ((ang / theta) + 0.5) % S
        s0 = int(round(s_float)) % S
        mid = (s0 + 0.5) * theta
        dev = ((ang - mid + math.pi) % (2.0*math.pi)) - math.pi

        delta_to_wall = (theta/2.0) - abs(dev)
        if delta_to_wall <= 0.0:
            continue

        perp = r * math.sin(delta_to_wall)
        if perp <= req_wall + TOL:
            continue

        kept.append((x, y))

    return kept

# --- Inset framing tunables ---
INSET_VIEW_SCALE       = 1.10
INITIAL_INSET_ZOOM     = 1.00
BACK_GAP_MIN           = 30.0
BACK_GAP_RATIO         = 0.15
BACK_CENTER_Y_NUDGE    = 0.0    # mm ( + = up )
BACK_CENTER_X_FACTOR   = 0.0    # fraction of plate_diameter ( + = right )
BACK_CENTER_X_NUDGE    = 0.0    # mm ( + = right )

# Focus behavior
INSET_FOCUS_MODE         = "auto"   # "auto" (holes_xy -> rows -> center), or "rows", or "holes", or "center"
INSET_FOCUS_ANGLE_DEG    = 0.0      # where on the ring to look (0° = +X/right)
INSET_FOCUS_X_NUDGE      = 0.0      # mm tweak at the end
INSET_FOCUS_Y_NUDGE      = 0.0      # mm tweak at the end (applied in inlet/back coordinates)

# How zoom blends from plate center -> focus center
INSET_FOCUS_BLEND_START  = 1.0      # zoom value where blending begins (t = 0)
INSET_FOCUS_BLEND_END    = 3.0      # zoom value where blending ends   (t = 1)

def _compute_focus_center(
    *,
    plate_diameter: float,
    holes_xy,                  # list[(x,y)] or None
    rows,                      # list[{"pcd": float, "num_holes": int}] or None
    back_offset_y: float,      # the inlet/back vertical offset you already compute
) -> tuple:
    """
    Returns an (x,y) on the inlet/back view to focus on.
    Strategy (auto):
      - If holes_xy present: use median of radii (robust if a few points stray).
      - Else if rows present: use middle PCD (avg of min & max).
      - Else: fall back to plate center on inlet/back (x=0, y=back_offset_y).
    Then place at angle INSET_FOCUS_ANGLE_DEG and add XY nudges.
    """
    import math
    mode   = str(globals().get("INSET_FOCUS_MODE", "auto")).lower()
    ang    = math.radians(float(globals().get("INSET_FOCUS_ANGLE_DEG", 0.0)))
    nx     = float(globals().get("INSET_FOCUS_X_NUDGE", 0.0))
    ny     = float(globals().get("INSET_FOCUS_Y_NUDGE", 0.0))

    # 1) radii candidates
    radii = []

    if mode in ("auto", "holes"):
        if holes_xy:
            try:
                radii = [math.hypot(x, y) for (x, y) in holes_xy]
                radii.sort()
            except Exception:
                radii = []

    if (not radii) and mode in ("auto", "rows"):
        try:
            pcs = [float(r.get("pcd", 0.0)) for r in (rows or []) if float(r.get("pcd", 0.0)) > 0.0]
            if pcs:
                rmin = min(pcs) * 0.5
                rmax = max(pcs) * 0.5
                rmid = 0.5 * (rmin + rmax)
                radii = [rmid]
        except Exception:
            pass

    # 2) choose radius
    if radii:
        # median if a list from holes; otherwise the single rmid
        if len(radii) > 2:
            mid = radii[len(radii)//2]
        else:
            mid = radii[0]
        rx = mid * math.cos(ang)
        ry = mid * math.sin(ang)
        x = rx + nx
        y = (ry + back_offset_y) + ny
        return (x, y)

    # 3) fallback: inlet plate center
    return (0.0 + nx, back_offset_y + ny)

# -- show a chip + temporarily disable the slider
def _inset_busy():
    return (
        gr.update(value="<div class='chip info'>Rendering inset…</div>"),
        gr.update(interactive=False),
    )

# -- clear chip + re-enable slider
def _inset_idle():
    return (
        gr.update(value=""),
        gr.update(interactive=True),
    )

def _reset_inset_zoom_control():
    # Clamp to your slider's [1.0, 15.0] range and guard against bad globals
    try:
        z = float(globals().get("INITIAL_INSET_ZOOM", 1.0))
    except Exception:
        z = 1.0
    z = max(1.0, min(15.0, z))
    return gr.update(value=z)

# ---- Button helpers (actions when waiting for images) --------------------------
def _render_target_holes(n):
    try:
        n_int = int(n or 0)
    except Exception:
        n_int = 0
    return (
        "<div style=\"display:flex;align-items:baseline;gap:10px;\">"
        "<span style=\"font-size:1.25rem;font-weight:600;\">Target number of holes:</span>"
        f"<span style=\"font-size:2rem;font-weight:800;line-height:1;\">{n_int:,}</span>"
        "</div>"
    )

def _btn_busy():
    import gradio as gr
    # disable DXF button + show status
    return (
        gr.update(interactive=False),
        gr.update(value="<em>Rendering DXF…</em>")
    )

def _btn_idle():
    import gradio as gr
    # re-enable DXF button + clear status
    return (
        gr.update(interactive=True),
        gr.update(value="")
    )

# --- last-known plate diameter for inset zoom handler ---
def _set_last_plate_diameter(pd):
    try:
        globals()["_LAST_PD"] = float(pd)
    except Exception:
        globals()["_LAST_PD"] = 0.0

def _get_last_plate_diameter():
    try:
        return float(globals().get("_LAST_PD", 0.0))
    except Exception:
        return 0.0

def _compute_back_center(plate_diameter, holes_xy,
                         back_gap_min, back_gap_ratio, y_nudge=0.0,
                         mode="auto"):
    """
    mode: "auto"  -> use holes centroid if available; else plate center
          "holes" -> always holes centroid (if none, falls back to plate center)
          "plate" -> always plate center
    """
    R = max(0.0, float(plate_diameter or 0.0)) / 2.0
    base_offset = -(2.0 * R + max(back_gap_min, back_gap_ratio * R)) + float(y_nudge)

    # plate center (before backside offset)
    plate_cx, plate_cy = 0.0, 0.0

    # holes centroid
    if holes_xy and len(holes_xy) > 0:
        hx = sum(x for x, _ in holes_xy) / float(len(holes_xy))
        hy = sum(y for _, y in holes_xy) / float(len(holes_xy))
    else:
        hx, hy = plate_cx, plate_cy

    if mode == "plate":
        cx, cy = plate_cx, plate_cy
    elif mode == "holes":
        cx, cy = hx, hy
    else:  # "auto"
        cx, cy = (hx, hy) if holes_xy else (plate_cx, plate_cy)

    return (float(cx), float(cy) + base_offset)

# ---- MTEXT helpers: split/join on DXF breaks or real newlines ----
import re

MTEXT_BR = r"\P"

def mtx_split(text: str):
    """Return logical lines (understands both MTEXT '\\P' and real newlines)."""
    return re.split(r"(?:\r?\n|\\P)", str(text or ""))

def mtx_join(lines):
    """Join lines with MTEXT break, trimming leading/trailing blanks nicely."""
    clean = [ln for ln in (lines or []) if ln is not None]
    # prevent accidental double blanks
    out = []
    last_blank = False
    for ln in clean:
        is_blank = (ln.strip() == "")
        if is_blank and last_blank:
            continue
        out.append(ln)
        last_blank = is_blank
    return MTEXT_BR.join(out).strip(MTEXT_BR)

# ---- Robust bolt-note normalizer (Inner/Outer only; MTEXT-safe; no segment text) ----
_BOLT_LINE_ANY = re.compile(r"^\s*(?:inner|outer|segment)\s+bolts?\b", re.I)

def _fmt1(x):
    try: return f"{float(x):.1f}"
    except Exception: return str(x)

def _valid_triplet(n, pcd, dia) -> bool:
    try:
        return int(n or 0) > 0 and float(pcd or 0) > 0 and float(dia or 0) > 0
    except Exception:
        return False

def apply_bolt_lines_mtext(
    text: str,
    *,
    inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
    outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
) -> str:
    """
    1) Remove ANY existing 'Inner/Outer/Segment bolts:' lines (anchored).
    2) Append fresh lines for Outer (first) and Inner (second) if valid.
    3) Returns MTEXT with '\\P' separators.
    """
    lines = mtx_split(text)

    # Strip all prior bolt-related lines so stale labels can't linger
    kept = [ln for ln in lines if not _BOLT_LINE_ANY.match(ln or "")]

    fresh = []
    if _valid_triplet(outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter):
        fresh.append(
            f"Outer bolts: {int(outer_bolt_count)} × Ø{_fmt1(outer_bolt_diameter)} on PCD {_fmt1(outer_bolt_pcd)} mm."
        )
    if _valid_triplet(inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter):
        fresh.append(
            f"Inner bolts: {int(inner_bolt_count)} × Ø{_fmt1(inner_bolt_diameter)} on PCD {_fmt1(inner_bolt_pcd)} mm."
        )

    if fresh:
        # keep non-bolt content, then add fresh bolt lines (Outer → Inner)
        return mtx_join(kept + ([""] if kept and kept[-1].strip() else []) + fresh)
    else:
        # no bolt lines at all
        return mtx_join(kept)
        
def write_notes_box(doc, plate_diameter_mm: float, text: str):
    """
    Remove old NOTES text (MTEXT/TEXT) and write one fresh MTEXT block.
    Uses your existing _add_summary_mtext / fallback.
    """
    try:
        msp = doc.modelspace()
        for e in list(msp.query("MTEXT[layer=='NOTES']")) + list(msp.query("TEXT[layer=='NOTES']")):
            try: e.destroy()
            except Exception: pass
    except Exception:
        pass

    try:
        _add_summary_mtext(doc, doc.modelspace(), float(plate_diameter_mm or 0.0), text)
    except Exception:
        import traceback
        log_error("write_notes_box: MTEXT failed, using fallback:\n" + traceback.format_exc())
        _add_summary_mtext_fallback(doc, float(plate_diameter_mm or 0.0), text)

#------ Backside segments with optional fillets (corner_radius) ----------------
def draw_backside_segments(
    doc,
    *,
    plate_diameter,
    die_style,
    segments,
    seg_inner_pcd,
    seg_outer_pcd,
    wall_width,
    corner_radius=0.0,           # when > 0, add fillets at r_in / r_out
    layer="BACK_SEGMENTS",
    force=False,
):
    """
    BACK / inlet view (STAGGERED):
      - Draw the two side walls per segment (constant width).
      - Trim ends at inner/outer PCD intersections.
      - If corner_radius > 0: add fillet arcs tangent to the side wall and the PCD.
        Otherwise: sharp corners (straight lines only).
    When ``force`` is True the walls are drawn even if ``die_style`` is not
    staggered (used for hybrid Tab 3 previews).
    """
    import math
    try:
        style = str(die_style or "").strip().lower()
        is_staggered = (die_style == DIE_STYLE_STAGGERED) or (style == "staggered")
        if not is_staggered and not force:
            return
        S = int(segments or 0)
        if S < 3:
            return

        r_in  = max(0.0, float(seg_inner_pcd or 0.0)) / 2.0
        r_out = max(0.0, float(seg_outer_pcd or 0.0)) / 2.0
        w     = max(0.0, float(wall_width    or 0.0))
        cr    = max(0.0, float(corner_radius or 0.0))
        if not (r_out > r_in and w > 0.0):
            return

        half_w = 0.5 * w
        msp = doc.modelspace()

        # Layer + light ref rings (optional)
        if layer not in doc.layers:
            doc.layers.add(layer, dxfattribs={"color": 8})
        BACK_SEG_RINGS = "BACK_SEG_RINGS"
        if BACK_SEG_RINGS not in doc.layers:
            doc.layers.add(BACK_SEG_RINGS, dxfattribs={"color": 8})

        # Backside offset (match add_backside_plate)
        Rplate   = max(0.0, float(plate_diameter or 0.0)) / 2.0
        offset_y = -(2.0 * Rplate + max(30.0, 0.15 * Rplate))
        def _add(x, y): return (x, y + offset_y)

        # Reference rings (thin)
        if r_in  > 0: msp.add_circle((0.0, offset_y), r_in,  dxfattribs={"layer": BACK_SEG_RINGS})
        if r_out > 0: msp.add_circle((0.0, offset_y), r_out, dxfattribs={"layer": BACK_SEG_RINGS})

        # Helpers
        def _add_line(p0, p1):
            msp.add_line(_add(*p0), _add(*p1), dxfattribs=ent_attrs(layer, 25))

        def _add_arc(center, radius, ang0_deg, ang1_deg):
            # draw the minor arc between ang0->ang1 (CCW); swap if >180°
            d = (ang1_deg - ang0_deg) % 360.0
            if d > 180.0:
                ang0_deg, ang1_deg = ang1_deg, ang0_deg
            msp.add_arc(_add(*center), radius, ang0_deg, ang1_deg, dxfattribs=ent_attrs(layer, 25))

        # Intersection of side line with circle r=R
        def line_circle_meet(R, theta, s):
            if R <= 0.0:
                return None
            ux, uy = math.cos(theta), math.sin(theta)
            tx, ty = -uy, ux
            beta = s * half_w
            if abs(beta) > R - 1e-9:
                return None
            alpha = math.sqrt(max(0.0, R*R - beta*beta))
            return (alpha*ux + beta*tx, alpha*uy + beta*ty)

        # OUTER fillet: internal tangency to outer PCD
        def fillet_outer(R, theta, s, r_c):
            if R <= 0.0 or r_c <= 0.0:
                return None
            ux, uy = math.cos(theta), math.sin(theta)
            tx, ty = -uy, ux
            beta   = s * half_w

            # center shift toward wedge interior
            gamma = beta + s * r_c
            Rf    = R - r_c                      # INTERNAL tangency (outer end)
            if Rf <= 0.0 or abs(gamma) > Rf - 1e-9:
                return None

            alpha = math.sqrt(max(0.0, Rf*Rf - gamma*gamma))
            Cx, Cy = (alpha*ux + gamma*tx, alpha*uy + gamma*ty)   # fillet center
            Lx, Ly = (alpha*ux +  beta*tx, alpha*uy +  beta*ty)   # tangency on line
            k = (R / Rf)
            Px, Py = (k*Cx, k*Cy)                                 # tangency on PCD
            return (Lx, Ly), (Px, Py), (Cx, Cy)

        # INNER fillet: external tangency to inner PCD (mirror of outer)
        def fillet_inner(R, theta, s, r_c):
            if R <= 0.0 or r_c <= 0.0:
                return None
            ux, uy = math.cos(theta), math.sin(theta)
            tx, ty = -uy, ux
            beta   = s * half_w

            # MIRROR of outer: flip lateral shift, external tangency
            gamma = beta + s * r_c
            Rf    = R + r_c                      # EXTERNAL tangency (inner end)
            if Rf <= 0.0 or abs(gamma) > Rf - 1e-9:
                return None

            alpha = math.sqrt(max(0.0, Rf*Rf - gamma*gamma))
            Cx, Cy = (alpha*ux + gamma*tx, alpha*uy + gamma*ty)   # fillet center
            Lx, Ly = (alpha*ux +  beta*tx, alpha*uy +  beta*ty)   # tangency on line
            k = (R / Rf)
            Px, Py = (k*Cx, k*Cy)                                 # tangency on inner PCD
            return (Lx, Ly), (Px, Py), (Cx, Cy)

        dtheta = (2.0 * math.pi) / S
        lines = arcs = 0

        # helper: safe arc add (uses your existing _add and _add_arc if present)
        def _arc_from_pts(C, r, Pstart, Pend):
            # compute start/end angles in degrees, arc drawn CCW by ezdxf
            a0 = math.degrees(math.atan2(Pstart[1]-C[1], Pstart[0]-C[0]))
            a1 = math.degrees(math.atan2(Pend[1]-C[1],  Pend[0]-C[0]))
            _add_arc(C, r, a0, a1)  # your existing helper adds layer/offset

        for seg in range(S):
            theta = seg * dtheta
            ux, uy = math.cos(theta), math.sin(theta)
            tx, ty = -uy, ux

            for s in (+1, -1):
                # default sharp endpoints (line–circle intersections)
                q_in  = line_circle_meet(r_in,  theta, s)
                q_out = line_circle_meet(r_out, theta, s)

                # only outer fillet for now (inner stays sharp)
                fin  = fillet_inner(r_in,  theta, s, cr) if cr > 0.0 else None
                fout = fillet_outer(r_out, theta, s, cr) if cr > 0.0 else None

                if q_in and q_out:
                	
                    p0 = fin[0]  if fin  else q_in
                    p1 = fout[0] if fout else q_out
                    _add_line(p0, p1); lines += 1

                    # draw fillets: line tangency -> circle tangency
                    if fin:
                        C = fin[2]
                        a0 = math.degrees(math.atan2(fin[1][1]-C[1], fin[1][0]-C[0]))  # from circle
                        a1 = math.degrees(math.atan2(fin[0][1]-C[1], fin[0][0]-C[0]))  # to line
                        _add_arc(C, cr, a0, a1); arcs += 1

                    if fout:
                        C = fout[2]
                        a0 = math.degrees(math.atan2(fout[0][1]-C[1], fout[0][0]-C[0]))  # from line
                        a1 = math.degrees(math.atan2(fout[1][1]-C[1], fout[1][0]-C[0]))  # to circle
                        _add_arc(C, cr, a0, a1); arcs += 1

                else:
                    # Fallback: trimmed straight line
                    beta = s * half_w
                    p_in  = (ux*r_in  + beta*tx, uy*r_in  + beta*ty)
                    p_out = (ux*r_out + beta*tx, uy*r_out + beta*ty)
                    _add_line(p_in, p_out); lines += 1

        try:
            mode = "filleted" if cr > 0.0 else "sharp"
            log_error(f"SEG: {mode} S={S} r_in={r_in:.2f} r_out={r_out:.2f} w={w:.2f}  lines={lines} arcs={arcs}")
        except Exception:
            pass

    except Exception:
        import traceback
        try:
            log_error("draw_backside_segments failed:\n" + traceback.format_exc())
        except Exception:
            pass

# ---- Unified countersink validator ------------------------------------------
def _normalize_countersink(
    *,
    opening_dia=None,
    cone_length=None,
    cone_width=None,
    cs_depth=None,
    cs_plus_radius=None,   # NEW UI: +mm to radius
    cs_diameter=None       # LEGACY: absolute diameter
):
    """
    Return (countersink_diameter_mm, countersink_depth_mm).

    Rules:
      - Depth is clamped to [0, cone_length].
      - If absolute diameter is provided, it WINS (legacy behavior).
      - Otherwise use: max(opening_dia, cone_width + 2*cs_plus_radius).
      - All inputs are coerced via _gval to handle raw numbers or Gradio components.
    """
    # Coerce
    od  = max(0.0, _gval(opening_dia, 0.0))
    Lk  = max(0.0, _gval(cone_length, 0.0))
    cw  = max(0.0, _gval(cone_width, 0.0)) if cone_width is not None else 0.0
    dep = max(0.0, _gval(cs_depth, 0.0))
    dep = min(dep, Lk)

    # Diameter: absolute wins; else derive from +radius
    if cs_diameter is not None:
        dia = max(od, _gval(cs_diameter, 0.0))
    else:
        plus_r = max(0.0, _gval(cs_plus_radius, 0.0))
        dia = max(od, cw + 2.0 * plus_r)

    return dia, dep

# ---- Back-compat shims (thin, non-duplicative) -------------------------------
def _validate_countersink_plus(
    opening_dia,
    cone_width,
    countersink_plus_radius,
    countersink_depth,
    cone_length,
) -> Tuple[float, float]:  # py3.8-safe
    """NEW UI path: user enters +mm to radius (countersink_plus_radius)."""
    return _normalize_countersink(
        opening_dia=opening_dia,
        cone_width=cone_width,
        cone_length=cone_length,
        cs_depth=countersink_depth,
        cs_plus_radius=countersink_plus_radius,
    )

def _validate_countersink(
    opening_dia,
    countersink_plus_radius=None,   # NEW UI path (preferred)
    countersink_depth=None,
    cone_length=None,
    cone_width=None,
    *,
    countersink_diameter=None,      # LEGACY absolute diameter (if given, wins)
) -> Tuple[float, float]:           # py3.8-safe
    """Return (cs_diameter_mm, cs_depth_mm); forwards to `_normalize_countersink`."""
    return _normalize_countersink(
        opening_dia=opening_dia,
        cone_length=cone_length,
        cone_width=cone_width,
        cs_depth=countersink_depth,
        cs_plus_radius=countersink_plus_radius,
        cs_diameter=countersink_diameter,
    )

from typing import Optional, Tuple

def add_detailed_inset(
    msp,
    *,
    opening_dia,
    cone_width,
    rows_data,
    inset_scale=1.8,
    holes_xy=None,
    # Prefer validated values when provided:
    cs_diameter=None,
    cs_depth=None,
    # Needed if we must derive CS from +radius (legacy UI) or clamp by cone
    cone_length=None,
    countersink_plus_radius=None,
    # Where to draw the inset (y is typically negative to go “below” plate)
    origin=(0.0, -150.0),
):
    """
    Draw the inlet-side 'detail' (hole, countersink, cone top) at a small scale,
    centered around `origin`. Returns the center used (x,y) so the preview can
    start there.
    """
    x0, y0 = float(origin[0]), float(origin[1])

    # 1) Resolve countersink once
    cone_len = float(cone_length or 0.0)
    cs_plus  = float(countersink_plus_radius or 0.0)
    if cs_diameter is None or cs_depth is None:
        cs_dia_res, cs_dep_res = _normalize_countersink(
            opening_dia=opening_dia,
            cone_length=cone_len,
            cone_width=cone_width,
            cs_depth=cs_depth,                  # may be None
            cs_plus_radius=cs_plus,             # may be 0.0
            cs_diameter=cs_diameter,            # may be None
        )
    else:
        cs_dia_res, cs_dep_res = float(cs_diameter), float(cs_depth)

    # 2) Layers (be explicit so zoom preview can find them reliably)
    if "INSET_DETAIL" not in msp.doc.layers:
        msp.doc.layers.add("INSET_DETAIL", dxfattribs={"color": 7})   # white
    if "INSET" not in msp.doc.layers:
        msp.doc.layers.add("INSET", dxfattribs={"color": 1})          # red

    # 3) Draw the three rings at origin (scaled visually by caller via zoom)
    try:
        # opening (channel)
        r_open = max(0.0, float(opening_dia or 0.0)) * 0.5
        if r_open > 0:
            msp.add_circle(center=(x0, y0), radius=r_open, dxfattribs={"layer": "INSET_DETAIL"})

        # countersink top diameter
        r_cs = max(0.0, float(cs_dia_res or 0.0)) * 0.5
        if r_cs > 0:
            msp.add_circle(center=(x0, y0), radius=r_cs, dxfattribs={"layer": "INSET_DETAIL"})
    except Exception:
        log_error("INSET_DETAIL ring draw failed:\n" + traceback.format_exc())

    # Optionally: sample a couple of inlet PCD guide arcs/circles if you like
    # (Keep them on INSET_DETAIL so they’re not left at 0,0 on PDF/DXF)
    # … omitted …

    return (x0, y0)

def create_dxf_doc(
    plate_diameter, opening_dia, cone_width,
    outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
    inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
    outer_opening_pcd, inner_opening_pcd, die_center_hole_diameter,
    chamfer_plate, chamfer_outer_opening, chamfer_inner_opening, chamfer_center_hole,
    rows_data, pellet_size, inset_scale,
    cone_length=None,
    countersink_plus_radius: float = 0.0,
    countersink_depth: float = 0.0,
    holes_xy=None,
    include_inset_samples: bool = False,
    **_,
):
    import ezdxf, math

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    try:
        doc.units = ezdxf.units.MM  # optional: make intent explicit
    except Exception:
        pass

    # Normalize cone presence
    cone_len_local, cone_top_local = _link_cone_len_and_width(cone_length, cone_width)
    has_cone = (cone_len_local > 0.0 and cone_top_local > 0.0)

    # If you’ll draw the countersink ring anywhere, compute the effective diameter/depth once:
    cs_dia_local, cs_depth_local = _normalize_countersink(
        opening_dia=opening_dia,
        cone_length=cone_len_local,
        cone_width=cone_top_local,
        cs_depth=countersink_depth,
        cs_plus_radius=countersink_plus_radius,  # use legacy cs_diameter=... here if needed
    )

    OUTLINE_LAYER   = "OUTLINE"
    HOLES_LAYER     = "HOLES"
    BOLTS_LAYER     = "BOLTS"
    PCD_GUIDE       = "PCD_GUIDE"
    INSET_OPENINGS  = "INSET_OPENINGS"
    INSET_CONE      = "INSET_CONE"
    INSET_CS        = "INSET_CS"

    for name, color in [
        (OUTLINE_LAYER,    7),
        (HOLES_LAYER,      5),
        (BOLTS_LAYER,      7),
        (PCD_GUIDE,        8),
        (INSET_OPENINGS,   5),
        (INSET_CONE,       4),
        (INSET_CS,         6),
    ]:
        # Use has_entry/new if available; otherwise your original check is fine
        try:
            if not doc.layers.has_entry(name):
                doc.layers.new(name, dxfattribs={"color": color})
        except Exception:
            if name not in doc.layers:
                doc.layers.add(name, dxfattribs={"color": color})

    # Plate outline + center
    R_plate  = max(0.0, float(plate_diameter or 0.0)) / 2.0
    R_center = max(0.0, float(die_center_hole_diameter or 0.0)) / 2.0
    if R_plate  > 0: msp.add_circle((0,0), R_plate,  dxfattribs={"layer": OUTLINE_LAYER})
    if R_center > 0: msp.add_circle((0,0), R_center, dxfattribs={"layer": OUTLINE_LAYER})

    # ---- Bolt rings (FRONT - KNIFE SIDE) ---------------------------------
    def _bolt_ring_front(count, pcd, dia, *, layer=BOLTS_LAYER):
        try:
            n = int(count or 0); p = float(pcd or 0.0); d = float(dia or 0.0)
        except Exception:
            return 0
        if n <= 0 or p <= 0.0 or d <= 0.0:
            return 0
        rad = p / 2.0
        rb  = d / 2.0
        added = 0
        for k in range(n):
            a = 2.0 * math.pi * (k / n)
            x, y = rad * math.cos(a), rad * math.sin(a)
            # lineweight ensures visibility in all viewers; color comes from layer
            msp.add_circle((x, y), rb, dxfattribs={"layer": layer, "lineweight": 25})
            added += 1
        return added

    fb = 0
    fb += _bolt_ring_front(outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter)
    fb += _bolt_ring_front(inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter)
    try:
        log_error(f"FRONT: bolt circles added = {fb} on layer {BOLTS_LAYER}")
    except Exception:
        pass

    # Opening PCD guides (optional)
    for pcd in (inner_opening_pcd, outer_opening_pcd):
        p = float(pcd or 0.0)
        if p > 0:
            msp.add_circle((0,0), p/2.0, dxfattribs={"layer": PCD_GUIDE})

    # Pellet openings (FRONT only)
    r_open = max(0.0, float(opening_dia or 0.0)) / 2.0

    if holes_xy and len(holes_xy) > 0:
        # explicit override provided AND non-empty
        for (x, y) in holes_xy:
            msp.add_circle(
                (x, y), r_open,
                dxfattribs={"layer": HOLES_LAYER, "lineweight": 25, "color": 5}
            )
    elif rows_data:
        # generate from rows (Tab 1 style)
        rows_for_front = _ensure_row_start_angles(rows_data)
        for row in rows_for_front:
            n = int(row.get("num_holes", 0))
            p = float(row.get("pcd", 0.0))
            if n <= 0 or p <= 0:
                continue
            rad = p / 2.0
            try:
                angle_offset = math.radians(float(row.get("start_deg", 0.0)))
            except Exception:
                angle_offset = 0.0
            for k in range(n):
                a = 2.0 * math.pi * (k / n) + angle_offset
                x, y = rad * math.cos(a), rad * math.sin(a)
                msp.add_circle(
                    (x, y), r_open,
                    dxfattribs={"layer": HOLES_LAYER, "lineweight": 25, "color": 5}
                )
    # else: neither override nor rows -> no holes (expected for some inputs)
    # ---- CHAMFER RINGS (front / knife side) ----
    try:
        ensure_chamfer_layers(doc)
    except Exception:
        for _nm in ("CHAMFER_PLATE","CHAMFER_OUTER_OPEN","CHAMFER_INNER_OPEN","CHAMFER_CENTER_HOLE"):
            try:
                if _nm not in doc.layers:
                    doc.layers.add(_nm, dxfattribs={"color": 8})
            except Exception:
                pass

    msp = doc.modelspace()

    def _ring_dia(base_d, chamfer, inward: bool) -> float:
        try:
            d = float(base_d or 0.0)
            c = float(chamfer or 0.0)
        except Exception:
            return 0.0
        if d <= 0.0 or c <= 0.0:
            return 0.0
        return max(0.0, (d - 2.0*c) if inward else (d + 2.0*c))

    def _add_chamfer_ring(center_xy, dia_mm, layer_name):
        try:
            dia = float(dia_mm or 0.0)
            if dia <= 0.0:
                return
            attrs = {"layer": layer_name, "lineweight": 13, "color": 256}  # ByLayer thin
            e = msp.add_circle(center_xy, dia / 2.0, dxfattribs=attrs)

            if layer_name in CHAMFER_GREY_LAYERS:
                # force a true grey for viewers that honor truecolor
                try:
                    from ezdxf.colors import rgb2int
                    e.dxf.true_color = rgb2int((128, 128, 128))
                except Exception:
                    try: e.dxf.color = 8
                    except Exception: pass
            else:
                # plate chamfer = black
                try: del e.dxf.true_color
                except Exception: pass
                try: e.dxf.color = 7
                except Exception: pass
        except Exception:
            pass

    _add_chamfer_ring((0.0, 0.0), _ring_dia(plate_diameter,           chamfer_plate,         inward=True),  L_CHAMFER_PLATE)
    _add_chamfer_ring((0.0, 0.0), _ring_dia(die_center_hole_diameter,  chamfer_center_hole,   inward=False), L_CHAMFER_CENTER)
    _add_chamfer_ring((0.0, 0.0), _ring_dia(outer_opening_pcd,         chamfer_outer_opening, inward=False), L_CHAMFER_OUTER)
    _add_chamfer_ring((0.0, 0.0), _ring_dia(inner_opening_pcd,         chamfer_inner_opening, inward=True),  L_CHAMFER_INNER)

    # --- Inset samples for preview / zoom (front preview only) ---
    inset_center = None
    if include_inset_samples:
        # Resolve countersink once so inset uses correct rings
        try:
            cs_dia_val, cs_dep_val = _normalize_countersink(
                opening_dia=opening_dia,
                cone_length=float(cone_length or 0.0),
                cone_width=cone_width,
                cs_depth=countersink_depth,
                cs_plus_radius=countersink_plus_radius,
            )
        except Exception:
            cs_dia_val, cs_dep_val = 0.0, 0.0

        inset_center = add_detailed_inset(
            msp,
            opening_dia=float(opening_dia or 0.0),
            cone_width=float(cone_width or 0.0),
            rows_data=rows_data,
            inset_scale=inset_scale,
            holes_xy=holes_xy,
            cs_diameter=cs_dia_val,                 # validated
            cs_depth=cs_dep_val,                    # validated
            cone_length=float(cone_length or 0.0),  # for fallback/clamp
            countersink_plus_radius=float(countersink_plus_radius or 0.0),
            origin=(0.0, -150.0),
        )

    return doc, inset_center

# ---------- PREVIEW RENDERING ------------------
def create_preview_image(
    doc, filename, mode="plate", plate_diameter=None,
    center=None, zoom=None, show_cone=False,
):
    """
    Render a lightweight PNG preview from DXF:
      - Honors layer visibility (plate vs inset; optional cone visibility)
      - Uses resolve_edgecolor(...) if available for friendlier colors
      - For 'inset' mode, culls offscreen entities for speed
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Arc
    from io import BytesIO
    from PIL import Image

    msp = doc.modelspace()
    context = "ui-plate" if mode == "plate" else "ui-inset"

    CHAMFER_LAYERS = {
        "CHAMFER_OUTER_OPEN",
        "CHAMFER_INNER_OPEN",
        "CHAMFER_CENTER_HOLE",
    }

    def _edge_and_lw(lyr: str):
        lyu = (lyr or "").upper()
        if lyu == "CHAMFER_PLATE":
            return (1.4, (0, 0, 0, 1))          # force black
        if lyu in CHAMFER_LAYERS:
            return (0.9, (0.55, 0.55, 0.55, 1)) # grey
        return (1.4, (0, 0, 0, 1))

    # ---- Layer visibility (prefer global helper if present; otherwise fallback here)
    def _hidden_layer(lyr: str) -> bool:
        try:
            return _is_hidden_layer(lyr, context=context, show_cone=show_cone)  # type: ignore[name-defined]
        except Exception:
            lyu = (lyr or "").upper()
            if context == "pdf":
                return False
            if not show_cone and lyu in {"CONE", "COUNTERSINK", "CONE_UNDERSIDE", "BACK_CONE", "BACK_CS"}:
                return True
            if context == "ui-plate" and lyu.startswith("INSET"):
                return True
            return False

    # -------------------------------------------------------------------------
    # Build a CULL window that matches the final camera (best), with a fallback
    # extents-based box if center is not provided. Keep `view` as a legacy box
    # for any downstream code that still checks it for framing.
    # -------------------------------------------------------------------------
    view = None        # legacy viewbox (xmin, xmax, ymin, ymax) used by older code
    cull_view = None   # the box we actually use for geometry culling

    if mode == "inset" and (center is not None) and (zoom is not None):
        # Use the SAME math as the final camera
        cx, cy = float(center[0]), float(center[1])
        fov  = max(10.0, float(plate_diameter or 0.0))       # field-of-view basis
        z    = max(1.0, float(zoom or 1.0))
        half = fov / (2.0 * z)
        pad  = max(1.0, 0.05 * half)                         # small safety margin
        cull_view = (cx - half - pad, cx + half + pad, cy - half - pad, cy + half + pad)
        # Also expose a legacy 'view' so older framing branches won't break
        view = (cx - half, cx + half, cy - half, cy + half)
        try:
            log_error(f"INSET CULL WINDOW: center=({cx:.1f},{cy:.1f}), fov={fov:.1f}, "
                      f"zoom={z:.2f}, half={half:.1f}, pad={pad:.1f}, "
                      f"cull={cull_view}")
        except Exception:
            pass

    elif mode == "inset" and (zoom is not None):
        # Fallback: derive a coarse window from INSET_/BACK_ extents
        ix0 = iy0 = float("+inf")
        ix1 = iy1 = float("-inf")

        def _upd_ix(b):
            nonlocal ix0, ix1, iy0, iy1
            x0, x1, y0, y1 = b
            if x0 < ix0: ix0 = x0
            if x1 > ix1: ix1 = x1
            if y0 < iy0: iy0 = y0
            if y1 > iy1: iy1 = y1

        for e in msp:
            try:
                lyr = (getattr(e.dxf, "layer", "") or "").upper()
                if not (lyr.startswith("INSET_") or lyr.startswith("BACK_")):
                    continue
                t = e.dxftype()
                if t == "CIRCLE":
                    c = e.dxf.center; r = float(e.dxf.radius)
                    _upd_ix((c[0] - r, c[0] + r, c[1] - r, c[1] + r))
                elif t == "ARC":
                    c = e.dxf.center; r = float(e.dxf.radius)
                    _upd_ix((c[0] - r, c[0] + r, c[1] - r, c[1] + r))  # coarse
                elif t in ("LWPOLYLINE", "POLYLINE"):
                    try:
                        pts = [(float(px), float(py)) for (px, py, *_rest) in e.get_points()]
                    except Exception:
                        try:
                            pts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in e.vertices]
                        except Exception:
                            pts = []
                    if len(pts) >= 2:
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        _upd_ix((min(xs), max(xs), min(ys), max(ys)))
                elif t == "LINE":
                    x0 = float(e.dxf.start[0]); y0 = float(e.dxf.start[1])
                    x1 = float(e.dxf.end[0]);   y1 = float(e.dxf.end[1])
                    _upd_ix((min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)))
            except Exception:
                continue

        if ix0 < ix1 and iy0 < iy1:
            icx = 0.5 * (ix0 + ix1)
            icy = 0.5 * (iy0 + iy1)
            span0 = max(ix1 - ix0, iy1 - iy0)
            base_span = max(80.0, span0 * 1.2)                # padding
            span = base_span / max(float(zoom or 1.0), 1e-6)
            view = (icx - span, icx + span, icy - span, icy + span)
            cull_view = view
            try:
                log_error(f"INSET CULL WINDOW (extents): center=({icx:.1f},{icy:.1f}), "
                          f"span0={span0:.1f}, base_span={base_span:.1f}, span={span:.1f}")
            except Exception:
                pass
        elif center is not None:
            # If we have a center but couldn't find extents, use the same math as above
            cx, cy = float(center[0]), float(center[1])
            fov  = max(10.0, float(plate_diameter or 0.0))
            z    = max(1.0, float(zoom or 1.0))
            half = fov / (2.0 * z)
            pad  = max(1.0, 0.05 * half)
            view = (cx - half, cx + half, cy - half, cy + half)
            cull_view = (cx - half - pad, cx + half + pad, cy - half - pad, cy + half + pad)

    def _outside(bbox, viewbox):
        if viewbox is None:
            return False
        bx0, bx1, by0, by1 = bbox
        vx0, vx1, vy0, vy1 = viewbox
        return (bx1 < vx0) or (bx0 > vx1) or (by1 < vy0) or (by0 > vy1)

    def _bb_circle(cx, cy, r):
        return (cx - r, cx + r, cy - r, cy + r)

    def _bb_line(x0, y0, x1, y1):
        return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    def _bb_poly(pts):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return (min(xs), max(xs), min(ys), max(ys))

    # --- collect simple geometry and extents (no text/dims)
    simple = []
    xmin = ymin = float("+inf")
    xmax = ymax = float("-inf")

    def _upd(x, y):
        nonlocal xmin, ymin, xmax, ymax
        if x < xmin: xmin = x
        if y < ymin: ymin = y
        if x > xmax: xmax = x
        if y > ymax: ymax = y

    # --- Collect with optional culling (only for inset)
    drop_tiny = (mode == "inset")
    if drop_tiny:
        # same offset as add_backside_plate
        R_plate = max(0.0, float(plate_diameter or 0.0)) / 2.0
        gap     = max(30.0, 0.15 * R_plate)
        back_y  = -(2.0 * R_plate + gap)

    for e in msp:
        try:
            lyr = getattr(e.dxf, "layer", "") or ""
            if _hidden_layer(lyr):
                continue

            t = e.dxftype()
            if t not in ("CIRCLE", "ARC", "LWPOLYLINE", "POLYLINE", "LINE"):
                continue

            try:
                color = resolve_edgecolor(e, doc, context="preview", mode=mode)
            except Exception:
                color = (0, 0, 0)

            if t == "CIRCLE":
                c = e.dxf.center
                r = float(e.dxf.radius)
                cx, cy = float(c[0]), float(c[1])

                # drop tiny center rings (preview-only)
                if drop_tiny and r <= 2.5 and (
                    (abs(cx) <= 1.0 and abs(cy) <= 1.0) or          # front center
                    (abs(cx) <= 1.0 and abs(cy - back_y) <= 1.0)    # back center
                ):
                    continue

                if cull_view and _outside(_bb_circle(cx, cy, r), cull_view):
                    continue

                simple.append(("CIRCLE", (cx, cy, r), color))
                _upd(cx - r, cy - r); _upd(cx + r, cy + r)

            elif t == "ARC":
                c = e.dxf.center
                r = float(e.dxf.radius)
                cx, cy = float(c[0]), float(c[1])
                if cull_view and _outside(_bb_circle(cx, cy, r), cull_view):
                    continue
                sa = float(getattr(e.dxf, "start_angle", 0.0))
                ea = float(getattr(e.dxf, "end_angle",   360.0))
                simple.append(("ARC", (cx, cy, r, sa, ea), color))
                _upd(cx - r, cy - r); _upd(cx + r, cy + r)

            elif t in ("LWPOLYLINE", "POLYLINE"):
                pts = []
                try:
                    pts = [(float(px), float(py)) for (px, py, *_rest) in e.get_points()]
                except Exception:
                    try:
                        pts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in e.vertices]
                    except Exception:
                        pts = []
                if len(pts) >= 2:
                    if cull_view and _outside(_bb_poly(pts), cull_view):
                        continue
                    simple.append(("POLY", pts, color))
                    for (x, y) in pts:
                        _upd(x, y)

            elif t == "LINE":
                x0 = float(e.dxf.start[0]); y0 = float(e.dxf.start[1])
                x1 = float(e.dxf.end[0]);   y1 = float(e.dxf.end[1])
                if cull_view and _outside(_bb_line(x0, y0, x1, y1), cull_view):
                    continue
                simple.append(("LINE", (x0, y0, x1, y1), color))
                _upd(x0, y0); _upd(x1, y1)

        except Exception:
            continue

    if not simple:
        try:
            log_error(f"preview: no drawable entities for mode={mode}, file={filename}")
        except Exception:
            pass

    # --- figure & draw
    fig, ax = plt.subplots(figsize=(6, 6))
    for kind, data, color in simple:
        if kind == "CIRCLE":
            cx, cy, r = data
            ax.add_patch(Circle((cx, cy), r, fill=False, linewidth=1.0, edgecolor=color))
        elif kind == "ARC":
            cx, cy, r, sa, ea = data
            ax.add_patch(Arc((cx, cy), 2*r, 2*r, angle=0.0, theta1=sa, theta2=ea,
                             linewidth=1.0, edgecolor=color))
        elif kind == "POLY":
            xs, ys = zip(*data)
            ax.plot(xs, ys, linewidth=1.0, color=color)
        elif kind == "LINE":
            x0, y0, x1, y1 = data
            ax.plot([x0, x1], [y0, y1], linewidth=1.0, color=color)

    def _frame_all():
        if xmin < xmax and ymin < ymax and (xmax - xmin) > 0 and (ymax - ymin) > 0:
            dx = xmax - xmin; dy = ymax - ymin
            pad = 0.05 * max(dx, dy)
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)
        else:
            ax.set_xlim(-100, 100); ax.set_ylim(-100, 100)
        ax.set_aspect("equal", adjustable="box")

    if mode == "plate" and plate_diameter:
        r = float(plate_diameter) / 2.0
        ax.set_xlim(-r * 1.1, r * 1.1)
        ax.set_ylim(-r * 1.1, r * 1.1)
        ax.set_aspect("equal", adjustable="box")
    elif mode == "inset" and view is not None:
        vx0, vx1, vy0, vy1 = view
        ax.set_xlim(vx0, vx1)
        ax.set_ylim(vy0, vy1)
        ax.set_aspect("equal", adjustable="box")
    else:
        _frame_all()
        
    # ---- FORCE VIEW FOR INSET (place AFTER drawing, BEFORE saving/return) ----
    try:
        if mode == "inset" and center is not None:
            cx, cy = float(center[0]), float(center[1])

            # 'plate_diameter' here is the field-of-view basis you pass in
            fov  = max(10.0, float(plate_diameter or 0.0))
            z    = max(1.0, float(zoom or 1.0))
            half = fov / (2.0 * z)

            # Lock camera to requested window
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_aspect("equal", adjustable="box")
            ax.margins(0)  # keep edges tight

            # IMPORTANT: do not call ax.autoscale() or ax.relim() after this
            try:
                log_error(
                    f"INSET VIEW FORCED: center=({cx:.1f},{cy:.1f}), "
                    f"fov={fov:.1f}, zoom={z:.2f}, half={half:.1f}, "
                    f"xlim={ax.get_xlim()}, ylim={ax.get_ylim()}"
                )
            except Exception:
                pass
    except Exception:
        # Never fail the preview if plotting limits error out
        pass

    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def _row_angle_offset(idx: int, rows: list) -> float:
    """Offset odd rows by half the pitch of the previous (inner) row."""
    if idx % 2 == 1 and idx - 1 >= 0:
        try:
            prev_n = int((rows[idx - 1] or {}).get("num_holes", 0) or 0)
        except Exception:
            prev_n = 0
        return (180.0 / prev_n) if prev_n > 0 else 0.0
    return 0.0


def _ensure_row_start_angles(rows, *, base_start_deg: float = 0.0):
    """Return a new list of row dicts where each row has a ``start_deg`` value.

    The first valid row starts at ``base_start_deg`` (default 0°). Every
    subsequent valid row is rotated half of the previous valid row's angular
    pitch so adjacent rows are staggered. If a row already defines ``start_deg``
    it is preserved. Rows with zero holes (or missing data) are passed through
    unchanged.
    """

    assigned = []
    last_start = None
    last_count = 0

    for row in list(rows or []):
        # Work on a shallow copy so callers' data isn't mutated unexpectedly.
        row_dict = dict(row or {})

        try:
            count = int(row_dict.get("num_holes") or row_dict.get("count") or 0)
        except Exception:
            count = 0

        if count <= 0:
            assigned.append(row_dict)
            continue

        start = row_dict.get("start_deg")
        if start is None:
            base = last_start if last_start is not None else base_start_deg
            if last_count > 0:
                offset = 180.0 / float(last_count)
                start = base + offset
            else:
                start = base_start_deg
        else:
            try:
                start = float(start)
            except Exception:
                start = base_start_deg

        start = float(start % 360.0)
        row_dict["start_deg"] = start
        assigned.append(row_dict)

        last_start = start
        last_count = count

    return assigned

    # --- Layers (your list kept as-is) ---
    BACK_OUTLINE   = "BACK_OUTLINE"
    BACK_HOLES     = "BACK_HOLES"
    BACK_CONE      = "BACK_CONE"
    BACK_CS        = "BACK_CS"
    BACK_BOLTS     = "BACK_BOLTS"
    PCD_GUIDE      = "PCD_GUIDE"
    BACK_INLET_PCD = "BACK_INLET_PCD"

    for name, color in [
        (BACK_OUTLINE,   7), (BACK_HOLES, 5), (BACK_CONE, 4), (BACK_CS, 6),
        (BACK_BOLTS,     7), (PCD_GUIDE,  8), (BACK_INLET_PCD, 7),
    ]:
        if name not in doc.layers:
            doc.layers.add(name, dxfattribs={"color": color})

def add_backside_plate(
    doc,
    *,
    plate_diameter,
    opening_dia,
    cone_width,
    rows_data,
    die_center_hole_diameter,
    outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
    inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
    holes_xy=None,
    # (drop outer_opening_pcd / inner_opening_pcd entirely if unused)
    die_style=None,
    segments=None,
    seg_inner_pcd=None,
    seg_outer_pcd=None,
    segment_wall_width=None,
    corner_radius_mm=0.0,
    force_segment_walls=False,
    cone_length=None,
    # inlet-side rings:
    inlet_inside_pcd=0.0,
    inside_taper_pcd=0.0,
    inlet_outside_pcd=0.0,
    outside_taper_pcd=0.0,
    # countersink:
    cs_diameter=None,
    cs_depth=None,
    **_extras,             # ← absorb any stray kwargs safely
):
    """
    Draw the BACK (inlet) view: outline, bolts, pellet holes, cone/CS rings,
    optional segment walls, and inlet PCD rings. This must mirror the
    backside offset used elsewhere (match preview math).
    """
    import math

    ensure_core_layers(doc)

    msp = doc.modelspace()    

    # Backside offset (MUST match your preview/_back_offset_y logic)
    R_plate  = max(0.0, float(plate_diameter or 0.0)) / 2.0
    gap      = max(30.0, 0.15 * R_plate)
    offset_y = -(2.0 * R_plate + gap)

    def _add(x, y):
        return (float(x), float(y) + offset_y)

    # ---- Outline + center hole on BACK ----
    if R_plate > 0.0:
        msp.add_circle(_add(0.0, 0.0), R_plate, dxfattribs={"layer": L_BACK_OUTLINE})
    R_center = max(0.0, float(die_center_hole_diameter or 0.0)) / 2.0
    if R_center > 0.0:
        msp.add_circle(_add(0.0, 0.0), R_center, dxfattribs={"layer": L_BACK_OUTLINE})

    # ---- Bolts on BACK ----
    def _bolt_ring_back(count, pcd, dia):
        try:
            n  = int(count or 0)
            rp = float(pcd or 0.0) / 2.0
            r  = float(dia or 0.0) / 2.0
        except Exception:
            return 0
        if n <= 0 or rp <= 0.0 or r <= 0.0:
            return 0

        attrs = ent_attrs(L_BACK_BOLTS, 25)  # unified helper

        for k in range(n):
            ang = 2.0 * math.pi * (k / n)
            x = rp * math.cos(ang)
            y = rp * math.sin(ang)
            msp.add_circle(_add(x, y), r, dxfattribs=attrs)
        return n

    try:
        _bolt_ring_back(outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter)
        _bolt_ring_back(inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter)
    except Exception:
        pass
    # ---- Pellet holes + cone + countersink on BACK ----
    r_open   = max(0.0, float(opening_dia or 0.0)) / 2.0
    has_cone = (float(cone_length or 0.0) > 0.0) and (float(cone_width or 0.0) > 0.0)
    r_cone   = (float(cone_width or 0.0) / 2.0) if has_cone else 0.0

    # Resolve countersink (prefer validated inputs if provided)
    try:
        cs_dia_eff, cs_dep_eff = _normalize_countersink(
            opening_dia=opening_dia,
            cone_length=cone_length,
            cone_width=cone_width,
            cs_depth=cs_depth,
            cs_plus_radius=None,          # if using +radius elsewhere, pass it in instead
            cs_diameter=cs_diameter,      # absolute wins if given
        )
    except Exception:
        cs_dia_eff, cs_dep_eff = (float(cs_diameter or 0.0), float(cs_depth or 0.0))
    r_cs = max(0.0, float(cs_dia_eff or 0.0)) / 2.0
    has_cs = has_cone and (r_cs > 0.0)

    def _draw_back_hole(x, y):
        if r_open > 0.0:
            msp.add_circle(_add(x, y), r_open, dxfattribs={"layer": L_BACK_HOLES})
        if has_cone and r_cone > 0.0:
            msp.add_circle(_add(x, y), r_cone, dxfattribs={"layer": L_BACK_CONE})
        if has_cs and (r_cs > r_cone + 1e-9):
            msp.add_circle(_add(x, y), r_cs, dxfattribs={"layer": L_BACK_CS})

    if holes_xy:
        for (x, y) in holes_xy:
            _draw_back_hole(float(x), float(y))
    else:
        rows_for_back = _ensure_row_start_angles(rows_data)
        for row in rows_for_back:
            try:
                n = int((row or {}).get("num_holes", 0) or 0)
                p = float((row or {}).get("pcd", 0.0) or 0.0)
            except Exception:
                continue
            if n <= 0 or p <= 0.0:
                continue
            rad = p / 2.0
            try:
                angle_offset = math.radians(float((row or {}).get("start_deg", 0.0)))
            except Exception:
                angle_offset = 0.0
            for k in range(n):
                a = 2.0 * math.pi * (k / n) + angle_offset
                x = rad * math.cos(a); y = rad * math.sin(a)
                _draw_back_hole(x, y)

    # ---- Optional: segment walls on BACK for staggered style ----
    try:
        style = str(die_style or "").strip().lower()
        use_staggered = (die_style == DIE_STYLE_STAGGERED) or (style == "staggered")
        force_segments = bool(force_segment_walls)
        if (use_staggered or force_segments) and "draw_backside_segments" in globals():
            S   = int(segments or 0)
            rin = float(seg_inner_pcd or 0.0)
            rout= float(seg_outer_pcd or 0.0)
            w   = float(segment_wall_width or 0.0)
            cr  = float(corner_radius_mm or 0.0)
            if S >= 3 and rout > rin > 0.0 and w > 0.0:
                draw_backside_segments(
                    doc,
                    plate_diameter=plate_diameter,
                    die_style="Staggered" if use_staggered else str(die_style or ""),
                    segments=S,
                    seg_inner_pcd=rin,
                    seg_outer_pcd=rout,
                    wall_width=w,
                    corner_radius=cr,
                    force=(force_segments and not use_staggered),
                )
    except Exception:
        pass

    # ---- Inlet & Taper PCD rings (split across layers) ----
    def _add_inlet_ring(d):
        """Bold/black inlet rings on BACK_INLET_PCD."""
        try:
            r = max(0.0, float(d or 0.0)) / 2.0
            if r <= 0.0:
                return
            e = msp.add_circle(
                _add(0.0, 0.0), r,
                dxfattribs={"layer": L_BACK_INLET_PCD, "lineweight": 25}
            )
            # Force black in DXF viewers that honor truecolor
            try:
                from ezdxf.colors import rgb2int as _rgb2int
                e.dxf.true_color = _rgb2int((0, 0, 0))
            except Exception:
                pass
        except Exception:
            pass

    def _add_taper_ring(d):
        """Thin/grey taper rings on BACK_TAPER_PCD."""
        try:
            r = max(0.0, float(d or 0.0)) / 2.0
            if r <= 0.0:
                return
            e = msp.add_circle(
                _add(0.0, 0.0), r,
                dxfattribs={"layer": L_BACK_TAPER, "lineweight": 13}
            )
            # Gentle grey for preview/PDF-capable viewers
            try:
                from ezdxf.colors import rgb2int as _rgb2int
                e.dxf.true_color = _rgb2int((128, 128, 128))
            except Exception:
                pass
        except Exception:
            pass

    # inlet (bold/black)
    _add_inlet_ring(inlet_inside_pcd)
    _add_inlet_ring(inlet_outside_pcd)

    # taper (thin/grey)
    _add_taper_ring(inside_taper_pcd)
    _add_taper_ring(outside_taper_pcd)
 
def generate_die_plate_dxf(
    pellet_size, plate_diameter, opening_dia, cone_width,
    chamfer_plate, chamfer_outer_opening, chamfer_inner_opening, chamfer_center_hole,
    outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
    inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
    outer_opening_pcd, inner_opening_pcd, die_center_hole_diameter,
    inset_zoom, inlet_inside, inside_taper, inlet_outside, outside_taper,
    throughput, calc_values_state,
    holes, staggered_pts_state, die_style_state,
    # NEW staggered-only UI:
    stag_segments, seg_bolts_chk,
    # Existing circular / geometry inputs:
    segments, wall_width, seg_inner_pcd, seg_outer_pcd,
    corner_radius, padding_adj,
    *row_inputs,
    **kwargs,
):
    import math, traceback, os
    from ezdxf.math import Matrix44
    
    global cached_doc, cached_filename, cached_inset_center, cached_inset_focus_center, cached_export_path

    # --- init calc store ---
    cv = calc_values_state or {}

    def _g(x, d=0.0):
        try:
            return float(x)
        except Exception:
            try:
                return float(getattr(x, "value"))
            except Exception:
                return float(d)

    # Keep cv fresh even if Calculate wasn’t clicked
    cv["inlet_inside_pcd"]  = _g(inlet_inside,  0.0)
    cv["inside_taper_pcd"]  = _g(inside_taper,  0.0)
    cv["inlet_outside_pcd"] = _g(inlet_outside, 0.0)
    cv["outside_taper_pcd"] = _g(outside_taper, 0.0)

    def _coerce_style(v):
        # Accept ints or strings like "Circular"/"Staggered"
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
        s = str(v).strip().lower()
        if s.startswith("stag"):
            return DIE_STYLE_STAGGERED
        return DIE_STYLE_CIRCULAR
    
    force_hybrid_tab3 = bool(kwargs.get("force_hybrid_tab3", False))
    tab3_show_segment_walls = bool(kwargs.get("tab3_show_segment_walls", True))

    die_style = _coerce_style(die_style_state)
    IS_STAGGERED = (die_style == DIE_STYLE_STAGGERED)

    # ---- Tab 3 override: circular + segmented walls (hybrid) ----
    if force_hybrid_tab3:
        die_style = DIE_STYLE_CIRCULAR
        IS_STAGGERED = False
        # IMPORTANT: use Tab 3's segment slider for all downstream code that reads S
        S = int(stag_segments or 0) or int(segments or 0)
        use_rows = True
    else:
        # Choose segment count from the right input for the current style
        if IS_STAGGERED:
            S = int(stag_segments or 0) or int(segments or 0)  # fallback safety
            use_rows = False  # rows ignored for staggered
        else:
            S = int(segments or 0)
            use_rows = True   # rows only for circular

    IS_HYBRID = bool(force_hybrid_tab3) and (int(stag_segments or 0) >= 3) and (
        float(seg_inner_pcd or 0.0) > 0.0 or float(seg_outer_pcd or 0.0) > 0.0)

    WILL_HAVE_SEGMENTS = IS_STAGGERED or IS_HYBRID
    draw_segment_walls = WILL_HAVE_SEGMENTS
    if force_hybrid_tab3 and not tab3_show_segment_walls:
        draw_segment_walls = False

    try:
        log_error(
            f"DIE STYLE: tab3_force={force_hybrid_tab3} "
            f"style={die_style} IS_STAGGERED={IS_STAGGERED} "
            f"IS_HYBRID={IS_HYBRID} S={S} use_rows={use_rows}"
        )
    except Exception:
        pass

    # Read tunables (defined at module scope, with safe fallbacks)
    inset_view_scale    = float(globals().get("INSET_VIEW_SCALE", 0.75))
    initial_inset_zoom  = float(globals().get("INITIAL_INSET_ZOOM", 1.00))
    back_gap_min        = float(globals().get("BACK_GAP_MIN", 30.0))
    back_gap_ratio      = float(globals().get("BACK_GAP_RATIO", 0.15))
    back_center_y_nudge = float(globals().get("BACK_CENTER_Y_NUDGE", 0.0))
    back_center_x_factor   = float(globals().get("BACK_CENTER_X_FACTOR", 0.0))
    back_center_x_nudge    = float(globals().get("BACK_CENTER_X_NUDGE", 0.0))

    # --- parse circular rows only when needed ---
    def _form_rows_from_inputs(row_inputs_tuple):
        rows = []
        for nh, pcd in zip(row_inputs_tuple[0::2], row_inputs_tuple[1::2]):
            try:
                n = int(nh or 0)
                p = float(pcd or 0.0)
            except Exception:
                continue
            if n > 0 and p > 0:
                rows.append({"num_holes": n, "pcd": p})
        try:
            log_error(f"ROWS DEBUG: len(row_inputs)={len(row_inputs_tuple)}, formed pairs={len(rows)}")
        except Exception:
            pass
        return _ensure_row_start_angles(rows)

    rows_data = [] if not use_rows else _form_rows_from_inputs(row_inputs)

    # --- optional holes override (e.g., from Tab 3) ---
    holes_xy = None
    stag_report = {}

    holes_override = kwargs.get("holes_override", None)
    if holes_override:
        try:
            holes_xy = [(float(x), float(y)) for (x, y, *_) in holes_override]
            log_error(f"OVERRIDE: using {len(holes_xy)} pre-culled holes (holes_override).")
        except Exception:
            log_error("OVERRIDE: invalid holes_override payload; ignoring.")
            holes_xy = None

    # --- pack/generate staggered points when requested (Tab 2) ---
    if holes_xy is None and IS_STAGGERED:
        desired_total = int(holes or 0)

        # Largest effective radius for clearance (packer uses max of opening / cone / countersink)
        open_dia = float(opening_dia or 0.0)
        cone_dia = float(cone_width  or 0.0)  # treat 'cone_width' as top diameter here
        cs_dia_for_pack = 0.0                  # packer doesn't need chamfer CS here; leave as 0

        try:
            holes_xy, stag_report = pack_segment_lattice(
                total_holes     = desired_total,
                segments        = int(S or 0),
                inner_pcd       = float(seg_inner_pcd or 0.0),
                outer_pcd       = float(seg_outer_pcd or 0.0),
                opening_dia     = open_dia,
                countersink_dia = cs_dia_for_pack,
                cone_dia        = cone_dia,
                padding_mm      = float(padding_adj or 0.0),
                wall_width_mm   = float(wall_width or 0.0),
                corner_radius_mm= float(corner_radius or 0.0),
            )
        except Exception:
            holes_xy, stag_report = [], {"hit_target": False, "notes": "lattice solver error"}

        # Staggered debug: geometric safety margins
        try:
            if holes_xy:
                r_h = 0.5 * max(0.0, open_dia, cone_dia, cs_dia_for_pack)
                pad = float(padding_adj or 0.0)
                fil = float(corner_radius or 0.0)
                Rin  = float(seg_inner_pcd or 0.0) / 2.0 + pad + fil + r_h
                Rout = float(seg_outer_pcd or 0.0) / 2.0 - pad - fil - r_h
                half_w_eff = 0.5 * float(wall_width or 0.0) - (pad + r_h)

                def _beta_for_seg(x, y, s):
                    theta = 2.0 * math.pi * s / max(1, S)
                    return -math.sin(theta) * x + math.cos(theta) * y

                min_wall_gap = min(
                    min(half_w_eff - abs(_beta_for_seg(x, y, s)) for s in range(max(1, S)))
                    for (x, y) in holes_xy
                ) if holes_xy else 0.0
                min_radial_gap = min(
                    min(abs(math.hypot(x, y) - Rin), abs(Rout - math.hypot(x, y)))
                    for (x, y) in holes_xy
                ) if holes_xy else 0.0

                log_error(
                    f"STAG GEN: pitch={stag_report.get('pitch_mm', 0.0):.3f} mm, "
                    f"holes={len(holes_xy)}, "
                    f"min_wall_gap={min_wall_gap:.3f} mm, "
                    f"min_radial_gap={min_radial_gap:.3f} mm"
                )
        except Exception:
            log_error("staggered debug logging failed:\n" + traceback.format_exc())

        # Expose a few things for the UI/debug
        cv.update({
            "stag_target_per_seg": stag_report.get("target_seg", 0),
            "stag_placed_per_seg": stag_report.get("placed_seg", 0),
            "stag_pitch_mm":       stag_report.get("pitch_mm", 0.0),
            "stag_hit_target":     stag_report.get("hit_target", False),
            "stag_note":           stag_report.get("notes", ""),
        })

    # --- segment-bolts: only for staggered AND S==3 ---
    seg_bolts_on = bool(seg_bolts_chk) and IS_STAGGERED and (S == 3)
    if bool(seg_bolts_chk) and IS_STAGGERED and (S != 3):
        # user ticked the box but S!=3 → disable silently and log
        seg_bolts_on = False
        try:
            log_error(f"SEG BOLTS: disabled because S={S} (requires S==3)")
        except Exception:
            pass

    # --- HYBRID (Tab 3): circular rows culled by segments, only if no override ---
    if holes_xy is None and IS_HYBRID:
        try:
            # ---- Prefer Tab 3 rows if HYBRID and rows_data empty ----
            # ---- If HYBRID and rows_data is empty, hydrate from Tab 3 capture ----
            if IS_HYBRID and not rows_data:
                t3_flat   = cv.get("t3_rows_flat") or []
                n_rows_t3 = int(cv.get("t3_n_rows") or (len(t3_flat) // 2))
                if t3_flat and n_rows_t3 > 0:
                    t3_rows   = _t3_rows_from_components(n_rows_t3, *t3_flat)
                    rows_data = _ensure_row_start_angles([
                        {
                            "num_holes": r.get("num_holes") or r.get("count", 0),
                            "pcd": r["pcd"],
                            "start_deg": r.get("start_deg"),
                        }
                        for r in t3_rows
                    ])
                    try:
                        log_error(f"T3 ROWS HYDRATE: using {len(rows_data)} rows from cv (flat={t3_flat})")
                    except Exception:
                        pass
            # 1) exact Tab-1 style points
            pts_xy = _circular_xy_from_rows(rows_data)

            # If there are no points, there’s nothing to cull
            if not pts_xy:
                holes_xy = []
            else:
                # 2) derive annulus if UI left it at 0
                row_pcds = [float(r.get("pcd", 0.0) or 0.0) for r in (rows_data or []) if float(r.get("pcd", 0.0) or 0.0) > 0]
                pcd_min = min(row_pcds) if row_pcds else 0.0
                pcd_max = max(row_pcds) if row_pcds else 0.0

                inner_pcd_mm = float(seg_inner_pcd or 0.0)
                outer_pcd_mm = float(seg_outer_pcd or 0.0)

                if inner_pcd_mm <= 0.0 and pcd_min > 0.0:
                    # make inner just below the smallest row
                    inner_pcd_mm = max(0.0, pcd_min - 2.0)   # 2 mm margin; adjust if you prefer

                if outer_pcd_mm <= 0.0 and pcd_max > 0.0:
                    # make outer just above the largest row
                    outer_pcd_mm = pcd_max + 2.0             # 2 mm margin; adjust if you prefer

                S_eff = max(3, int(stag_segments or S or 0))

                # --- Build annulus from UI or row PCDs (no duplication) ---
                row_pcds = [float(r.get("pcd", 0.0) or 0.0) for r in (rows_data or []) if float(r.get("pcd", 0.0) or 0.0) > 0]
                pcd_min = min(row_pcds) if row_pcds else 0.0
                pcd_max = max(row_pcds) if row_pcds else 0.0

                inner_pcd_mm = float(seg_inner_pcd or 0.0)
                outer_pcd_mm = float(seg_outer_pcd or 0.0)

                # Gentle default margins around rings if UI left 0s
                _m = max(0.5, 0.5 * float(opening_dia or 0.0))  # ≥0.5 mm, ~hole radius if known
                if inner_pcd_mm <= 0.0 and pcd_min > 0.0:
                    inner_pcd_mm = max(0.0, pcd_min - 2.0 * _m)
                if outer_pcd_mm <= 0.0 and pcd_max > 0.0:
                    outer_pcd_mm = pcd_max + 2.0 * _m

                pad_eff = max(0.0, float(padding_adj or 0.0))
                fil_eff = max(0.0, float(corner_radius or 0.0))
                w_eff   = max(0.0, float(wall_width or 0.0))

                # Effective annulus the culler will use
                Rin_eff  = 0.5 * inner_pcd_mm + pad_eff + fil_eff
                Rout_eff = 0.5 * outer_pcd_mm - pad_eff - fil_eff

                # If collapsed, relax it around actual rows and reduce aggressiveness
                if not (Rout_eff > Rin_eff + 0.25):
                    if pcd_min > 0.0 and pcd_max > 0.0 and pcd_max > pcd_min:
                        inner_pcd_mm = max(0.0, pcd_min - 1.0)
                        outer_pcd_mm = pcd_max + 1.0
                        pad_eff = 0.0
                        fil_eff = min(fil_eff, 0.25 * (outer_pcd_mm - inner_pcd_mm))
                        Rin_eff  = 0.5 * inner_pcd_mm + pad_eff + fil_eff
                        Rout_eff = 0.5 * outer_pcd_mm - pad_eff - fil_eff

                # --- Log config BEFORE cull (safe) ---
                try:
                    log_error(
                        f"HYBRID CULL cfg: S={S_eff}  Rin={0.5*inner_pcd_mm:.2f}  "
                        f"Rout={0.5*outer_pcd_mm:.2f}  pad={pad_eff:.2f}  "
                        f"fil={fil_eff:.2f}  w={w_eff:.2f}"
                    )
                except Exception:
                    pass

                # --- Main Cull ---
                holes_xy = _cull_points_by_segments_xy(
                    pts_xy,
                    segments=S_eff,
                    inner_pcd_mm=inner_pcd_mm,
                    outer_pcd_mm=outer_pcd_mm,
                    wall_width_mm=w_eff,
                    corner_radius_mm=fil_eff,
                    padding_mm=pad_eff,
                    rotation_deg=0.0,
                )

                # ---- Wide-annulus retry (angle-only) ----
                if not holes_xy and pts_xy:
                    try:
                        log_error("HYBRID: annulus too tight; retrying with wide annulus for angle-only cull.")
                    except Exception:
                        pass
                    holes_xy = _cull_points_by_segments_xy(
                        pts_xy,
                        segments=S_eff,
                        inner_pcd_mm=1e-3 * 2.0,   # ~= 0 PCD
                        outer_pcd_mm=5e5 * 2.0,    # huge PCD
                        wall_width_mm=w_eff,
                        corner_radius_mm=0.0,
                        padding_mm=max(0.0, pad_eff - 0.5),
                        rotation_deg=0.0,
                    )

                # ---- Snap survivors back onto PCD rings ----
                if holes_xy:
                    try:
                        log_error(f"HYBRID SNAP: snapping {len(holes_xy)} pts to {len(rows_data)} rows.")
                    except Exception:
                        pass
                    holes_xy = _snap_to_rows_pcd(holes_xy, rows_data)

                # ---- Fallback if still nothing ----
                if not holes_xy and pts_xy:
                    try:
                        log_error("HYBRID: cull produced 0 holes; falling back to unculled circular rows.")
                    except Exception:
                        pass
                    holes_xy = list(pts_xy)

                # ---- Final summary ----
                try:
                    log_error(f"HYBRID: built {len(pts_xy)} ring pts; kept {len(holes_xy)} after cull.")
                except Exception:
                    pass

        except Exception:
            log_error("HYBRID build failed:\n" + traceback.format_exc())
            holes_xy = []    

    # Stash a small config blob in cv so your existing draw/export code can pick it up.
    # (Knife/outside: hole+CS; Cone/inlet: hole only.)
    cv.update({
        "seg_bolts_on":    seg_bolts_on,
        "seg_bolts_S":     int(S or 0),
        "seg_bolts_pcds":  (150.0, 200.0, 250.0),  # mm
        "seg_bolt_hole_d": 15.0,                   # mm (draw both sides)
        "seg_bolt_cs_d":   22.0,                   # mm (knife/outside only)
    })

    # ----- unique base name for this render/export -----
    import uuid
    filename = f"die_plate_{uuid.uuid4().hex}"

    # ---- derive values used by drawing helpers ----
    cone_top = float(cone_width or 0.0)
    cone_len = float(cv.get("cone_length", 0.0))
    cs_plus  = float(cv.get("countersink_plus_radius", 0.0))
    cs_dep   = float(cv.get("countersink_depth", 0.0))

    #-----validate countersink once (absolute diameter/depth)
    cs_dia_local, cs_depth_local = _normalize_countersink(
        opening_dia=opening_dia,
        cone_length=cone_len,
        cone_width=cone_top,
        cs_depth=cs_dep,
        cs_plus_radius=cs_plus,
    )

    # keep naming consistent for downstream calls
    rows = _ensure_row_start_angles(rows_data)

    # ---- summary text (build ONCE, before writing it anywhere) ----
    rows_for_summary = [(i+1, float(r["pcd"]), int(r["num_holes"])) for i, r in enumerate(rows)]
    holes_count = (len(holes_xy) if holes_xy else sum(n for _, _, n in rows_for_summary))

    text = _build_summary_text(
        ps=pellet_size, throughput=throughput, cw=cone_top,
        total_holes=holes_count, calc=cv, rows=rows_for_summary,
    )
    text = apply_bolt_lines_mtext(
        text,
        inner_bolt_count=inner_bolt_count,
        inner_bolt_pcd=inner_bolt_pcd,
        inner_bolt_diameter=inner_bolt_diameter,
        outer_bolt_count=outer_bolt_count,
        outer_bolt_pcd=outer_bolt_pcd,
        outer_bolt_diameter=outer_bolt_diameter,
    )

    # Pull current values (inlet-side rings) -----------#
    in_in  = float(cv.get("inlet_inside_pcd")  or 0.0)
    in_tp  = float(cv.get("inside_taper_pcd")  or 0.0)
    out_in = float(cv.get("inlet_outside_pcd") or 0.0)
    out_tp = float(cv.get("outside_taper_pcd") or 0.0)

    # Fallback: if all 4 are zero, derive a clear inside ring from countersink
    if (in_in <= 0.0 and in_tp <= 0.0 and out_in <= 0.0 and out_tp <= 0.0):
        cs_d_fallback, _ = _normalize_countersink(
            opening_dia=cv.get("opening_dia"),
            cone_length=cv.get("cone_length"),
            cone_width=cv.get("cone_width"),
            cs_depth=cv.get("countersink_depth"),
            cs_plus_radius=cv.get("countersink_plus_radius"),
        )
        in_in = float(cs_d_fallback)
        # persist so downstream reads & UI stay consistent
        cv["inlet_inside_pcd"]  = in_in
        cv["inside_taper_pcd"]  = in_tp
        cv["inlet_outside_pcd"] = out_in
        cv["outside_taper_pcd"] = out_tp

    # ---------- FRONT doc (for UI previews) ----------
    front_doc, _ = create_dxf_doc(
        plate_diameter, opening_dia, cone_top,
        outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
        inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
        outer_opening_pcd, inner_opening_pcd, die_center_hole_diameter,
        chamfer_plate, chamfer_outer_opening, chamfer_inner_opening, chamfer_center_hole,
        rows, pellet_size, inset_zoom,
        cone_length=cone_len,
        countersink_plus_radius=cs_plus,
        countersink_depth=cs_dep,
        holes_xy=holes_xy,
        include_inset_samples=False,
    )

    try:
        if draw_segment_walls:
            force_segments = str(die_style or "").strip().lower() != str(DIE_STYLE_STAGGERED).strip().lower()
            draw_backside_segments(
                front_doc,
                plate_diameter=plate_diameter,
                die_style=die_style,
                segments=int(S),
                seg_inner_pcd=seg_inner_pcd,
                seg_outer_pcd=seg_outer_pcd,
                wall_width=wall_width,
                corner_radius=corner_radius,
                force=force_segments,
            )
    except Exception:
        log_error("BACK segments (front_doc) failed:\n" + traceback.format_exc())

    ensure_ttf_text_style(front_doc)
    try:
        strip_center_marks(front_doc)
    except Exception:
        pass

    # WRITE NOTES to front_doc ONCE (original summary + inner/outer bolt lines)
    write_notes_box(front_doc, plate_diameter, text)

    # (optional) segment-wall bolts geometry (no notes)
    if IS_STAGGERED and cv.get("seg_bolts_on", False):
        add_segment_wall_bolts(front_doc, segments=S, plate_diameter=float(plate_diameter or 0.0))

    # cache for previews
    cached_filename = filename
    cached_inset_center = None

    export_doc, _ = create_dxf_doc(
        plate_diameter, opening_dia, cone_top,
        outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
        inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
        outer_opening_pcd, inner_opening_pcd, die_center_hole_diameter,
        chamfer_plate, chamfer_outer_opening, chamfer_inner_opening, chamfer_center_hole,
        rows, pellet_size, inset_zoom,
        cone_length=cone_len,
        countersink_plus_radius=cs_plus,
        countersink_depth=cs_dep,
        holes_xy=holes_xy,
        include_inset_samples=False,
    )

    # ---- pre-add logging
    try:
        msp = export_doc.modelspace()
        c_exp_before = len(msp.query("CIRCLE"))
        rows_count = len(rows or [])
        holes_count = len(holes_xy or []) if holes_xy is not None else 0
        ii = float(cv.get("inlet_inside_pcd",  0.0))
        it = float(cv.get("inside_taper_pcd",  0.0))
        io = float(cv.get("inlet_outside_pcd", 0.0))
        ot = float(cv.get("outside_taper_pcd", 0.0))
        log_error(
            f"GEN: export_doc ready (base={c_exp_before} circles); "
            f"rows={rows_count}, holes_xy={holes_count}; "
            f"inletPCD(in={ii:.2f},{it:.2f}; out={io:.2f},{ot:.2f})"
        )
    except Exception:
        pass

    # ---- add backside (this creates BACK_INLET_PCD circles)
    try:
        add_backside_plate(
            export_doc,
            plate_diameter=plate_diameter,
            opening_dia=opening_dia,
            cone_width=cone_top,
            rows_data=rows,
            die_center_hole_diameter=die_center_hole_diameter,

            outer_bolt_count=outer_bolt_count, outer_bolt_pcd=outer_bolt_pcd, outer_bolt_diameter=outer_bolt_diameter,
            inner_bolt_count=inner_bolt_count, inner_bolt_pcd=inner_bolt_pcd, inner_bolt_diameter=inner_bolt_diameter,
            holes_xy=holes_xy,
            outer_opening_pcd=outer_opening_pcd,
            inner_opening_pcd=inner_opening_pcd,
            die_style=die_style,
            segments=S,
            seg_inner_pcd=seg_inner_pcd,
            seg_outer_pcd=seg_outer_pcd,
            segment_wall_width=wall_width,
            corner_radius_mm=corner_radius,
            force_segment_walls=(draw_segment_walls and str(die_style or "").strip().lower() != str(DIE_STYLE_STAGGERED).strip().lower()),
            cone_length=cone_len,

            # inlet-side rings (possibly with fallback)
            inlet_inside_pcd=float(cv.get("inlet_inside_pcd", in_in)),
            inside_taper_pcd=float(cv.get("inside_taper_pcd", in_tp)),
            inlet_outside_pcd=float(cv.get("inlet_outside_pcd", out_in)),
            outside_taper_pcd=float(cv.get("outside_taper_pcd", out_tp)),

            # pass validated countersink values so the backside has the right CS too
            cs_diameter=cs_dia_local,
            cs_depth=cs_depth_local,
        )
    except NameError:
        log_error("add_backside_plate is not defined; skipping backside drawing.")
    except Exception:
        log_error("add_backside_plate failed:\n" + traceback.format_exc())

    # ---- post-add logging
    try:
        msp = export_doc.modelspace()
        c_inlet = len(msp.query("CIRCLE[layer=='BACK_INLET_PCD']"))
        c_total = len(msp.query("CIRCLE"))
        layer_exists = ("BACK_INLET_PCD" in export_doc.layers)
        log_error(
            f"GEN: inlet PCD circles on export: {c_inlet} "
            f"(total now {c_total}, layer_exists={layer_exists})"
        )
        # Helpful warning if inputs were >0 but no circles got created
        ii = float(cv.get("inlet_inside_pcd",  0.0))
        it = float(cv.get("inside_taper_pcd",  0.0))
        io = float(cv.get("inlet_outside_pcd", 0.0))
        ot = float(cv.get("outside_taper_pcd", 0.0))
        if c_inlet == 0 and any(v > 0.0 for v in (ii, it, io, ot)):
            log_error("WARN: inlet PCD inputs > 0 but BACK_INLET_PCD circles not found.")
    except Exception:
        log_error("post-add logging failed:\n" + traceback.format_exc())

    # --- remove any stray center ring on the inlet/back plate ---
    try:
        _purge_back_center_circle(
            export_doc,
            plate_diameter=plate_diameter,
            back_gap_min=float(cv.get("back_gap_min",  float(globals().get("BACK_GAP_MIN", 30.0)))),
            back_gap_ratio=float(cv.get("back_gap_ratio", float(globals().get("BACK_GAP_RATIO", 0.15)))),
            opening_dia=opening_dia,   # size the ghost threshold
            tol_xy=0.6
        )
    except Exception:
        log_error("center-ring purge skipped:\n" + traceback.format_exc())

    # (optional) remove any front center marks on the export doc too
    try:
        strip_center_marks(export_doc)
    except Exception:
        pass

    # (optional) segment graphics on the EXPORT doc (no labels/notes added)
    try:
        if draw_segment_walls:  # covers Tab 2 and Tab 3 (hybrid)
            force_segments = str(die_style or "").strip().lower() != str(DIE_STYLE_STAGGERED).strip().lower()
            draw_backside_segments(
                export_doc,
                plate_diameter=plate_diameter,
                die_style=die_style,
                segments=S,
                seg_inner_pcd=seg_inner_pcd,
                seg_outer_pcd=seg_outer_pcd,
                wall_width=wall_width,
                corner_radius=corner_radius,
                force=force_segments,
            )
    except Exception:
        log_error("BACK segments (export_doc) failed:\n" + traceback.format_exc())
    
    if cv.get("seg_bolts_on", False):
        add_segment_wall_bolts(export_doc, segments=S, plate_diameter=float(plate_diameter or 0.0))

    ensure_ttf_text_style(export_doc)

    # WRITE NOTES to export_doc ONCE (original summary + inner/outer bolt lines)
    write_notes_box(export_doc, plate_diameter, text)

    # Cache for previews and later exports (center is set later in the RESET block)
    cached_doc = export_doc or front_doc
    cached_filename = filename
    cached_inset_center = None
    _set_last_plate_diameter(float(plate_diameter or 0.0))

    # ---- AutoCAD cross-section inset (right & down) -----------------------
    try:
        channel_dia_mm  = float(opening_dia)
        cone_top_dia_mm = float(cone_width)
        cs_dia_mm       = cs_dia_local  # validated absolute diameter

        cone_len_mm     = float(cv.get("cone_length",    0.0))
        channel_len_mm  = float(cv.get("channel_length", 0.0))
        cs_depth_mm_in  = float(cs_depth_local)

        # Put the cross-section on a layer that starts with INSET_
        INSET_XSEC_LAYER = "INSET_XSEC"
        try:
            if INSET_XSEC_LAYER not in export_doc.layers:
                export_doc.layers.add(INSET_XSEC_LAYER, dxfattribs={"color": 1})
        except Exception:
            # tolerate older ezdxf versions
            try:
                export_doc.layers.get(INSET_XSEC_LAYER)
            except Exception:
                pass

        max_radius = max(channel_dia_mm, cone_top_dia_mm, cs_dia_mm) / 2.0
        offset_x   = max(50.0, 3.0 * max_radius + 10.0)
        y_top      = 0.0

        inset_poly = add_cross_section_inset(
            export_doc.modelspace(),
            opening_dia=channel_dia_mm,
            countersink_dia=cs_dia_mm,
            cone_dia=cone_top_dia_mm,
            cone_len=cone_len_mm,
            channel_len=channel_len_mm,
            countersink_depth=cs_depth_mm_in,
            inset_offset_x=offset_x,
            y_top=y_top,
            layer=INSET_XSEC_LAYER,
        )

        # place the cross-section inset to the right and a bit below
        try:
            R_plate       = max(0.0, float(plate_diameter or 0.0)) / 2.0
            used_cs_depth = max(0.0, min(float(cs_depth_mm_in or 0.0), max(0.0, float(cone_len_mm or 0.0))))
            cone_taper_len= max(0.0, float(cone_len_mm or 0.0) - used_cs_depth)
            total_h       = used_cs_depth + cone_taper_len + max(0.0, float(channel_len_mm or 0.0))
            shift_right   = (2.0 * R_plate) + max(50.0, 0.25 * R_plate)
            shift_down    = -(total_h + 40.0)
            inset_poly.transform(Matrix44.translate(shift_right, shift_down, 0.0))
            log_error(f"GEN: cross-section inset placed (dx={shift_right:.1f}, dy={shift_down:.1f})")
        except Exception:
            log_error("Cross-section inset placement failed:\n" + traceback.format_exc())
    except Exception:
        log_error("Cross-section inset creation failed:\n" + traceback.format_exc())

    #----- save export DXF and cache path ----------------------------------
    dxf_path = None
    try:
        out_dir = _get_writable_tmp_dir()
        filename_base = f"die_plate_{filename}" if not filename.startswith("die_plate_") else filename
        dxf_path = os.path.join(out_dir, f"{filename_base}.dxf")
        export_doc.saveas(dxf_path)

        # cache path for later PDF export + UI
        cached_export_path = dxf_path
        _set_last_dxf_path(dxf_path)
        log_error(f"GEN: DXF saved -> {dxf_path}")
    except Exception as e:
        try:
            log_error(f"DXF save failed (dir={out_dir if 'out_dir' in locals() else 'n/a'}): {e}\n" + traceback.format_exc())
        except Exception:
            pass
        dxf_path = None

    # ---------- UI previews ----------
    try:
        plate_img = create_preview_image(
            front_doc, filename,
            mode="plate",
            plate_diameter=plate_diameter,
            show_cone=False,
        )
        log_error("GEN: plate preview OK")
    except Exception:
        log_error("plate preview failed:\n" + traceback.format_exc())
        plate_img = None

    # Inset preview (RESET on each Generate): show the full BACK plate, then let the slider zoom in
    try:
        # Tunables (prefer locals set earlier; fall back to globals)
        inset_view_scale     = locals().get("inset_view_scale",     float(globals().get("INSET_VIEW_SCALE", 0.75)))
        initial_inset_zoom   = locals().get("initial_inset_zoom",   float(globals().get("INITIAL_INSET_ZOOM", 1.00)))
        back_gap_min         = locals().get("back_gap_min",         float(globals().get("BACK_GAP_MIN", 30.0)))
        back_gap_ratio       = locals().get("back_gap_ratio",       float(globals().get("BACK_GAP_RATIO", 0.15)))
        back_center_y_nudge  = locals().get("back_center_y_nudge",  float(globals().get("BACK_CENTER_Y_NUDGE", 0.0)))
        back_center_x_factor = locals().get("back_center_x_factor", float(globals().get("BACK_CENTER_X_FACTOR", 0.0)))
        back_center_x_nudge  = locals().get("back_center_x_nudge",  float(globals().get("BACK_CENTER_X_NUDGE", 0.0)))

        # Centroid of holes if present, else origin (x=0,y=0)
        if holes_xy and len(holes_xy) > 0:
            cx = sum(x for x, _ in holes_xy) / float(len(holes_xy))
            cy = sum(y for _, y in holes_xy) / float(len(holes_xy))
        else:
            cx, cy = 0.0, 0.0

        # Vertical offset down to the BACK plate (must match add_backside_plate)
        R_plate     = max(0.0, float(plate_diameter or 0.0)) / 2.0
        back_offset = -(2.0 * R_plate + max(back_gap_min, back_gap_ratio * R_plate))

        # Apply horizontal (factor × plate_diameter + mm) and vertical (mm) nudges
        x = float(cx) + float(plate_diameter or 0.0) * float(back_center_x_factor) + float(back_center_x_nudge)
        y = float(cy) + back_offset + float(back_center_y_nudge)
        back_center = (x, y)

        focus_center = _compute_focus_center(
            plate_diameter=float(plate_diameter or 0.0),
            holes_xy=holes_xy,
            rows=rows,
            back_offset_y=back_offset,
        )

        # Force reset zoom each Generate
        zoom_used = float(initial_inset_zoom)
        # Field-of-view for the preview (smaller ⇒ tighter around the BACK plate)
        view_diam = max(10.0, float(plate_diameter or 0.0) * float(inset_view_scale))

        # Make the center impossible to ignore:
        # 1) set the global cache that your zoom-change handler uses
        cached_doc = export_doc
        cached_filename = filename
        cached_inset_center = back_center
        cached_inset_focus_center = focus_center

        # 2) also pass the center explicitly to the renderer
        inset_img = create_preview_image(
            export_doc, filename,
            mode="inset",
            plate_diameter=view_diam,     # main FOV knob
            center=back_center,           # explicit center
            zoom=zoom_used,
            show_cone=True,
        )

        # Log everything so we can see what's actually being used
        log_error(
            "RESET INSET: "
            f"scale={inset_view_scale:.3f}, zoom={zoom_used:.2f}, "
            f"x_factor={back_center_x_factor:.3f}, x_nudge={back_center_x_nudge:.1f}, "
            f"y_nudge={back_center_y_nudge:.1f}, center=({back_center[0]:.1f},{back_center[1]:.1f}), "
            f"view_diam={view_diam:.1f}"
        )
    except Exception:
        log_error("inset preview failed:\n" + traceback.format_exc())
        inset_img = plate_img

    return plate_img, inset_img, dxf_path

# --------- Helper to calculate staggered hole pattern geometry ---------------
#-------Segment bolt settings--------------------
def add_segment_wall_bolts(target, *, segments: int, plate_diameter: float,
                           pcds=(150.0, 200.0, 250.0), hole_d=15.0, cs_d=22.0) -> int:
    """
    Add 3 bolts on each segment wall (only when S==3).
      - FRONT (knife/outside):  Ø{hole_d} on layer 'BOLTS' + Ø{cs_d} on 'BOLTS_CS'
      - BACK  (inlet/cone):     Ø{hole_d} on layer 'BACK_BOLTS', offset downward
    Works with either a DXF Document or a Modelspace passed as `target`.

    Returns: total number of circles added across both views.
    """
    import math

    S = int(segments or 0)
    if S != 3:
        return 0

    # ---- Resolve doc / modelspace ----
    if hasattr(target, "modelspace"):  # a DXF Document
        doc = target
        msp = doc.modelspace()
    else:                               # a Modelspace
        msp = target
        doc = msp.doc

    # ---- Layers (ensure exist) ----
    def _ensure_layer(name, color=7):
        try:
            if name not in doc.layers:
                doc.layers.add(name, dxfattribs={"color": color})
        except Exception:
            # tolerate older ezdxf versions
            try:
                doc.layers.get(name)
            except Exception:
                pass

    L_FRONT_HOLE = "BOLTS"
    L_FRONT_CS   = "BOLTS_CS"
    L_BACK_HOLE  = "BACK_BOLTS"

    _ensure_layer(L_FRONT_HOLE, color=7)
    _ensure_layer(L_FRONT_CS,   color=7)
    _ensure_layer(L_BACK_HOLE,  color=7)

    # If creating the CS layer fails for any reason, fall back to the hole layer
    try:
        _ = doc.layers.get(L_FRONT_CS)
    except Exception:
        L_FRONT_CS = L_FRONT_HOLE

    # ---- Backside vertical offset (match add_backside_plate) ----
    R_plate  = max(0.0, float(plate_diameter or 0.0)) / 2.0
    gap      = max(30.0, 0.15 * R_plate)
    offset_y = -(2.0 * R_plate + gap)

    # ---- Geometry ----
    hole_r = float(hole_d) / 2.0
    cs_r   = float(cs_d)   / 2.0
    theta  = 2.0 * math.pi / S

    count = 0
    for s in range(S):
        wall_angle = s * theta  # segment wall (ray)
        c, s_ = math.cos(wall_angle), math.sin(wall_angle)

        for pcd in pcds:
            r = 0.5 * float(pcd)
            x = r * c
            y = r * s_

            # FRONT (knife/outside): hole + countersink ring
            try:
                msp.add_circle((x, y), hole_r, dxfattribs={"layer": L_FRONT_HOLE, "lineweight": 25}); count += 1
                msp.add_circle((x, y), cs_r,   dxfattribs={"layer": L_FRONT_CS,   "lineweight": 25}); count += 1
            except Exception:
                pass

            # BACK (inlet/cone): only hole, offset downward
            try:
                msp.add_circle((x, y + offset_y), hole_r, dxfattribs={"layer": L_BACK_HOLE, "lineweight": 25}); count += 1
            except Exception:
                pass

    # Optional: log for debugging if you have log_error(...)
    try:
        log_error(f"SEG_BOLTS: S=3, PCDs={tuple(pcds)}, hole={hole_d}, cs={cs_d} → added {count} circles")
    except Exception:
        pass

    return count

def calc_staggered_pattern(
    opening_dia, cone_width, calc_values,
    segments, wall_width, seg_inner_pcd, seg_outer_pcd,
    corner_radius, padding_adj,
    desired_total_holes,
):
    import gradio as gr

    # --- envelope / derived diameters (for clearance)
    cv = dict(calc_values or {})
    cs_plus  = float(cv.get("countersink_plus_radius", 0.0))
    has_cone = (float(cv.get("cone_length", 0.0) or 0.0) > 0.0 and float(cone_width or 0.0) > 0.0)
    cs_dia   = (float(cone_width) + 2.0 * cs_plus) if has_cone else 0.0
    cone_dia = float(cone_width) if has_cone else 0.0

    # --- solve: exact-N if possible, else best ≤ N
    pts, rep = pack_segment_lattice(
        total_holes     = int(desired_total_holes or 0),
        segments        = int(segments or 0),
        inner_pcd       = float(seg_inner_pcd or 0.0),
        outer_pcd       = float(seg_outer_pcd or 0.0),
        opening_dia     = float(opening_dia or 0.0),
        countersink_dia = cs_dia,
        cone_dia        = cone_dia,
        padding_mm      = float(padding_adj or 0.0),
        wall_width_mm   = float(wall_width or 0.0),
        corner_radius_mm= float(corner_radius or 0.0),
    )

    # --- nothing fits
    if not pts:
        msg_text = ("Holes don’t fit. Try increasing segment size or reducing hole size"
                    "Holes that fit = 0.")
        return (
            [],                                        # staggered_pts_state
            gr.update(value="<em>No holes</em>", visible=True),  # stag_counts (HTML)
            gr.update(value=msg_text, visible=True),   # stag_msg (Markdown)
        )

    # --- chips row + message
    counts = segment_counts(pts, int(segments or 0))
    chips_html, _total_text = render_counts_html(counts)

    placed = sum(counts)
    target = int(desired_total_holes or 0)

    if rep.get("hit_target", False) or placed == target:
        msg_text = f"All holes fit (target {target}). Pitch = {rep.get('pitch_mm', 0.0):.3f} mm."
    else:
        msg_text = (
            f"Holes don’t fit (needed {target}, placed {placed}). "
            f"Try increasing segment size or reduce hole size/padding. "
            f"Holes that fit = {placed}."
        )

    return (
        pts,                                        # staggered_pts_state
        gr.update(value=chips_html, visible=True),  # stag_counts
        gr.update(value=msg_text, visible=True),    # stag_msg
    )

def compute_staggered_points(
    *,
    segments: int,
    inner_pcd_mm: float,
    outer_pcd_mm: float,
    wall_width_mm: float,
    corner_radius_mm: float,
    padding_mm: float,
    clearance_radius_mm: float,
    line_angle_offset_deg: float = 0.0,   # global rotation if your drawn segments are rotated
    pitch_mm: Optional[float] = None,     # explicit pitch if provided
    min_gap_mm: float = 0.2,              # edge-to-edge between adjacent holes
) -> List[Tuple[float, float]]:
    """
    Triangular lattice inside each segment wedge, mirrored about each segment midline.
    Never places a center inside the walls, never touches walls or radial bounds.
    """
    S = max(3, int(segments or 0))
    if outer_pcd_mm <= 0 or outer_pcd_mm <= inner_pcd_mm:
        return []

    # --- geometry envelope (strict) ---
    r_h = float(clearance_radius_mm or 0.0)
    pad = float(padding_mm or 0.0)
    fil = float(corner_radius_mm or 0.0)
    w   = float(wall_width_mm or 0.0)

    r_in_raw  = max(0.0, float(inner_pcd_mm) / 2.0)
    r_out_raw = max(0.0, float(outer_pcd_mm) / 2.0)
    Rin  = r_in_raw  + pad + fil + r_h
    Rout = r_out_raw - pad - fil - r_h
    if Rout <= Rin:
        return []

    # Strict "no touching" tolerance (tiny, but non-zero)
    TOL = 1e-4  # mm

    # Triangular lattice spacing
    if pitch_mm is None:
        P = 2.0 * r_h + float(min_gap_mm or 0.0)
    else:
        P = max(1e-6, float(pitch_mm))
    Q = (math.sqrt(3.0) / 2.0) * P

    # Required perpendicular distance from the *inner face* of a wall ray
    req_wall = pad + r_h + 0.5 * w

    theta = 2.0 * math.pi / S                   # wedge angle
    phi0  = math.radians(float(line_angle_offset_deg or 0.0))  # global rotation if needed

    pts: List[Tuple[float, float]] = []

    # For each segment, build rows perpendicular to its midline
    for s in range(S):
        mid = phi0 + (s + 0.5) * theta          # segment midline angle

        # Row direction = perpendicular to midline
        phi = mid + math.pi / 2.0
        ux, uy = math.cos(phi), math.sin(phi)   # along-row unit vector
        nx, ny = -uy, ux                        # row normal (toward +midline)

        # Sweep rows symmetrically about the midline
        tmin, tmax = -Rout, Rout
        nmin = int(math.floor(tmin / Q))
        nmax = int(math.ceil (tmax / Q))

        for n in range(nmin, nmax + 1):
            t = n * Q
            offset = 0.0 if (n % 2 == 0) else (0.5 * P)

            k = -Rout + offset
            while k <= Rout + 1e-9:
                x = nx * t + ux * k
                y = ny * t + uy * k
                r = math.hypot(x, y)

                # strict radial bounds (no touching)
                if r < Rin + TOL or r > Rout - TOL:
                    k += P
                    continue

                # Angular deviation from midline (wrap to [-pi, pi])
                ang = math.atan2(y, x)
                dev = ((ang - mid + math.pi) % (2.0*math.pi)) - math.pi
                # angle to nearest wall (>= 0); walls at mid ± theta/2
                delta_to_wall = (theta / 2.0) - abs(dev)
                if delta_to_wall <= 0.0:
                    k += P
                    continue

                # Perpendicular distance from the wall ray at radius r
                perp = r * math.sin(delta_to_wall)

                # strict wall clearance (no touching)
                if perp <= req_wall + TOL:
                    k += P
                    continue

                pts.append((x, y))
                k += P

    return pts

# --- Fallback packer that adapts pitch to hit the target (or max out) ---
def pack_segment_lattice(
    *,
    total_holes: int,
    segments: int,
    inner_pcd: float,
    outer_pcd: float,
    opening_dia: float,
    countersink_dia: float = 0.0,
    cone_dia: float = 0.0,
    padding_mm: float = 0.0,
    wall_width_mm: float = 0.0,
    corner_radius_mm: float = 0.0,
    line_angle_offset_deg: float = 0.0,
):
    """
    Choose a uniform pitch for a triangular lattice that yields EXACTLY N points,
    without removing any. If N==0, returns an empty set. If N > capacity at the
    tightest legal pitch, returns the capacity (hit_target=False).
    """
    import math

    # --- Largest circle radius that must clear everything (open + cone + CS)
    r_h = 0.5 * max(
        0.0,
        float(opening_dia or 0.0),
        float(cone_dia or 0.0),
        float(countersink_dia or 0.0),
    )
    min_gap = 0.2  # mm, edge-to-edge minimum

    def _grid(P: float):
        # NOTE: compute_staggered_points must enforce walls & annulus correctly.
        return compute_staggered_points(
            segments=int(segments or 0),
            inner_pcd_mm=float(inner_pcd or 0.0),
            outer_pcd_mm=float(outer_pcd or 0.0),
            wall_width_mm=float(wall_width_mm or 0.0),
            corner_radius_mm=float(corner_radius_mm or 0.0),
            padding_mm=float(padding_mm or 0.0),
            clearance_radius_mm=r_h,
            line_angle_offset_deg=float(line_angle_offset_deg or 0.0),
            pitch_mm=float(P),
            min_gap_mm=min_gap,
        )

    # ---- densest legal pitch & capacity
    P_min = 2.0 * r_h + min_gap
    pts_min = _grid(P_min)
    capacity = len(pts_min)
    N = max(0, int(total_holes or 0))

    # 0) Explicit "no holes" request → return 0 holes
    if N <= 0:
        try:
            log_error(f"STAG PACK: target_N=0 → returning 0 holes (no fallback). "
                      f"P_min={P_min:.3f}, capacity={capacity}")
        except Exception:
            pass
        return [], {
            "hit_target": True,
            "target_seg": 0,
            "placed_seg": 0,
            "pitch_mm":   P_min,
            "notes":      "N=0 requested; returned 0 holes",
        }

    # 1) If even the tightest pitch can't fit N → return capacity (no removals possible)
    if capacity < N:
        try:
            log_error(f"STAG PACK: target N={N} exceeds capacity={capacity} at P_min={P_min:.3f}")
        except Exception:
            pass
        counts = segment_counts(pts_min, int(segments or 0))
        return pts_min, {
            "hit_target": False,
            "target_seg": N // max(1, int(segments or 1)),
            "placed_seg": min(counts) if counts else 0,
            "pitch_mm":   P_min,
            "notes":      "target exceeds maximum capacity (tightest legal pitch)",
        }

    # 2) We can reach the target → find a pitch whose generated lattice has EXACTLY N points.
    #    Strategy:
    #      a) Exponential grow pitch until count drops below N → bracket [loP (>=N), hiP (<N)]
    #      b) Binary-search to the threshold
    #      c) Fine sweep & micro-step to hit an exact count (no removals)
    loP = P_min
    loC = capacity
    loPts = pts_min

    # a) grow to find the first infeasible pitch (count < N)
    grow = 1.12
    hiP = None
    hiC = None
    for _ in range(64):
        if loC == N:
            # already exact at the current pitch
            counts = segment_counts(loPts, int(segments or 0))
            return loPts, {
                "hit_target": True,
                "target_seg": N // max(1, int(segments or 1)),
                "placed_seg": min(counts) if counts else 0,
                "pitch_mm":   loP,
                "notes":      "exact-N at bracket start",
            }
        testP = loP * grow
        testPts = _grid(testP)
        c = len(testPts)
        if c >= N:
            loP, loC, loPts = testP, c, testPts
        else:
            hiP, hiC = testP, c
            break

    if hiP is None:
        # In practice we should always find a hiP; guard just in case.
        hiP = loP * grow
        hiPts = _grid(hiP)
        hiC = len(hiPts)

    # b) binary search the transition region
    for _ in range(36):
        mid = 0.5 * (loP + hiP)
        midPts = _grid(mid)
        cmid = len(midPts)
        if cmid == N:
            counts = segment_counts(midPts, int(segments or 0))
            return midPts, {
                "hit_target": True,
                "target_seg": N // max(1, int(segments or 1)),
                "placed_seg": min(counts) if counts else 0,
                "pitch_mm":   mid,
                "notes":      "exact-N via binary search",
            }
        if cmid >= N:
            loP, loC, loPts = mid, cmid, midPts
        else:
            hiP, hiC = mid, cmid

    # c1) fine sweep in log-space between loP (>=N) and hiP (<N) to try to hit EXACT N
    #     (counts change stepwise; a small move in P often flips right to N)
    for i in range(1, 121):  # ~120 probes
        t = i / 121.0
        P_try = math.exp((1.0 - t) * math.log(loP) + t * math.log(hiP))
        pts_try = _grid(P_try)
        c = len(pts_try)
        if c == N:
            counts = segment_counts(pts_try, int(segments or 0))
            return pts_try, {
                "hit_target": True,
                "target_seg": N // max(1, int(segments or 1)),
                "placed_seg": min(counts) if counts else 0,
                "pitch_mm":   P_try,
                "notes":      "exact-N via fine sweep",
            }
        if c > N:
            # track the best (closest above N) in case we need micro-steps next
            if c < loC:
                loP, loC, loPts = P_try, c, pts_try
        else:
            # track the best (closest below N) too
            if hiC is None or c > hiC:
                hiP, hiC = P_try, c

    # c2) micro-step OUTWARD from the best >=N pitch to try to drop one-by-one to EXACT N
    #     (small multiplicative bumps; usually flips border inclusions cleanly)
    P_bump = loP
    pts_bump = loPts
    c_bump = loC
    for _ in range(400):  # very small ~0.1% outward steps
        if c_bump == N:
            counts = segment_counts(pts_bump, int(segments or 0))
            return pts_bump, {
                "hit_target": True,
                "target_seg": N // max(1, int(segments or 1)),
                "placed_seg": min(counts) if counts else 0,
                "pitch_mm":   P_bump,
                "notes":      "exact-N via micro-step",
            }
        if c_bump < N:
            break
        P_bump *= 1.001
        pts_bump = _grid(P_bump)
        c_bump = len(pts_bump)

    # If we STILL didn't land exactly on N (rare), prefer the closest **below** N rather than overfilling.
    # This respects the user's "no removals" rule (never return >N).
    if hiC is not None and hiC <= N:
        pts_final = _grid(hiP)  # this is <= N
        c_final = len(pts_final)
        note = "could not hit exact-N; returned closest ≤ N without removals"
        try:
            log_error(f"STAG PACK: exact-N miss; N={N}, chosen={c_final}, P≈{hiP:.6f}")
        except Exception:
            pass
        counts = segment_counts(pts_final, int(segments or 0))
        return pts_final, {
            "hit_target": (c_final == N),
            "target_seg": N // max(1, int(segments or 1)),
            "placed_seg": min(counts) if counts else 0,
            "pitch_mm":   hiP,
            "notes":      note,
        }

    # Ultimate fallback: return the best >=N (still uniform, but over target).
    # If you *never* want to overfill, you can change this to return [] instead,
    # but it's usually better to indicate we couldn't hit N exactly.
    pts_final = loPts
    c_final = len(pts_final)
    try:
        log_error(f"STAG PACK: fallback overfill; N={N}, chosen={c_final}, P≈{loP:.6f}")
    except Exception:
        pass
    counts = segment_counts(pts_final, int(segments or 0))
    return pts_final, {
        "hit_target": (c_final == N),
        "target_seg": N // max(1, int(segments or 1)),
        "placed_seg": min(counts) if counts else 0,
        "pitch_mm":   loP,
        "notes":      "exact-N not reachable; minimal overfill at uniform pitch",
    }

    try:
        log_error(
            "STAG PACK: "
            f"capacity={capacity} target_N={N} "
            f"P_min={P_min:.3f} bestP={bestP:.3f} "
            f"count={len(pts)} exact={exact}"
        )
    except Exception:
        pass

    return pts, report

    # --- debug helper: worst-case margins for a set of points (matches generator math)
    def _wedge_margins(pts):
        try:
            import math
            S = max(3, int(segments or 0))
            theta = 2.0 * math.pi / S
            r_h = 0.5 * max(
                0.0,
                float(opening_dia or 0.0),
                float(cone_dia or 0.0),
                float(countersink_dia or 0.0),
            )
            pad = float(padding_mm or 0.0)
            fil = float(corner_radius_mm or 0.0)
            Rin  = float(inner_pcd or 0.0)/2.0 + fil + pad + r_h
            Rout = float(outer_pcd or 0.0)/2.0 - fil - pad - r_h
            req_wall = pad + r_h + 0.5 * float(wall_width_mm or 0.0)

            min_wall = float("+inf")
            min_rad  = float("+inf")

            for (x, y) in (pts or []):
                r   = math.hypot(x, y)
                ang = math.atan2(y, x)
                # segment index for angle; walls at multiples of theta
                s   = int(math.floor((ang % (2.0*math.pi)) / theta))
                mid = s * theta + 0.5 * theta
                dev = ((ang - mid + math.pi) % (2.0*math.pi)) - math.pi
                delta_to_wall = (theta / 2.0) - abs(dev)
                if delta_to_wall <= 0.0:
                    wall_gap = -1e9
                else:
                    wall_gap = (r * math.sin(max(0.0, delta_to_wall))) - req_wall
                if wall_gap < min_wall: min_wall = wall_gap

                rad_gap = min(r - Rin, Rout - r)
                if rad_gap < min_rad:  min_rad  = rad_gap

            if min_wall is float("+inf"): min_wall = 0.0
            if min_rad  is float("+inf"): min_rad  = 0.0
            return (min_wall, min_rad)
        except Exception:
            return (0.0, 0.0)

    # Densest legal pitch first
    P_min = 2.0 * r_h + min_gap
    pts = _grid(P_min)
    capacity = len(pts)
    N = max(0, int(total_holes or 0))

    if N == 0:
        counts = segment_counts(pts, int(segments or 0))
        report = {
            "hit_target": True,
            "target_seg": 0,
            "placed_seg": min(counts) if counts else 0,
            "pitch_mm":   P_min,
            "notes":      "no explicit target; used densest legal pitch",
        }
        try:
            mw, mr = _wedge_margins(pts)
            log_error(f"STAG PACK: N=0 capacity={capacity} P_min={P_min:.3f} "
                      f"half_w_eff={(0.5*float(wall_width_mm or 0.0) - (float(padding_mm or 0.0) + r_h)):.3f} "
                      f"min_wall_gap={mw:.3f} mm min_radial_gap={mr:.3f} mm")
        except Exception:
            pass
        return pts, report

    if capacity < N:
        counts = segment_counts(pts, int(segments or 0))
        report = {
            "hit_target": False,
            "target_seg": N // max(1, int(segments or 1)),
            "placed_seg": min(counts) if counts else 0,
            "pitch_mm":   P_min,
            "notes":      "target exceeds maximum capacity (tightest legal pitch)",
        }
        try:
            mw, mr = _wedge_margins(pts)
            log_error(f"STAG PACK: capacity miss N={N} capacity={capacity} P_min={P_min:.3f} "
                      f"min_wall_gap={mw:.3f} mm min_radial_gap={mr:.3f} mm")
        except Exception:
            pass
        return pts, report

        # We can reach the target → find the largest pitch with count ≥ N (minimal overfill)

    # 1) Exponentially grow until we FIRST drop below N → that gives a bracket [loP (≥N), hiP (<N)]
    grow = 1.12  # 1.10–1.18 is fine; smaller = tighter bracket, a bit more work
    loP = P_min
    hiP = None

    while True:
        testP = loP * grow
        c = len(_grid(testP))
        if c >= N:
            loP = testP               # still feasible; keep growing
            if testP > 1e6:           # hard guard
                break
        else:
            hiP = testP               # first infeasible pitch
            break

    if hiP is None:
        # never fell under N while growing — fabricate a fail point to close the bracket
        hiP = loP * grow

    # 2) Binary search between loP (≥N) and hiP (<N)
    lo, hi = loP, hiP
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if len(_grid(mid)) >= N:
            lo = mid                  # still ≥ N → try to grow spacing
        else:
            hi = mid                  # too sparse

    P_star = lo
    pts_star = _grid(P_star)

    # Select exactly N, balanced across segments
    chosen = select_staggered_subset(pts_star, int(segments or 0), N)
    counts = segment_counts(chosen, int(segments or 0))
    return chosen, {
        "hit_target": True,
        "target_seg": N // max(1, int(segments or 1)),
        "placed_seg": min(counts) if counts else 0,
        "pitch_mm":   P_star,
        "notes":      "adaptive pitch (tight bracket) — minimal overfill before selection",
    }
   
    # --- debug log BEFORE returning
    try:
        mw, mr = _wedge_margins(chosen)
        log_error(
            "STAG PACK: "
            f"S={int(segments or 0)} "
            f"Rin={float(inner_pcd)/2:.2f}→{(float(inner_pcd)/2 + float(corner_radius_mm or 0.0) + float(padding_mm or 0.0) + r_h):.2f} "
            f"Rout={(float(outer_pcd)/2 - float(corner_radius_mm or 0.0) - float(padding_mm or 0.0) - r_h):.2f} "
            f"half_w_eff={(0.5*float(wall_width_mm or 0.0) - (float(padding_mm or 0.0) + r_h)):.3f} "
            f"P_min={P_min:.3f} P_star={P_star:.3f} "
            f"N={N} capacity={capacity} chosen={len(chosen)} "
            f"min_wall_gap={mw:.3f} mm min_radial_gap={mr:.3f} mm"
        )
    except Exception:
        pass

    return chosen, report

from typing import List, Tuple
import math

def segment_counts(holes_xy, segments: int) -> List[int]:
    if not holes_xy or not segments:
        return [0] * max(1, int(segments or 1))
    S = int(segments)
    base = 360.0 / S
    counts = [0] * S
    for (x, y) in holes_xy:
        ang = (math.degrees(math.atan2(y, x)) % 360.0)
        idx = int(ang // base) % S
        counts[idx] += 1
    return counts

def render_counts_html(counts: List[int]) -> Tuple[str, str]:
    total = sum(counts)
    chips = "".join(
        f'<span style="display:inline-block;padding:4px 8px;border-radius:9999px;border:1px solid #ccc;'
        f'font-size:12px;background:#f6f6f6">S{i+1}: {c}</span>'
        for i, c in enumerate(counts)
    )
    placed_chip = (
        f'<span style="display:inline-block;padding:4px 8px;border-radius:9999px;'
        f'border:1px solid #777;font-weight:600;background:#eee;margin-left:8px">'
        f'Placed: {total}</span>'
    )
    html = f'<div style="display:flex;gap:8px;flex-wrap:wrap">{chips}{placed_chip}</div>'
    return html, f"Placed: {total}"   

import math
from typing import List, Tuple  # keep if you're on Python 3.8

def select_staggered_subset(
    pts: List[Tuple[float, float]],
    segments: int,
    desired_total: int,
) -> List[Tuple[float, float]]:
    """
    From an overfull candidate grid (pts), select up to desired_total centers,
    distributed as evenly as possible across segments.
    """
    S = max(1, int(segments or 0))
    N = max(0, int(desired_total or 0))
    if not pts or N == 0:
        return []

    # Bucket candidates by segment index (angle)
    base_deg = 360.0 / S
    buckets: List[List[Tuple[float, float]]] = [[] for _ in range(S)]
    for (x, y) in pts:
        ang = (math.degrees(math.atan2(y, x)) % 360.0)
        idx = int(ang // base_deg) % S
        buckets[idx].append((x, y))

    # Target per segment (even split + remainder to early segments)
    base = N // S
    rem  = N % S
    targets = [base + (1 if i < rem else 0) for i in range(S)]

    selected: List[Tuple[float, float]] = []
    remaining: List[List[Tuple[float, float]]] = [[] for _ in range(S)]

    for s in range(S):
        cand = buckets[s]
        k = targets[s]
        if not cand:
            continue
        # Orient "across-face" similar to compute_staggered_points for segment s
        seg_start = s * base_deg
        seg_end   = (s + 1) * base_deg
        seg_mid   = 0.5 * (seg_start + seg_end)
        phi = seg_mid + 90.0
        pr  = math.radians(phi)
        ux, uy = math.cos(pr), math.sin(pr)  # along-line unit vector

        # Sort by along-line coordinate to spread selection
        cand_sorted = sorted(cand, key=lambda p: p[0]*ux + p[1]*uy)
        m = len(cand_sorted)
        if m <= k:
            take = cand_sorted
            extra = []
        else:
            # pick k evenly spaced indices across the sorted list
            step = m / k
            idxs = [min(m-1, max(0, int(round((j + 0.5) * step - 1)))) for j in range(k)]
            seen = set()
            take = []
            for idx in idxs:
                if idx not in seen:
                    take.append(cand_sorted[idx])
                    seen.add(idx)
            # remainder for possible fill if other segments are short
            extra = [p for i, p in enumerate(cand_sorted) if i not in seen]

        selected.extend(take)
        remaining[s] = extra

    # If some segments had fewer than target, try to top-up from others’ extras
    missing = N - len(selected)
    if missing > 0:
        for s in range(S):
            if not remaining[s]:
                continue
            grab = remaining[s][:missing]
            selected.extend(grab)
            missing -= len(grab)
            if missing <= 0:
                break

    # Cap to N just in case
    return selected[:N]

#----------- DXF Summary info Helpers and Inset sample herlers-----------------------

import ezdxf

# --- preview zoom cache  ---
try:
    cached_doc
except NameError:
    cached_doc = None

try:
    cached_inset_center
except NameError:
    cached_inset_center = None
    
try:
    cached_inset_focus_center
except NameError:
    cached_inset_focus_center = None

# --- gradio callback used by the inset zoom slider/button ---
def update_inset_zoom(zoom):
    global cached_doc, cached_inset_center, cached_inset_focus_center, cached_filename
    doc = cached_doc
    if doc is None:
        return None

    pd = float(_get_last_plate_diameter() or 0.0)

    # existing framing tunables
    inset_view_scale     = float(globals().get("INSET_VIEW_SCALE", 0.75))
    back_gap_min         = float(globals().get("BACK_GAP_MIN", 30.0))
    back_gap_ratio       = float(globals().get("BACK_GAP_RATIO", 0.15))
    back_center_y_nudge  = float(globals().get("BACK_CENTER_Y_NUDGE", 0.0))
    back_center_x_factor = float(globals().get("BACK_CENTER_X_FACTOR", 0.0))
    back_center_x_nudge  = float(globals().get("BACK_CENTER_X_NUDGE", 0.0))

    # focus blend tunables (use the ones defined at top)
    z_focus_start = float(globals().get("INSET_FOCUS_BLEND_START", 1.0))
    z_focus_full  = float(globals().get("INSET_FOCUS_BLEND_END",   3.0))

    # base center from Generate
    base_center = cached_inset_center
    if not base_center:
        # defensive fallback
        R = max(0.0, pd) / 2.0
        back_offset = -(2.0 * R + max(back_gap_min, back_gap_ratio * R))
        base_center = (
            (pd * back_center_x_factor) + back_center_x_nudge,
            back_offset + back_center_y_nudge,
        )

    focus_center = cached_inset_focus_center or base_center

    z = float(zoom or 1.0)
    view_diam = max(10.0, pd * inset_view_scale)

    # guard: if someone sets START >= END, treat as “always focus”
    if z_focus_full <= z_focus_start:
        center_to_use = focus_center
        blend_alpha = 1.0
    elif z <= z_focus_start:
        center_to_use = base_center
        blend_alpha = 0.0
    elif z >= z_focus_full:
        center_to_use = focus_center
        blend_alpha = 1.0
    else:
        # smoothstep blend
        t = (z - z_focus_start) / (z_focus_full - z_focus_start)
        t = t * t * (3.0 - 2.0 * t)
        cx = (1.0 - t) * float(base_center[0]) + t * float(focus_center[0])
        cy = (1.0 - t) * float(base_center[1]) + t * float(focus_center[1])
        center_to_use = (cx, cy)
        blend_alpha = t

    img = create_preview_image(
        doc, cached_filename or "inset",
        mode="inset",
        plate_diameter=view_diam,
        center=center_to_use,
        zoom=z,
        show_cone=True,
    )

    which = (
        "FOCUS" if blend_alpha >= 0.999 else
        "BASE"  if blend_alpha <= 0.001 else
        f"BLEND({blend_alpha:.2f})"
    )
    log_error(
        f"ZOOM CHANGE: zoom={z:.2f}, view_diam={view_diam:.1f}, "
        f"center={which} ({center_to_use[0]:.1f},{center_to_use[1]:.1f}), "
        f"z_start={z_focus_start:.2f}, z_full={z_focus_full:.2f}"
    )
    return img

def add_cross_section_inset(
    msp,
    *,
    opening_dia: float,           # NARROW: straight channel diameter (die opening)
    countersink_dia: float,       # WIDE: countersink outer diameter (cone_dia + 2*radius)
    cone_dia: float,              # WIDE: cone TOP diameter (at CS bottom)
    cone_len: float,              # USER Cone Length = countersink_depth_used + cone_taper_len
    channel_len: float,           # straight channel AFTER cone, at opening_dia
    countersink_depth: float = 0.0,
    inset_offset_x: float = 200.0,
    y_top: float = 0.0,
    layer: str = "INSET",
):
    """
    Single-ended funnel (top → down), with Cone Length respected as:
        Cone Length = used_countersink_depth + cone_taper_len

      - Countersink taper: countersink_dia @ y_top  → cone_dia @ (y_top - used_cs_depth)
      - Cone taper:        cone_dia       → opening_dia over cone_taper_len
      - Channel:           opening_dia straight for channel_len
    """
    # radii
    cs_r       = float(countersink_dia) / 2.0          # widest (top)
    cone_top_r = float(cone_dia)        / 2.0          # cone diameter at top of cone
    open_r     = float(opening_dia)     / 2.0          # narrow channel dia

    # lengths (enforce "cone_len = used_cs_depth + cone_taper_len")
    cone_len_in      = max(0.0, float(cone_len))
    cs_depth_in      = max(0.0, float(countersink_depth))
    used_cs_depth    = min(cs_depth_in, cone_len_in)          # cannot exceed total
    cone_taper_len   = cone_len_in - used_cs_depth

    ch_len = max(0.0, float(channel_len))

    # Y levels (top → down)
    y0            = float(y_top)                         # top surface
    y_cs_bot      = y0 - used_cs_depth                  # bottom of countersink taper
    y_cone_bot    = y_cs_bot - cone_taper_len           # bottom of cone taper
    y_channel_bot = y_cone_bot - ch_len                 # end of channel

    ox = float(inset_offset_x)

    # Left side down, right side up, then close — two angled sections then straight
    pts = [
        (ox - cs_r,       y0),             # CS outer (top-left)
        (ox - cone_top_r, y_cs_bot),       # CS → cone (first taper)
        (ox - open_r,     y_cone_bot),     # cone → opening (second taper)
        (ox - open_r,     y_channel_bot),  # straight channel down
        (ox + open_r,     y_channel_bot),  # bottom right
        (ox + open_r,     y_cone_bot),     # up channel right
        (ox + cone_top_r, y_cs_bot),       # up cone right
        (ox + cs_r,       y0),             # CS outer right
    ]

    poly = msp.add_lwpolyline(pts, dxfattribs={"layer": layer, "closed": True})
    return poly

# ---- Geometry: cross-section path (shared by UI + PDF) ----
def build_cross_section_path(
    *,
    opening_dia: float,          # narrow channel diameter (die opening)
    cone_top_dia: float,         # cone TOP diameter (at countersink bottom)
    cs_plus_radius: float,       # countersink extra radius (radius, not diameter)
    countersink_depth: float,    # user countersink depth
    cone_len: float,             # user cone length (will be split into cs + cone)
    channel_len: float,          # user channel length
    x_offset: float = 0.0,
    y_top: float = 0.0,
):
    """
    Builds a single-ended funnel outline (top→down):
      Countersink (cs_dia) → Cone (cone_top_dia→opening_dia) → Channel (opening_dia).
      Cone Length is respected as: used_cs_depth + cone_taper_len = cone_len.
    """
    # Normalize cone pair: if either is ~0, both become 0
    cone_len, cone_top_dia = _link_cone_len_and_width(cone_len, cone_top_dia)

    # (Optional, recommended) if cone is off, kill CS too to avoid stray top width
    if cone_len == 0.0 and cone_top_dia == 0.0:
        countersink_depth = 0.0
        cs_plus_radius    = 0.0

    cs_dia       = float(cone_top_dia) + 2.0 * max(0.0, float(cs_plus_radius))
    opening_d    = float(opening_dia)
    cone_top_d   = float(cone_top_dia)

    cs_r         = cs_dia     / 2.0
    cone_top_r   = cone_top_d / 2.0
    open_r       = opening_d  / 2.0

    cone_len_in    = max(0.0, float(cone_len))
    cs_depth_in    = max(0.0, float(countersink_depth))
    used_cs_depth  = min(cs_depth_in, cone_len_in)     # cap CS by total cone length
    cone_taper_len = cone_len_in - used_cs_depth
    ch_len         = max(0.0, float(channel_len))

    y0            = float(y_top)                       # top surface
    y_cs_bot      = y0 - used_cs_depth                 # bottom of countersink
    y_cone_bot    = y_cs_bot - cone_taper_len          # bottom of cone taper
    y_channel_bot = y_cone_bot - ch_len                # end of channel
    ox            = float(x_offset)

    pts = [
        (ox - cs_r,       y0),             # cs outer (top-left)
        (ox - cone_top_r, y_cs_bot),       # cs → cone
        (ox - open_r,     y_cone_bot),     # cone → channel
        (ox - open_r,     y_channel_bot),  # channel down
        (ox + open_r,     y_channel_bot),  # bottom right
        (ox + open_r,     y_cone_bot),     # up channel
        (ox + cone_top_r, y_cs_bot),       # up cone
        (ox + cs_r,       y0),             # cs outer right
    ]

    metrics = {
        "y_top": y0,
        "y_cs_bot": y_cs_bot,
        "y_cone_bot": y_cone_bot,
        "y_channel_bot": y_channel_bot,
        "cs_dia": cs_dia,
        "cone_top_dia": cone_top_dia,
        "opening_dia": opening_dia,
        "used_cs_depth": used_cs_depth,
        "cone_taper_len": cone_taper_len,
        "channel_len": ch_len,
    }
    return pts, metrics

# ----Cross section for PDF - helpers: render cross-section image with colors/annotations ----

def render_cross_section_for_pdf(
    *,
    opening_dia,
    cone_top_dia,           # cone width (at CS bottom)
    cs_plus_radius,         # countersink radius (extra per side)
    countersink_depth,
    cone_len,
    channel_len,
    colors=None,
    line_w=1.5,
    fig_size_in=(4.8, 2.2),
    dpi=300,
    annot=None,  # annotation config
):
    """
    Cross-section image with neat annotations:
      - Top-center diameter = cone_top_dia + 2*cs_plus_radius (true top dia when CS present)
      - Left list (no lines): CS depth, width after countersink (== cone_top_dia), CS angle, Cone angle
      - Right labels (text only, same x): Total Cone Length (wrappable), Channel length (wrappable)
      - Tick lines: CS-top & Channel thin grey; Cone-width thin grey if CS present, else black
    """
    # ---- palette ----
    pal = {
        "outline": "#111111",
        "labels":  "#222222",
        "guides":  "#888888",  # "thin grey"
    }
    if colors:
        pal.update(colors)

    # ---- annotation config ----
    acfg = {
        # spacing around the shape (fractions of shape width/height)
        "left_pad":    0.26,   # a touch more room on the left column
        "right_pad":   0.26,
        "right_nudge": -0.20, # pull right labels slightly toward the shape
        "top_pad":     0.12,
        "bottom_pad":  0.12,

        "fontsize":   8,
        "label_color": None,     # defaults to pal["labels"]

        # wrapping for right-side labels
        "wrap_total_cone_len": True,
        "wrap_channel_len":     True,

        # diameter labels (symbol first)
        "channel_dia_format":    "⌀{:.2f}",
        "channel_dia_fontweight":"bold",
        "top_dia_format":        "⌀{:.2f}",
        "top_dia_fontweight":    "bold",
        "top_label_offset_mm":    0.1,   # +mm up from the top of the geometry (avoids touching)

        # move the top label *away* from the edge
        "top_pad":               0.16,  # was 0.12
        "top_label_offset_mm":   2.0,   # was 0.5–1.0

        # extra safety margin in the axes limits so text never clips
        "top_headroom_frac":     0.12,  # new: extra top space as a fraction of shape height

        # optional: push the top label downward by N mm (positive moves it down)
        "top_dia_down_mm":       2.5,   # new: set to 0.0 if you don't want this nudge

        # left-side list (no lines)
        "label_cs_depth":        "depth of cs",
        "label_cs_width":        "width after cs",
        "label_cs_angle":        "cs angle",
        "label_cone_angle":      "cone angle",
        "left_list_spacing_frac": 0.095,  # more vertical spacing so it’s not cramped
    }
    if annot:
        acfg.update(annot)

    acfg.setdefault("label_total_cone_len", "Total Cone Length")
    acfg.setdefault("label_channel_len", "Channel length")

    label_color = acfg["label_color"] or pal["labels"]

    import math
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from PIL import Image
    import io

    # ---- geometry (shared) ----
    pts, m = build_cross_section_path(
        opening_dia=opening_dia,
        cone_top_dia=cone_top_dia,
        cs_plus_radius=cs_plus_radius,
        countersink_depth=countersink_depth,
        cone_len=cone_len,
        channel_len=channel_len,
        x_offset=0.0,
        y_top=0.0,
    )

    xs, ys = zip(*pts)
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # derived dimensions & angles
    top_true_dia = m["cone_top_dia"] + 2.0 * max(0.0, float(cs_plus_radius))   # TRUE top diameter
    used_cs      = m["used_cs_depth"]
    cone_taper   = m["cone_taper_len"]

    has_cone = (float(cone_len) > 0.0 and float(cone_top_dia) > 0.0)

    def included_angle_deg(d_top, d_bot, length):
        if length <= 0.0:
            return None
        dr = 0.5 * abs(d_top - d_bot)
        return math.degrees(2.0 * math.atan(dr / length)) if length > 0 else None

    cs_angle_deg   = included_angle_deg(top_true_dia, m["cone_top_dia"], used_cs)
    cone_angle_deg = included_angle_deg(m["cone_top_dia"], m["opening_dia"], cone_taper)

    fig = plt.figure(figsize=fig_size_in, dpi=dpi)
    ax  = fig.add_axes([0, 0, 1, 1])

    # outline
    poly = Polygon(pts, closed=True, fill=False, linewidth=line_w, edgecolor=pal["outline"], zorder=2)
    ax.add_patch(poly)

    # width ticks
    def width_tick(y, width_val, color, lw):
        x0 = -width_val/2.0; x1 = width_val/2.0
        ax.plot([x0, x1], [y, y], color=color, linewidth=lw, zorder=3)
        
    thin_lw = max(0.6, line_w * 0.55)

    # Channel tick (always)
    width_tick(m["y_cone_bot"], m["opening_dia"],  pal["guides"], thin_lw)

    # Cone/countersink tick(s) — only if there IS a cone
    if has_cone:
        if used_cs > 0:
            width_tick(m["y_cs_bot"], m["cone_top_dia"], pal["guides"], thin_lw)
        else:
            width_tick(m["y_cs_bot"], m["cone_top_dia"], pal["outline"], line_w)

    # right-side aligned labels (text only)
    right_x  = x_max + acfg["right_pad"] * w
    right_xn = right_x + acfg["right_nudge"] * w

    # Total Cone Length label (3-line style) — only if there IS a cone
    if has_cone:
        tcl_label = acfg.get("label_total_cone_len", "Total Cone Length")
        tcl_label = tcl_label.replace("Total Cone Length", "Total Cone\nLength")
        total_cone_len = used_cs + cone_taper
        ax.text(
            right_xn, (m["y_cs_bot"] + m["y_cone_bot"]) / 2.0,
            f"{tcl_label}\n= {total_cone_len:.2f} mm",
            color=label_color, fontsize=acfg["fontsize"],
            va="center", ha="left", zorder=5
        )
    else:
        # Either hide entirely, or show a short note
        if acfg.get("show_no_cone_text", True):
            ax.text(
                right_xn, (m["y_cs_bot"] + m["y_cone_bot"]) / 2.0,
                acfg.get("no_cone_text", "No cone"),
                color=label_color, fontsize=acfg["fontsize"],
                va="center", ha="left", zorder=5
            )

    # Channel length — force 2 lines: "Channel\nlength\n= XX.X mm"
    ch_label = acfg.get("label_channel_len", "Channel length").replace("Channel length", "Channel\nlength")
    ch_txt = f"{ch_label}\n= {m['channel_len']:.2f} mm"

    ax.text(
        right_xn,
        (m["y_cone_bot"] + m["y_channel_bot"]) / 2.0,
        ch_txt,
        color=label_color,
        fontsize=acfg["fontsize"],
        va="center",
        ha="left",
        zorder=5,
        linespacing=1.05,
        clip_on=False,
    )  

    # left-side list (no lines): conditional on countersink presence
    left_x   = x_min - acfg["left_pad"] * w
    y_anchor = (m["y_top"] + m["y_cs_bot"]) / 2.0
    dy       = acfg["left_list_spacing_frac"] * h

    left_items = []
    if used_cs > 0:
        left_items.append((f"{acfg['label_cs_depth']} = {used_cs:.2f}", y_anchor))
        left_items.append((f"{acfg['label_cs_width']} = {m['cone_top_dia']:.2f}", y_anchor - 1*dy))
        if cs_angle_deg is not None:
            left_items.append((f"{acfg['label_cs_angle']} = {cs_angle_deg:.1f}°", y_anchor - 2*dy))
    if cone_angle_deg is not None:
        left_items.append((f"{acfg['label_cone_angle']} = {cone_angle_deg:.1f}°",
                           y_anchor - (3 if used_cs > 0 else 1)*dy))

    for txt, yy in left_items:
        ax.text(left_x, yy, txt, color=label_color, fontsize=acfg["fontsize"],
                va="center", ha="right", zorder=5)

    # bottom-center: channel diameter (symbol first)
    ch_txt = acfg["channel_dia_format"].format(m["opening_dia"])
    y_text_bottom = y_min - acfg["bottom_pad"] * h
    ax.text(0.0, y_text_bottom, ch_txt, color=label_color, fontsize=acfg["fontsize"],
            fontweight=acfg.get("channel_dia_fontweight", "normal"),
            va="top", ha="center", zorder=5)

    # top-center: TRUE top diameter — only if there IS a cone
    if has_cone:
        top_txt = acfg["top_dia_format"].format(top_true_dia)
        y_text_top = (
            y_max
            + acfg["top_pad"] * h
            + float(acfg.get("top_label_offset_mm", 1.0))
            - float(acfg.get("top_dia_down_mm", 0.0))
        )
        ax.text(
            0.0, y_text_top, top_txt,
            color=label_color,
            fontsize=acfg.get("top_dia_fontsize", acfg["fontsize"]),
            fontweight=acfg.get("top_dia_fontweight", "normal"),
            va="bottom", ha="center", zorder=5, clip_on=False,
        )

    # frame (ensure enough headroom for the top label)
    ax.set_aspect("equal", adjustable="datalim")

    # horizontal padding unchanged
    x_pad_left  = 0.30 * w
    x_pad_right = 0.35 * w
    ax.set_xlim(x_min - x_pad_left, x_max + x_pad_right)

    # top margin includes pad + label offset + extra headroom
    top_margin = (
        acfg["top_pad"] * h
        + float(acfg.get("top_label_offset_mm", 1.0))
        + float(acfg.get("top_headroom_frac", 0.12)) * h
    )
    ax.set_ylim(y_min - 0.30 * h, y_max + top_margin)

    # belt & suspenders: turn off clipping for any text
    for _t in ax.texts:
        try:
            _t.set_clip_on(False)
        except Exception:
            pass

    ax.axis("off")

    # -> PIL.Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# --- PDF-cross-section style (single source of truth) ---
CS_STYLE = {
    "colors": {
        "outline": "#000000",
        "cs":      "#1f77b4",
        "cone":    "#ff7f0e",
        "chan":    "#2ca02c",
        "labels":  "#111111",
        "guides":  "#888888",
    },
    "line_w": 1.5,
    "fig_size_in": (5.0, 2.3),
    "dpi": 300,
    "annot": {
        "left_pad": 0.24,
        "label_total_cone_len": "Total Cone Length",
        "label_channel_len": "Channel length",
        "wrap_channel_len": True,
        "channel_dia_format": "⌀{:.1f}",
        "channel_dia_fontweight": "bold",
        "show_no_cone_text": True,
        "no_cone_text": "No cone",

        # formatting for right-side labels
        "newline_before_value": True,          # put the value on its own line
        "total_cone_format": "= {:.1f} mm",
        "channel_len_format": "= {:.1f} mm",
    },
}

from PIL import ImageDraw, ImageFont
import re

def _extract_cs_inputs(V: dict):
    """Pull CS inputs with safe fallbacks; compute +radius if only diameter provided."""
    opening_dia  = float((V.get("opening_dia") or V.get("opening_diameter") or 0) or 0)
    cone_top_dia = float((V.get("cone_width")  or 0) or 0)
    channel_len  = float((V.get("channel_length") or 0) or 0)
    cone_len     = float((V.get("cone_length") or 0) or 0)
    cs_depth     = float((V.get("countersink_depth") or V.get("cs_depth") or 0) or 0)
    cs_plus      = float((V.get("countersink_plus_radius") or 0) or 0)

    cone_len, cone_top_dia = _link_cone_len_and_width(cone_len, cone_top_dia)

    if cs_plus <= 0:
        cs_dia_abs = float((V.get("countersink_diameter") or 0) or 0)
        if cs_dia_abs > 0 and cone_top_dia > 0:
            cs_plus = max(0.0, 0.5 * (cs_dia_abs - cone_top_dia))

    return opening_dia, cone_top_dia, channel_len, cone_len, cs_depth, cs_plus

def _render_cs_image(vals_dict, **style_overrides):
    """Render the annotated CS image using unified style; optional overrides per call."""
    style = dict(CS_STYLE)
    # deep-merge subdicts
    style["colors"] = {**CS_STYLE["colors"], **style_overrides.get("colors", {})}
    style["annot"]  = {**CS_STYLE["annot"],  **style_overrides.get("annot",  {})}
    for k in ("line_w", "fig_size_in", "dpi"):
        if k in style_overrides:
            style[k] = style_overrides[k]

    od, cd, Lc, Lk, cs_h, cs_plus = _extract_cs_inputs(vals_dict)

    return render_cross_section_for_pdf(
        opening_dia       = od,
        cone_top_dia      = cd,
        cs_plus_radius    = cs_plus,
        countersink_depth = cs_h,
        cone_len          = Lk,
        channel_len       = Lc,
        colors    = style["colors"],
        line_w    = style["line_w"],
        fig_size_in = style["fig_size_in"],
        dpi       = style["dpi"],
        annot     = style["annot"],
    )

def _place_cs_image(fig, img,
                    base_left=0.68, base_bottom=0.06, box_w=0.30, box_h=0.32,
                    nudge_right_mm=0.0, nudge_down_mm=14.0, margin=0.01,
                    target_ppi=600, interpolation="none"):
    """
    Place a bitmap (PIL image or NumPy array) on the figure in a fixed box.
    - interpolation="none" keeps edges crisp in the exported PDF.
    - Returns the inset Axes (you can add vector overlays later).
    """
    import numpy as np
    from PIL import Image

    # Convert mm nudges to figure fractions
    fig_w_in, fig_h_in = fig.get_size_inches()
    dx_frac = (nudge_right_mm / 25.4) / max(fig_w_in, 1e-6)
    dy_frac = (nudge_down_mm  / 25.4) / max(fig_h_in, 1e-6)

    left   = base_left  + dx_frac + margin
    bottom = base_bottom - dy_frac + margin
    width  = max(1e-4, box_w  - 2*margin)
    height = max(1e-4, box_h  - 2*margin)

    ax_cs = fig.add_axes([left, bottom, width, height])
    ax_cs.set_axis_off()

    # --- Normalize to an array we can imshow reliably ---
    try:
        if isinstance(img, np.ndarray):
            arr = img
        else:
            # Assume PIL-like object; ensure sensible mode
            if hasattr(img, "mode"):
                if img.mode in ("P", "1", "L"):
                    img = img.convert("RGB")
                elif img.mode in ("LA",):
                    img = img.convert("RGBA")
            arr = np.asarray(img)
    except Exception:
        # Last resort: a 1×1 white pixel so we don't crash
        arr = np.ones((1, 1, 3), dtype=np.uint8) * 255

    # Grayscale → RGB; single-channel → 3-channel
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    # Make sure dtype is uint8 (handle floats 0..1 and other ranges)
    if arr.dtype != np.uint8:
        a_min = float(np.nanmin(arr)) if np.size(arr) else 0.0
        a_max = float(np.nanmax(arr)) if np.size(arr) else 1.0
        if a_max <= 1.0 and a_min >= 0.0:
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    # --- Draw without resampling to keep edges crisp ---
    im = ax_cs.imshow(arr, origin="upper", interpolation=interpolation, resample=False)
    try:
        im.set_rasterized(False)  # keep bitmap as-is in vector PDF
    except Exception:
        pass

    h, w = arr.shape[0], arr.shape[1]
    ax_cs.set_xlim(0, w)
    ax_cs.set_ylim(h, 0)
    ax_cs.set_aspect("equal", adjustable="box")
    return ax_cs

#----------------------------------------------------------------------------------

def _format_summary_text(
    pellet_size,
    throughput,           # slider value
    cone_width,
    calc_values,          # dict from calculate step
    rows                  # list of {"num_holes": int, "pcd": float}
):
    def _n(v, nd=2):
        try:
            return round(float(v), nd)
        except Exception:
            return None

    oa_total = _n((calc_values or {}).get("total_plate_open_area"), 0)
    oa_per_t = _n((calc_values or {}).get("open_area_per_tonne"), 0)
    total_len = _n((calc_values or {}).get("die_total_length"), 1)

    lines = []
    if pellet_size is not None:
        lines.append(f"Target pellet size: {_n(pellet_size, 1)} mm")
    if throughput is not None:
        lines.append(f"Designed throughput: {_n(throughput, 1)} t/h")
    if total_len is not None:
        lines.append(f"Total hole length: {total_len:.1f} mm")
    if cone_width is not None:
        lines.append(f"Cone width: {_n(cone_width, 1)} mm")
    if oa_total is not None:
        lines.append(f"OA total: {int(oa_total)} mm²")
    if oa_per_t is not None:
        lines.append(f"OA per t: {int(oa_per_t)} mm²/t/h")

    safe_rows = []
    for i, r in enumerate(rows, start=1):
        nh = r.get("num_holes")
        pcd = r.get("pcd")
        try:
            if nh and pcd is not None:
                nh_i = int(nh)
                pcd_f = float(pcd)
                safe_rows.append((i, nh_i, pcd_f))
        except Exception:
            continue

    for i, nh_i, pcd_f in safe_rows:
        lines.append(f"Row {i}: PCD {pcd_f:.0f} mm — {nh_i} holes")

    return "\\P".join(lines)  # DXF MTEXT line breaks

def _safe_num(x, nd=1, default="–"):
    try:
        v = float(x)
        if nd is None:
            return str(int(round(v)))
        return f"{v:.{nd}f}"
    except Exception:
        return default

def _collect_rows_from_inputs(*row_inputs, max_rows=50):
    """row_inputs = [num_holes1, pcd1, num_holes2, pcd2, ...]"""
    rows = []
    pairs = min(len(row_inputs)//2, max_rows)
    for i in range(pairs):
        n = row_inputs[2*i]
        p = row_inputs[2*i+1]
        try:
            n = int(n)
            p = float(p)
        except Exception:
            continue
        if n > 0 and p > 0:
            rows.append((i+1, p, n))  # (row_index, pcd_mm, hole_count)
    return rows

def _build_summary_text(ps, throughput, cw, total_holes, calc, rows):
    import math

    def fmt(x, nd=1):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "-"
    def fmt0(x):
        try:
            return str(int(round(float(x))))
        except Exception:
            return "-"

    total_len_mm = calc.get("die_total_length")
    oa_total     = calc.get("total_plate_open_area")
    oa_per_t     = calc.get("open_area_per_tonne")

    lines = [
        f"Target pellet size: {fmt(ps)} mm",
        f"Designed Throughput: {fmt(throughput)} t/h",
        f"Cone Width: {fmt(cw)} mm",
        f"Total holes: {fmt0(total_holes)}",
        f"Total hole length: {fmt(total_len_mm)} mm",
        f"OA total: {fmt0(oa_total)} mm²" if oa_total else None,
        f"OA/t: {fmt0(oa_per_t)} mm²/t/h" if oa_per_t else None,
        "Rows:",
    ]
    lines = [ln for ln in lines if ln]

    # rows = iterable of (row_index, pcd_mm, holes)
    for idx, pcd, n in rows:
        try:
            n_i   = int(n)
            pcd_f = float(pcd)
        except Exception:
            n_i, pcd_f = 0, 0.0

        if n_i > 0 and pcd_f > 0:
            # distance between holes on this PCD
            dist = (math.pi * pcd_f) / n_i
            # PCD at 1 dp (compact), spacing at 2 dp (your request)
            lines.append(
                f"Row {idx}: PCD {fmt(pcd_f, 1)} mm - {fmt0(n_i)} holes → {fmt(dist, 2)} mm between holes"
            )
        else:
            lines.append(f"Row {idx}: PCD {fmt(pcd, 1)} mm - {fmt0(n)} holes")

    return r"\P".join(lines)
    
#------Notes helpers--------------------------------------------------

def _overlay_back_segments_on_pdf(ax, doc):
    """
    Matplotlib overlay to ensure BACK_SEGMENTS (walls/fillets) and BACK_SEG_RINGS (PCDs)
    are visible in the PDF regardless of ezdxf backend quirks.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Arc

    try:
        msp = doc.modelspace()

        # Rings (inner/outer segment PCD)
        for e in msp.query("CIRCLE[layer=='BACK_SEG_RINGS']"):
            c = e.dxf.center
            r = float(e.dxf.radius)
            ax.add_patch(Circle((float(c[0]), float(c[1])), r,
                                fill=False, linewidth=1.2, edgecolor=(0.5, 0.5, 0.5), zorder=5000))

        # Walls (lines)
        for e in msp.query("LINE[layer=='BACK_SEGMENTS']"):
            x0, y0 = float(e.dxf.start[0]), float(e.dxf.start[1])
            x1, y1 = float(e.dxf.end[0]),   float(e.dxf.end[1])
            ax.plot([x0, x1], [y0, y1], linewidth=1.2, color=(0.2, 0.2, 0.2), zorder=5000)

        # Fillets (arcs)
        for e in msp.query("ARC[layer=='BACK_SEGMENTS']"):
            c = e.dxf.center
            r = float(e.dxf.radius)
            sa = float(getattr(e.dxf, "start_angle", 0.0))
            ea = float(getattr(e.dxf, "end_angle",   360.0))
            ax.add_patch(Arc((float(c[0]), float(c[1])), 2*r, 2*r, angle=0.0,
                             theta1=sa, theta2=ea, linewidth=1.2, edgecolor=(0.2, 0.2, 0.2), zorder=5000))
                             
        # NEW: inlet PCD rings overlay (drawn black)
        for e in msp.query("CIRCLE[layer=='BACK_INLET_PCD']"):
            c = e.dxf.center
            r = float(e.dxf.radius)
            ax.add_patch(Circle((float(c[0]), float(c[1])), r,
                             fill=False, linewidth=1.2, edgecolor=(0,0,0), zorder=5000))
                             
    except Exception:
        # stay silent in release; you already log inside exporter if needed
        pass

def _ensure_notes_layer(doc):
    """
    Ensure a 'NOTES' layer exists, prints, and uses a near-black plotting color.
    Safe + idempotent across ezdxf versions.
    """
    try:
        lyr = doc.layers.get("NOTES")
    except Exception:
        lyr = None

    if lyr is None:
        # Use ACI 7 (white/black) so it plots as black on PDFs; 250 is another “near-black” option
        try:
            lyr = doc.layers.add("NOTES", dxfattribs={"color": 7})
        except Exception:
            # Fallback for very old ezdxf
            lyr = doc.layers.new("NOTES")
            try:
                lyr.dxf.color = 7
            except Exception:
                pass
    else:
        # Make sure color stays readable in print
        try:
            lyr.dxf.color = 7   # ACI 7 -> plots black on white
        except Exception:
            pass

    # Make sure it’s plottable (API varies slightly by version)
    try:
        lyr.set_plot(True)
    except Exception:
        try:
            lyr.dxf.plot = True
        except Exception:
            pass

def _add_summary_mtext(doc, msp, plate_diameter_mm, text):
    """Place a wrapped MTEXT box just to the right of the plate, in black."""
    try:
        from ezdxf.enums import MTextEntityAlignment as MT_ALIGN
    except Exception:
        MT_ALIGN = type("MT_ALIGN", (), {"TOP_LEFT": 2})

    _ensure_notes_layer(doc)

    R = float(plate_diameter_mm) / 2.0
    margin = max(5.0, plate_diameter_mm * 0.05)
    x = R + margin
    y = R - margin * 0.5

    char_h = max(2.2, min(5.5, plate_diameter_mm * 0.014))   # ~10–15% smaller font
    box_w  = max(60.0, plate_diameter_mm * 0.42)             # ~+10% wider box

    # Explicit entity color so we’re not relying on ByLayer
    # Style is optional but keeps some renderers happy
    dxfattrs = {
        "layer": "NOTES",
        "char_height": char_h,
        "color": 7,             # <<— force dark text
        "style": "Standard",
    }

    # Ensure Standard exists (older ezdxf files can lack it)
    try:
        doc.styles.get("Standard")
    except Exception:
        try:
            doc.styles.add("Standard")
        except Exception:
            dxfattrs.pop("style", None)

    m = msp.add_mtext(text, dxfattribs=dxfattrs)
    try:
        m.set_location(insert=(x, y), align=MT_ALIGN.TOP_LEFT)
    except Exception:
        m.dxf.insert = (x, y)

    try:
        m.set_text_width(box_w)
    except Exception:
        m.dxf.width = box_w

def _add_summary_mtext_fallback(doc, plate_diameter_mm, text):
    """
    Bullet-proof fallback: render each MTEXT line as a simple TEXT entity.
    Used only if MTEXT creation fails for any reason.
    """
    _ensure_notes_layer(doc)
    msp = doc.modelspace()

    R = float(plate_diameter_mm) / 2.0
    margin = max(5.0, plate_diameter_mm * 0.05)
    x = R + margin
    y = R - margin * 0.5
    h = max(2.5, min(6.0, plate_diameter_mm * 0.016))
    dy = h * 1.35

    try:
        from ezdxf.enums import TextEntityAlignment as TXT_ALIGN
    except Exception:
        TXT_ALIGN = type("TXT_ALIGN", (), {"TOP_LEFT": 3})

    lines = text.replace(r"\P", "\n").splitlines()
    for i, ln in enumerate(lines):
        t = msp.add_text(
            ln,
            dxfattribs={"layer": "NOTES", "height": h, "color": 7, "style": "Standard"},
        )
        try:
            t.set_pos((x, y - i * dy), align=TXT_ALIGN.TOP_LEFT)
        except Exception:
            t.dxf.insert = (x, y - i * dy)

# -------- Persistence -------------------------------------------

def list_saved_designs(save_dir=SAVE_DIR):
        os.makedirs(save_dir, exist_ok=True)
        files = glob.glob(os.path.join(save_dir, "*.json"))
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)  # newest first
        return [os.path.basename(f) for f in files]

from pathlib import Path
from typing import Dict, Any

def load_die_design(filename: str, save_dir: Path = SAVE_DIR) -> Dict[str, Any]:
    """
    Load a saved die design JSON from `save_dir/filename`, backfill new inputs,
    and return the dict. Raises on errors.
    """
    # Normalize and guard against path traversal
    save_dir = Path(save_dir).resolve()
    path = (save_dir / filename).resolve()
    if not str(path).startswith(str(save_dir)):
        raise ValueError(f"Illegal path outside save_dir: {path}")

    if path.suffix.lower() != ".json":
        log_error(f"load_die_design: forcing .json extension for {path}")
    if not path.exists():
        log_error(f"load_die_design: file not found: {path}")
        raise FileNotFoundError(path)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        log_error(f"load_die_design: invalid JSON in {path}: {e}")
        raise
    except Exception as e:
        log_error(f"load_die_design: read error for {path}: {e}")
        raise

    # Backfill new inputs safely
    try:
        if isinstance(data, dict):
            inputs = data.get("inputs")
            if isinstance(inputs, dict):
                data["inputs"] = _backfill_new_inputs(inputs)
    except Exception as e:
        # Don’t fail load for a backfill hiccup; just log it.
        log_error(f"load_die_design: backfill failed for {path}: {e}")

    return data

def save_die_design(data, save_dir=SAVE_DIR, filename=None):
    try:
        os.makedirs(save_dir, exist_ok=True)
        if not filename:
            filename = os.path.join(save_dir, f"die_design_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log_error(f"Design saved to {filename}")
        return filename
    except Exception as e:
        log_error(f"Error saving design: {e}")
        raise

def _backfill_new_inputs(inp: dict) -> dict:
    inp.setdefault("countersink_diameter", 0.0)
    inp.setdefault("countersink_depth", 0.0)
    return inp

# --------- UI --------------------------------------
def build_ui():
    with gr.Blocks(
        css="""
/* ===== App Header / Banner ===== */
.app-header {
  position: sticky;         /* sticks to top on scroll */
  top: 0;
  z-index: 50;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 10px 16px;
  background: #3f4e73;      /* dark navy */
  color: #e5e7eb;           /* slate-200 */
  border-bottom: 1px solid rgba(255,255,255,.06);
}

.app-title {
  font-size: 1.25rem;       /* ~20px */
  font-weight: 700;
  letter-spacing: .2px;
}

.app-meta {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: .9rem;
  color: #FDFDFD;           /* slate-300 */
  white-space: nowrap;
}

.app-meta .badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(255,255,255,.08);
  border: 1px solid rgba(255,255,255,.12);
  color: #e5e7eb;
  font-weight: 600;
  line-height: 1.4;
}

.app-meta .sep { opacity: .35;
}
/* Make the header title pure white (overrides .app-header's color) */
.app-header .app-title {
  color: #f8fafc;
}

/* If something else keeps overriding it, force it: */
/* .app-header .app-title { color: #ffffff !important; } */


/* PANELS & LAYOUT */
.card-panel,
.button-panel,
.metadata-panel {
  background-color: #fcfcfc;
  border: 2px solid #c5d9f0;
  border-radius: 10px;
  padding: 12px 14px;
  margin: 12px 0;
}
.panel-spacer { height: 6px; }

.responsive-container {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
@media (min-width: 768px) {
  .responsive-container { flex-direction: row; align-items: stretch; }
}

.soft-bg {
  background: #e4ecf5;
  border: 1px solid #c5d9f0;
  border-radius: 10px;
  padding: 10px;
}
.dark-bg {
  background: #cbe5f7;
  border: 1px solid #c5d9f0;
  border-radius: 10px;
  padding: 10px;
}

/* REMOVE LABEL BACKGROUNDS INSIDE .no-label-bg */
.no-label-bg {
  --block-title-background: transparent !important;
  --block-label-background: transparent !important;
  --block-title-background-fill: transparent !important;
  --block-label-background-fill: transparent !important;
}
.no-label-bg .label,
.no-label-bg .gr-label,
.no-label-bg .block-title,
.no-label-bg .wrap > label,
.no-label-bg label {
  background: transparent !important;
  box-shadow: none !important;
  border: 0 !important;
  padding: 0 !important;
  margin-bottom: 6px !important;
}

/* METADATA PANEL (COMPACT) */
.metadata-panel label {
  font-size: var(--meta-font) !important;
  line-height: 1.2 !important;
  margin-bottom: 2px !important;
}
.metadata-panel .compact-input input,
.metadata-panel .compact-input select,
.metadata-panel .compact-input textarea { box-sizing: border-box !important; }
.metadata-panel .compact-input .gr-textbox input,
.metadata-panel .compact-input .gr-textbox textarea,
.metadata-panel .compact-input .gr-text-input input {
  height: var(--meta-h) !important;
  line-height: var(--meta-h) !important;
  padding: 0 var(--meta-pad-x) !important;
  font-size: var(--meta-font) !important;
}

/* ===========================
   TABLE CELLS IN CARD PANELS
   =========================== */
.card-panel .panel-table table thead th,
.card-panel .panel-table table tbody td {
  padding: 6px 8px !important;
  font-size: 14px !important;
  line-height: 22px !important;
  height: 24px !important;
  text-align: center;
  vertical-align: middle;
}


/* ===========================
   SAVED DIE DESIGNS – COMPACT HTML MIRROR
   =========================== */

/* Scroll host for the HTML mirror table */
.saved-grid-html {
  /* Layout */
  position: relative;
  width: 100%;
  contain: layout paint;

  /* Tunables */
  --col-w: 7rem;
  --font: 12px;
  --row-h: 24px;

  /* Scrolling + size (~25 rows + header) */
  max-height: calc(var(--row-h) * 36 + 8px);
  overflow-x: auto;
  overflow-y: auto;
  scroll-behavior: smooth;
  overscroll-behavior: contain;
  /* Prevent tiny jiggles when scrollbars appear/disappear */
  scrollbar-gutter: stable both-edges;
}

/* Table layout */
.saved-grid-html table {
  width: 100%;
  table-layout: fixed;
  border-collapse: collapse;
}

/* Column widths (match your generator) */
.saved-grid-html table colgroup col {
  width: var(--col-w);
}

/* Cells (body + header shared) */
.saved-grid-html thead th,
.saved-grid-html tbody td {
  padding: 2px 4px;
  font-size: var(--font);
  line-height: 1.15;
  height: 22px;
  text-align: center;
  vertical-align: middle;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  /* smoothen hover/selection color changes a touch */
  transition: background-color 120ms linear;
}

/* Sticky header */
.saved-grid-html thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  font-weight: 600;
  white-space: normal;
  word-break: break-word;
  overflow-wrap: anywhere;
  hyphens: auto;
  background: linear-gradient(#f4f7fb, #e9f0f8);
  border-bottom: 1px solid #c9d7ea;
}

/* Zebra rows */
.saved-grid-html tbody tr:nth-child(even) td {
  background: #fafbfc;
}
/* Hover highlight (doesn’t override selected row) */
.saved-grid-html tbody tr:not(.selected):hover td {
  background: #f0f4fa;
}

/* Padded rows */
.saved-grid-html tbody tr.pad td {
  opacity: 0.7;
}

/* Selected/highlighted row */
.saved-grid-html tbody tr.selected td {
  background: linear-gradient(#fffbe6, #ffeab0);
  border-bottom: 1px solid #f0d67a;
  box-shadow: inset 0 0 0 1px #f2cf5b;
}

/* Small “DEL” pill style (even though that column is hidden in mirror) */
.saved-grid-html .del-pill {
  display: inline-block;
  height: 16px;
  padding: 0 6px;
  border: 1px solid #c33;
  border-radius: 9999px;
  font-size: 10px;
  line-height: 16px;
  color: #c33;
  background: #fff5f5;
}

/* Hide first column (DEL) in the HTML mirror */
.saved-grid-html colgroup col:first-child,
.saved-grid-html th:first-child,
.saved-grid-html td:first-child {
  display: none;
}

/* Chips */
.chip {
  display: inline-block;
  padding: 2px 8px;
  margin: 2px 4px;
  border-radius: 12px;
  border: 1px solid #ccc;
  font-size: 12px;
}
.chip.ok   { background: #e8f5e9; border-color: #81c784; }
.chip.warn { background: #fff3e0; border-color: #ffb74d; }

/* Muted tail text (e.g., filename · timestamp) */
.muted { opacity: .6; }

/* (Optional) Hide the accordion that wraps the Gradio DataFrame */
#selector_acc { display: none; }

/* Reduce “bounce” when messages change */
#selected_header_html, #manager_status { min-height: 1.4em; }

/* Selected header: darker + bigger + bolder */
#selected_header_html{
  color: #1b5e20;      /* darker green */
  font-weight: 900;    /* max bold that browsers honor */
  font-size: 1.8rem;   /* bump as you like: 1.8rem if needed */
  line-height: 1.35;
}

/* Keep the “Selected:” label bold, too */
#selected_header_html strong{ font-weight: 900; }

/* Make the tail (short filename + timestamp) subtle but readable */
#selected_header_html .muted{
  opacity: .55;
  font-weight: 400;
}
/* Info chip for in-progress messages */
.chip.info { background:#e3f2fd; border-color:#64b5f6; color:#0b57d0; font-weight:600; }

/* Check icon */
.chip .check{
  display:inline-block; width:1em; text-align:center;
  font-weight:900; margin-right:6px;
}

""",

        title="Extruder Die Design Tool") as demo:
        # --- App Header / Banner ---
        APP_NAME = "Extruder Die Design Tool"
        APP_VERSION = "v2.0.1"

        header_html = f"""
        <div class="app-header">
        <div class="app-title">{APP_NAME}</div>
        <div class="app-meta">
        <span class="badge">{APP_VERSION}</span>
        <span class="sep">•</span>
        <span>Build ready</span>
        </div>
        </div>
        """
        gr.HTML(header_html)
        
        ##-----------Favicon Fun-------------------------           
        try:
            _data_url, _mime, _src = _favicon_data_url()
            js = """
            () => {{
              ['link[rel="icon"]','link[rel="shortcut icon"]','link[rel="apple-touch-icon"]']
                .forEach(sel => document.querySelectorAll(sel).forEach(n => n.remove()));

              const href = '{href}';
              const type = '{mime}';

              const add = (rel) => {{
                const l = document.createElement('link');
                l.rel = rel; l.type = type; l.href = href;
                document.head.appendChild(l);
              }};
              add('icon');
              add('shortcut icon');
            }}
            """.format(href=_data_url, mime=_mime)

            demo.load(fn=None, inputs=None, outputs=None, js=js)
            logger.info("Injected favicon via demo.load(js=...) (%s) from %s", _mime, _src)
        except Exception as e:
            logger.info("Favicon JS inject skipped: %s", e)

        with gr.Row():
            with gr.Column(elem_classes=["soft-bg"]):
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

            with gr.Column(elem_classes=["soft-bg"]):
                gr.Markdown("### Die Design Parameters")
                opening_dia = gr.Slider(0.3, 22, value=6.3, step=0.01, label="Die Diameter (mm)")
                channel_length = gr.Slider(0, 30, value=14.0, step=0.1, label="Channel/Land Length (mm)")
                holes = gr.Slider(0, 7000, step=1, value=64, label="Number of Holes")

            with gr.Column(elem_classes=["soft-bg"]):
                gr.Markdown("### Cone Parameters")
                cone_width = gr.Slider(0, 22, value=13.0, step=0.1, label="Cone Width (mm)")
                cone_length = gr.Slider(0, 30, value=4.0, step=0.1, label="Cone Length (mm)")
                countersink_diameter = gr.Slider(label="Countersink Radius (mm)", minimum=0.0, maximum=5.0, step=0.1, value=0.0, interactive=True)
                countersink_depth = gr.Slider(label="Countersink Depth (mm)", minimum=0.0, maximum=12.0, step=0.1, value=0.0, interactive=True)

        with gr.Row(elem_classes=["responsive-container"]):
            with gr.Column(elem_classes=["card-panel"]):
                gr.Markdown("<h4>Hole Cross Section</h4>")
                output_image = gr.HTML()

            with gr.Column(elem_classes=["card-panel", "panel-table"]):
                gr.Markdown("<h4>Your Design vs Suggestions</h4>")
                gr.HTML("<div class='panel-spacer'></div>")
                output_table = gr.HTML()
                gr.HTML("<div class='panel-spacer'></div>")
                with gr.Row():
                    calculate_btn = gr.Button("Calculate")
                with gr.Row(): 
                    reset_btn = gr.Button("Reset Design")
                    save_button = gr.Button("Save", elem_classes=["icon-save-button"])                            
                save_status = gr.HTML(value="<div></div>", visible=True)

            with gr.Column(elem_classes=["metadata-panel"]):
                gr.Markdown("<h4>Design Metadata</h4>")
                extruder_type = gr.Textbox(label="Extruder Type", placeholder="Wenger X-185", lines=1, elem_classes=["compact-input"])
                production_line = gr.Textbox(label="Line", placeholder="Line 3", lines=1, elem_classes=["compact-input"])
                plant_name = gr.Textbox(label="Plant", placeholder="Plant or place name", lines=1, elem_classes=["compact-input"])
                product_type = gr.Textbox(label="Material", placeholder="Die material", lines=1, elem_classes=["compact-input"])
                die_performance = gr.Textbox(label="Die Marking", placeholder="Die ID or marking", lines=1, elem_classes=["compact-input"])
                comments = gr.Textbox(label="Comments", placeholder="Comments or die performance notes", lines=1, elem_classes=["compact-input"])
        with gr.Row():
            with gr.Column(elem_classes=["soft-bg"]):
                gr.Markdown("#### Plate sizing")
                plate_diameter = gr.Slider(label="Plate Outer Diameter", minimum=100, maximum=600, step=0.1, value=250)
                chamfer_plate = gr.Slider(label="Outer Chamfer 45°, if any", minimum=0, maximum=10, step=0.1, value=2)
                gr.HTML("<div style='height:10px'></div>")  # <-- small vertical spacer
                die_center_hole_diameter = gr.Slider(label="Die Center Hole Diameter", minimum=0, maximum=200, step=0.1, value=48.5)
                chamfer_center_hole = gr.Slider(label="Hole Chamfer 45°, if any", minimum=0, maximum=10, step=0.1, value=2)

            with gr.Column(elem_classes=["soft-bg"]):
                gr.Markdown("#### Cutting Track")
                outer_opening_pcd = gr.Slider(label="Outer cutting track - PCD", minimum=0, maximum=350, step=0.1, value=189)
                chamfer_outer_opening = gr.Slider(label="Chamfer 45°, if any", minimum=0, maximum=10, step=0.1, value=0)
                gr.HTML("<div style='height:10px'></div>")  # <-- small vertical spacer
                inner_opening_pcd = gr.Slider(label="Inner cutting track - PCD", minimum=0, maximum=250, step=0.1, value=128)
                chamfer_inner_opening = gr.Slider(label="Chamfer 45°, if any", minimum=0, maximum=10, step=0.1, value=0)

            with gr.Column(elem_classes=["soft-bg"]):
                gr.Markdown("#### Bolts")
                outer_bolt_count = gr.Slider(label="Outer Bolts - Number of holes", minimum=0, maximum=MAX_BOLT_HOLES, value=8, step=1)
                outer_bolt_pcd = gr.Slider(label="Outer Bolts - PCD", minimum=0, maximum=400, step=0.1, value=230)
                outer_bolt_diameter = gr.Slider(label="Outer Bolts - Hole Diameter", minimum=0, maximum=20, step=0.1, value=10.5)
                gr.HTML("<div style='height:10px'></div>")  # <-- small vertical spacer
                inner_bolt_count = gr.Slider(label="Inner Bolts - Number of holes", minimum=0, maximum=MAX_BOLT_HOLES, value=4, step=1)
                inner_bolt_pcd = gr.Slider(label="Inner Bolts - PCD", minimum=0, maximum=200, step=0.1, value=75)
                inner_bolt_diameter = gr.Slider(label="Inner Bolts - Hole Diameter", minimum=0, maximum=20, step=0.1, value=8.2)

            with gr.Column(elem_classes=["soft-bg"]):
                gr.Markdown("#### Extra circles - cone side")
                inlet_inside  = gr.Slider(label="Inlet inside - PCD (mm)",  minimum=0.0, maximum=300.0, step=0.1, value=113.0)
                inside_taper  = gr.Slider(label="Taper towards holes - PCD (mm)",  minimum=0.0, maximum=300.0, step=0.1, value=121.0)
                gr.HTML("<div style='height:10px'></div>")  # <-- small vertical spacer
                inlet_outside = gr.Slider(label="Outer PCD (mm)", minimum=0.0, maximum=600.0, step=0.1, value=204.0)
                outside_taper = gr.Slider(label="Taper leading to holes - PCD (mm)", minimum=0.0, maximum=600.0, step=0.1, value=194.0) 

        # ---- Holes & Rows: keep header + accordion inside ONE column
        die_style_state = gr.State(DIE_STYLE_CIRCULAR)
        with gr.Row():
            with gr.Column(elem_classes=["dark-bg"], scale=2, min_width=520):
                gr.Markdown("#### Holes and Rows")
                visible_rows = gr.State(ROWS_STEP)
                row_components = []

                # ---- Layout selection (tabs, no accordions) ----
                with gr.Tabs() as hole_pattern_tabs:  

                    # ----------------- Circular pattern tab - TAB 1 -----------------
                    with gr.Tab("Circular pattern") as tab_circ:
                        # collect per-row controls (holes + PCD) so we can toggle visibility
                        row_components = []

                        # Number of rows control (inside this tab)
                        line_rows = gr.Slider(0, MAX_ROWS, step=1, value=2, label="Number of Rows")

                        current_row = None
                        for i in range(MAX_ROWS):
                            if i % 3 == 0:
                                current_row = gr.Row()
                            with current_row:
                                # One column per row/PCD pair; holes stacked above PCD
                                with gr.Column(min_width=220):
                                    num_holes = gr.Slider(
                                        label=f"Row {i+1} - Holes",
                                        minimum=0, maximum=250, step=1, value=0,
                                        visible=True if i < ROWS_STEP else False,  # initial; slider will update
                                    )
                                    pcd = gr.Slider(
                                        label=f"Row {i+1} - PCD (mm)",
                                        minimum=90, maximum=350, step=0.1, value=0.0,
                                        visible=True if i < ROWS_STEP else False,
                                    )
                                # >>> this must be INSIDE the for-loop <<<
                                row_components.append((num_holes, pcd))

                        # Buttons for circular pattern
                        with gr.Row():
                            generate_btn = gr.Button("Generate Drawings", variant="primary", scale=0)
                            autofill_btn = gr.Button("Autofill Rows & PCDs")
                            add_rows_btn = gr.Button("Add Rows")
                            pcd_reset_btn = gr.Button("PCD reset", variant="secondary")

                        # ---- wiring: slider controls which row widgets are visible ----
                        def _toggle_row_visibility(n_rows: int):
                            updates = []
                            n = int(n_rows or 0)
                            for idx, (holes_row, p_row) in enumerate(row_components):
                                vis = idx < n
                                updates.extend([gr.update(visible=vis), gr.update(visible=vis)])
                            return updates

                        # Flatten outputs (holes, pcd) for every row
                        row_vis_outputs = []
                        for holes_row, p_row in row_components:
                            row_vis_outputs += [holes_row, p_row]

                        # Single change handler — this is the only one you need
                        line_rows.change(
                            fn=_toggle_row_visibility,
                            inputs=[line_rows],
                            outputs=row_vis_outputs,
                        )
                        
                        # Add exactly ONE row per click
                        def _add_rows(n_rows: int):
                            n = min(int(n_rows or 0) + 1, MAX_ROWS)
                            return [gr.update(value=n)] + _toggle_row_visibility(n)

                        add_rows_btn.click(
                            fn=_add_rows,
                            inputs=[line_rows],
                            outputs=[line_rows] + row_vis_outputs,
                        )

                    # ----------------- Staggered pattern tab - TAB 2  -----------------
                    with gr.Tab("Staggered pattern") as tab_stag:
                        with gr.Row():
                            # ---- Column 1: inner/outer PCD ----
                            with gr.Column():
                                gr.HTML("<div class='panel-spacer'></div>")
                                seg_inner_pcd = gr.Slider(
                                    label="Segment inner PCD (mm)",
                                    minimum=50.0,
                                    maximum=250.0,
                                    step=0.1,
                                    value=120.0,
                                    interactive=True,
                                )
                                gr.HTML("<div style='height:6px'></div>")
                                seg_outer_pcd = gr.Slider(
                                    label="Segment outer PCD (mm)",
                                    minimum=100.0,
                                    maximum=450.0,
                                    step=0.1,
                                    value=220.0,
                                    interactive=True,
                                )
                                gr.HTML("<div style='height:18px'></div>")  # <-- small vertical spacer
                                stag_generate_btn = gr.Button("Generate Staggered", variant="primary")

                            # ---- Column 2: segments / wall width ----
                            with gr.Column():
                                gr.HTML("<div class='panel-spacer'></div>")
                                stag_segments = gr.Slider(
                                    label="Number of segments (N)",
                                    minimum=3,
                                    maximum=6,
                                    step=1,
                                    value=5,
                                    interactive=True,
                                )
                                gr.HTML("<div style='height:6px'></div>")
                                wall_width = gr.Slider(
                                    label="Wall segment width (mm)",
                                    minimum=5.0,
                                    maximum=30.0,
                                    step=0.1,
                                    value=10.0,
                                    interactive=True,
                                )

                            # ---- Column 3: radius / padding ----
                            with gr.Column():
                                gr.HTML("<div class='panel-spacer'></div>")
                                corner_radius = gr.Slider(
                                    label="Corner radius R (mm)",
                                    minimum=0.0, maximum=15.0, step=0.1, value=5.0,
                                    interactive=True,
                                )
                                gr.HTML("<div style='height:6px'></div>")
                                padding_adj = gr.Slider(
                                    label="Padding adjustment (mm)",
                                    minimum=0.0, maximum=10.0, step=0.1, value=0.0,
                                    interactive=True,
                                )

                             # ---- Column 4: bolts and bits ----
                            with gr.Column():
                                gr.HTML("<div class='panel-spacer'></div>")
                                seg_bolts_chk = gr.Checkbox(
                                    label="Add segment bolts (requires 3 segments)",
                                    value=False,
                                    interactive=True,
                                )
                                
                                stag_total = gr.HTML(
                                    value=_render_target_holes(getattr(holes, "value", 64)),
                                    label=""
                                )

                        # alias to keep compatibility with places that expect `segments`
                        segments = stag_segments

                        with gr.Row():
                            gr.HTML("<div class='panel-spacer'></div>")
                            gr.HTML("<div class='panel-spacer'></div>")
                            btn_stag_calc = gr.Button("Check fit", variant="secondary")
                            stag_counts = gr.HTML()
                            stag_msg   = gr.Markdown()

                            # Force the generator down the staggered code path without relying on other tabs
                            die_style_state_stag = gr.State("Staggered")
                            staggered_pts_state = gr.State(value=None)

                        def _toggle_seg_bolts_chk(s):
                            enable = int(s or 0) == 3
                            # when disabling, also uncheck so it doesn’t leak a True
                            return gr.update(interactive=enable, value=False if not enable else None)

                        # keep the checkbox synced with the current Segments input
                        stag_segments.change(_toggle_seg_bolts_chk, inputs=stag_segments, outputs=seg_bolts_chk)
 
                    # ----------------- Circular on Segmented (Hybrid) tab --TAB 3 ---------------
                    with gr.Tab("Circular on Segmented") as tab_circseg:
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### Segments")
                                stag_segments_t3 = gr.Slider(3, 18, value=8, step=1, label="Segment numbers", interactive=True)
                                gr.HTML("<div style='height:10px'></div>")  # small vertical spacer
                                wall_width_t3    = gr.Slider(0.0, 50.0, value=8.0, step=0.5, label="Wall thickness of segment (mm)", interactive=True)
                                gr.HTML("<div style='height:10px'></div>")  # small vertical spacer
                                seg_inner_pcd_t3 = gr.Slider(0, 500, value=120, step=1, label="Inner PCD (mm)", interactive=True)

                            with gr.Column():
                                gr.Markdown("#### More")
                                seg_outer_pcd_t3 = gr.Slider(0, 500, value=198, step=1, label="Outer PCD (mm)", interactive=True)
                                gr.HTML("<div style='height:10px'></div>")  # small vertical spacer
                                corner_radius_t3 = gr.Slider(0.0, 30.0, value=5.0, step=0.5, label="Segment corner radius (mm)", interactive=True)
                                gr.HTML("<div style='height:10px'></div>")  # small vertical spacer
                                padding_adj_t3   = gr.Slider(0.0, 20.0, value=2.0, step=0.5, label="Padding (mm)",  interactive=True)
                                show_segment_walls_t3 = gr.Checkbox(
                                    label="Show segment walls",
                                    value=True,
                                    interactive=True,
                                )

                                seg_bolts_chk_t3 = gr.State(False)

                        # --- Tab 3's own Circular rows (independent of Tab 1) ---
                        gr.Markdown("#### Holes and Rows (Tab 3)")
                        circseg_line_rows = gr.Slider(0, MAX_ROWS, step=1, value=2, label="Number of Rows (Tab 3)")

                        circseg_row_components = []     # [(num_holes_slider, pcd_slider), ...]
                        flattened_row_inputs_tab3 = []  # [num1, pcd1, num2, pcd2, ...]

                        current_row = None
                        for i in range(MAX_ROWS):
                            if i % 3 == 0:
                                current_row = gr.Row()
                            with current_row:
                                with gr.Column(min_width=220):
                                    t3_num_holes = gr.Slider(
                                        label=f"[T3] Row {i+1} - Holes",
                                        minimum=0, maximum=250, step=1, value=0,
                                        visible=True if i < ROWS_STEP else False,
                                    )
                                    t3_pcd = gr.Slider(
                                        label=f"[T3] Row {i+1} - PCD (mm)",
                                        minimum=90, maximum=350, step=0.1, value=0.0,
                                        visible=True if i < ROWS_STEP else False,
                                    )
                                circseg_row_components.append((t3_num_holes, t3_pcd))
                                flattened_row_inputs_tab3.extend([t3_num_holes, t3_pcd])

                        # Visibility wiring for Tab 3 row widgets
                        def _t3_toggle_row_visibility(n_rows: int):
                            updates = []
                            n = int(n_rows or 0)
                            for idx, (holes_row, p_row) in enumerate(circseg_row_components):
                                vis = idx < n
                                updates.extend([gr.update(visible=vis), gr.update(visible=vis)])
                            return updates

                        _t3_row_vis_outputs = []
                        for holes_row, p_row in circseg_row_components:
                            _t3_row_vis_outputs += [holes_row, p_row]

                        circseg_line_rows.change(
                            fn=_t3_toggle_row_visibility,
                            inputs=[circseg_line_rows],
                            outputs=_t3_row_vis_outputs,
                        )

                        with gr.Row():
                            circseg_generate_btn = gr.Button("Generate Hybrid", variant="primary")
                            circseg_autofill_btn = gr.Button("Autofill Rows & PCDs", variant="secondary")
                            circseg_counts = gr.Markdown()
                            circseg_msg = gr.Markdown()

                        # stores culled holes for preview/export (from Tab 3 logic, if you use it)
                        circseg_pts_state = gr.State(value=None)

        # ---- Previews on their own row (clean separation)
        with gr.Row():
            with gr.Column():
                plate_preview = gr.Image(label="Shown from knife side", type="pil")
                with gr.Row(equal_height=True):

                    dxf_dl       = gr.DownloadButton("Open Autocad Drawing", interactive=False, visible=False, scale=0, min_width=140)
                    open_pdf_btn = gr.Button("Open PDF", interactive=False, scale=0, min_width=140)
                    quick_pdf    = gr.File(label="PDF", visible=False)

                    stale_banner = gr.Markdown("Drawings need to be Generated again", visible=False)

                with gr.Row():
                    status_html = gr.HTML(value="", label="", elem_id="gen-status")
                    pdf_status = gr.HTML(value="", visible=True)

            with gr.Column():
                inset_preview = gr.Image(label="Inlet side showing cones", type="pil")
                inset_zoom = gr.Slider(
                    label="Zoom in on holes",
                    minimum=1.0, maximum=15.0, step=0.1,
                    value=float(globals().get("INITIAL_INSET_ZOOM", 1.20))  # safe fallback
                )
                inset_status = gr.HTML(value="", elem_id="inset_status")

        def _previews_only(*params):
            """Call your generator and return only the heavy preview bits."""
            plate_img, inset_img, dxf_path = generate_die_plate_dxf(*params)
            return (
                gr.update(value=plate_img, visible=True),
                gr.update(value=inset_img,  visible=True),
                dxf_path,
            )

        def _hybrid_previews_only(*args):
            """
            Same idea but accepts circseg_pts_state as the *last* arg (holes_override).
            Returns only (plate_preview, inset_preview, dxf_path_state).
            """
            if not args:
                return gr.update(), gr.update(), ""

            params, raw_points = args[:-1], args[-1]

            # Separate Tab 3 row sliders (fixed count) and the visibility toggle
            params = list(params)
            ROW_INPUT_COUNT = MAX_ROWS * 2
            row_values = []
            if len(params) >= ROW_INPUT_COUNT:
                row_values = params[-ROW_INPUT_COUNT:]
                params = params[:-ROW_INPUT_COUNT]
            tab3_show_walls = True
            if params:
                candidate = params.pop()
                cand_type = type(candidate).__name__
                if isinstance(candidate, bool) or cand_type in {"bool_", "bool8"}:
                    tab3_show_walls = bool(candidate)
                else:
                    params.append(candidate)
            all_params = params + row_values

            def _normalize_points(v):
                v = getattr(v, "value", v)
                if isinstance(v, dict):
                    v = v.get("points") or v.get("kept") or v.get("holes") or v.get("data")
                try:
                    import pandas as _pd
                    if isinstance(v, _pd.DataFrame):
                        cols = {c.lower(): c for c in v.columns}
                        if "x" in cols and "y" in cols:
                            return [(float(x), float(y)) for x, y in zip(v[cols["x"]], v[cols["y"]])]
                except Exception:
                    pass
                try:
                    import numpy as _np
                    if isinstance(v, _np.ndarray) and v.ndim == 2 and v.shape[1] >= 2:
                        return [(float(x), float(y)) for x, y in v[:, :2]]
                except Exception:
                    pass
                out = []
                if isinstance(v, (list, tuple)):
                    for item in v:
                        try:
                            if isinstance(item, dict) and "x" in item and "y" in item:
                                out.append((float(item["x"]), float(item["y"])))
                            else:
                                x, y = item[0], item[1]
                                out.append((float(x), float(y)))
                        except Exception:
                            continue
                return out or None

            kept_points = _normalize_points(raw_points)

            if kept_points:
                plate_img, inset_img, dxf_path = generate_die_plate_dxf(
                    *all_params,
                    holes_override=kept_points,
                    force_hybrid_tab3=True,
                    tab3_show_segment_walls=tab3_show_walls,
                )
            else:
                plate_img, inset_img, dxf_path = generate_die_plate_dxf(
                    *all_params,
                    force_hybrid_tab3=True,
                    tab3_show_segment_walls=tab3_show_walls,
                )

            return (
                gr.update(value=plate_img, visible=True),
                gr.update(value=inset_img,  visible=True),
                dxf_path,
            )

        # ---- States -----------------------------
        visible_rows      = gr.State(ROWS_STEP)

        # ---- Internal states & file outputs (define BEFORE reset_outputs & wiring) ----
        calc_ok_state     = gr.State(False)
        calc_values_state = gr.State({})   # primary dict returned from on_calculate
        calc_state        = gr.State({})   # legacy alias; keeps old code paths happy
        
        #---- Table Selection States--------------------------------------------
        selected_row_idx_state   = gr.State(None)   # 0-based index of highlighted row
        selected_filename_state  = gr.State("")     # key/filename for highlighted row

        # Hidden bridge the JS writes to (MUST be defined once)
        sel_idx_from_html = gr.Number(value=None, visible=False, elem_id="sel_idx_html")

        dxf_ready_state   = gr.State(False)
        dxf_path_state     = gr.State("")

        selected_index_state = selected_row_idx_state

        # File outputs (DXF auto-download helper)
        quick_dxf = gr.File(label="DXF", visible=False)

        def _sel_circ(): return DIE_STYLE_CIRCULAR
        def _sel_stag(): return DIE_STYLE_STAGGERED
        
        # --- Tab selection wiring  ---
        def _clear_on_tab_switch():
            import gradio as gr
            # clear any cached export/doc you track
            try:
                global cached_doc, cached_filename, cached_inset_center, cached_inset_focus_center, cached_export_path
                cached_doc = None
                cached_filename = ""
                cached_inset_center = None
                cached_inset_focus_center = None
                cached_export_path = None
            except Exception:
                pass

            # reset zoom safely
            try:
                inz = _reset_inset_zoom_control()  # returns gr.update(value=...)
            except Exception:
                inz = gr.update(value=1.0)

            return (
                False,                                        # dxf_ready_state
                gr.update(visible=False, interactive=False),  # dxf_dl (Open AutoCAD)
                gr.update(visible=False, interactive=False),  # open_pdf_btn
                gr.update(value=None, visible=False),         # quick_pdf (hidden File)
                gr.update(value=None),                        # plate_preview
                gr.update(value=None),                        # inset_preview
                "",                                           # dxf_path_state
                gr.update(visible=True, value="Switched tabs — please Generate again."),  # stale_banner
                inz,                                          # inset_zoom
            )

        # Tab 1: Circular
        tab_circ.select(
            fn=_sel_circ, inputs=[], outputs=[die_style_state],
            queue=False, show_progress=False,
        ).then(
            fn=_clear_on_tab_switch, inputs=[],
            outputs=[dxf_ready_state, dxf_dl, open_pdf_btn, quick_pdf,
                     plate_preview, inset_preview, dxf_path_state, stale_banner, inset_zoom],
            queue=False, show_progress=False,
        )

        # Tab 2: Staggered
        tab_stag.select(
            fn=_sel_stag, inputs=[], outputs=[die_style_state],
            queue=False, show_progress=False,
        ).then(
            fn=_clear_on_tab_switch, inputs=[],
            outputs=[dxf_ready_state, dxf_dl, open_pdf_btn, quick_pdf,
                     plate_preview, inset_preview, dxf_path_state, stale_banner, inset_zoom],
            queue=False, show_progress=False,
        )

        # Tab 3: Hybrid — only clear artifacts here (Tab 3 has its own dropdown)
        tab_circseg.select(
            fn=_clear_on_tab_switch, inputs=[],
            outputs=[dxf_ready_state, dxf_dl, open_pdf_btn, quick_pdf,
                     plate_preview, inset_preview, dxf_path_state, stale_banner, inset_zoom],
            queue=False, show_progress=False,
        )    

        # ---- Manager helpers (must be defined BEFORE wiring and charts) ----
        FEED_DISPLAY_MAP = {
            "Floating": "Floating",
            "Shrimp_Prawn": "Shrimp/Prawn",
            "Shrimp Prawn": "Shrimp/Prawn",
            "Salmonid": "Salmonid",
            "Salmon id": "Salmonid",
            "Other sinking": "Other sinking",
            "Other Sinking": "Other sinking",
        }

        def norm_feed(val: str) -> str:
            s = str(val or "").strip()
            return FEED_DISPLAY_MAP.get(s, s)

        def uniq(series):
            vals = series.dropna().astype(str).unique().tolist()
            try:
                vals = sorted(vals, key=lambda x: float(x))
            except Exception:
                vals = sorted(vals)
            return vals

        #----- help to render Table DataFrame as HTML and track selected rows --------

        def _initial_saved_df():
            import pandas as pd
            try:
                # If your builder tolerates None inputs, call it here; otherwise skip it.
                df = build_table_df(None, None, None, None)
            except Exception:
                df = None

            # Fallback to an empty but valid frame with the key column
            if df is None or getattr(df, "empty", True):
                df = pd.DataFrame(columns=[KEY_COL])

            # Keep your usual transforms
            df = add_delete_col(df)
            df = _pad_display_rows(df, 4)
            return df
            
        def on_delete_clicked(confirm, sel_idx, df, p_min, p_max, species, plant):
            """
            Delete the HTML-highlighted row.
            Inputs:
              confirm (bool), sel_idx (int or None), df (DataFrame or list-of-lists), filters...
            Outputs:
              saved_table, manager_status, selected_row_idx_state, selected_filename_state,
              saved_table_html, selected_preview_html
            """
            import pandas as pd

            # Normalize DF from possible list-of-lists
            if isinstance(df, list) and df:
                df = pd.DataFrame(df[1:], columns=df[0])

            # Resolve selection
            try:
                i = int(sel_idx) if sel_idx is not None else None
            except Exception:
                i = None

            # Guard: nothing selected
            if df is None or getattr(df, "empty", True) or i is None or i < 0 or i >= len(df):
                warn = "<div class='chip warn'>Pick a row first.</div>"
                return (
                    df,                          # saved_table unchanged
                    gr.update(value=warn),       # manager_status (single warning spot)
                    sel_idx,                     # selected_row_idx_state unchanged
                    "",                          # selected_filename_state cleared
                    gr.update(),                 # saved_table_html unchanged (no blanking)
                    gr.update(value="<div class='chip warn'>No selection</div>"),
                )

            key_col = "File" if ("File" in df.columns) else df.columns[0]
            file_key = str(df.iloc[i][key_col]).strip()

            # Guard: must confirm
            if not confirm:
                chip = f"<div class='chip ok'>SELECTED</div><div class='chip'>{file_key}</div>" if file_key else "<div class='chip warn'>No selection</div>"
                return (
                    df,
                    gr.update(value="<div class='chip warn'>Tick ‘Confirm delete’.</div>"),
                    i,
                    file_key,
                    gr.update(),                 # HTML unchanged
                    gr.update(value=chip),       # preview shows selection (not another warning)
                )

            # --- Perform delete using your existing helper (by key) ---
            new_df, status_html = delete_focused_row(file_key, p_min, p_max, species, plant)

            # Choose next selection (stay on same index when possible)
            n = 0 if new_df is None else len(new_df)
            if n == 0:
                new_idx, new_key = None, ""
            else:
                new_idx = min(i, n - 1)
                try:
                    new_key = str(new_df.iloc[new_idx][key_col]).strip()
                except Exception:
                    new_idx, new_key = 0, (str(new_df.iloc[0][key_col]).strip() if n else "")

            # Rebuild HTML ONLY now (data changed)
            new_df = add_delete_col(new_df)
            html = df_to_compact_html(new_df, selected_index=new_idx)

            chip_deleted = f"<div class='chip ok'>DELETED</div><div class='chip'>{file_key}</div>"
            chip_selected = f"<div class='chip ok'>SELECTED</div><div class='chip'>{new_key}</div>" if new_key else "<div class='chip'>No selection</div>"

            return (
                new_df,                                 # saved_table
                gr.update(value=chip_deleted + " " + status_html),  # manager_status
                new_idx,                                # selected_row_idx_state
                new_key,                                # selected_filename_state
                gr.update(value=html),                  # saved_table_html
                gr.update(value=chip_selected),         # selected_preview_html
            )

        import pandas as pd

        def add_delete_col(df, *args, **kwargs):
            if isinstance(df, dict):
                out = pd.DataFrame(df)
            else:
                out = df.copy()
            if DELETE_COL not in out.columns:
                out.insert(0, DELETE_COL, False)
            return out

        def df_to_compact_html(df, selected_index=None):
            from html import escape as html_escape
            import pandas as pd

            # Normalize Gradio list-of-lists payload → DataFrame
            try:
                if isinstance(df, list):
                    if not df:                       # handle empty list payloads
                        return "<em>No data</em>"
                    df = pd.DataFrame(df[1:], columns=df[0])
            except Exception as e:
                try:
                    log_error(f"df_to_compact_html normalize fail: {e}")
                except Exception:
                    pass
                df = None

            if df is None:
                return "<em>No data</em>"

            # Clamp selected_index (no auto-select to top row)
            try:
                if selected_index is not None:
                    i = int(selected_index)
                    if i < 0 or i >= len(df):
                        selected_index = None
                    else:
                        selected_index = i
            except Exception:
                selected_index = None

            headers = list(df.columns)
            colgroup = "".join("<col />" for _ in headers)
            thead = "".join(f"<th>{html_escape(str(h))}</th>" for h in headers)

            rows_html = []
            try:
                for ridx, row in enumerate(df.itertuples(index=False)):
                    # mark “pad” rows when first col is blank; JS treats these as unselectable
                    is_pad = str(row[0]).strip() == ""
                    cls = " class='pad'" if is_pad else ""
                    if selected_index is not None and ridx == selected_index:
                        cls = " class='selected'"
                    tds = "".join(f"<td>{html_escape(str(val))}</td>" for val in row)
                    rows_html.append(f"<tr{cls}>{tds}</tr>")
            except Exception as e:
                try:
                    log_error(f"df_to_compact_html rows fail: {e}")
                except Exception:
                    pass

            # Ensure stable height
            try:
                MIN_ROWS = 4
                if len(rows_html) < MIN_ROWS:
                    blank = "<tr class='pad'>" + "".join("<td></td>" for _ in headers) + "</tr>"
                    rows_html += [blank] * (MIN_ROWS - len(rows_html))
            except Exception:
                pass

            return (
                "<div class=\"saved-grid-html\">"
                "<table>"
                f"<colgroup>{colgroup}</colgroup>"
                f"<thead><tr>{thead}</tr></thead>"
                f"<tbody>{''.join(rows_html)}</tbody>"
                "</table>"
                "</div>"
            )

        KEY_COL = "File"

        def _pellet_bounds(df):
            """
            Return (min, max) pellet size in mm from the current table.
            Safe for list payloads and empty frames.
            """
            import pandas as pd
            try:
                if isinstance(df, list) and df:
                    df = pd.DataFrame(df[1:], columns=df[0])
                if df is None or getattr(df, "empty", True):
                    return 0.0, 20.0
                col = "Pellet Size (mm)"
                if col not in df.columns:
                    # Optional: try a couple fallbacks if names ever drift
                    for alt in ("Pellet Size", "Pellet size (mm)", "pellet_size_mm"):
                        if alt in df.columns:
                            col = alt
                            break
                    else:
                        return 0.0, 20.0
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if vals.empty:
                    return 0.0, 20.0
                return float(vals.min()), float(vals.max())
            except Exception as e:
                try: log_error(f"_pellet_bounds fail: {e}")
                except Exception: pass
                return 0.0, 20.0

        def _ensure_del_column(df):
                """
                Normalize gr.DataFrame payload to DataFrame and guarantee DEL column exists.
                """
                import pandas as pd
                if df is None:
                        return df
                if isinstance(df, list) and df:
                        try:
                                df = pd.DataFrame(df[1:], columns=df[0])
                        except Exception as e:
                                try: log_error(f"_ensure_del_column normalize fail: {e}")
                                except Exception: pass
                                return df
                try:
                        _ = df.columns
                except Exception:
                        return df
                try:
                        return add_delete_col(df)
                except Exception as e:
                        try: log_error(f"_ensure_del_column add_delete_col fail: {e}")
                        except Exception: pass
                        return df

        def _pos_index_from_key(df, key_val):
                """Return 0-based iloc where KEY_COL == key_val, else None."""
                try:
                        if df is None or getattr(df, "empty", True) or not key_val or KEY_COL not in df.columns:
                                return None
                        key = str(key_val).strip()
                        mask = df[KEY_COL].astype(str).str.strip().eq(key)
                        pos = int(mask.to_numpy().nonzero()[0][0])
                        return pos
                except Exception:
                        return None

        def _key_at_pos(df, pos):
                """Return KEY_COL value at iloc pos or empty string."""
                import pandas as pd
                try:
                        if isinstance(df, list) and df:
                                df = pd.DataFrame(df[1:], columns=df[0])
                        return str(df.iloc[int(pos)][KEY_COL]).strip()
                except Exception:
                        return ""

        def _mirror_saved_table(df):
                """Simple mirror (no selection highlight)."""
                try:
                        return df_to_compact_html(df, selected_index=None)
                except Exception as e:
                        try: log_error(f"_mirror_saved_table fail: {e}")
                        except Exception: pass
                        return "<div class='saved-grid-html'><table><tbody></tbody></table></div>"

        def _mirror_saved_table_with_key(df, sel_key):
                """
                Mirror with selection by key; falls back to top-row if key not found.
                """
                import pandas as pd
                try:
                        if isinstance(df, list) and df:
                                df = pd.DataFrame(df[1:], columns=df[0])
                        df = _ensure_del_column(df)
                        pos = _pos_index_from_key(df, sel_key)
                        pos = pos if isinstance(pos, int) else None
                        return df_to_compact_html(df, selected_index=pos)
                except Exception as e:
                        try: log_error(f"_mirror_saved_table_with_key fail: {e}")
                        except Exception: pass
                        return "<div class='saved-grid-html'><table><tbody></tbody></table></div>"

        def _valid_positions(df):
            """
            Return list of iloc positions that have a non-empty key in KEY_COL.
            Skips padded/blank rows. Safe for list payloads.
            """
            import pandas as pd
            try:
                if isinstance(df, list) and df:
                    df = pd.DataFrame(df[1:], columns=df[0])
                if df is None or getattr(df, "empty", True):
                    return []
                if KEY_COL not in df.columns:
                    return []
                s = df[KEY_COL].astype(str).str.strip()
                return [i for i, ok in enumerate((s != "") & s.notna()) if ok]
            except Exception as e:
                try: log_error(f"_valid_positions fail: {e}")
                except Exception: pass
                return []

        def _select_by_offset(current_df, current_key, offset):
            """
            Move selection by 'offset' among valid rows.
            Returns: (selected_row_idx_state, selected_filename_state)
            Only updates server state; HTML mirror is handled elsewhere.
            """
            import pandas as pd
            try:
                df = current_df
                if isinstance(df, list) and df:
                    df = pd.DataFrame(df[1:], columns=df[0])

                # Ensure DEL column / invariants if you rely on it elsewhere
                df = _ensure_del_column(df)

                if df is None or getattr(df, "empty", True) or KEY_COL not in df.columns:
                    return None, ""

                valid = _valid_positions(df)
                if not valid:
                    return None, ""

                cur_pos = _pos_index_from_key(df, current_key)
                if cur_pos is None or cur_pos not in valid:
                    target = valid[0]  # start from first valid
                else:
                    idx_in_valid = valid.index(cur_pos)
                    idx_in_valid = max(0, min(len(valid) - 1, idx_in_valid + int(offset)))
                    target = valid[idx_in_valid]

                new_key = _key_at_pos(df, target)
                return int(target), new_key
            except Exception as e:
                try: log_error(f"_select_by_offset fail: {e}")
                except Exception: pass
                return None, ""

        def _select_first(current_df, current_key):
            """Clamp to first valid row (state-only)."""
            return _select_by_offset(current_df, current_key, 0)

        def _select_last(current_df, current_key):
            """Clamp to last valid row (state-only)."""
            return _select_by_offset(current_df, current_key, 10**6)

        def on_select_current(sel_idx, df):
            """Confirm selection: copy highlighted row → states, update header with details."""
            import pandas as pd
            if isinstance(df, list) and df:
                try:
                    df = pd.DataFrame(df[1:], columns=df[0])
                except Exception:
                    df = None

            # Resolve key from current index
            key = ""
            try:
                i = int(sel_idx) if sel_idx is not None else None
            except Exception:
                i = None

            if df is not None and getattr(df, "empty", True) is False and i is not None and 0 <= i < len(df):
                key_col = "File" if "File" in df.columns else df.columns[0]
                key = str(df.iloc[i][key_col]).strip()

            header = _describe_selected(key) if key else "<strong>Selected:</strong> <em>none</em>"

            # Return: index, filename, header, (clear chip), clear manager_status
            return (
                i,
                key,
                gr.update(value=header),
                gr.update(value="", visible=False),   # selected_preview_html hidden/cleared
                gr.update(value=""),
            )

        def _on_saved_table_click_or_select(evt: gr.SelectData, current_df, p_min, p_max, species, plant, select_all):
            import pandas as pd

            # normalize
            df = current_df
            if isinstance(df, list) and df:
                df = pd.DataFrame(df[1:], columns=df[0])
            df = _ensure_del_column(df)

            if df is None or df.empty:
                html = df_to_compact_html(df, selected_index=None)
                return (
                    NO_CHANGE,                # saved_table (no change)
                    SET("<em>No rows</em>"), # manager_status
                    None,                    # selected_row_idx_state
                    "",                      # selected_filename_state
                    SET("<em>No rows</em>"), # selected_preview_html
                    SET(html),               # saved_table_html
                )

            # parse click...
            try:
                if isinstance(evt.index, (list, tuple)):
                    row_idx, col_idx = int(evt.index[0]), int(evt.index[1])
                else:
                    row_idx, col_idx = int(evt.index), 0
            except Exception:
                html = df_to_compact_html(df, selected_index=None)
                return (NO_CHANGE, NO_CHANGE, None, "", SET(""), SET(html))

            cols = df.columns.tolist()
            if not (0 <= row_idx < len(df) and 0 <= col_idx < len(cols)):
                html = df_to_compact_html(df, selected_index=None)
                return (NO_CHANGE, NO_CHANGE, None, "", SET(""), SET(html))

            col_name = cols[col_idx]
            file_key = str(df.iloc[row_idx].get(KEY_COL, "")).strip()

            if col_name == DELETE_COL:
                try:
                    _delete_saved_design(file_key)
                except Exception as e:
                    status = f"<span class='err'>Delete failed: {e}</span>"
                    html = _mirror_saved_table_with_key(df, file_key)
                    return (
                        NO_CHANGE,         # <- was gr.update(value=df) (caused jitter); use NO_CHANGE
                        SET(status),
                        None,
                        "",
                        SET(""),
                        SET(html),
                    )

                new_df = build_table_df(p_min, p_max, species, plant)
                new_df = add_delete_col(new_df)
                new_df = _pad_display_rows(new_df, 4)

                new_key = ""
                html = _mirror_saved_table_with_key(new_df, new_key)
                status = f"Deleted: {html_escape(str(file_key))}"

                return (
                    gr.update(value=new_df, headers=new_df.columns.tolist(), datatype=_df_datatypes(new_df)),
                    SET(status),
                    None,
                    new_key,
                    SET(""),
                    SET(html),
                )

            # selection path
            preview = f"<span class='chip ok'>SELECTED:</span> <code>{html_escape(str(file_key))}</code>"
            html = _mirror_saved_table_with_key(df, file_key)
            return (
                NO_CHANGE,     # no re-render ⇒ smooth scroll
                SET(""),
                int(row_idx),
                file_key,
                SET(preview),
                SET(html),
            )

        def _df_datatypes(df):
            dtypes = []
            for c in df.columns.tolist():
                if c == "Select":
                    dtypes.append("bool")
                elif c in MANAGER_NUMBER_COLS:
                    dtypes.append("number")
                else:
                    dtypes.append("str")
            return dtypes

        def ensure_manager_columns(df):
            out = df.copy()
            # Make sure every canonical column exists
            for col in MANAGER_COLUMNS:
                if col not in out.columns:
                    out[col] = (pd.NA if col in MANAGER_NUMBER_COLS else "")
            # Types for numeric columns
            for col in MANAGER_NUMBER_COLS:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce")

            # Tidy Saved column for display/filters
            if "Saved" in out.columns:
                out["Saved"] = out["Saved"].fillna("").astype(str)

            # Order exactly as MANAGER_COLUMNS
            out = out[MANAGER_COLUMNS]
            return out

        # --- canonical filters for the table & charts ---
        def apply_filters(df, pellet_min: float,
                          pellet_max: float,
                          species_choices,
                          plant_choices):
            out = df.copy()

            # Pellet size range (robust to bad data, order-insensitive)
            if "Pellet Size (mm)" in out.columns:
                ps = pd.to_numeric(out["Pellet Size (mm)"], errors="coerce")
                lo = float(min(pellet_min, pellet_max)) if pellet_min is not None else None
                hi = float(max(pellet_min, pellet_max)) if pellet_max is not None else None
                if lo is not None:
                    out = out[ps >= lo]
                if hi is not None:
                    out = out[ps <= hi]

            # Feed Type (normalize to consistent labels)
            if species_choices:
                normalized = {norm_feed(s) for s in species_choices}
                if "Feed Type" in out.columns:
                    out["Feed Type"] = out["Feed Type"].map(norm_feed)
                    out = out[out["Feed Type"].isin(normalized)]

            # Plant
            if plant_choices and "Plant" in out.columns:
                out = out[out["Plant"].astype(str).str.strip().isin([str(x).strip() for x in plant_choices])]

            return out

        def build_table_df(pellet_min, pellet_max, species_choices, plant_choices, select_all=False):
            base = clean_table_view(collect_manager_rows())
            view = apply_filters(base, pellet_min, pellet_max, species_choices or [], plant_choices or [])
            view = ensure_manager_columns(view)  # ensures columns exist + right order

            # Ensure metadata/chart columns exist
            for col in ["Extruder", "Line", "Plant", "Type", "Performance", "Comments"]:
                if col not in view.columns:
                    view[col] = ""
            if "Bulk Density" not in view.columns:
                view["Bulk Density"] = 0.0
            if "Expansion (%)" not in view.columns:
                view["Expansion (%)"] = 0.0

            # Move 'File' to the far right
            cols = list(view.columns)
            if "File" in cols:
                cols.remove("File")
                cols = cols + ["File"]
            return view[cols]
            
        def refresh_table(pellet_min_val, pellet_max_val, species_choices, plant_choices, select_all=False):
            df = build_table_df(pellet_min_val, pellet_max_val, species_choices or [], plant_choices or [], select_all=bool(select_all))
            df = add_delete_col(df)                  # << ensure DEL column exists
            df = _pad_display_rows(df, 3)            # (keep if you use padding)
            return gr.update(value=df, headers=df.columns.tolist(), datatype=_df_datatypes(df))

        def refresh_table_and_mirror(pellet_min, pellet_max, species_filter, plant_filter, select_all):
            df = build_table_df(pellet_min, pellet_max, species_filter or [], plant_filter or [], select_all=bool(select_all))
            df = add_delete_col(df)                  # << ensure DEL column exists
            # (optional) df = _pad_display_rows(df, 3)
            return (
                gr.update(value=df, headers=df.columns.tolist(), datatype=_df_datatypes(df)),
                gr.update(value=df_to_compact_html(df)),   # or _mirror_saved_table_with_key(df, selected_filename_state) in a .then
            )

        def _export_pdf_from_cached(calc_vals=None):
            import os, tempfile, traceback
            import gradio as gr
            vals = dict(calc_vals or {})

            global cached_export_path, cached_doc

            try:
                # If we already have the combined/saved DXF file, use it directly.
                if cached_export_path and os.path.exists(cached_export_path):
                    return export_pdf_from_dxf(cached_export_path, calc_vals=vals)

                # Otherwise, fall back to the in-memory DXF doc: save it to a temp file first.
                if cached_doc is not None:
                    tmp = tempfile.NamedTemporaryFile(suffix=".dxf", delete=False)
                    tmp_path = tmp.name
                    tmp.close()
                    try:
                        cached_doc.saveas(tmp_path)
                        return export_pdf_from_dxf(tmp_path, calc_vals=vals)
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

                # Nothing to export
                return gr.update(value=None, visible=False)

            except Exception:
                log_error("cached wrapper -> export_pdf_from_dxf failed:\n" + traceback.format_exc())
                return gr.update(value=None, visible=False)

        def refresh_table_and_filters(pellet_min_val, pellet_max_val, species_choices, plant_choices, select_all=False):
            # Build the filtered table first (using incoming slider values)
            df = build_table_df(pellet_min_val, pellet_max_val, species_choices, plant_choices, select_all=bool(select_all))

            # Recompute global bounds for sliders from the whole (unfiltered) manager view
            base_all = clean_table_view(collect_manager_rows())
            new_min, new_max = _pellet_bounds(base_all)

            # Clamp current slider values into new bounds
            cur_min = float(min(max(pellet_min_val, new_min), new_max)) if pellet_min_val is not None else new_min
            cur_max = float(min(max(pellet_max_val, new_min), new_max)) if pellet_max_val is not None else new_max
            if cur_min > cur_max:
                cur_min, cur_max = cur_max, cur_min

            plant_opts  = [p for p in uniq(base_all["Plant"]) if str(p).strip()] if not base_all.empty else []
            species_opts = ["Floating", "Shrimp/Prawn", "Salmonid", "Other sinking"]

            # Keep only still-valid selections
            def _norm_str_list(xs): return [str(x).strip() for x in (xs or [])]
            plant_sel  = [v for v in _norm_str_list(plant_choices)  if v in _norm_str_list(plant_opts)]
            species_sel = [v for v in (species_choices or []) if v in species_opts]
            
            df_disp = _pad_display_rows(df, 3)

            return (
                gr.update(value=df_disp, headers=df_disp.columns.tolist(), datatype=_df_datatypes(df_disp)),   # saved_table
                gr.update(minimum=new_min, maximum=new_max, value=cur_min, step=0.1),           # pellet_min
                gr.update(minimum=new_min, maximum=new_max, value=cur_max, step=0.1),           # pellet_max
                gr.update(choices=species_opts, value=species_sel),                             # species_filter
                gr.update(choices=plant_opts, value=plant_sel),                                 # plant_filter
            )

        def on_pcd_reset():
            """
            Clear ONLY:
              - Number of Rows (to 0)
              - All Circular row fields (holes & PCD) and hide them
              - Bolt details (inner/outer counts, PCDs, diameters)
            """
            updates = []

            # 1) rows slider → 0
            updates.append(gr.update(value=0))  # line_rows

            # 2) bolt details → 0
            updates += [
                gr.update(value=0),   # outer_bolt_count
                gr.update(value=0.0), # outer_bolt_pcd
                gr.update(value=0.0), # outer_bolt_diameter
                gr.update(value=0),   # inner_bolt_count
                gr.update(value=0.0), # inner_bolt_pcd
                gr.update(value=0.0), # inner_bolt_diameter
            ]

            # 3) per-row fields → 0 and hide
            for (num_h, pcd) in row_components:
                updates.append(gr.update(value=0,   visible=False))  # num_holes
                updates.append(gr.update(value=0.0, visible=False))  # pcd

            return updates

        # ----- Wire the reset button -----
        # NOTE: Replace bolt component names below if yours differ.
        reset_outputs = [
            line_rows,
            outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
            inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
        ]
        for (num_h, p) in row_components:
            reset_outputs += [num_h, p]

        pcd_reset_btn.click(
            on_pcd_reset,
            inputs=[],
            outputs=reset_outputs,
        )

        # --- Feed type: allow only one (or none) in the designer ---
        def on_fl_change(fl, sh, sa, osk):
            # If Floating is checked, turn the others off; if unchecked, all off
            return (
                bool(fl),      # floating
                False,         # shrimp
                False,         # salmonid
                False          # other_sinking
            ) if fl else (False, False, False, False)

        def on_sh_change(fl, sh, sa, osk):
            return (False, bool(sh), False, False) if sh else (False, False, False, False)

        def on_sa_change(fl, sh, sa, osk):
            return (False, False, bool(sa), False) if sa else (False, False, False, False)

        def on_os_change(fl, sh, sa, osk):
            return (False, False, False, bool(osk)) if osk else (False, False, False, False)
            
        floating.change(
            fn=on_fl_change,
            inputs=[floating, shrimp, salmonid, other_sinking],
            outputs=[floating, shrimp, salmonid, other_sinking],
            show_progress=False
        )
        shrimp.change(
            fn=on_sh_change,
            inputs=[floating, shrimp, salmonid, other_sinking],
            outputs=[floating, shrimp, salmonid, other_sinking],
            show_progress=False
        )
        salmonid.change(
            fn=on_sa_change,
            inputs=[floating, shrimp, salmonid, other_sinking],
            outputs=[floating, shrimp, salmonid, other_sinking],
            show_progress=False
        )
        other_sinking.change(
            fn=on_os_change,
            inputs=[floating, shrimp, salmonid, other_sinking],
            outputs=[floating, shrimp, salmonid, other_sinking],
            show_progress=False
        )
    
        def derive_feed_type(fl, sh, sa, osk):
            if fl: return "Floating"
            if sh: return "Shrimp_Prawn"
            if sa: return "Salmonid"
            if osk: return "Other sinking"
            return ""
     
        def get_selected_feed_type(floating, shrimp, salmonid, other_sinking):
            if floating: return "Floating"
            if shrimp: return "Shrimp_Prawn"
            if salmonid: return "Salmonid"
            if other_sinking: return "Other sinking"
            return ""

        def on_calculate(ps, th, bd, ft, fl, sh, sa, osk,
                         cl, col, od, cw,
                         cs_dia, cs_dep,
                         nh,
                         inlet_inside, inside_taper, inlet_outside, outside_taper):

            # small helper so this works even if a Gradio component slips through
            def _g(x, default=0.0):
                try:
                    return float(x)
                except Exception:
                    try:
                        return float(getattr(x, "value"))
                    except Exception:
                        return float(default)

            try:
                # 1) Run your existing calculation
                image_html, table_html, calc_values = calculate_die_design(
                    ps, th, bd, ft, get_selected_feed_type(fl, sh, sa, osk),
                    cl, col, od, cw, nh,
                    countersink_plus_radius_mm=max(0.0, _g(cs_dia, 0.0)),             # UI is +radius
                    countersink_depth_mm=min(max(0.0, _g(cs_dep, 0.0)), _g(col, 0.0)) # clamp to cone length
                )

                # 2) Persist everything the DXF exporter needs (including the 4 PCDs)
                cv = dict(calc_values or {})
                cv["channel_length"]          = _g(cl, 0.0)
                cv["cone_length"]             = _g(col, 0.0)
                cv["opening_dia"]             = _g(od, 0.0)
                cv["cone_width"]              = _g(cw, 0.0)
                cv["countersink_plus_radius"] = max(0.0, _g(cs_dia, 0.0))
                cv["countersink_depth"]       = min(max(0.0, _g(cs_dep, 0.0)), _g(col, 0.0))

                # NEW inlet-side PCD rings
                cv["inlet_inside_pcd"]  = _g(inlet_inside,  0.0)
                cv["inside_taper_pcd"]  = _g(inside_taper,  0.0)
                cv["inlet_outside_pcd"] = _g(inlet_outside, 0.0)
                cv["outside_taper_pcd"] = _g(outside_taper, 0.0)

                # log for sanity
                try:
                    log_error(
                        "CALC: inletPCD -> "
                        f"in={cv['inlet_inside_pcd']:.2f},{cv['inside_taper_pcd']:.2f}; "
                        f"out={cv['inlet_outside_pcd']:.2f},{cv['outside_taper_pcd']:.2f}"
                    )
                except Exception:
                    pass

                ok = True
                return image_html, table_html, ok, cv

            except Exception as e:
                log_error(f"on_calculate failed: {e}")
                return "<div>Error</div>", "<div></div>", False, {}

        def clear_save_message():
            return gr.update(value="")

        def on_any_input_change():
            return False  # forces recalc before save

        # ------- Saved Designs Tabel row control helpers -
        def _pad_display_rows(df, min_rows=3):
            import pandas as pd
            if df is None:
                return df
            n = len(df)
            if n >= min_rows:
                return df
            pad = pd.DataFrame([{c: None for c in df.columns}] * (min_rows - n))
            return pd.concat([df, pad], ignore_index=True)
        
        # ------- Save helpers ----------------------------------------

        def build_payload_inputs(
            ps, th, bd, ft, fl, sh, sa, osk,
            cl, col, od, cw,
            countersink_diameter,           # NOTE: historical name; this is actually +RADIUS from the UI
            countersink_depth, nh,
            plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
            outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
            outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
            inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
            inlet_inside, inside_taper, inlet_outside, outside_taper,
            *row_inputs
        ):
            """
            Build the 'inputs' payload and row list.
            - Uses unified _normalize_countersink() (no duplicate validator here).
            - 'countersink_diameter' param name is legacy: it is a +RADIUS value from the UI.
            """

            # rows: (num_holes, pcd) pairs, robust to Gradio components
            hole_rows = []
            pairs = min(len(row_inputs) // 2, MAX_ROWS)  # ensure MAX_ROWS is defined
            for i in range(pairs):
                n_raw = row_inputs[2 * i]
                p_raw = row_inputs[2 * i + 1]
                n = _to_int(n_raw, 0)
                p = _to_float(p_raw, 0.0)
                if n > 0 and p > 0.0:
                    hole_rows.append({"row": i + 1, "num_holes": n, "pcd": p})

            # Resolve countersink using the UNIFIED helper.
            # IMPORTANT: 'countersink_diameter' here is the UI "+radius" (legacy naming).
            cs_plus_radius_ui = _g(countersink_diameter, 0.0)
            cs_dia, cs_dep = _normalize_countersink(
                opening_dia=od,
                cone_length=col,
                cone_width=cw,
                cs_depth=countersink_depth,
                cs_plus_radius=cs_plus_radius_ui,
                # (If someday you pass an absolute diameter, give it as cs_diameter=...)
            )

            payload_inputs = {
                "pellet_size": ps,
                "throughput": th,
                "bulk_density": bd,
                "fat": ft,
                "feed_type": get_selected_feed_type(fl, sh, sa, osk),

                "channel_length": cl,
                "cone_length": col,
               "opening_dia": od,
                "cone_width": cw,
                "number_of_holes": nh,

                # Store both the UI +radius and the resolved absolute diameter/depth:
                "countersink_plus_radius": cs_plus_radius_ui,
                "countersink_diameter": cs_dia,
                "countersink_depth": cs_dep,

                "plate_diameter": plate_diameter,
                "chamfer_plate": chamfer_plate,
                "die_center_hole_diameter": die_center_hole_diameter,
                "chamfer_center_hole": chamfer_center_hole,

                "outer_opening_pcd": outer_opening_pcd,
                "chamfer_outer_opening": chamfer_outer_opening,
                "inner_opening_pcd": inner_opening_pcd,
                "chamfer_inner_opening": chamfer_inner_opening,

                "outer_bolt_count": outer_bolt_count,
                "outer_bolt_pcd": outer_bolt_pcd,
               "outer_bolt_diameter": outer_bolt_diameter,
                "inner_bolt_count": inner_bolt_count,
                "inner_bolt_pcd": inner_bolt_pcd,
                "inner_bolt_diameter": inner_bolt_diameter,

                # Coerce these with _g to avoid the earlier "Slider" TypeError:
                "inlet_inside_pcd":  _g(inlet_inside,  0.0),
                "inside_taper_pcd":  _g(inside_taper,  0.0),
                "inlet_outside_pcd": _g(inlet_outside, 0.0),
                "outside_taper_pcd": _g(outside_taper, 0.0),

                "hole_rows": hole_rows,
            }

            return payload_inputs, hole_rows

        def do_save(calc_ok, calc_values, payload_inputs, metadata=None):
            import datetime, os
            if not calc_ok:
                return gr.update(
                    value="<div style='color:red; font-weight:bold;'>Please click 'Calculate' before saving.</div>",
                    visible=True
                )
            payload = {
                "timestamp": datetime.datetime.now().isoformat(),
                "inputs": payload_inputs,
                "calculated": calc_values,
                "metadata": (metadata or {})
            }

            # Build smart filename from inputs/calculated and pass it through
            try:
                smart_name = make_smart_filename(payload_inputs, calc_values)
                target_path = os.path.join(SAVE_DIR, smart_name)
            except Exception:
                target_path = None  # fall back to default naming

            try:
                filename = save_die_design(payload, SAVE_DIR, filename=target_path)
                return gr.update(
                    value=f"<div style='color:green; font-weight:bold;'>Saved! ({os.path.basename(filename)})</div>",
                    visible=True
                )
            except Exception as e:
                return gr.update(
                    value=f"<div style='color:red; font-weight:bold;'>Save failed: {e}</div>",
                    visible=True
                )
        def on_save_with_meta(calc_ok, calc_values,
                              ps, th, bd, ft, fl, sh, sa, osk,
                              cl, col, od, cw,
                              countersink_diameter, countersink_depth,
                              nh,
                              plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                              outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                              outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                              inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                              # NEW sliders (inlet PCDs)
                              inlet_inside, inside_taper, inlet_outside, outside_taper,
                              *row_inputs_and_meta):

            # trailing meta fields (must be 6 at the end of the inputs list)
            *row_inputs, extruder_type, production_line, plant_name, product_type, die_performance, comments = row_inputs_and_meta

            if not calc_ok:
                # IMPORTANT: outputs order must match your .click(..., outputs=[...])
                return (
                    gr.update(value="<div style='color:red;font-weight:bold;'>Please click 'Calculate' before saving.</div>", visible=True),
                    gr.update(value=pd.DataFrame()),
                    None,
                )

            payload_inputs, _ = build_payload_inputs(
                ps, th, bd, ft, fl, sh, sa, osk,
                cl, col, od, cw,
                countersink_diameter, countersink_depth,
                nh,
                plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                inlet_inside, inside_taper, inlet_outside, outside_taper,
                *row_inputs
            )

            metadata = {
                "extruder_type":   extruder_type or "",
                "production_line": production_line or "",
                "plant_name":      plant_name or "",
                "product_type":    product_type or "",
                "die_performance": die_performance or "",
                "comments":        comments or ""
            }

            # do_save MUST return (status_html_update, saved_table_df, saved_file_path_or_None)
            return do_save(calc_ok, calc_values, payload_inputs, metadata)

        def on_quick_save(calc_ok, calc_values,
                          ps, th, bd, ft, fl, sh, sa, osk,
                          cl, col, od, cw,
                          countersink_diameter, countersink_depth,
                          nh,
                          plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                          outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                          outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                          inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                          inlet_inside, inside_taper, inlet_outside, outside_taper,
                          *row_inputs):

            if not calc_ok:
                return (
                    gr.update(value="<div style='color:red;font-weight:bold;'>Please click 'Calculate' before saving.</div>", visible=True),
                    gr.update(value=pd.DataFrame()),
                    None,
                )

            payload_inputs, _ = build_payload_inputs(
                ps, th, bd, ft, fl, sh, sa, osk,
                cl, col, od, cw,
                countersink_diameter, countersink_depth,
                nh,
                plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                inlet_inside, inside_taper, inlet_outside, outside_taper,
                *row_inputs
            )
            return do_save(calc_ok, calc_values, payload_inputs, metadata=None)


        def on_save(calc_ok, calc_values,
                    ps, th, bd, ft, fl, sh, sa, osk,
                    cl, col, od, cw,
                    countersink_diameter, countersink_depth,
                    nh,
                    plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                    outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                    outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                    inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                    inlet_inside, inside_taper, inlet_outside, outside_taper,
                    *row_inputs):
            """Legacy save -> route into the unified path to avoid drift."""
            if not calc_ok:
                return (
                    gr.update(value="<div style='color:red;font-weight:bold;'>Please click 'Calculate' before saving.</div>", visible=True),
                    gr.update(value=pd.DataFrame()),
                    None,
                )

            payload_inputs, _ = build_payload_inputs(
                ps, th, bd, ft, fl, sh, sa, osk,
                cl, col, od, cw,
                countersink_diameter, countersink_depth,
                nh,
                plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                inlet_inside, inside_taper, inlet_outside, outside_taper,
                *row_inputs
            )
            return do_save(calc_ok, calc_values, payload_inputs, metadata=None)

        #---------------display a sensible file name on table select and delete-----------------
        # Helper: compact decimal string (2 dp, strip trailing zeros)
        def _mm2(x, nd=2):
            try:
                s = f"{float(x):.{nd}f}"
                return s.rstrip("0").rstrip(".")
            except Exception:
                return ""

        # Build a descriptive, filesystem-safe name from INPUTS + CALCULATED
        # Example: 9mm_435_holes_6.3mm_opening_saved_2025-08-20_23-34.json
        def make_smart_filename(inputs: dict = None, calc: dict = None, original: str = "") -> str:
            import re
            from datetime import datetime

            inputs = inputs or {}
            calc   = calc or {}

            # Prefer INPUTS for pellet & opening, CALC for total holes
            ps   = inputs.get("pellet_size") or inputs.get("pellet_size_mm") or calc.get("pellet_size") or calc.get("ps")
            holes = calc.get("total_holes") or calc.get("holes") or inputs.get("number_of_holes") or inputs.get("holes")
            opn  = inputs.get("opening_dia") or inputs.get("die_opening_diameter") or calc.get("opening_dia") or calc.get("die_opening_diameter")

            ps_s    = f"{_mm2(ps)}mm" if ps not in (None, "") else "NAmm"
            holes_s = f"{int(float(holes))}_holes" if holes not in (None, "", "-") else "NA_holes"
            opn_s   = f"{_mm2(opn)}mm_opening" if opn not in (None, "") else "NAmm_opening"

            # Safe timestamp (no colons)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M")

            tail = ""
            if original:
                base = re.sub(r"[\\/:*?\"<>|]", "_", str(original).rsplit(".", 1)[0])
                tail = f"_{base}"

            name = f"{ps_s}_{holes_s}_{opn_s}_saved_{ts}{tail}.json"
            # Normalize spaces/unsafe chars
            name = re.sub(r"\s+", " ", name).strip().replace(" ", "_")
            name = re.sub(r"[\\/:*?\"<>|]", "_", name)
            return name

        def _fmt_mm(x, nd=2):
            try:
                s = f"{float(x):.{nd}f}"
                return s.rstrip("0").rstrip(".")
            except Exception:
                return ""

        def _describe_selected(file_key):
            """
            Selected header text, e.g.:
              <strong>Selected:</strong> Floating — 9mm · 435 holes · 6.3mm opening
              <span class='muted'>(114208.json · 2025-08-20 11:42)</span>
            """
            import os, json, re, datetime

            if not file_key:
                return "<strong>Selected:</strong> <em>none</em>"

            path = os.path.join(SAVE_DIR, os.path.basename(str(file_key)))

            # Defaults
            ps = holes = opn = None
            feed = ""
            ts_disp = None
            short_name = os.path.basename(path)

            # Helpers
            def _mm2(x, nd=2):
                try:
                    s = f"{float(x):.{nd}f}"
                    return s.rstrip("0").rstrip(".")
                except Exception:
                    return ""

            def _truthy(v):
                s = str(v).strip().lower()
                return s in ("1", "true", "yes", "y", "on")

            def _norm_feed_label(s: str) -> str:
                t = re.sub(r"[^a-z]+", "", (s or "").lower())
                if t.startswith("float"):     return "Floating"
                if t.startswith("shrimp") or t.startswith("prawn") or t == "shrimpprawn":  return "Shrimp/Prawn"
                if t.startswith("salmon"):    return "Salmonid"
                if "sink" in t or t == "other": return "Other sinking"
                return s.strip() if s else ""

            def _short_fname_from_ts(dt: "datetime.datetime" = None, fallback_name: str = "") -> str:
                try:
                    if dt:
                        return dt.strftime("%H%M%S") + ".json"
                except Exception:
                    pass
                base = os.path.basename(fallback_name or path)
                if len(base) <= 14:
                    return base
                root, ext = os.path.splitext(base)
                return (root[:8] + "…" + ext) if ext else (base[:12] + "…")

            # --- 1) Prefer details from the JSON payload
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                inputs = data.get("inputs") or {}
                calc   = data.get("calculated") or {}
                meta   = data.get("metadata") or {}

                # Pellet / holes / opening
                ps    = inputs.get("pellet_size") or inputs.get("pellet_size_mm") or calc.get("pellet_size") or calc.get("ps")
                holes = calc.get("total_holes") or calc.get("holes") or inputs.get("number_of_holes") or inputs.get("holes")
                opn   = inputs.get("opening_dia") or inputs.get("die_opening_diameter") or calc.get("opening_dia") or calc.get("die_opening_diameter")

                # --- Feed Type: string fields first (handles "Feed Type": "Floating", etc.)
                for key in ("Feed Type", "feed_type", "feedType", "species", "Species", "type"):
                    val = inputs.get(key) or calc.get(key) or meta.get(key)
                    if isinstance(val, str) and val.strip():
                        feed = _norm_feed_label(val)
                        break

                # If still empty, check boolean flags with common synonyms
                if not feed:
                    for key, label in (
                        ("floating", "Floating"),
                        ("shrimp", "Shrimp/Prawn"),
                        ("shrimp_prawn", "Shrimp/Prawn"),
                        ("prawn", "Shrimp/Prawn"),
                        ("salmonid", "Salmonid"),
                        ("other_sinking", "Other sinking"),
                        ("other", "Other sinking"),
                        ("sinking", "Other sinking"),
                    ):
                        v = inputs.get(key, calc.get(key))
                        if _truthy(v):
                            feed = label
                            break

                # Timestamp (prefer payload)
                ts_raw = data.get("timestamp")
                dt = None
                if ts_raw:
                    try:
                        dt = datetime.datetime.fromisoformat(ts_raw)
                    except Exception:
                        dt = None
                if not dt:
                    try:
                        dt = datetime.datetime.fromtimestamp(os.path.getmtime(path))
                    except Exception:
                        dt = None

                short_name = _short_fname_from_ts(dt, os.path.basename(path))
                ts_disp = dt.strftime("%Y-%m-%d %H:%M") if dt else None

            except Exception:
                # --- 2) Fallbacks from filename + file mtime
                base = os.path.basename(path).replace("_", " ")
                m = re.search(
                    r"(?i)(\d+(?:\.\d+)?)\s*mm\s+(\d+)\s*holes?\s+(\d+(?:\.\d+)?)\s*mm\s*opening",
                    base
                )
                if m:
                    ps, holes, opn = m.group(1), m.group(2), m.group(3)
                try:
                    dt = datetime.datetime.fromtimestamp(os.path.getmtime(path))
                    short_name = dt.strftime("%H%M%S") + ".json"
                    ts_disp = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass

            # --- 3) Format pieces
            ps_s    = f"{_mm2(ps)}mm" if ps not in (None, "") else "NAmm"
            holes_s = f"{int(float(holes))} holes" if holes not in (None, "", "-") else "NA holes"
            opn_s   = f"{_mm2(opn)}mm opening" if opn not in (None, "") else "NAmm opening"

            lead = (feed + " — ") if feed else ""
            tail = short_name + (f" · {ts_disp}" if ts_disp else "")

            return f"<strong>Selected:</strong> {lead}{ps_s} · {holes_s} · {opn_s} <span class='muted'>({tail})</span>"

        # ---- Build flattened_row_inputs for row controls  ----

        try:

            flattened_row_inputs = [ctrl for pair in zip(row_holes_fields, row_pcd_fields) for ctrl in pair]
        except NameError:
            # Legacy: you might already have a single flat list you used elsewhere
            flattened_row_inputs = locals().get("row_inputs") or []

            # Last-resort: auto-discover by component labels so we can still wire the button
            if not flattened_row_inputs:
                import re
                def _autodiscover_row_inputs(ns: dict):
                    holes, pcds = [], []
                    for obj in ns.values():
                        # Only consider Gradio components that could be numeric inputs
                        clsname = getattr(obj, "__class__", type("X",(object,),{})).__name__.lower()
                        if not any(k in clsname for k in ("number", "slider", "textbox")):
                            continue
                        label = (getattr(obj, "label", "") or "")
                        L = label.lower()
                        if ("row" in L and "pcd" in L) or ("pcd" in L and re.search(r"\brow\b|\brow\s*\d+", L)):
                            pcds.append(obj)
                        elif ("row" in L and "hole" in L) or ("# holes" in L) or ("holes per pcd" in L):
                            holes.append(obj)

                    def _row_idx(o):
                        m = re.search(r"(\d+)", getattr(o, "label", "") or "")
                        return int(m.group(1)) if m else 0

                    holes.sort(key=_row_idx)
                    pcds.sort(key=_row_idx)
                    # interleave (holes1, pcd1, holes2, pcd2, ...)
                    flat = []
                    for h, p in zip(holes, pcds):
                        flat.extend([h, p])
                    return flat

                flattened_row_inputs = _autodiscover_row_inputs(locals()) or []
        # ---- end build flattened_row_inputs ----
        # ---- Bindings (Designer) --------------------------------
        # Flatten Tab 3 row sliders for input plumbing: [nh1, pcd1, nh2, pcd2, ...]
        _t3_flat_row_inputs = []
        for nh, p in circseg_row_components:
            _t3_flat_row_inputs += [nh, p]

        circseg_generate_btn.click(
            fn=_btn_busy,  # disables button + shows message in circseg_msg
            inputs=[],
            outputs=[circseg_generate_btn, circseg_msg],
            show_progress=False,
            queue=False,
            concurrency_limit=1,
        ).then(
            fn=_t3_build_kept_points,  # compute circular holes from Tab 3 rows, then cull by segments
            inputs=[calc_values_state, circseg_line_rows] + _t3_flat_row_inputs + [
                stag_segments_t3, wall_width_t3, seg_inner_pcd_t3, seg_outer_pcd_t3, corner_radius_t3, padding_adj_t3
            ],
            outputs=[calc_values_state, circseg_pts_state, circseg_counts, circseg_msg],
            show_progress=False,
            queue=False,
        ).then(
            fn=_hybrid_previews_only,  # heavy work: ONLY previews + path (spinners show in the two Images)
            inputs=[
                # core (same as your normal generator)
                pellet_size, plate_diameter, opening_dia, cone_width,
                chamfer_plate, chamfer_outer_opening, chamfer_inner_opening, chamfer_center_hole,
                outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                outer_opening_pcd, inner_opening_pcd, die_center_hole_diameter,
                inset_zoom, inlet_inside, inside_taper, inlet_outside, outside_taper,
                throughput, calc_values_state,
                holes, staggered_pts_state, die_style_state,

                # Tab 3 geometry (kept for titles/layers; holes were already culled above)
                stag_segments_t3, seg_bolts_chk_t3,
                stag_segments_t3,  # segments
                wall_width_t3, seg_inner_pcd_t3, seg_outer_pcd_t3,
                corner_radius_t3, padding_adj_t3,
                show_segment_walls_t3,

            ] + _t3_flat_row_inputs + [
                # IMPORTANT: pass the culled points last → holes_override
                circseg_pts_state,
            ],
            outputs=[plate_preview, inset_preview, dxf_path_state],
            show_progress=True,   # <-- spinners only in Image components
        ).then(
            fn=_mark_ready_and_show_dxf,
            inputs=[dxf_path_state],
            outputs=[dxf_ready_state, dxf_dl, open_pdf_btn, stale_banner],
            show_progress=False,
            queue=False,
        ).then(
            fn=_reset_inset_zoom_control,
            inputs=[],
            outputs=[inset_zoom],
            show_progress=False,
            queue=False,
        ).then(
            fn=_btn_idle,  # re-enable button + clear message
            inputs=[],
            outputs=[circseg_generate_btn, circseg_msg],
            show_progress=False,
            queue=False,
        ).then(
            fn=_hide_pdf_link,  # clear any stale quick_pdf
            inputs=[],
            outputs=[quick_pdf],
            show_progress=False,
            queue=False,
        )

        calculate_btn.click(
            fn=on_calculate,
            inputs=[
                pellet_size, throughput, density, fat,
                floating, shrimp, salmonid, other_sinking,
                channel_length, cone_length, opening_dia, cone_width,
                countersink_diameter, countersink_depth,
                holes,
                # NEW inlet PCD sliders
                inlet_inside, inside_taper, inlet_outside, outside_taper,
            ],
            outputs=[output_image, output_table, calc_ok_state, calc_values_state]
        )

        calculate_btn.click(fn=clear_save_message, outputs=[save_status])

        # NOTE: 'Save this Design' button removed; no save_button.click wiring here.
        # Change handlers to mark calc state as dirty when any input changes
        components_to_watch = [
            pellet_size, throughput, density, fat, floating, shrimp, salmonid, other_sinking,
            channel_length, cone_length, opening_dia, cone_width,
            countersink_diameter, countersink_depth,   # <-- add these
            holes, line_rows,                           # ensure this is your actual "rows count" control
            plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
            outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
            outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
            inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
            inlet_inside, inside_taper, inlet_outside, outside_taper,  # <-- add these four
        ]

        for comp in components_to_watch:
            comp.change(
                fn=on_any_input_change,
                inputs=[],
                outputs=[calc_ok_state],
                show_progress=False,
                queue=False,
                concurrency_limit=1,
            )

        def on_add_rows(n_visible):
            new_visible = min(n_visible + ROWS_STEP, MAX_ROWS)
            updates = []
            for idx, (nh, p) in enumerate(row_components):
                show = idx < new_visible
                updates.extend([gr.update(visible=show), gr.update(visible=show)])
            return [new_visible] + updates

        def _clear_dxf_preview():
            return gr.update(value=None), gr.update(value=None)

        def on_autofill_rows_and_pcds_t3(
            total_holes,        # from global "holes" slider
            n_rows,             # from circseg_line_rows (Tab 3)
            opening_dia_mm,     # from opening_dia
            plate_dia_mm,       # from plate_diameter
            cv,
        ):
        	
            import gradio as gr
            # --- Normalize rows (0 allowed to hide all) ---
            try:
                n_rows = int(n_rows or 0)
                n_rows = max(0, min(n_rows, MAX_ROWS))
            except Exception:
                n_rows = 0

            if n_rows == 0:
                # 1st output updates the Tab 3 "Number of Rows (local)" slider to 0
                updates = [gr.update(value=0)]
                # Then zero & hide every row pair
                for _nh, _pcd in circseg_row_components:
                    updates.append(gr.update(value=0,   visible=False))  # num holes
                    updates.append(gr.update(value=0.0, visible=False))  # pcd
                    
                cv = _t3_capture_rows_to_cv(cv, 0)
                return updates + [cv]

            # --- Normalize total holes ---
            try:
                total_holes = max(0, int(total_holes or 0))
            except Exception:
                total_holes = 0

            # Even split across rows (remainder to the last row)
            base = total_holes // n_rows
            rem  = total_holes %  n_rows
            holes_by_row = [base] * (n_rows - 1) + [base + rem]

            # --- Suggest PCDs (same recipe as Tab 1) ---
            spacing = (3.7298 * float(opening_dia_mm or 0.0)) + 0.7151
            center  = (0.5243 * float(plate_dia_mm   or 0.0)) + 28.998

            if n_rows == 1:
                pcds = [center]
            elif n_rows % 2 == 1:
                k = (n_rows - 1) // 2
                pcds = [center + (i - k) * spacing for i in range(n_rows)]
            else:
                half = spacing / 2.0
                k = n_rows // 2
                lefts  = [center - (half + (k - 1 - i) * spacing) for i in range(k)]
                rights = [center + (half + i * spacing) for i in range(k)]
                pcds = lefts + rights

            # Clamp + round to match Tab 3 PCD sliders (min/max & step=0.1)
            PCD_MIN, PCD_MAX = 90.0, 350.0
            pcds = [round(max(PCD_MIN, min(PCD_MAX, float(p))), 1) for p in pcds]

            # Build updates:
            #   - First: set the Tab 3 rows slider to the same n_rows (keeps UI in sync)
            updates = [gr.update(value=n_rows)]
            flat_vals = []

            #   - Then: set values & visibility for each row pair
            for idx, (_nh_comp, _pcd_comp) in enumerate(circseg_row_components):
                if idx < n_rows:
                    nh_val = max(0, min(holes_by_row[idx], 250))  # cap per-row if needed
                    pc_val = pcds[idx]
                    updates.append(gr.update(value=nh_val,    visible=True))  # holes
                    updates.append(gr.update(value=pcds[idx], visible=True))  # pcd
                    flat_vals += [nh_val, pc_val]
                else:
                    updates.append(gr.update(visible=False))
                    updates.append(gr.update(visible=False))

            # --- Persist to cv so the generator sees Tab 3 rows ---
            try:
                cv = _t3_capture_rows_to_cv(cv, n_rows, *flat_vals)
                log_error(f"T3 AUTOFILL persisted: n_rows={n_rows}, flat={flat_vals}")
            except Exception:
                pass

            # Return UI updates + updated cv (must be wired as an output)
            return updates + [cv]

        def on_autofill_rows_and_pcds(
            total_holes,        # from holes slider
            n_rows,             # from line_rows slider
            opening_dia_mm,     # from opening_dia slider
            plate_dia_mm,       # from plate_diameter slider
            n_visible,          # from visible_rows state (unused now, but OK to keep in signature)
        ):
            # --- Safeguard & normalize rows (allows 0 to hide all) ---
            try:
                n_rows = int(n_rows or 0)
                n_rows = max(0, min(n_rows, MAX_ROWS))
            except Exception:
                n_rows = 0

            if n_rows == 0:
                updates = [0]
                for _nh, _pcd in row_components:
                    updates.append(gr.update(value=0,   visible=False))  # num_holes
                    updates.append(gr.update(value=0.0, visible=False))  # pcd
                return updates
            # ----------------------------------------------------------

            # Normalize total holes
            try:
                total_holes = max(0, int(total_holes or 0))
            except Exception:
                total_holes = 0

            # Distribute holes across rows (remainder into last row, e.g., 101→50,51 for 2 rows)
            base = total_holes // n_rows
            rem  = total_holes %  n_rows
            holes_by_row = [base] * (n_rows - 1) + [base + rem]

            # --- Suggest PCDs ---
            # Spacing between rows: y = 3.7298*x + 0.7151
            spacing = (3.7298 * float(opening_dia_mm or 0.0)) + 0.7151

            # Middle/anchor PCD from plate diameter: y = 0.5243*x + 28.998
            center = (0.5243 * float(plate_dia_mm or 0.0)) + 28.998

            if n_rows == 1:
               pcds = [center]
            elif n_rows % 2 == 1:
                k = (n_rows - 1) // 2
                pcds = [center + (i - k) * spacing for i in range(n_rows)]
            else:
                half = spacing / 2.0
                k = n_rows // 2
                lefts  = [center - (half + (k - 1 - i) * spacing) for i in range(k)]
                rights = [center + (half + i * spacing) for i in range(k)]
                pcds = lefts + rights

            # Clamp & round to 1 decimal to match PCD slider step=0.1
            PCD_MIN, PCD_MAX = 90.0, 350.0
            pcds = [round(max(PCD_MIN, min(PCD_MAX, float(p))), 1) for p in pcds]

            # Show exactly n_rows rows
            new_visible = n_rows

            # Build updates: first output is visible_rows, then each (num_holes, pcd)
            updates = [new_visible]
            for idx, (_nh_comp, _pcd_comp) in enumerate(row_components):
                if idx < n_rows:
                    nh_val = holes_by_row[idx]
                    nh_val = max(0, min(nh_val, 250))  # adjust if your per-row max != 250
                    updates.append(gr.update(value=nh_val,      visible=True))   # num_holes
                    updates.append(gr.update(value=pcds[idx],   visible=True))   # pcd
                else:
                    updates.append(gr.update(visible=False))
                    updates.append(gr.update(visible=False))

            return updates

        def _render_inset_from_existing(dxf_path, zoom_inset):
            """
            Re-render ONLY the inset preview using the current on-disk DXF.
            Does not touch quick_pdf, stale flags, or any other outputs.
            Returns a gr.Image-compatible value (filepath or numpy array).
            """
            import gradio as gr
            try:
                if not dxf_path or str(dxf_path).strip() == "":
                    return gr.update()  # nothing to render

                # --- Try your renderer by common names (keep whichever you actually have) ---
                try:
                    # If you have a dedicated inset renderer
                    return render_inset_png(dxf_path, zoom_inset)           # <- keep if exists
                except NameError:
                    pass

                try:
                    # Alternative naming
                    return render_inset_from_dxf(dxf_path, zoom_inset)      # <- keep if exists
                except NameError:
                    pass

                try:
                    # If you have a combined preview function returning (plate_img, inset_img)
                    plate_img, inset_img = render_previews(dxf_path, zoom_inset)  # <- keep if exists
                    return inset_img
                except NameError:
                    pass

                # If none of the known renderers exist, do nothing (keeps current preview)
                return gr.update()
            except Exception as e:
                try:
                    log_error(f"_render_inset_from_existing fail: {e}")
                except Exception:
                    pass
                return gr.update()

        def _clear_selection():
            """Clear server-side selection + header + chip."""
            return (
                None,  # selected_index_state
                "",    # selected_filename_state
                gr.update(value="<strong>Selected:</strong> <em>none</em>"),  # header
                gr.update(value="", visible=False),                           # chip hidden
            )

        def on_reset_all():
            updates = []
            # zero main sliders
            updates.append(gr.update(value=0))  # holes
            updates.append(gr.update(value=0))  # line_rows
            # hide all rows, zero values
            vis = 0
            row_updates = []
            for _nh, _pcd in row_components:
                row_updates.append(gr.update(value=0, visible=False))
                row_updates.append(gr.update(value=0.0, visible=False))
            # status/ok-state if needed
            return [vis] + row_updates + updates

        holes.change(
            fn=_render_target_holes,
            inputs=[holes],
            outputs=[stag_total],
            show_progress=False,
            queue=False,
        )

        btn_stag_calc.click(
            fn=calc_staggered_pattern,
            inputs=[
                opening_dia, cone_width, calc_values_state,
                segments, wall_width, seg_inner_pcd, seg_outer_pcd,
                corner_radius, padding_adj, holes,
            ],
            outputs=[staggered_pts_state, stag_counts, stag_msg],
            show_progress=False,
        )

        stag_generate_btn.click(
            fn=_btn_busy,  # disables button + shows "Rendering DXF…" in stag_msg
            inputs=[],
            outputs=[stag_generate_btn, stag_msg],
            show_progress=False,
            queue=False,
            concurrency_limit=1,
        ).then(
            fn=_previews_only,   # heavy work: ONLY previews + path
            inputs=[
                pellet_size, plate_diameter, opening_dia, cone_width,
                chamfer_plate, chamfer_outer_opening, chamfer_inner_opening, chamfer_center_hole,
                outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                outer_opening_pcd, inner_opening_pcd, die_center_hole_diameter,
                inset_zoom, inlet_inside, inside_taper, inlet_outside, outside_taper,
                throughput, calc_values_state,
                holes, staggered_pts_state, die_style_state_stag,  # force staggered path
                # segment & geometry inputs (Tab 2)
                stag_segments, seg_bolts_chk,
                stag_segments,  # segments
                wall_width, seg_inner_pcd, seg_outer_pcd,
                corner_radius, padding_adj,
                # no row inputs for staggered
            ],
            outputs=[plate_preview, inset_preview, dxf_path_state],
            show_progress=True,   # <-- spinners only in the two Image components
            # (omit queue here; keep app’s default so progress overlays work)
        ).then(
            fn=_mark_ready_and_show_dxf,
            inputs=[dxf_path_state],
            outputs=[dxf_ready_state, dxf_dl, open_pdf_btn, stale_banner],
            show_progress=False,
            queue=False,
        ).then(
            fn=_reset_inset_zoom_control,
            inputs=[],
            outputs=[inset_zoom],
            show_progress=False,
            queue=False,
        ).then(
            fn=_btn_idle,  # re-enable button + clear message
            inputs=[],
            outputs=[stag_generate_btn, stag_msg],
            show_progress=False,
            queue=False,
        ).then(
            fn=_hide_pdf_link,  # clear any stale quick_pdf
            inputs=[],
            outputs=[quick_pdf],
            show_progress=False,
            queue=False,
        )  

        generate_btn.click(
            fn=_btn_busy,
            inputs=[],
            outputs=[generate_btn, status_html],
            show_progress=False,
            queue=False,
            concurrency_limit=1,
        ).then(
            fn=generate_die_plate_dxf,
            inputs=[
                pellet_size, plate_diameter, opening_dia, cone_width,
                chamfer_plate, chamfer_outer_opening, chamfer_inner_opening, chamfer_center_hole,
                outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                outer_opening_pcd, inner_opening_pcd, die_center_hole_diameter,
                # include zoom + the 4 NEW PCD sliders (must match function signature)
                inset_zoom, inlet_inside, inside_taper, inlet_outside, outside_taper,
                throughput, calc_values_state,
                holes, staggered_pts_state, die_style_state,
                stag_segments, seg_bolts_chk,
                # Existing (used by circular; ignored by staggered):
                segments, wall_width, seg_inner_pcd, seg_outer_pcd,
                corner_radius, padding_adj,
                *[comp for pair in row_components for comp in pair],
            ],
            outputs=[plate_preview, inset_preview, dxf_path_state],
            show_progress=True,   # heavy work
            # (omit queue here; leave default so progress works if app queue is enabled)
        ).then(
            fn=_mark_ready_and_show_dxf,
            inputs=[dxf_path_state],
            outputs=[dxf_ready_state, dxf_dl, open_pdf_btn, stale_banner],
            show_progress=False,
            queue=False,
        ).then(
            fn=_reset_inset_zoom_control,
            inputs=[],
            outputs=[inset_zoom],
            show_progress=False,
            queue=False,
        ).then(
            fn=_btn_idle,
            inputs=[],
            outputs=[generate_btn, status_html],
            show_progress=False,
            queue=False,
        ).then(
            fn=_hide_pdf_link,
            inputs=[],
            outputs=[quick_pdf],
            show_progress=False,
            queue=False,
        )

        # ——— Zoom inset status (single binding) ———
        # Render only when the user releases the slider (no programmatic retrigger)
        inset_zoom.release(
            fn=_inset_busy,
            inputs=[],
            outputs=[inset_status, inset_zoom],
            queue=False,
            show_progress=False,
        ).then(
            fn=update_inset_zoom,
            inputs=[inset_zoom],
            outputs=[inset_preview],
            queue=True,           # keep the heavy redraw queued
            show_progress=True,
        ).then(
            fn=_inset_idle,
            inputs=[],
            outputs=[inset_status, inset_zoom],
            queue=False,
            show_progress=False,
        )

        # Clicking the tiny button feeds the path into the hidden File widget to trigger the browser's download
        dxf_dl.click(
            fn=lambda p: p,
            inputs=[dxf_path_state],
            outputs=[quick_dxf],     # hidden gr.File component
            show_progress=False,
        )

        # --- Mark drawings stale when any relevant input changes -----------------
        stale_triggers = [
                pellet_size, plate_diameter, opening_dia, cone_width,
                chamfer_plate, chamfer_outer_opening, chamfer_inner_opening, chamfer_center_hole,
                outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                outer_opening_pcd, inner_opening_pcd, die_center_hole_diameter,
                throughput,
                countersink_diameter, countersink_depth,
                inlet_inside, inside_taper, inlet_outside, outside_taper,
        ]

        # Circular tab row sliders (if defined): [(num_holes_i, pcd_i), ...]
        try:
                stale_triggers += [comp for pair in row_components for comp in pair]
        except NameError:
                pass

        # Staggered (Tab 2) controls (if defined)
        try:
                stale_triggers += [
                        stag_segments, wall_width, seg_inner_pcd, seg_outer_pcd,
                        corner_radius, padding_adj, seg_bolts_chk,
                ]
        except NameError:
                pass

        # Hybrid "Circular on Segmented" (Tab 3) controls (if defined)
        try:
                stale_triggers += [
                        stag_segments_t3, wall_width_t3, seg_inner_pcd_t3, seg_outer_pcd_t3,
                        corner_radius_t3, padding_adj_t3, seg_bolts_chk_t3,
                ]
        except NameError:
                pass

        # Any Tab 3 local row controls you created (if defined)
        try:
                stale_triggers += flattened_row_inputs_tab3
        except NameError:
                pass

        # Attach the same stale handler to all triggers
        for ctrl in stale_triggers:
                ctrl.change(
                        fn=_mark_stale,
                        inputs=[],
                        outputs=[dxf_ready_state, dxf_dl, open_pdf_btn, stale_banner],
                        queue=False,
                        show_progress=False,
                )

        # --- One reset to rule them all ------------------------------------
        def on_reset():
            # Clear any cached artifacts so buttons don't try to reuse stale files
            try:
                global cached_doc, cached_filename, cached_inset_center, cached_inset_focus_center, cached_export_path
                cached_doc = None
                cached_filename = None
                cached_inset_center = None
                cached_export_path = None
            except Exception:
                pass

            updates = []

            # Top specs
            updates += [
                gr.update(value=9.0),  # pellet_size
                gr.update(value=9.0),  # throughput
                gr.update(value=500),  # density
                gr.update(value=24),   # fat
                gr.update(value=False),  # floating
                gr.update(value=False),  # shrimp
                gr.update(value=False),  # salmonid
                gr.update(value=False),  # other_sinking
            ]

            # Design (opening_dia, channel_length, cone_length, cone_width,
            #         countersink_plus_radius (UI field is named countersink_diameter), countersink_depth, holes)
            updates += [
                gr.update(value=6.3),   # opening_dia
                gr.update(value=14.0),  # channel_length
                gr.update(value=4.0),   # cone_length
                gr.update(value=13.0),  # cone_width
                gr.update(value=0.0),   # countersink_diameter  (UI = +radius mm)
                gr.update(value=0.0),   # countersink_depth
                gr.update(value=500),   # holes / number_of_holes
            ]

            # Plate / track / bolts
            updates += [
                gr.update(value=250),  # plate_diameter
                gr.update(value=2),    # chamfer_plate
                gr.update(value=48.5), # die_center_hole_diameter
                gr.update(value=0),    # chamfer_center_hole
                gr.update(value=189),  # outer_opening_pcd
                gr.update(value=0),    # chamfer_outer_opening
                gr.update(value=128),  # inner_opening_pcd
                gr.update(value=0),    # chamfer_inner_opening
                gr.update(value=12),   # outer_bolt_count
                gr.update(value=230),  # outer_bolt_pcd
                gr.update(value=10.5), # outer_bolt_diameter
                gr.update(value=4),    # inner_bolt_count
                gr.update(value=75),   # inner_bolt_pcd
                gr.update(value=8.2),  # inner_bolt_diameter
            ]

            # Metadata (keep order in sync with reset_outputs)
            updates += [
                gr.update(value=""),  # extruder_type
                gr.update(value=""),  # production_line
                gr.update(value=""),  # plant_name
                gr.update(value=""),  # product_type
                gr.update(value=""),  # die_performance
                gr.update(value=""),  # comments
            ]

            # Rows (show first ROWS_STEP, hide/zero the rest)
            for idx, (nh, p) in enumerate(row_components):
                show = idx < ROWS_STEP
                updates.append(gr.update(value=0, visible=show))  # num_holes
                updates.append(gr.update(value=0, visible=show))  # pcd

            # Extras and state
            updates += [
                gr.update(value=None, visible=False),   # quick_dxf
                gr.update(value=None, visible=False),   # quick_pdf
                gr.update(value=""),                    # save_status
                gr.update(value=None),                  # plate_preview
                gr.update(value=None),                  # inset_preview
                None,                                   # dxf_path_state
                False,                                  # calc_ok_state
                {},                                     # calc_values_state
                ROWS_STEP,                               # visible_rows
            ]
            return updates

        # Keep outputs order EXACTLY aligned with the updates list above
        reset_outputs = [
            # Top specs
            pellet_size, throughput, density, fat, floating, shrimp, salmonid, other_sinking,
            # Design
            opening_dia, channel_length, cone_length, cone_width,
            countersink_diameter, countersink_depth,  # UI field 'countersink_diameter' = +radius
            holes,
            # Plate / track / bolts
            plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
            outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
            outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
            inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
            # Metadata
            extruder_type, production_line, plant_name, product_type, die_performance, comments,
            # Rows
            *[comp for pair in row_components for comp in pair],
            # Extras / state
            quick_dxf, quick_pdf, save_status, plate_preview, inset_preview,
            dxf_path_state, calc_ok_state,
            calc_values_state,
            visible_rows,
        ]

        # ============= Inline Die Manager (TABLE) =============

        import pandas as pd
        
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
            import pandas as pd
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

                    # Always use the saved feed_type, never metadata Type
                    species_label = norm_feed(inputs.get("feed_type", ""))

                    rows.append({
                        "Saved": fmt_time(data.get("timestamp","")),
                        "Feed Type": species_label,
                        "Extruder": meta.get("extruder_type", ""),
                        "Line": meta.get("production_line", ""),
                        "Plant": meta.get("plant_name", ""),
                        "Type": meta.get("product_type", ""),
                        "Performance": meta.get("die_performance", ""),
                        "Comments": meta.get("comments", ""),
                        "Pellet Size (mm)": _r1(inputs.get("pellet_size")),
                        "# Rows": _r0(d.get("rows_count", 0)),
                        "# Holes": _r0(inputs.get("number_of_holes")),
                        "Die Ø (mm)": _r2(inputs.get("opening_dia")),
                        "Land L (mm)": _r1(inputs.get("channel_length")),
                        "Cone L (mm)": _r1(inputs.get("cone_length")),
                        "Cone W (mm)": _r1(inputs.get("cone_width")),
                        "OA 1 hole (mm²)": _r2(d.get("open_area_one_hole", 0)),
                        "OA total (mm²)": _r0(d.get("total_plate_open_area", 0)),
                        "OA/t (mm²/t/h)": _r0(d.get("open_area_per_tonne", 0)),
                        "L/D": _r2(d.get("ld_ratio", 0)),
                        "Die L (mm)": _r1(d.get("die_total_length", 0)),
                        "Cone °": _r1(d.get("cone_angle", 0)),
                        "Bulk Density": _r0(d.get("bulk_density", 0)),
                        "Expansion (%)": _r1(d.get("expansion_percent", 0)),
                        "Inlet inside PCD (mm)":  _r1(inputs.get("inlet_inside_pcd")),
                        "Inside taper PCD (mm)":  _r1(inputs.get("inside_taper_pcd")),
                        "Inlet outside PCD (mm)": _r1(inputs.get("inlet_outside_pcd")),
                        "Outside taper PCD (mm)": _r1(inputs.get("outside_taper_pcd")),
                        "File": fname,
                    })
                except Exception as e:
                    log_error(f"collect_manager_rows() failed for {fname}: {e}")
            return pd.DataFrame(rows)

        def clean_table_view(df):
            import pandas as pd
            if df is None or df.empty:
                return pd.DataFrame(columns=MANAGER_COLUMNS)

            out = ensure_manager_columns(df)

            # keep rows with a timestamp and either >0 rows or >0 holes
            holes_ok = out["# Holes"].fillna(0) > 0
            rows_ok  = out["# Rows"].fillna(0) > 0
            out = out[(out["Saved"].str.len() > 0) & (rows_ok | holes_ok)]
            return out

        def delete_selected_now(file_key, sel_idx, df, p_min, p_max, species, plant):
            """
            Delete the selected file (from header/selection). No checkbox, no blink.
            Returns:
              saved_table, manager_status, selected_index_state, selected_filename_state,
              saved_table_html, selected_preview_html, selected_header_html
            """
            import pandas as pd

            # Normalize DF
            if isinstance(df, list):
                if df:
                    try:
                        df = pd.DataFrame(df[1:], columns=df[0])
                    except Exception:
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()

            # Guard: must have a key
            if not file_key or not str(file_key).strip():
                warn = "<div class='chip warn'>Click <strong>Select</strong> first.</div>"
                return (
                    df,                                  # saved_table unchanged
                    gr.update(value=warn),               # manager_status (ONLY place we show the warn)
                    sel_idx,                             # selected_index unchanged
                    file_key or "",                      # selected_filename unchanged
                    gr.update(),                         # saved_table_html unchanged
                    gr.update(value="", visible=False),  # selected_preview_html cleared/hidden
                    gr.update(value="<strong>Selected:</strong> <em>none</em>"),
                )

            # Delete
            new_df, status_html = delete_focused_row(file_key, p_min, p_max, species, plant)

            # Rebuild HTML (data changed) with NO selection
            new_df = add_delete_col(new_df)
            html = df_to_compact_html(new_df, selected_index=None)

            # ONE clean line: green chip + compact file name chip (no trailing text)
            try:
                short = _short_display_name(file_key)
            except Exception:
                short = file_key
            msg = f"<div class='chip ok'><span class='check'>✓</span>Deleted</div><div class='chip'>{short}</div>"

            return (
                new_df,                                 # saved_table
                gr.update(value=msg),                   # manager_status (single line)
                None,                                   # selected_index cleared
                "",                                     # selected_filename cleared
                gr.update(value=html),                  # saved_table_html
                gr.update(value="", visible=False),     # selected_preview_html cleared/hidden
                gr.update(value="<strong>Selected:</strong> <em>none</em>"),  # header cleared
            )

        def delete_focused_row(selected_filename, pellet_min, pellet_max, species_choices, plant_choices):
            import os
            if not selected_filename:
                # Return current filtered DF unchanged + message
                df = build_table_df(pellet_min, pellet_max, species_choices or [], plant_choices or [], select_all=False)
                return df, "No row selected."

            try:
                target = selected_filename
                if not os.path.isabs(target):
                    target = os.path.join(SAVE_DIR, selected_filename)
                if os.path.exists(target):
                    os.remove(target)
                    msg = f"Deleted 1 row ({os.path.basename(target)})."
                else:
                    msg = "File not found."
            except Exception as e:
                log_error(f"Delete failed for {selected_filename}: {e}")
                msg = "Delete failed."

            # Rebuild filtered table AFTER delete
            df = build_table_df(pellet_min, pellet_max, species_choices or [], plant_choices or [], select_all=False)
            return df, msg

        def load_focused_to_designer(selected_filename):
            """
            Must return exactly as many outputs as wired in load_btn.click(...).
            Appends the 6 metadata fields at the end:
              extruder_type, production_line, plant_name, product_type, die_performance, comments
            """
            if not selected_filename:
                # Keep the existing placeholder sizing logic, but add 6 blanks at the end for metadata
                base = [gr.update()] * (8 + 5 + 18 + (MAX_ROWS*2) + 7) + ["No selection."]
                return base + ["", "", "", "", "", ""]

            try:
                data = load_die_design(selected_filename)
                inp  = data.get("inputs", {}) or {}

                # --- Top specs (8)
                ps    = inp.get("pellet_size", 9.0)
                th    = inp.get("throughput", 9.0)
                bd    = inp.get("bulk_density", 500.0)
                fatv  = inp.get("fat", 24.0)
                ft    = inp.get("feed_type", "Salmonid")
                fl    = (ft == "Floating")
                sh    = (ft == "Shrimp/Prawn") or (ft == "Shrimp_Prawn")
                sa    = (ft == "Salmonid")
                osk   = (ft == "Other sinking")

                # --- Design (5)
                od    = inp.get("opening_dia", 6.3)
                cl    = inp.get("channel_length", 14.0)
                col   = inp.get("cone_length", 4.0)
                cw    = inp.get("cone_width", 8.0)
                nh    = inp.get("holes", inp.get("number_of_holes", 300))

                # --- Plate/track/bolts (14)
                plate_diameter            = inp.get("plate_diameter", 700.0)
                chamfer_plate             = inp.get("chamfer_plate", 0.0)
                die_center_hole_diameter  = inp.get("die_center_hole_diameter", 20.0)
                chamfer_center_hole       = inp.get("chamfer_center_hole", 0.0)

                outer_opening_pcd         = inp.get("outer_opening_pcd", 500.0)
                chamfer_outer_opening     = inp.get("chamfer_outer_opening", 0.0)
                inner_opening_pcd         = inp.get("inner_opening_pcd", 200.0)
                chamfer_inner_opening     = inp.get("chamfer_inner_opening", 0.0)

                outer_bolt_count          = inp.get("outer_bolt_count", 24)
                outer_bolt_pcd            = inp.get("outer_bolt_pcd", 600.0)
                outer_bolt_diameter       = inp.get("outer_bolt_diameter", 14.0)

                inner_bolt_count          = inp.get("inner_bolt_count", 12)
                inner_bolt_pcd            = inp.get("inner_bolt_pcd", 300.0)
                inner_bolt_diameter       = inp.get("inner_bolt_diameter", 12.0)
                
                vals_plate_track_bolts = [
                    # ...existing 14 plate/bolt outputs...
                    float(inp.get("inlet_inside_pcd", 0.0)),
                    float(inp.get("inside_taper_pcd", 0.0)),
                    float(inp.get("inlet_outside_pcd", 0.0)),
                    float(inp.get("outside_taper_pcd", 0.0)),
                ]

                # --- Rows (2 per row)
                rows     = (inp.get("hole_rows", []) or [])
                row_vals = []
                for r in rows:
                    n_holes = r.get("n_holes", r.get("num_holes", 0))
                    pcd     = r.get("pcd", 0.0)
                    row_vals.extend([n_holes, pcd])

                # --- UI placeholders & status (8)
                quick_dxf       = gr.update()
                save_status     = gr.update()
                plate_preview   = gr.update()
                inset_preview   = gr.update()
                dxf_path_state  = ""
                calc_ok_state   = True
                visible_rows    = len(rows)
                manager_status  = "Loaded."

                # --- Metadata (6)
                meta             = data.get("metadata", {}) or {}
                extruder_type    = meta.get("extruder_type", "")
                production_line  = meta.get("production_line", "")
                plant_name       = meta.get("plant_name", "")
                product_type     = meta.get("product_type", "")
                die_performance  = meta.get("die_performance", "")
                comments         = meta.get("comments", "")

                return [
                    # Top specs (8)
                    ps, th, bd, fatv, fl, sh, sa, osk,
                    # Design (5)
                    od, cl, col, cw, nh,
                    # Plate/track/bolts (14)
                    plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                    outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                    outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                    inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                    # NEW: the 4 inlet sliders so UI restores them
                    float(inp.get("inlet_inside_pcd", 0.0)),
                    float(inp.get("inside_taper_pcd", 0.0)),
                    float(inp.get("inlet_outside_pcd", 0.0)),
                    float(inp.get("outside_taper_pcd", 0.0)),
                ] + row_vals + [
                    # UI + status (8)
                    quick_dxf, save_status, plate_preview, inset_preview,
                    dxf_path_state, calc_ok_state, visible_rows, manager_status,
                    # NEW: metadata (6)
                    extruder_type, production_line, plant_name, product_type, die_performance, comments,
                ]

            except Exception as e:
                log_error(f"load_focused_to_designer() failed: {e}")
                # same count as success: placeholders, then "Load failed." for manager_status, then 6 blanks
                base = [gr.update()] * (8 + 5 + 14 + (MAX_ROWS*2) + 7) + ["Load failed."]
                return base + ["", "", "", "", "", ""]

        # ------- Die Manager ---------------------------------------------------------
        # ---- Initial data for Selector grid & HTML mirror ----

        base_all = clean_table_view(collect_manager_rows())
        ps_min0, ps_max0 = _pellet_bounds(base_all)

        # Build the first table with NO filters and a fall back-----
        base_df = build_table_df(ps_min0, ps_max0, None, None, select_all=True)

        if base_df is None or getattr(base_df, "empty", True):
            base_df = base_all.copy()

        # Ensure DEL column exists and pad to at least 4 rows so the table doesn't look tiny
        base_df = add_delete_col(base_df)
        base_df = _pad_display_rows(base_df, 4)   # <- this is the bit you were missing
         
        species_options = ["Floating", "Shrimp/Prawn", "Salmonid", "Other sinking"]
        plant_options = (
            [p for p in uniq(base_all["Plant"]) if str(p).strip()]
            if (base_all is not None and not base_all.empty and "Plant" in base_all.columns)
            else []
        )

        try:
            log_error(f"BOOT: rows base_df={len(base_df)} base_all={len(base_all)} ps=({ps_min0},{ps_max0}) plants={len(plant_options)}")
        except Exception:
            pass    
            
        with gr.Column(elem_classes=["manager-panel"]):       
            gr.Markdown("**Filters**")
            with gr.Row():
                with gr.Column(scale=1, min_width=220):
                    pellet_min = gr.Slider(minimum=ps_min0, maximum=ps_max0, step=0.1,
                                           value=ps_min0, label="Pellet size MIN (mm)")
                    pellet_max = gr.Slider(minimum=ps_min0, maximum=ps_max0, step=0.1,
                                           value=ps_max0, label="Pellet size MAX (mm)")
                with gr.Column(scale=1, min_width=220):
                    species_filter = gr.CheckboxGroup(
                        choices=species_options, label="Feed Type", value=[]
                    )
                with gr.Column(scale=3, min_width=360):
                    plant_filter = gr.CheckboxGroup(
                        choices=plant_options, label="Plant", value=[]
                    )
                select_all = gr.Checkbox(value=False, label="Select All / None", visible=False)

            # --- Actions -------------------------------------------------------------
            gr.Markdown("")
            # --- Status block (above the table/buttons) -----------------------------------
            with gr.Row():
                selected_header_html = gr.HTML(
                    value="<strong>Selected:</strong> <em>none</em>",
                    elem_id="selected_header_html",
                    visible=True,
                )
                selected_preview_html = gr.HTML(
                    value="<em>No row selected</em>",
                    elem_id="selected_preview_html",
                    elem_classes=["selected-row-preview"],
                    visible=False,
                )
                exported_excel = gr.File(label="Excel export", visible=False)

            # A dedicated status bar you can update from events/callbacks
            manager_status = gr.HTML(value="", elem_id="manager_status", visible=True)

            # --- Buttons ------------------------------------------------------------------
            with gr.Row():
                select_btn        = gr.Button("Select", elem_id="select_btn")
                reset_filters_btn = gr.Button("Reset Filters")
                delete_btn        = gr.Button("Delete")
                load_btn          = gr.Button("Load back to Designer")
                save_excel_btn    = gr.Button("Export to Excel")

            # --- Selector: accordion ONLY wraps the grid -------------------------------
            with gr.Accordion("Selector: click Select to Delete or Return a design", open=False, elem_id="selector_acc"):
                saved_table = gr.Dataframe(
                    value=base_df,
                    headers=base_df.columns.tolist(),
                    datatype=_df_datatypes(base_df),
                    interactive=True,
                    height=240,
                    wrap=True,
                    elem_id="saved_table",
                    elem_classes=["saved-grid","selector-grid"],
                    label="",
                    visible=False,
                )

            with gr.Row():
                sel_top_btn    = gr.Button("⤒ Top")
                sel_prev_btn   = gr.Button("▲ Prev")
                sel_next_btn   = gr.Button("▼ Next")
                sel_bottom_btn = gr.Button("⤓ Bottom")

            # --- Client-side row movers (no re-render, no blink) ---
            from textwrap import dedent

            js_move_prev = dedent("""\
            () => {    
    
              const host = document.getElementById('saved_table_html');   // was querySelector('.saved-grid-host')
              if (!host) return;
              const grid = host.querySelector('.saved-grid-html table');

              if (!grid || !grid.tBodies || !grid.tBodies[0]) return;
              const tbody = grid.tBodies[0];

              const rows = Array.from(tbody.querySelectorAll('tr'));
              if (!rows.length) return;

              let cur = tbody.querySelector('tr.selected');
              if (!cur) cur = rows.find(r => !r.classList.contains('pad')) || rows[0];

              let i = rows.indexOf(cur) - 1;
              while (i >= 0 && rows[i].classList.contains('pad')) i -= 1;
              if (i < 0) return;

              cur.classList.remove('selected');
              rows[i].classList.add('selected');

              // >>> NEW: push the selection index into the hidden Number so server state updates
              const box = document.querySelector('#sel_idx_html input');
              if (box) {
                box.value = String(i);
                box.dispatchEvent(new Event('input',  { bubbles: true }));
                box.dispatchEvent(new Event('change', { bubbles: true }));
              }
              // <<< END NEW

              requestAnimationFrame(() => rows[i].scrollIntoView({ block: 'nearest', behavior: 'smooth' }));

            }
            """).strip()

            js_move_next = dedent("""\
            () => {
   
              const host = document.getElementById('saved_table_html');   // was querySelector('.saved-grid-host')
              if (!host) return;
              const grid = host.querySelector('.saved-grid-html table');

              if (!grid || !grid.tBodies || !grid.tBodies[0]) return;
              const tbody = grid.tBodies[0];

              const rows = Array.from(tbody.querySelectorAll('tr'));
              if (!rows.length) return;

              let cur = tbody.querySelector('tr.selected');
              if (!cur) cur = rows.find(r => !r.classList.contains('pad')) || rows[0];

              let i = rows.indexOf(cur) + 1;
              while (i < rows.length && rows[i].classList.contains('pad')) i += 1;
              if (i >= rows.length) return;

              cur.classList.remove('selected');
              rows[i].classList.add('selected');

              // >>> NEW: sync hidden Number so server sees the selection
              const box = document.querySelector('#sel_idx_html input');
              if (box) {
                  box.value = String(i);
                  box.dispatchEvent(new Event('input',  { bubbles: true }));
                  box.dispatchEvent(new Event('change', { bubbles: true }));
              }
              // <<< END NEW

              requestAnimationFrame(() => rows[i].scrollIntoView({ block: 'nearest', behavior: 'smooth' }));

            }
            """).strip()

            js_move_top = dedent("""\
            () => {
              const host = document.getElementById('saved_table_html');   // was querySelector('.saved-grid-host')
              if (!host) return;
              const grid = host.querySelector('.saved-grid-html table');

              if (!grid || !grid.tBodies || !grid.tBodies[0]) return;
              const tbody = grid.tBodies[0];

              const rows = Array.from(tbody.querySelectorAll('tr'));
              if (!rows.length) return;

              const target = rows.find(r => !r.classList.contains('pad')) || rows[0];
              const cur = tbody.querySelector('tr.selected');
              if (cur !== target) { if (cur) cur.classList.remove('selected'); target.classList.add('selected'); }

              // >>> NEW: sync hidden Number so server sees the selection (top row)
              const i = rows.indexOf(target);
              const box = document.querySelector('#sel_idx_html input');
              if (box) {
                box.value = String(i);
                box.dispatchEvent(new Event('input',  { bubbles: true }));
                box.dispatchEvent(new Event('change', { bubbles: true }));
              }
              // <<< END NEW

              requestAnimationFrame(() => target.scrollIntoView({ block: 'nearest', behavior: 'smooth' }));
            }
            """).strip()

            js_move_bottom = dedent("""\
            () => {
              const host = document.getElementById('saved_table_html');   // was querySelector('.saved-grid-host')
              if (!host) return;
              const grid = host.querySelector('.saved-grid-html table');

              if (!grid || !grid.tBodies || !grid.tBodies[0]) return;
              const tbody = grid.tBodies[0];

              const rows = Array.from(tbody.querySelectorAll('tr'));
              if (!rows.length) return;

              let i = rows.length - 1;
              while (i >= 0 && rows[i].classList.contains('pad')) i -= 1;
              if (i < 0) return;

              const target = rows[i];
              const cur = tbody.querySelector('tr.selected');
              if (cur !== target) { if (cur) cur.classList.remove('selected'); target.classList.add('selected'); }

              // >>> NEW: sync hidden Number so server sees the selection (bottom row)
              const box = document.querySelector('#sel_idx_html input');
              if (box) {
                box.value = String(i);
                box.dispatchEvent(new Event('input',  { bubbles: true }));
                box.dispatchEvent(new Event('change', { bubbles: true }));
              }
              // <<< END NEW

              requestAnimationFrame(() => target.scrollIntoView({ block: 'nearest', behavior: 'smooth' }));

            }
            """).strip()

            from textwrap import dedent

            js_bind_row_clicks = dedent("""
            () => {
              const host = document.getElementById('saved_table_html');
              if (!host) return;
              const tbody = host.querySelector('.saved-grid-html tbody');
              if (!tbody) return;

              // Clear any old handlers to avoid duplicates
              tbody.onclick = null;

              tbody.onclick = (ev) => {
                const tr = ev.target.closest('tr');
                if (!tr) return;
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const idx = rows.indexOf(tr);

                // Toggle highlight (no rebuild)
                rows.forEach((r, i) => r.classList.toggle('selected', i === idx));

                // Push index into hidden Number & trigger a change → server callback runs
                const box = document.querySelector('#sel_idx_html input');
                if (box) {
                  box.value = String(idx);
                  box.dispatchEvent(new Event('input',  { bubbles: true }));
                  box.dispatchEvent(new Event('change', { bubbles: true }));
                }
              };
            }
            """)

            js_highlight_row = dedent("""
            (selIdx) => {
              const host  = document.getElementById('saved_table_html');
              const tbody = host?.querySelector('.saved-grid-html tbody');
              if (!tbody) return;

              const rows = Array.from(tbody.querySelectorAll('tr'));
              // Always clear any previous highlight
              rows.forEach(tr => tr.classList.remove('selected'));

              // Only (re)apply if selIdx is a valid integer within bounds
              const i = Number.parseInt(selIdx, 10);
              if (Number.isInteger(i) && i >= 0 && i < rows.length) {
                rows[i].classList.add('selected');
                rows[i].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
              }
            }
            """).strip()

            # HTML mirror lives OUTSIDE the accordion
            saved_table_html = gr.HTML(
                value=df_to_compact_html(base_df, selected_index=None),
                elem_id="saved_table_html",
                elem_classes=["saved-grid-host"]
            )

            # --- Charts (lifted above tables) ---------------------------------------
            gr.Markdown("**Insights**")
            with gr.Row():
                chart1 = gr.Plot(label="")
                chart2 = gr.Plot(label="")
            with gr.Row():
                chart3 = gr.Plot(label="")
                chart4 = gr.Plot(label="")

            # --- States (invisible; placement doesn’t affect layout) --------

            manager_df_state        = gr.State(None)
            selected_row_idx_state  = gr.State(None)
            selected_filename_state = gr.State("")

            # --- Selector Button Controls (JS-only) ---------------------------------
            nav_outputs = [selected_row_idx_state, selected_filename_state]

            def _noop():
                return None

            sel_prev_btn.click(_noop, [], [], show_progress=False, queue=False
            ).then(js=js_move_prev, inputs=[], outputs=[])

            sel_next_btn.click(_noop, [], [], show_progress=False, queue=False
            ).then(js=js_move_next, inputs=[], outputs=[])

            sel_top_btn.click(_noop, [], [], show_progress=False, queue=False
            ).then(js=js_move_top, inputs=[], outputs=[])

            sel_bottom_btn.click(_noop, [], [], show_progress=False, queue=False
            ).then(js=js_move_bottom, inputs=[], outputs=[])

            # Select → copy current highlight into header & chip (no HTML rebuild)
            select_btn.click(
                fn=on_select_current,
                inputs=[selected_index_state, saved_table],
                outputs=[selected_index_state, selected_filename_state, selected_header_html, selected_preview_html, manager_status],
                show_progress=False,
                queue=False,
            )

            def _guard_have_selection(file_key):
                if not file_key or not str(file_key).strip():
                    return gr.update(value="<div class='chip warn'>Click <strong>Select</strong> first.</div>")
                return gr.update(value="")

            load_btn.click(
                fn=_guard_have_selection,
                inputs=[selected_filename_state],
                outputs=[manager_status],
                show_progress=False,
                queue=False,
            ).then(
                fn=load_focused_to_designer,  # your existing function
                inputs=[selected_filename_state],
                outputs=[  # keep your current outputs list for the designer panel

                ],
                show_progress=False,
                queue=False,
            )

            def charts_log(*args, **kwargs):
                pass

            def update_charts_multi(pellet_min_val, pellet_max_val, specieses, plants):
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                import re

                df = build_table_df(pellet_min_val, pellet_max_val, specieses, plants, select_all=False).copy()

                def find_col(exacts=(), patterns=()):
                    for name in exacts:
                        if name in df.columns:
                            return name
                    for col in df.columns:
                        for pat in patterns:
                            if re.search(pat, col, re.IGNORECASE):
                                return col
                    return None

                oa_exacts = ("OA/t", "OA/t (mm²/t/h)", "OA/t (mm2/t/h)", "OA per t", "OA_per_t",
                             "Open Area per t", "OpenAreaPerT")
                oa_patterns = (r'\boa\s*/?\s*t\b', r'open\s*area.*(per|/)\s*t', r'\boa[/\s]*per')

                bd_exacts = ("Bulk Density", "BulkDensity", "bulk_density", "Density (kg/m3)")
                bd_patterns = (r'bulk\s*density', r'^density(\s*\(.*\))?$')

                exp_exacts = ("Expansion (%)", "Expansion %", "Expansion percent", "Expansion", "expansion_percent")
                exp_patterns = (r'^expansion\b', r'expansion\s*%')

                holes_exacts = ("# Holes", "Holes", "number_of_holes", "num_holes", "n_holes")
                feed_exacts  = ("Feed Type", "FeedType", "Species", "Species/Type", "feed_type")

                ld_exacts   = ("L/D", "L-D", "LD", "L/D ratio", "L over D")
                ld_patterns = (r'^\s*l\s*[/\-]?\s*d\s*$', r'\bl\s*/\s*d\b', r'\bl[\W_]*d\b')
                dio_exacts   = ("Die Ø (mm)", "Die O (mm)", "Opening Dia (mm)", "Die opening (mm)", "Die opening diameter", "opening_dia", "Die Ø")
                dio_patterns = (r'die.*(ø|o|opening).*', r'opening.*dia', r'opening_dia')
                
                ps_exacts = ("Pellet Size (mm)", "Pellet size", "pellet_size", "Pellet_Size_mm")
                ps_patterns = (r'pellet.*size',)

                oat_exacts = ("OA total (mm²)", "OA total (mm2)", "Total Open Area", "total_plate_open_area", "OA_total")
                oat_patterns = (r'oa.*total', r'total.*open.*area')         

                # Resolve for charts 1 & 2
                y_key = find_col(oa_exacts, oa_patterns) or "OA/t"
                if y_key not in df.columns:
                    df[y_key] = np.nan

                x1 = find_col(bd_exacts, bd_patterns) or "Bulk Density"
                x2 = find_col(exp_exacts, exp_patterns) or "Expansion (%)"
                holes = next((c for c in holes_exacts if c in df.columns), None)
                feed  = next((c for c in feed_exacts  if c in df.columns), None)

                charts_log(f"columns={list(df.columns)}")
                charts_log(f"resolved: y={y_key}, x1(BulkDensity?)={x1}, x2(Expansion?)={x2}, feed={feed}, holes={holes}")

                for c in [y_key, x1, x2, holes]:
                    if c and c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                fx = df.dropna(subset=[x1, y_key]) if (x1 in df.columns and y_key in df.columns) else pd.DataFrame()
                gx = df.dropna(subset=[x2, y_key]) if (x2 in df.columns and y_key in df.columns) else pd.DataFrame()
 
                # ------ chart 3: L/D vs Die Ø ------
                y3 = find_col(ld_exacts, ld_patterns) or "L/D"
                x3 = find_col(dio_exacts, dio_patterns) or "Die Ø (mm)"
                if y3 in df.columns:
                    df[y3] = pd.to_numeric(df[y3], errors="coerce")
                if x3 in df.columns:
                    df[x3] = pd.to_numeric(df[x3], errors="coerce")
                lx = df.dropna(subset=[x3, y3]) if (x3 in df.columns and y3 in df.columns) else pd.DataFrame()

                # ------ chart 4: OA total vs Pellet Size ------
                x4 = find_col(ps_exacts, ps_patterns) or "Pellet Size (mm)"
                y4 = find_col(oat_exacts, oat_patterns) or "OA total (mm²)"
                if x4 in df.columns:
                    df[x4] = pd.to_numeric(df[x4], errors="coerce")
                if y4 in df.columns:
                    df[y4] = pd.to_numeric(df[y4], errors="coerce")
                tx = df.dropna(subset=[x4, y4]) if (x4 in df.columns and y4 in df.columns) else pd.DataFrame()

                charts_log(f"counts: df={len(df)}, fx={len(fx)}, gx={len(gx)}, lx={len(lx)}, tx={len(tx)}")

                def make_scatter_mpl(data, x, y, title):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor("white")
                    fixed_size = 48.0  # uniform marker size (points^2). Try 36–64 to taste.

                    if data is None or data.empty:
                        ax.set_title(title)
                    else:
                        if feed and feed in data.columns:
                            for name, grp in data.groupby(data[feed].fillna("Unknown")):
                                ax.scatter(grp[x], grp[y], s=fixed_size, alpha=0.85, edgecolors="none", label=str(name))
                            ax.legend(title=str(feed), fontsize=8, frameon=False)
                        else:
                            ax.scatter(data[x], data[y], s=fixed_size, alpha=0.85, edgecolors="none")

                    # fixed axes like you prefer, with smart Y expansion if needed
                    if x == x1:
                        ax.set_xlim(200, 800)
                    elif x == x2:
                        ax.set_xlim(0, 40)

                    ymax = None
                    if data is not None and not data.empty:
                        try:
                            ymax = float(np.nanmax(pd.to_numeric(data[y], errors="coerce")))
                        except Exception:
                            ymax = None
                    if ymax is not None and ymax > 1500:
                        upper = int((np.ceil(ymax / 100.0) + 1) * 100)
                        ax.set_ylim(0, upper)
                    else:
                        ax.set_ylim(0, 1500)

                    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)
                    ax.grid(True, alpha=0.25)
                    fig.tight_layout()
                    return fig

                #nested helper for chart holes/feed
                def make_scatter_ld(data, x, y, title):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fixed_size = 48.0  # uniform marker size (points^2). Try 36–64 to taste.

                    if data is None or data.empty:
                        ax.set_title(title)
                    else:
                        if feed and feed in data.columns:
                            for name, grp in data.groupby(data[feed].fillna("Unknown")):
                                ax.scatter(grp[x], grp[y], s=fixed_size, alpha=0.85, edgecolors="none", label=str(name))
                            ax.legend(title=str(feed), fontsize=8, frameon=False)
                        else:
                            ax.scatter(data[x], data[y], s=fixed_size, alpha=0.85, edgecolors="none")

                    # Axes for L/D vs Die Ø
                    ax.set_xlim(0, 30)  # tweak if your openings differ
                    ymax = None
                    if data is not None and not data.empty:
                        try:
                            ymax = float(np.nanmax(pd.to_numeric(data[y], errors="coerce")))
                        except Exception:
                            ymax = None
                    if ymax is not None and ymax > 15:
                        upper = int((np.ceil(ymax / 1.0) + 1) * 1)
                        ax.set_ylim(0, upper)
                    else:
                        ax.set_ylim(0, 15)

                    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)
                    ax.grid(True, alpha=0.25)
                    fig.tight_layout()
                    return fig
                    
                #nested helper for chart pellet size and total OA
                def make_scatter_ps(data, x, y, title):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fixed_size = 48.0  # uniform marker size
                    if data is None or data.empty:
                        ax.set_title(title)
                    else:
                        if feed and feed in data.columns:
                            for name, grp in data.groupby(data[feed].fillna("Unknown")):
                                ax.scatter(grp[x], grp[y], s=fixed_size, alpha=0.85, edgecolors="none", label=str(name))
                            ax.legend(title=str(feed), fontsize=8, frameon=False)
                        else:
                            ax.scatter(data[x], data[y], s=fixed_size, alpha=0.85, edgecolors="none")

                    # Pellet size axis (tweak if you prefer another span)
                    ax.set_xlim(0, 30)

                    # Smart Y expansion for OA total (round up to a nice step)
                    ymax = None
                    if data is not None and not data.empty:
                        try:
                            ymax = float(np.nanmax(pd.to_numeric(data[y], errors="coerce")))
                        except Exception:
                            ymax = None
                    if ymax is not None and ymax > 0:
                        import numpy as _np
                        step = 500 if ymax <= 5000 else (1000 if ymax <= 20000 else 5000)
                        upper = int((_np.ceil(ymax / step) + 1) * step)
                        ax.set_ylim(0, upper)

                    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)
                    ax.grid(True, alpha=0.25)
                    fig.tight_layout()
                    return fig

                fig1 = make_scatter_mpl(fx, x1, y_key, f"{y_key} vs {x1}")
                fig2 = make_scatter_mpl(gx, x2, y_key, f"{y_key} vs {x2}")
                fig3 = make_scatter_ld(lx, x3, y3, f"{y3} vs {x3}")
                fig4 = make_scatter_ps(tx, x4, y4, f"{y4} vs {x4}")
                return fig1, fig2, fig3, fig4

            def on_reset_filters():
                base_all = clean_table_view(collect_manager_rows())
                new_min, new_max = _pellet_bounds(base_all)
                plant_opts   = [p for p in uniq(base_all["Plant"]) if str(p).strip()] if not base_all.empty else []
                species_opts = ["Floating", "Shrimp/Prawn", "Salmonid", "Other sinking"]
                df = build_table_df(new_min, new_max, [], [], select_all=False)
                df = _pad_display_rows(df, 3)
                return (
                    gr.update(value=df, headers=df.columns.tolist(), datatype=_df_datatypes(df)),
                    gr.update(minimum=new_min, maximum=new_max, value=new_min, step=0.1),
                    gr.update(minimum=new_min, maximum=new_max, value=new_max, step=0.1),
                    gr.update(choices=species_opts, value=[]),
                    gr.update(choices=plant_opts, value=[]),
                )
                
            #-----Loaded message popup helper

            def _status_loading_designer(_file_key: str):
                html = "<div class='chip info'>Die loading into designer…</div>"
                return gr.update(value=html)

            def _status_loaded_designer(_file_key: str):
                html = "<div class='chip ok'><span class='check'>✓</span>Die loaded into designer</div>"
                return gr.update(value=html) 

            # --- Excel export helpers (flatten full JSON) ----------------------------------------
            def _flatten_for_excel(data: dict, filename: str) -> dict:
                import json, os
                row = {
                    "file": os.path.basename(filename),
                    "timestamp": data.get("timestamp", ""),
                }
                for section in ("inputs", "calculated", "metadata"):
                    sec = data.get(section, {}) or {}
                    for k, v in sec.items():
                        key = f"{section}.{k}"
                        # lists/dicts -> JSON string so Excel gets the full content
                        if isinstance(v, (list, dict)):
                            row[key] = json.dumps(v, ensure_ascii=False)
                        else:
                            row[key] = v
                return row

            def export_all_designs_to_excel():
                import os, datetime as dt
                import pandas as pd
                rows = []
                for f in list_saved_designs():
                    try:
                        data = load_die_design(f)
                        rows.append(_flatten_for_excel(data, f))
                    except Exception as e:
                        log_error(f"export flatten failed for {f}: {e}")

                # Build a DataFrame even if empty
                df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["file","timestamp"])

                # Nice-ish column order: keep a few up front, then everything else sorted
                front = [c for c in [
                    "file","timestamp",
                    "metadata.plant_name","metadata.production_line","metadata.extruder_type",
                    "inputs.feed_type","inputs.pellet_size"
                ] if c in df.columns]
                rest = [c for c in df.columns if c not in front]
                if front:
                    df = df[front + sorted(rest)]

                # Write Excel next to your JSONs
                ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_path = os.path.join(SAVE_DIR, f"die_designs_export_{ts}.xlsx")
                try:
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    df.to_excel(excel_path, index=False)  # engine="openpyxl" if you want to be explicit
                    # Show the download link + a friendly status line
                    return (
                        gr.update(value=excel_path, visible=True),
                        gr.update(value=f"Exported {len(df)} rows to {os.path.basename(excel_path)}")    
                    )
                except Exception as e:
                    log_error(f"Excel export failed: {e}")
                    return gr.update(visible=False), gr.update(    
                        value=f"<div style='color:red; font-weight:bold;'>Excel export failed: {e}</div>"
                    )                

            def clear_manager_status():
                return gr.update(value="")

            def clear_export_link_and_status():
                return gr.update(value=None, visible=False), gr.update(value="")

            def _hide_file():
                return gr.update(value=None, visible=False)
                # Back-compat aliases (use any or all as needed)
            hide_pdf_link = _hide_file
            hide_dxf_link = _hide_file      
            
            def export_pdf_from_dxf(dxf_path, calc_vals=None):
                import ezdxf, os, traceback, re
                import matplotlib.pyplot as plt
                from matplotlib import patches
                import gradio as gr
                from ezdxf import colors as dxfcolors
                
                log_error("PDF path: export_pdf_from_dxf active")

                # ---------------- Page/Layout tunables ----------------
                FIGSIZE = (11.69, 8.27)   # A4 landscape, inches

                # NOTES grid (bottom-left of axes, in axes fractions)
                NOTES_FONT_SIZE = 8
                NOTES_GRID = True
                NOTES_COL_XS = (0.06, 0.22, 0.52)  # column X positions (axes fraction, inside the axes)
                NOTES_ROW_DY = 0.020               # line spacing (~2% of axes height)
                TITLE_POS = (0.05, 0.940)  # (x,y) in FIGURE fractions; raise/lower Y or move X
                # Small headings above each plate (axes fractions)
                LABEL_FONT_SIZE   = 9
                LABEL_KNIFE_POS   = (0.23, 0.88)   # left plate
                LABEL_INLET_POS   = (0.66, 0.88)   # right plate

                # NOTES position (inside the axes, 0..1). Lower = closer to page bottom.
                NOTES_BASE_Y = 0.013

                # Axes rectangle (page margins). Lower AX_BOTTOM to push axes nearer the page bottom.
                AX_LEFT, AX_RIGHT, AX_BOTTOM, AX_TOP = 0.01, 0.995, 0.045, 0.985  # bottom margin 4.5%

                # Plates: spacing, zoom, and whole-pair nudge (fractions of geometry span)
                PLATES_GAP_FRAC = 0.16        # smaller = plates closer together; larger = further apart
                PLATES_ZOOM     = 1.50        # 0.90 = bigger plates; 1.10 = smaller plates
                PLATES_HBIAS    = 0.00        # + moves both plates to the RIGHT (in data units)
                PLATES_VBIAS    = 0.62        # + moves both plates UP (in data units)

                # Extra whitespace around everything (as fraction of span)
                PAD_FRAC        = 0.08
                # Handles line thickness and rounding
                EDGE_PAD_FRAC = 0.02   # 2% extra on each side

                # ---------------- input values (for your legacy title block) ----------------
                V = dict(calc_vals or {})
                vals = V  # your title builder expects "vals"

                def _rgb01(r, g, b):
                    return (r/255.0, g/255.0, b/255.0)

                # Friendly layer colors on white
                BLUE_OPEN = _rgb01(0, 102, 204)
                BLUE_CONE = _rgb01(40, 140, 230)
                BLUE_CS   = _rgb01(90, 180, 255)
                GREY_PCD  = _rgb01(80, 80, 80)
                GREY_CHAM = _rgb01(60, 60, 60)
                GREY_SEG  = _rgb01(51, 51, 51)     # BACK_SEGMENTS (walls/fillets)
                GREY_SEGR = _rgb01(128, 128, 128)  # BACK_SEG_RINGS
                BLACK     = _rgb01(0, 0, 0)

                def _aci_to_rgb01(aci: int):
                    try:
                        aci = int(aci)
                    except Exception:
                        aci = 7
                    if aci == 7:
                        return BLACK
                    try:
                        r, g, b = dxfcolors.aci2rgb(aci if 1 <= aci <= 255 else 7)
                        return _rgb01(r, g, b)
                    except Exception:
                        return BLACK

                def _layer_color(layer_name: str, doc):
                    ly = (layer_name or "").upper()
                    if ly in {"OUTLINE", "BACK_OUTLINE", "BOLTS", "BACK_BOLTS"}: return BLACK
                    if ly in {"HOLES", "BACK_HOLES", "INSET_OPENINGS"}:         return BLUE_OPEN
                    if ly in {"CONE", "BACK_CONE", "INSET_CONE"}:               return BLUE_CONE
                    if ly in {"COUNTERSINK", "BACK_CS", "INSET_CS"}:            return BLUE_CS
                    if ly == "PCD_GUIDE":                                       return GREY_PCD
                    if ly == "CHAMFERS":                                        return GREY_CHAM
                    if ly == "BACK_SEGMENTS":                                   return GREY_SEG
                    if ly == "BACK_SEG_RINGS":                                  return GREY_SEGR
                    try:
                        aci = int(doc.layers.get(ly).dxf.color or 7)
                    except Exception:
                        aci = 7
                    return _aci_to_rgb01(aci)

                def _is_back_layer(ly: str) -> bool:
                    return (ly or "").upper().startswith("BACK_")

                # ---------------- load dxf ----------------
                try:
                    doc = ezdxf.readfile(dxf_path)
                except Exception:
                    log_error("export_pdf_from_dxf: readfile failed:\n" + traceback.format_exc())
                    return gr.update(value=None, visible=False)
                msp = doc.modelspace()

                # ---------------- collect geometry ----------------
                circles, arcs, lines = [], [], []  # we'll only draw BACK_SEGMENTS lines

                for e in msp:
                    try:
                        ly = (getattr(e.dxf, "layer", "") or "")
                        t  = e.dxftype()
                        if t == "CIRCLE":
                            c = e.dxf.center; r = float(e.dxf.radius)
                            circles.append((float(c[0]), float(c[1]), r, ly))
                        elif t == "ARC":
                            c = e.dxf.center; r = float(e.dxf.radius)
                            sa = float(getattr(e.dxf, "start_angle", 0.0))
                            ea = float(getattr(e.dxf, "end_angle",   360.0))
                            arcs.append((float(c[0]), float(c[1]), r, sa, ea, ly))
                        elif t == "LINE":
                            p0 = e.dxf.start; p1 = e.dxf.end
                            lines.append((float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]), ly))
                        elif t in ("LWPOLYLINE", "POLYLINE"):
                            pts = []
                            try:
                                pts = [(float(x), float(y)) for (x, y, *_rest) in e.get_points()]
                            except Exception:
                                try:
                                    pts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in e.vertices]
                                except Exception:
                                    pts = []
                            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                                lines.append((x1, y1, x2, y2, ly))
                            if getattr(e, "closed", False) and len(pts) >= 2:
                                x1, y1 = pts[-1]; x2, y2 = pts[0]
                                lines.append((x1, y1, x2, y2, ly))
                    except Exception:
                        continue

                lines_back_segments = [l for l in lines if (l[4] or "").upper() == "BACK_SEGMENTS"]

                # ---------------- split left/right (front/back) ----------------
                left_c  = [c for c in circles if not _is_back_layer(c[3])]
                left_a  = [a for a in arcs    if not _is_back_layer(a[5])]
                right_c = [c for c in circles if _is_back_layer(c[3])]
                right_a = [a for a in arcs    if _is_back_layer(a[5])]
                right_l = lines_back_segments

                # ---------------- compute bboxes/centers ----------------
                def _bbox_group(crcs, ars, lns=None):
                    inf = float("inf")
                    minx, miny, maxx, maxy = inf, inf, -inf, -inf
                    def _upd(x, y):
                        nonlocal minx, miny, maxx, maxy
                        if x < minx: minx = x
                        if y < miny: miny = y
                        if x > maxx: maxx = x
                        if y > maxy: maxy = y
                    for x, y, r, *_ in crcs:
                        _upd(x - r, y - r); _upd(x + r, y + r)
                    for x, y, r, sa, ea, *_ in ars:
                        _upd(x - r, y - r); _upd(x + r, y + r)
                    for x1, y1, x2, y2, *_ in (lns or []):
                        _upd(x1, y1); _upd(x2, y2)
                    if not (minx < maxx and miny < maxy):
                        minx, miny, maxx, maxy = -50, -50, 50, 50
                    cx = (minx + maxx) / 2.0
                    cy = (miny + maxy) / 2.0
                    return (minx, miny, maxx, maxy, cx, cy)

                Lminx, Lminy, Lmaxx, Lmaxy, Lcx, Lcy = _bbox_group(left_c, left_a, None)
                Rminx, Rminy, Rmaxx, Rmaxy, Rcx, Rcy = _bbox_group(right_c, right_a, right_l)

                Lw = max(1e-6, Lmaxx - Lminx)
                Rw = max(1e-6, Rmaxx - Rminx)
                
                # Side-by-side target centers + pair nudge
                span = max(max(Lmaxx - Lminx, Rmaxx - Rminx), max(Lmaxy - Lminy, Rmaxy - Rminy))
                gap  = PLATES_GAP_FRAC * max(Lw, Rw)

                target_L = (-0.5 * (Lw + gap/2.0), 0.0)
                target_R = ( 0.5 * (Rw + gap/2.0), 0.0)

                xshift_left  = (target_L[0] - Lcx) + (PLATES_HBIAS * span)
                yshift_left  = (target_L[1] - Lcy) + (PLATES_VBIAS * span)
                xshift_right = (target_R[0] - Rcx) + (PLATES_HBIAS * span)
                yshift_right = (target_R[1] - Rcy) + (PLATES_VBIAS * span)

                # ---------------- figure/axes ----------------
                fig = plt.figure(figsize=FIGSIZE)
                ax = fig.add_axes([AX_LEFT, AX_BOTTOM, AX_RIGHT - AX_LEFT, AX_TOP - AX_BOTTOM])
                ax.set_aspect("equal", adjustable="datalim")

                def _draw_group(crcs, ars, lns, dx=0.0, dy=0.0):
                    for x, y, r, ly in crcs:
                        col = _layer_color(ly, doc)
                        ax.add_patch(patches.Circle((x + dx, y + dy), r,
                                                    fill=False, linewidth=1.0, edgecolor=col))
                    for x, y, r, sa, ea, ly in ars:
                        col = _layer_color(ly, doc)
                        ax.add_patch(patches.Arc((x + dx, y + dy), 2*r, 2*r,
                                                 angle=0.0, theta1=sa, theta2=ea,
                                                 linewidth=1.0, edgecolor=col))
                    for x1, y1, x2, y2, ly in (lns or []):
                        col = _layer_color(ly, doc)
                        ax.plot([x1 + dx, x2 + dx], [y1 + dy, y2 + dy], linewidth=1.0, color=col)

                # draw geometry (side-by-side)
                _draw_group(left_c,  left_a,  None,    xshift_left,  yshift_left)
                _draw_group(right_c, right_a, right_l, xshift_right, yshift_right)
  
                # Overall bounds after shifting both groups
                minx = min(Lminx + xshift_left,  Rminx + xshift_right)
                maxx = max(Lmaxx + xshift_left,  Rmaxx + xshift_right)
                miny = min(Lminy + yshift_left,  Rminy + yshift_right)
                maxy = max(Lmaxy + yshift_left,  Rmaxy + yshift_right)

                # Span + center of the shifted geometry
                dx_all = maxx - minx
                dy_all = maxy - miny
                cx_all = 0.5 * (minx + maxx)
                cy_all = 0.5 * (miny + maxy)

                # Base half-span with padding (keep aspect square)
                base_half = 0.5 * max(dx_all, dy_all) * (1.0 + 2.0 * PAD_FRAC)

                # Apply zoom: >1 = zoom out (more margin), <1 = zoom in (less margin)
                half = base_half / max(PLATES_ZOOM, 1e-6)

                # Extra edge pad so strokes don’t kiss the frame
                half *= (1.0 + EDGE_PAD_FRAC)

                # Fix the viewport
                ax.set_autoscale_on(False)
                ax.set_xlim(cx_all - half, cx_all + half)
                ax.set_ylim(cy_all - half, cy_all + half)

                # Fill the page width and prevent auto-centering squeeze
                ax.set_position([0.00, 0.06, 1.00, 0.88])     # [left, bottom, width, height] in figure coords
                ax.set_aspect("equal", adjustable="datalim")  # critical: preserve box size, adjust data limits
                ax.set_anchor("W")                             # anchor left

                # Debug: you should see l≈0.000, w≈1.000
                try:
                    l, b, w, h = ax.get_position().bounds
                    log_error(f"PDF: AX POS (l={l:.3f}, b={b:.3f}, w={w:.3f}, h={h:.3f}), FIGSIZE={FIGSIZE}")
                except Exception:
                    pass

                    # Axes-space frame: the visible axes rectangle on the figure
                    ax.add_patch(
                        patches.Rectangle(
                            (0, 0), 1, 1,
                            transform=ax.transAxes, fill=False, linewidth=0.8, linestyle=":",
                            edgecolor="orange", zorder=5001, clip_on=False
                        )
                    )

                # --- Small headings (axes fractions; draw after limits/position) ---
                try:
                    ax.text(LABEL_KNIFE_POS[0], LABEL_KNIFE_POS[1], "Top View - knife side",
                            transform=ax.transAxes, ha="left", va="bottom",
                            fontsize=LABEL_FONT_SIZE, zorder=10000, clip_on=False)

                    ax.text(LABEL_INLET_POS[0], LABEL_INLET_POS[1], "Top View - inlet side",
                            transform=ax.transAxes, ha="left", va="bottom",
                            fontsize=LABEL_FONT_SIZE, zorder=10000, clip_on=False)
                except Exception:
                    pass

                # -------- Title / NOTES / cross-section come AFTER this ------

                try:
                    notes_mtexts = [e for e in msp.query("MTEXT")
                                    if (getattr(e.dxf, "layer", "") or "").upper() == "NOTES"]

                    def _mtxt_str(mt):
                        s = getattr(mt, "plain_text", None)
                        if callable(s):
                            try: return s()
                            except Exception: pass
                        if isinstance(s, str) and s:
                            return s
                        return getattr(mt, "text", "")

                    notes_txt = "\n".join(_mtxt_str(mt) for mt in notes_mtexts)

                    def _find_float(pat, s):
                        m = re.search(pat, s, re.I)
                        if not m: return None
                        val = m.group(1).replace(",", ".")
                        try: return float(val)
                        except: return None

                    def _find_int(pat, s):
                        m = re.search(pat, s, re.I)
                        if not m: return None
                        try: return int(m.group(1))
                        except: return None

                    pellet_mm_notes   = _find_float(r"Target\s*pellet\s*size:\s*([0-9]+(?:[.,][0-9]+)?)\s*mm", notes_txt)
                    holes_total_notes = _find_int(r"Total\s*holes:\s*([0-9]+)", notes_txt)

                    pellet_mm  = pellet_mm_notes if pellet_mm_notes is not None else float(
                        (V.get("pellet") or V.get("pellet_size") or V.get("pellet_mm") or V.get("pellet_size_mm") or 0) or 0
                    )
                    holes_total = holes_total_notes if holes_total_notes is not None else int(
                        (V.get("holes") or V.get("number_of_holes") or V.get("total_holes") or 0) or 0
                    )

                    opening_mm = float((V.get("opening_dia") or V.get("opening_diameter") or 0) or 0)
                    land_mm    = float((V.get("channel_length") or 0) or 0)

                    title = f"Die Design {pellet_mm:g}mm pellet ({opening_mm:g} x {holes_total} - {land_mm:g})"
                    fig.text(TITLE_POS[0], TITLE_POS[1], title,
                             ha="left", va="top",
                             fontsize=12, fontweight="bold", color="#111111")

                except Exception:
                    log_error("TITLE build failed:\n" + traceback.format_exc())

                # ---------------- NOTES bottom-left (axes coords) ----------------
                try:
                    import math, re
                    from collections import Counter

                    # 1) Collect raw note lines from DXF (MTEXT + TEXT on NOTES layer)
                    raw_lines = []
                    for e in msp.query("MTEXT"):
                        if (getattr(e.dxf, "layer", "") or "").upper() != "NOTES":
                            continue
                        try:
                            txt = e.plain_text() if hasattr(e, "plain_text") and callable(e.plain_text) else (e.text or "")
                        except Exception:
                            txt = getattr(e, "text", "") or ""
                        raw_lines.extend([ln for ln in txt.replace(r"\P", "\n").splitlines() if ln.strip()])
                    for t in msp.query("TEXT"):
                        if (getattr(t.dxf, "layer", "") or "").upper() != "NOTES":
                            continue
                        s = getattr(t.dxf, "text", "") or ""
                        if s.strip():
                            raw_lines.append(s)

                    def _clean(s: str) -> str:
                        s = s.strip()
                        s = re.sub(r"\s{2,}", " ", s)
                        return s.strip(" :;,-")

                    lines = [_clean(s) for s in raw_lines if _clean(s)]
                    low   = [s.lower() for s in lines]

                    # 2) Compute bolt-ring PCDs from FRONT (left) bolt holes
                    def _bolt_rings_front():
                        bolts = [(x, y, r) for (x, y, r, ly) in left_c
                                 if (ly or "").upper() == "BOLTS" and r > 0]
                        if not bolts:
                            return []
                        cx = sum(x for x, _, _ in bolts) / len(bolts)
                        cy = sum(y for _, y, _ in bolts) / len(bolts)
                        # (ring radius, hole_diameter)
                        radii = [(math.hypot(x - cx, y - cy), 2.0 * r) for (x, y, r) in bolts]
                        # bin by radius to find rings
                        bins = {}
                        for ring_r, hole_d in radii:
                            key = round(ring_r, 1)  # 0.1 mm bin
                            bins.setdefault(key, []).append(round(hole_d, 2))
                        rings = []
                        for key in sorted(bins.keys()):
                            ds = bins[key]
                            hole_d = Counter(ds).most_common(1)[0][0] if ds else 0.0
                            pcd = 2.0 * float(key)
                            n = len(ds)
                            rings.append((pcd, hole_d, n))
                        return rings[:2]  # up to inner+outer

                    # 3) Column 1: key specs (ordered) + computed plate diameter
                    def _pick_once(key_substr: str, label_out: str):
                        for i, s in enumerate(low):
                            if key_substr in s:
                                raw = lines[i]
                                rhs = raw.split(":", 1)[1].strip() if ":" in raw else raw
                                return f"{label_out}: {rhs}"
                        return None

                    col1 = []
                    for key, label in [
                        ("target pellet size", "Target pellet size"),
                        ("die opening diameter", "Die opening diameter"),
                        ("designed throughput", "Ave throughput"),
                        ("cone width", "Cone width"),
                        ("total holes", "Total holes"),
                        ("total hole length", "Total hole length"),
                        ("oa total", "Total open area"),
                        ("oa per tonne", "OA per tonne"),
                    ]:
                        v = _pick_once(key, label)
                        if v: col1.append(v)

                    # estimate plate diameter from left group bbox (works even if no OUTLINE)
                    try:
                        d_plate = max(Lmaxx - Lminx, Lmaxy - Lminy)
                        col1.append(f"Plate diameter: {d_plate:.2f} mm")
                    except Exception:
                        pass

                    # 4) Columns 2 & 3: all "Row ... PCD ..." lines, robust parsing
                    # Accepts things like: "Row 3 - PCD 125 mm - 24 holes" or "Row 3: PCD=125mm, 24 holes"
                    row_re = re.compile(
                        r"\brow\s*0*(\d+)\b.*?\bpcd\b\s*([0-9]+(?:\.[0-9]+)?)\s*mm"
                        r"(?:.*?\b([0-9]+)\s*holes?)?",
                        re.IGNORECASE
                    )

                    rows_found = []
                    for s in lines:
                        m = row_re.search(s)
                        if not m:
                            continue
                        n = int(m.group(1))
                        pcd_mm = float(m.group(2))
                        n_holes = (int(m.group(3)) if m.group(3) else None)
                        tail_parts = [f"PCD {pcd_mm:.1f} mm"]
                        if n_holes:
                            tail_parts.append(f"{n_holes} holes")
                        rows_found.append((n, f"Row {n}: " + " - ".join(tail_parts)))

                    rows_found.sort(key=lambda t: t[0])
                    col2 = [txt for n, txt in rows_found if 1 <= n <= 8]
                    col3 = [txt for n, txt in rows_found if n >= 9]

                    # 5) Prepend bolt ring info to Column 2 (if present)
                    try:
                        rings = _bolt_rings_front()
                        bolt_lines = []
                        if len(rings) >= 1:
                            pcd, d, n = rings[0]
                            bolt_lines.append(f"Inner bolts PCD = {pcd:.1f} mm, bolt opening {d:.1f} mm (x{n})")
                        if len(rings) >= 2:
                            pcd, d, n = rings[1]
                            bolt_lines.append(f"Outer bolts PCD = {pcd:.1f} mm, bolt opening {d:.1f} mm (x{n})")
                        if bolt_lines:
                            col2 = bolt_lines + col2
                    except Exception:
                        pass

                    # 6) Draw the three columns at bottom-left (axes coords)
                    x1, x2, x3 = NOTES_COL_XS
                    y0 = NOTES_BASE_Y
                    dy = NOTES_ROW_DY
                    fs = NOTES_FONT_SIZE

                    for i, s in enumerate(col1):
                        ax.text(x1, y0 + i * dy, s, transform=ax.transAxes,
                                ha="left", va="bottom", fontsize=fs, family="DejaVu Sans",
                                color="black", zorder=9999, clip_on=False)

                    for i, s in enumerate(col2):
                        ax.text(x2, y0 + i * dy, s, transform=ax.transAxes,
                                ha="left", va="bottom", fontsize=fs, family="DejaVu Sans",
                                color="black", zorder=9999, clip_on=False)

                    for i, s in enumerate(col3):
                        ax.text(x3, y0 + i * dy, s, transform=ax.transAxes,
                                ha="left", va="bottom", fontsize=fs, family="DejaVu Sans",
                                color="black", zorder=9999, clip_on=False)

                except Exception:
                    log_error("NOTES 3-column render failed:\n" + traceback.format_exc())

                    # ---------------- Cross-section inset (unchanged) ----------------
                try:
                    if '_render_cs_image' in globals() and '_place_cs_image' in globals():
                        cs_img = _render_cs_image(V)
                        _place_cs_image(
                            fig, cs_img,
                            base_left=0.68, base_bottom=0.06, box_w=0.30, box_h=0.32,
                            nudge_right_mm=20.0, nudge_down_mm=14.0, margin=0.01
                        )
                except Exception:
                    pass

                ax.axis("off")
                pdf_path = f"/tmp/{os.path.splitext(os.path.basename(dxf_path))[0]}.pdf"
                fig.savefig(pdf_path, format="pdf")
                plt.close(fig)
                return gr.update(value=pdf_path, visible=True)

            # --- Helpers used by export_pdf_from_dxf -------------------------------------
            def _ensure_ttf_text_style(doc):
                """Force the 'Standard' style to a TTF so Matplotlib can draw glyphs."""
                try:
                    st = doc.styles.get("Standard")
                except KeyError:
                    st = doc.styles.add("Standard")
                try:
                    st.dxf.font = "DejaVuSans.ttf"  # Matplotlib includes DejaVu Sans
                    st.dxf.bigfont = ""             # avoid SHX bigfont
                except Exception:
                    pass

            def _dump_notes_debug(doc, tag=""):
                """Log what NOTES text exists and where."""
                try:
                    msp = doc.modelspace()
                    mts = [e for e in msp.query("MTEXT") if (getattr(e.dxf, "layer", "") or "").upper() == "NOTES"]
                    txs = [e for e in msp.query("TEXT")  if (getattr(e.dxf, "layer", "") or "").upper() == "NOTES"]
                    log_error(f"Overlay{tag}: NOTES MTEXT={len(mts)} TEXT={len(txs)}")
                    for e in mts[:5]:
                        ins = getattr(e.dxf, "insert", (0,0))
                        ch  = getattr(e.dxf, "char_height", None)
                        try:
                            sample = (e.plain_text() if hasattr(e, "plain_text") else (e.text or "")).replace(r"\\P","\n")
                        except Exception:
                            sample = getattr(e, "text", "") or ""
                        sample = sample[:120]
                        log_error(f"  MTEXT at {ins}, ch={ch}, sample={sample!r}")
                    for t in txs[:5]:
                        ins = getattr(t.dxf, "insert", (0,0))
                        h   = getattr(t.dxf, "height", None)
                        s   = getattr(t.dxf, "text", "")[:120]
                        log_error(f"  TEXT  at {ins}, h={h}, sample={s!r}")
                except Exception:
                    import traceback
                    log_error("Overlay debug failed:\n" + traceback.format_exc())

            # Hover/arrow selection only updates the index; no chips, no filename, no UI reflow.
            def on_html_row_hovered(idx, df):
                import pandas as pd
                if isinstance(df, list) and df:
                    try:
                        df = pd.DataFrame(df[1:], columns=df[0])
                    except Exception:
                        df = None
                try:
                    i = int(idx) if idx is not None else None
                except Exception:
                    i = None
                if df is None or getattr(df, "empty", True) or i is None or i < 0 or i >= len(df):
                    return None
                return i

            # Hidden-bridge change → ONLY update the index state (no chip / no filename)
            sel_idx_from_html.change(
                fn=on_html_row_hovered,
                inputs=[sel_idx_from_html, saved_table],
                outputs=[selected_index_state],
                show_progress=False,
                queue=False,
            )

            #------------Table force left on updates ----------------------------------------        
            from textwrap import dedent

            js_left_and_top = dedent("""\
            () => {
              const host = document.getElementById('saved_table_html');
              if (!host) return;
              const scroller = host.querySelector('.saved-grid-html');
              if (scroller) scroller.scrollLeft = 0;     // <- snap fully left

              const grid = host.querySelector('.saved-grid-html table');
              if (!grid || !grid.tBodies || !grid.tBodies[0]) return;
              const tbody = grid.tBodies[0];
              const rows = Array.from(tbody.querySelectorAll('tr'));
              if (!rows.length) return;

              // pick first non-pad row, or fallback to first row
              const target = rows.find(r => !r.classList.contains('pad')) || rows[0];
              const cur = tbody.querySelector('tr.selected');
              if (cur !== target) { if (cur) cur.classList.remove('selected'); target.classList.add('selected'); }
              requestAnimationFrame(() => target.scrollIntoView({ block: 'nearest', behavior: 'smooth' }));
            }
            """).strip()
            #----------------Selector Button Controls ---------------------------------------

            nav_outputs = [selected_row_idx_state, selected_filename_state]

            sel_prev_btn.click(
                fn=lambda df, key: _select_by_offset(df, key, -1),
                inputs=[saved_table, selected_filename_state],
                outputs=nav_outputs,
                show_progress=False,
                queue=False,
                concurrency_limit=1,
                js=js_move_prev,
            )
            sel_next_btn.click(
                fn=lambda df, key: _select_by_offset(df, key, +1),
                inputs=[saved_table, selected_filename_state],
                outputs=nav_outputs,
                show_progress=False,
                queue=False,
                concurrency_limit=1,
                js=js_move_next,
            )
            sel_top_btn.click(
                fn=_select_first,
                inputs=[saved_table, selected_filename_state],
                outputs=nav_outputs,
                show_progress=False,
                queue=False,
                concurrency_limit=1,
                js=js_move_top,
            )
            sel_bottom_btn.click(
                fn=_select_last,
                inputs=[saved_table, selected_filename_state],
                outputs=nav_outputs,
                show_progress=False,
                queue=False,
                concurrency_limit=1,
                js=js_move_bottom,
            )

            # --- PDF: simple “Generate → show DownloadButton” chain ---

            open_pdf_btn.click(
                fn=_pdf_busy,
                inputs=[],
                outputs=[pdf_status, open_pdf_btn],
                show_progress=False,
            ).then(
                fn=export_pdf_from_dxf,
                inputs=[dxf_path_state, calc_values_state],
                outputs=[quick_pdf],                     # gr.File value becomes the file dict/path
                show_progress=True,
             ).then(
                fn=_pdf_done_from_component,
                inputs=[quick_pdf],
                outputs=[pdf_status, open_pdf_btn],
                show_progress=False,
            ).then(
                fn=_pdf_clear_status,
                inputs=[],
                outputs=[pdf_status],
                show_progress=False,
            )

            # Save -> refresh table/filters -> mirror HTML -> update charts -> (optional) clear exports & hide PDF

            def _wire_table_select(table_comp):
                table_comp.select(
                    fn=_on_saved_table_click_or_select,
                    inputs=[table_comp, pellet_min, pellet_max, species_filter, plant_filter, select_all],
                    outputs=[
                        saved_table,                 # table may change on delete
                        manager_status,              # status
                        selected_row_idx_state,      # index for load-back
                        selected_filename_state,     # key for load-back/delete
                        selected_preview_html,       # "SELECTED: ..." chip
                        saved_table_html,            # HTML mirror updates/highlight
                    ],
                    show_progress=False,
                    queue=False,
                    concurrency_limit=1,
                ).then(
                    fn=update_charts_multi,
                    inputs=[pellet_min, pellet_max, species_filter, plant_filter],
                    outputs=[chart1, chart2, chart3, chart4],
                    show_progress=False,
                    queue=False,
                )

            # Bind it to the DataFrame you actually click:
            _wire_table_select(saved_table)

            save_button.click(
                fn=on_save_with_meta,
                inputs=[calc_ok_state, calc_values_state,
                        pellet_size, throughput, density, fat, floating, shrimp, salmonid, other_sinking,
                        channel_length, cone_length, opening_dia, cone_width, countersink_diameter, countersink_depth,
                        holes, plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                        outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                        outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                        inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter] +
                       [comp for pair in row_components for comp in pair] +
                       [extruder_type, production_line, plant_name, product_type, die_performance, comments],
                outputs=[save_status],
                show_progress=False,
                queue=False,
            ).then(
                fn=refresh_table_and_filters,
                inputs=[pellet_min, pellet_max, species_filter, plant_filter, select_all],
                outputs=[saved_table, pellet_min, pellet_max, species_filter, plant_filter],
                show_progress=False,
                queue=False,
            ).then(
                fn=_ensure_del_column,
                inputs=[saved_table],
                outputs=[saved_table],
                show_progress=False,
                queue=False,
            ).then(
                fn=_mirror_saved_table_with_key,
                inputs=[saved_table, selected_filename_state],
                outputs=[saved_table_html],
                show_progress=False,
                queue=False,
            ).then(
                fn=_select_first,
                inputs=[saved_table, selected_filename_state],
                outputs=[selected_row_idx_state, selected_filename_state],
                show_progress=False,
                queue=False,
            ).then(
                fn=update_charts_multi,
                inputs=[pellet_min, pellet_max, species_filter, plant_filter],
                outputs=[chart1, chart2, chart3, chart4],
                show_progress=False,
                queue=False,
            )

            # Save to Excel
            save_excel_btn.click(
                fn=export_all_designs_to_excel,
                inputs=[],
                outputs=[exported_excel, manager_status],
                show_progress=False,
                queue=False,
            )

            # --- Refresh and Reset: table+filters first, then rows/pcds, then charts if you want ---
            # Outputs for your existing on_reset (table/filters/exports) — keep your original list here
            reset_outputs_main = reset_outputs
            # Outputs for our rows+counts reset (visible_rows + all row comps + holes + line_rows)
            rows_reset_outputs = [visible_rows] + [c for pair in row_components for c in pair] + [holes, line_rows]

            # One click → first on_reset, then on_reset_all
            reset_btn.click(
                fn=on_reset,
                inputs=[],
                outputs=reset_outputs_main,
                show_progress=False,
                queue=False,
                concurrency_limit=1,
            ).then(
                fn=on_reset_all,
                inputs=[],
                outputs=rows_reset_outputs,
                show_progress=False,
                queue=False,
                concurrency_limit=1,
            )

            # Add Rows / Autofill
            # Flatten Tab 3 row components into [nh1, pcd1, nh2, pcd2, ...] for outputs
            _t3_row_outputs = []
            for nh, p in circseg_row_components:
                _t3_row_outputs += [nh, p]

            circseg_autofill_btn.click(
                fn=on_autofill_rows_and_pcds_t3,
                inputs=[holes, circseg_line_rows, opening_dia, plate_diameter, calc_values_state],  # +cv in
                outputs=[circseg_line_rows] + _t3_row_outputs + [calc_values_state],               # +cv out (last)
                queue=False,
                show_progress=False,
            )

            add_rows_outputs = [visible_rows] + [comp for pair in row_components for comp in pair]
            add_rows_btn.click(
                fn=on_add_rows,
                inputs=[visible_rows],
                outputs=add_rows_outputs,
                show_progress=False,
                queue=False,
            )
            autofill_btn.click(
                fn=on_autofill_rows_and_pcds,
                inputs=[holes, line_rows, opening_dia, plate_diameter, visible_rows],
                outputs=add_rows_outputs,
                show_progress=False,
                queue=False,   # keeps it snappy, avoids "busy" banners
            ).then(
                fn=_clear_dxf_preview,
                inputs=[],
                outputs=[plate_preview, inset_preview],
                show_progress=False,
                queue=False,
            )

            # Reset filters -> rebuild table/filters -> update charts (unchanged)
            
            reset_filters_btn.click(
                fn=on_reset_filters,
                inputs=[],
                outputs=[saved_table, pellet_min, pellet_max, species_filter, plant_filter],
                show_progress=False,
                queue=False,
            ).then(
                fn=_ensure_del_column,                     # 1) ensure DEL first
                inputs=[saved_table],
                outputs=[saved_table],
                show_progress=False,
                queue=False,
            ).then(
                fn=_mirror_saved_table_with_key,           # 2) rebuild HTML mirror
                inputs=[saved_table, selected_filename_state],
                outputs=[saved_table_html],
                show_progress=False,
                queue=False,
                js=js_left_and_top,
            ).then(
                fn=_select_first,
                inputs=[saved_table, selected_filename_state],
                outputs=[selected_row_idx_state, selected_filename_state],
                show_progress=False,
                queue=False,
            ).then(
                fn=update_charts_multi,
                inputs=[pellet_min, pellet_max, species_filter, plant_filter],
                outputs=[chart1, chart2, chart3, chart4],
                show_progress=False,
                queue=False,
            ).then(
                fn=clear_export_link_and_status,
                inputs=[],
                outputs=[exported_excel, manager_status],
                show_progress=False,
                queue=False,
            ).then(
                fn=_hide_pdf_link,
                inputs=[],
                outputs=[quick_pdf],
                show_progress=False,
                queue=False,
            ).then(
                fn=_clear_selection,
                inputs=[],
                outputs=[selected_index_state, selected_filename_state, selected_header_html, selected_preview_html],
                show_progress=False,
                queue=False,
            ).then(
                js=js_highlight_row, inputs=[selected_index_state], outputs=[]
            )

            # ---------- Filter -> Table -> HTML -> Charts -> Cleanup ----------

            def _none_if_empty(v):
                return None if v in (None, "", []) else v

            def on_filters_change(p_min, p_max, species, plant):
                try:
                    species = _none_if_empty(species)
                    plant   = _none_if_empty(plant)
                    df = build_table_df(p_min, p_max, species, plant,
                                        select_all=(species is None and plant is None))
                    df = add_delete_col(df)
                    df = _pad_display_rows(df, 4)
                    # quick debug line:
                    log_error(f"filters → rows={0 if df is None else len(df)} species={species} plant={plant}")
                    return gr.update(value=df, headers=df.columns.tolist(), datatype=_df_datatypes(df))
                except Exception as e:
                    log_error(f"on_filters_change failed: {e}")
                    raise

            def _wire_filter_change(ctrl):
                ctrl.change(
                    fn=refresh_table_and_mirror,   # rebuild BOTH table + HTML in one go
                    inputs=[pellet_min, pellet_max, species_filter, plant_filter, select_all],
                    outputs=[saved_table, saved_table_html],
                    show_progress=False,
                    queue=False,
                    concurrency_limit=1,
                ).then(
                    fn=_select_first,             # fix server-side selection
                    inputs=[saved_table, selected_filename_state],
                    outputs=[selected_row_idx_state, selected_filename_state],
                    show_progress=False,
                    queue=False,
                    js=js_move_top,              # move the client-side highlight w/o re-render
                ).then(
                    fn=update_charts_multi,
                    inputs=[pellet_min, pellet_max, species_filter, plant_filter],
                    outputs=[chart1, chart2, chart3, chart4],
                    show_progress=False,
                    queue=False,
                ).then(
                    fn=clear_export_link_and_status,
                    inputs=[],
                    outputs=[exported_excel, manager_status],
                    show_progress=False,
                    queue=False,
                ).then(
                    fn=_hide_pdf_link,
                    inputs=[],
                    outputs=[quick_pdf],
                    show_progress=False,
                    queue=False,
                )

            # --- Load a die design back the the designer -------------------
            load_btn.click(
                fn=_status_loading_designer,                 # blue "Die loading into designer…"
                inputs=[selected_filename_state],
                outputs=[manager_status],
                show_progress=False,
                queue=False,
            ).then(
                fn=load_focused_to_designer,                # must return values in THIS exact order/count
                inputs=[selected_filename_state],
                outputs=[
                    pellet_size, throughput, density, fat, floating, shrimp, salmonid, other_sinking,
                    opening_dia, channel_length, cone_length, cone_width, holes,
                    plate_diameter, chamfer_plate, die_center_hole_diameter, chamfer_center_hole,
                    outer_opening_pcd, chamfer_outer_opening, inner_opening_pcd, chamfer_inner_opening,
                    outer_bolt_count, outer_bolt_pcd, outer_bolt_diameter,
                    inner_bolt_count, inner_bolt_pcd, inner_bolt_diameter,
                ] + [comp for pair in row_components for comp in pair] + [
                    quick_dxf, save_status, plate_preview, inset_preview,
                    dxf_path_state, calc_ok_state, visible_rows, manager_status,
                    # NEW: metadata fields appended at the end (6)
                    extruder_type, production_line, plant_name, product_type, die_performance, comments,
                ],
                show_progress=False,
                queue=False,
            ).then(
                fn=_status_loaded_designer,                 # green ✓ "Die loaded into designer"
                inputs=[selected_filename_state],
                outputs=[manager_status],
                show_progress=False,
                queue=False,
            ).then(
                fn=_clear_selection,                        # clear selected row & header after load
                inputs=[],
                outputs=[selected_index_state, selected_filename_state, selected_header_html, selected_preview_html],
                show_progress=False,
                queue=False,
            ).then(
                js=js_highlight_row, inputs=[selected_index_state], outputs=[]
            )

            # Attach exactly once per control:
            for _ctrl in (pellet_min, pellet_max, species_filter, plant_filter):
                _wire_filter_change(_ctrl)

            # --- Delete: HTML-selected row -> delete -> rebind clicks -> rehighlight -> charts ---
            delete_btn.click(
                fn=delete_selected_now,
                inputs=[selected_filename_state, selected_index_state, saved_table, pellet_min, pellet_max, species_filter, plant_filter],
                outputs=[saved_table, manager_status, selected_index_state, selected_filename_state, saved_table_html, selected_preview_html, selected_header_html],
                show_progress=False,
                queue=False,
            ).then(
                js=js_bind_row_clicks, inputs=[], outputs=[]   # 4) rebind row clicks on fresh HTML
            ).then(
                js=js_highlight_row, inputs=[selected_index_state], outputs=[]  # 5) clears highlight (None)
            ).then(
                fn=update_charts_multi,                        # 6) refresh charts (filters unchanged)
                inputs=[pellet_min, pellet_max, species_filter, plant_filter],
                outputs=[chart1, chart2, chart3, chart4],
                show_progress=False,
                queue=False,
            )

            # --- Initial load: build filters & table -> mirror -> select first -> charts -> tidy ---
            demo.load(
                fn=refresh_table_and_filters,
                inputs=[pellet_min, pellet_max, species_filter, plant_filter, select_all],
                outputs=[saved_table, pellet_min, pellet_max, species_filter, plant_filter],
                show_progress=False,
            ).then(
                fn=_ensure_del_column,                 # keep invariants
                inputs=[saved_table],
                outputs=[saved_table],
                show_progress=False,
                queue=False,
            ).then(
                fn=_mirror_saved_table_with_key,       # rebuild the HTML mirror from NEW df
                inputs=[saved_table, selected_filename_state],
                outputs=[saved_table_html],
                show_progress=False,
                queue=False,
            ).then(
                fn=_select_first,                      # update server selection states (no re-render)
                inputs=[saved_table, selected_filename_state],
                outputs=[selected_row_idx_state, selected_filename_state],
                show_progress=False,
                queue=False,
            ).then(
                fn=update_charts_multi,                # charts reflect current filters/table
                inputs=[pellet_min, pellet_max, species_filter, plant_filter],
                outputs=[chart1, chart2, chart3, chart4],
                show_progress=False,
                queue=False,
            ).then(
                fn=clear_export_link_and_status,       # nice-to-have: clear stale status
                inputs=[],
                outputs=[exported_excel, manager_status],
                show_progress=False,
                queue=False,
            ).then(
                fn=_hide_pdf_link,                     # hide quick PDF until user regenerates
                inputs=[],
                outputs=[quick_pdf],
                show_progress=False,
                queue=False,
            )

        return demo

# -------- App Startup ------------------------------------------
def _start():
    t0 = time.monotonic()
    port = int(os.getenv("DIE_TOOL_PORT", "7866"))
    share = os.getenv("GRADIO_SHARE", "").strip().lower() in ("1", "true", "yes", "y")

    # Use favicon only if present; None is fine and Gradio will ignore it
    fav_file = APP_BASE / "favicon.ico"
    fav_path = str(fav_file) if fav_file.exists() else None

    try:
        logger.info("Startup: Calling build_ui()")
        ui = build_ui()
        logger.info("Startup: build_ui() returned: %s", type(ui).__name__)

        app = ui.queue(max_size=8)

        logger.info(
            "Startup: launching Gradio (port=%s, share=%s, favicon=%s)…",
            port, share, (fav_path or "none")
        )

        app.launch(
            server_name="0.0.0.0",
            server_port=port,
            inbrowser=False,
            show_api=False,
            favicon_path=fav_path,  # None is ok
            share=share,
            # root_path="/your-subpath",  # only if you serve under a subpath
        )

        logger.info("Shutdown: app.launch() returned (server stopped). ran=%.2fs", time.monotonic() - t0)
    except Exception:
        logger.exception("Startup: failed")
        raise

if __name__ == "__main__":
    _start()
