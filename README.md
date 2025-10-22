# Die Design Tool

A web-based Python application for extrusion die plate design and visualization.  
Supports three design types — **Circular**, **Segmented**, and **Hybrid Circular/Segmented** — with real-time calculations, auto-fill features, and DXF export.

---

## Features

### Design Tabs
| Tab | Type | Description |
|-----|------|--------------|
| **1** | Circular | Classic circular rows with automatic staggering between rows. |
| **2** | Segmented | Wall-based staggered patterns inside defined segment boundaries. |
| **3** | Hybrid | Circular rows culled by segmented walls for complex hybrid dies. |

### Automation
- **Auto-Fill Rows:** Automatically distribute hole counts and PCDs.
- **Mirrored Table UI:** Tab 3 uses a synchronized, mirrored control layout for faster edits.
- **Real-Time Logging:** Tracks every build, export, and geometry computation.

### Visual Outputs
- **Cross-Section View:** Displays channel, cone, and countersink dimensions.
- **Top View:** Complete die layout including bolt circles and cutting tracks.
- **Inset Zoom View:** Auto-centering zoom of the die’s active region.

### File Outputs
- **DXF Export:** Ready for AutoCAD/CAM import.
- **PNG Preview:** Auto-generated top-view preview for documentation.
- **JSON Save:** Persistent save format for Die Manager integration.

### Advanced Features
- **Historical Suggestion Engine:** Intelligent starting values from Excel database.
- **State Synchronization:** All design tabs share a unified calculation context (`cv`).
- **Error Recovery:** Graceful fallbacks and hybrid retries in geometry generation.
- **Logging:** Structured debug-level logs for all design events.

---

## Technologies

- **Python 3**
- **Gradio** – UI Framework
- **Pandas / OpenPyXL** – Data & Excel integration
- **Matplotlib / Pillow** – Image generation
- **ezdxf** – DXF creation
- **Bash / Synology** – Deployment & monitoring

---

## Deployment

1. **Start App**
   ```bash
   ./deploy_merged_design_tool.sh
