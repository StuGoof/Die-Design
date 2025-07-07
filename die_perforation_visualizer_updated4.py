import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(layout="wide")
st.title("Die Perforation Visualizer")

# Target Inputs
st.sidebar.header("Target Parameters")
pellet_size = st.sidebar.slider("Pellet Size (mm)", 0.3, 20.0, 3.0, step=0.1)
bulk_density = st.sidebar.slider("Bulk Density (g/l)", 300, 750, 500)
final_fat = st.sidebar.slider("Final Fat (%)", 7, 44, 10)

# Die Geometry Inputs
st.sidebar.header("Die Geometry")
plate_thickness = st.sidebar.slider("Total Plate Thickness (mm)", 1.0, 40.0, 20.0, step=0.5)
final_diameter = st.sidebar.slider("Final Hole Diameter (mm)", 0.3, 20.0, 3.0, step=0.05)
cone_diameter = st.sidebar.slider("Cone Opening Diameter (mm)", 0.3, 25.0, 6.0, step=0.05)
channel_length = st.sidebar.slider("Channel (Land) Length (mm)", 0.1, min(35.0, plate_thickness), 5.0, step=0.1)

# Hole Layout Inputs
st.sidebar.header("Hole Layout")
total_holes = st.sidebar.number_input("Total Number of Holes Required", min_value=1, value=100)
dry_meal_throughput = st.sidebar.number_input("Dry Meal Throughput (tonne/h)", min_value=0.1, value=10.0)
number_of_rows = st.sidebar.number_input("Number of Rows", min_value=1, value=5)

# PCD Inputs
st.sidebar.header("Pitch Circle Diameters (PCD)")
pcd_values = []
for i in range(number_of_rows):
    pcd = st.sidebar.slider(f"PCD for Row {i+1} (mm)", 50.0, 600.0, 100.0, step=1.0)
    pcd_values.append(pcd)

# Calculated Outputs
cone_length = plate_thickness - channel_length
cone_radius = abs((cone_diameter - final_diameter) / 2)
cone_angle_rad = np.arctan(cone_radius / cone_length) if cone_length > 0 else 0
cone_angle_deg = np.degrees(cone_angle_rad)
holes_per_row = int(total_holes / number_of_rows)
open_area_one_hole = np.pi * (final_diameter / 2) ** 2
total_open_area = open_area_one_hole * total_holes
open_area_per_tonne = total_open_area / dry_meal_throughput
expansion = (1 - (final_diameter / pellet_size)) * 100 if pellet_size > 0 else 0

# Calculate space between holes (edge-to-edge) for each row
space_between_holes_list = []
for pcd in pcd_values:
    radius = pcd / 2
    arc_length = 2 * np.pi * radius / holes_per_row
    spacing = arc_length - final_diameter
    space_between_holes_list.append(spacing)

# Calculate space between rows from PCDs
space_between_rows = np.mean(np.diff(sorted(pcd_values))) if len(pcd_values) > 1 else 0

# Display Outputs
st.subheader("Calculated Outputs")
col1, col2 = st.columns(2)
with col1:
    st.write(f"Cone Length: {cone_length:.2f} mm")
    st.write(f"Cone Angle: {cone_angle_deg:.2f}°")
    st.write(f"Open Area of One Hole: {open_area_one_hole:.2f} mm²")
    st.write(f"Total Plate Open Area: {total_open_area:.2f} mm²")
    st.write(f"Open Area per Tonne: {open_area_per_tonne:.2f} mm²/t/h")
with col2:
    st.write(f"Number of Holes per Row: {holes_per_row}")
    for i, spacing in enumerate(space_between_holes_list):
        st.write(f"Row {i+1} - Space Between Holes: {spacing:.2f} mm")
    st.write(f"Space Between Rows (calculated): {space_between_rows:.2f} mm")
    st.write(f"Expansion: {expansion:.2f} %")

# 2D Cross-Section Visualization
fig2d, ax2d = plt.subplots(figsize=(4, 6))
ax2d.set_xlim(-cone_diameter, cone_diameter)
ax2d.set_ylim(0, plate_thickness + 5)
ax2d.set_aspect('equal')

# Cone section
cone_patch = patches.Polygon([
    [-cone_diameter/2, 0],
    [-final_diameter/2, cone_length],
    [-final_diameter/2, plate_thickness],
    [final_diameter/2, plate_thickness],
    [final_diameter/2, cone_length],
    [cone_diameter/2, 0]
], closed=True, facecolor='lightblue', edgecolor='black', label='Cone')

# Channel section
channel_patch = patches.Rectangle((-final_diameter/2, cone_length), final_diameter, channel_length,
                                  facecolor='lightgreen', edgecolor='black', label='Channel')

ax2d.add_patch(cone_patch)
ax2d.add_patch(channel_patch)
ax2d.set_xlabel("Width (mm)")
ax2d.set_ylabel("Depth (mm)")
ax2d.legend()
st.pyplot(fig2d)

# 3D Visualization
fig3d = plt.figure(figsize=(6, 6))
ax3d = fig3d.add_subplot(111, projection='3d')
z_cone = np.linspace(0, cone_length, 30)
z_channel = np.linspace(cone_length, plate_thickness, 10)
theta = np.linspace(0, 2 * np.pi, 30)
theta_grid, z_cone_grid = np.meshgrid(theta, z_cone)
theta_grid2, z_channel_grid = np.meshgrid(theta, z_channel)

r_cone = (cone_diameter/2 - (cone_diameter - final_diameter)/2 * (z_cone / cone_length)) if cone_length > 0 else np.full_like(z_cone, final_diameter/2)
x_cone = r_cone[:, np.newaxis] * np.cos(theta_grid)
y_cone = r_cone[:, np.newaxis] * np.sin(theta_grid)
z_cone_plot = z_cone_grid

r_channel = np.full_like(z_channel, final_diameter/2)
x_channel = r_channel[:, np.newaxis] * np.cos(theta_grid2)
y_channel = r_channel[:, np.newaxis] * np.sin(theta_grid2)
z_channel_plot = z_channel_grid

ax3d.plot_surface(x_cone, y_cone, z_cone_plot, color='lightblue', alpha=0.8)
ax3d.plot_surface(x_channel, y_channel, z_channel_plot, color='lightgreen', alpha=0.8)
ax3d.set_xlabel("X (mm)")
ax3d.set_ylabel("Y (mm)")
ax3d.set_zlabel("Depth (mm)")
st.pyplot(fig3d)

# Ring Layout Visualization
fig_ring, ax_ring = plt.subplots(figsize=(6, 6))
ax_ring.set_aspect('equal')
for row, pcd in enumerate(pcd_values):
    radius = pcd / 2
    for i in range(holes_per_row):
        angle = 2 * np.pi * i / holes_per_row
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        hole = patches.Circle((x, y), final_diameter / 2, color='gray', edgecolor='black')
        ax_ring.add_patch(hole)
    ax_ring.set_xlim(-radius - 10, radius + 10)
    ax_ring.set_ylim(-radius - 10, radius + 10)
ax_ring.axis('off')
st.pyplot(fig_ring)
