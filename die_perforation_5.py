
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

st.title("Die Perforation Visualizer")

# User Inputs
plate_thickness = st.slider("Total Plate Thickness (mm)", 1, 100, 20)
final_diameter = st.slider("Final Hole Diameter (mm)", 1, 50, 10)
cone_diameter = st.slider("Cone Opening Diameter (mm)", final_diameter + 1, 100, 20)
channel_length = st.slider("Channel (Land) Length (mm)", 1, plate_thickness, 10)
total_holes = st.number_input("Total Number of Holes Required", min_value=1, value=100)
dry_meal_throughput = st.number_input("Dry Meal Throughput (tonne/h)", min_value=0.1, value=10.0)
space_between_holes = st.number_input("Space Between Holes (mm)", min_value=1.0, value=5.0)
space_between_rows = st.number_input("Space Between Rows (mm)", min_value=1.0, value=5.0)
number_of_rows = st.number_input("Number of Rows", min_value=1, value=5)

# Calculated Outputs
cone_length = plate_thickness - channel_length
cone_radius = (cone_diameter - final_diameter) / 2
cone_angle_rad = np.arctan(cone_radius / cone_length)
cone_angle_deg = np.degrees(cone_angle_rad)
holes_per_row = int(total_holes / number_of_rows)
open_area_one_hole = np.pi * (final_diameter / 2) ** 2
total_open_area = open_area_one_hole * total_holes
open_area_per_tonne = total_open_area / dry_meal_throughput

# Display Outputs
st.markdown("### Calculated Outputs")
st.write(f"Cone Length: {cone_length:.2f} mm")
st.write(f"Cone Angle: {cone_angle_deg:.2f}°")
st.write(f"Open Area of One Hole: {open_area_one_hole:.2f} mm²")
st.write(f"Total Plate Open Area: {total_open_area:.2f} mm²")
st.write(f"Open Area per Tonne: {open_area_per_tonne:.2f} mm²/t/h")
st.write(f"Number of Holes per Row: {holes_per_row}")

# 2D Cross-Section Visualization
st.markdown("### 2D Cross-Section of Die Perforation")
fig, ax = plt.subplots(figsize=(4, 6))
ax.set_xlim(-cone_diameter, cone_diameter)
ax.set_ylim(0, plate_thickness + 5)
ax.set_aspect('equal')
ax.set_title("Die Cross-Section")

# Cone
cone = patches.Polygon([[0, 0], [-cone_diameter/2, cone_length], [-final_diameter/2, cone_length], [-final_diameter/2, plate_thickness],
                        [final_diameter/2, plate_thickness], [final_diameter/2, cone_length], [cone_diameter/2, cone_length], [0, 0]],
                       closed=True, color='lightblue', edgecolor='black')
ax.add_patch(cone)
ax.set_xlabel("Width (mm)")
ax.set_ylabel("Depth (mm)")
st.pyplot(fig)

# 3D Visualization
st.markdown("### 3D Visualization of Die Perforation")
fig3d = plt.figure(figsize=(6, 6))
ax3d = fig3d.add_subplot(111, projection='3d')
z_cone = np.linspace(0, cone_length, 30)
z_channel = np.linspace(cone_length, plate_thickness, 10)
theta = np.linspace(0, 2 * np.pi, 30)
theta_grid, z_cone_grid = np.meshgrid(theta, z_cone)
theta_grid2, z_channel_grid = np.meshgrid(theta, z_channel)

r_cone = (cone_diameter/2 - (cone_diameter - final_diameter)/2 * (z_cone / cone_length))
x_cone = r_cone[:, np.newaxis] * np.cos(theta_grid)
y_cone = r_cone[:, np.newaxis] * np.sin(theta_grid)
z_cone_plot = z_cone_grid

r_channel = np.full_like(z_channel, final_diameter/2)
x_channel = r_channel[:, np.newaxis] * np.cos(theta_grid2)
y_channel = r_channel[:, np.newaxis] * np.sin(theta_grid2)
z_channel_plot = z_channel_grid

ax3d.plot_surface(x_cone, y_cone, z_cone_plot, color='lightblue', alpha=0.8)
ax3d.plot_surface(x_channel, y_channel, z_channel_plot, color='lightgreen', alpha=0.8)
ax3d.set_title("3D Die Perforation")
ax3d.set_xlabel("X (mm)")
ax3d.set_ylabel("Y (mm)")
ax3d.set_zlabel("Depth (mm)")
st.pyplot(fig3d)

# Ring Layout Visualization
st.markdown("### Ring Layout of Holes")
fig_ring, ax_ring = plt.subplots(figsize=(6, 6))
ax_ring.set_aspect('equal')
ax_ring.set_title("Die Hole Layout")
for row in range(number_of_rows):
    radius = 20 + row * space_between_rows
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
