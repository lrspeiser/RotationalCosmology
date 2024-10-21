import xml.etree.ElementTree as ET
import csv
import math
import numpy as np
from scipy.optimize import minimize
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import base64
import markdown
from io import BytesIO

# Constants
SPEED_OF_LIGHT = 299792.458  # km/s
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
UNIVERSE_RADIUS_LY = 14e9  # light years
UNIVERSE_RADIUS = UNIVERSE_RADIUS_LY / 3.26e6  # Convert to Mpc (~4297 Mpc)
H0 = 70  # Hubble constant (km/s/Mpc)

# CMB dipole direction (Galactic coordinates)
CMB_DIPOLE_L = 264.021  # degrees
CMB_DIPOLE_B = 48.253   # degrees

# CMB quadrupole alignment axis (Galactic coordinates)
CMB_QUADRUPOLE_L = 240.0  # degrees
CMB_QUADRUPOLE_B = 60.0   # degrees

# Global variables
total_galaxies = 0
galaxies_processed = 0
galaxies_with_redshift = 0
distance_min = float('inf')
distance_max = float('-inf')
maser_data = []
distance_methods = {
    'DMsnIa': 0, 'DMtf': 0, 'DMfp': 0, 'DMsbf': 0,
    'DMsnII': 0, 'DMtrgb': 0, 'DMcep': 0, 'DMmas': 0
}

def galactic_to_cartesian(l, b):
    l_rad = np.radians(l)
    b_rad = np.radians(b)
    x = np.cos(b_rad) * np.cos(l_rad)
    y = np.cos(b_rad) * np.sin(l_rad)
    z = np.sin(b_rad)
    return np.array([x, y, z])

def calculate_cmb_axes():
    dipole_axis = galactic_to_cartesian(CMB_DIPOLE_L, CMB_DIPOLE_B)
    quadrupole_axis = galactic_to_cartesian(CMB_QUADRUPOLE_L, CMB_QUADRUPOLE_B)
    return dipole_axis, quadrupole_axis

def calculate_universe_rotation_axis():
    dipole_axis, quadrupole_axis = calculate_cmb_axes()
    rotation_axis = np.cross(dipole_axis, quadrupole_axis)
    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize
    return rotation_axis

def logarithmic_scale(value, base=10):
    """Apply a logarithmic transformation to a value."""
    return np.sign(value) * np.log10(1 + np.abs(value))

def calculate_earth_position(rotation_axis, cmb_dipole_axis):
    perpendicular_component = cmb_dipole_axis - np.dot(cmb_dipole_axis, rotation_axis) * rotation_axis
    perpendicular_component /= np.linalg.norm(perpendicular_component)
    displacement = 0.01 * UNIVERSE_RADIUS  # 1% of universe radius in Mpc
    return displacement * perpendicular_component

def adjust_coordinates(position, earth_position, rotation_axis):
    relative_position = position - earth_position
    axial_component = np.dot(relative_position, rotation_axis) * rotation_axis
    perpendicular_component = relative_position - axial_component
    return perpendicular_component + axial_component

def calculate_maser_axis_distance(position, earth_position, rotation_axis):
    """
    Calculate the distance from a maser to the rotation axis.

    :param position: Maser's position vector
    :param earth_position: Earth's position vector
    :param rotation_axis: Unit vector of the rotation axis
    :return: Distance to the rotation axis in Mpc
    """
    adjusted_position = adjust_coordinates(position, earth_position, rotation_axis)
    perpendicular_component = adjusted_position - np.dot(adjusted_position, rotation_axis) * rotation_axis
    return np.linalg.norm(perpendicular_component)

def create_plotly_3d_visualization(maser_data, rotation_axis, earth_position):
    """
    Create a 3D Plotly visualization with the maser data, rotation axis, and Earth.
    """
    fig = go.Figure()

    # Plot Earth
    fig.add_trace(go.Scatter3d(
        x=[earth_position[0]],
        y=[earth_position[1]],
        z=[earth_position[2]],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=['Earth'],
        name='Earth'
    ))

    # Plot masers
    maser_x, maser_y, maser_z = [], [], []
    for maser in maser_data:
        position = equatorial_to_cartesian(float(maser['ra']), float(maser['dec']), maser['maser_distance'])
        adjusted_position = adjust_coordinates(position, earth_position, rotation_axis)
        maser_x.append(adjusted_position[0])
        maser_y.append(adjusted_position[1])
        maser_z.append(adjusted_position[2])

    fig.add_trace(go.Scatter3d(
        x=maser_x, y=maser_y, z=maser_z,
        mode='markers',
        marker=dict(size=5, color='red'),
        text=[maser['identifier'] for maser in maser_data],
        name='Masers'
    ))

    # Plot rotation axis
    axis_length = max(max(abs(np.array(maser_x))), max(abs(np.array(maser_y))), max(abs(np.array(maser_z))))
    fig.add_trace(go.Scatter3d(
        x=[-rotation_axis[0] * axis_length, rotation_axis[0] * axis_length],
        y=[-rotation_axis[1] * axis_length, rotation_axis[1] * axis_length],
        z=[-rotation_axis[2] * axis_length, rotation_axis[2] * axis_length],
        mode='lines',
        line=dict(color='green', width=5),
        name='Rotation Axis'
    ))

    # Add rotation direction indicator
    arrow_length = axis_length * 0.2
    arrow_end = np.cross(rotation_axis, [0, 0, 1]) * arrow_length
    fig.add_trace(go.Scatter3d(
        x=[0, arrow_end[0]],
        y=[0, arrow_end[1]],
        z=[0, arrow_end[2]],
        mode='lines',
        line=dict(color='orange', width=5),
        name='Rotation Direction'
    ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (Mpc)',
            yaxis_title='Y (Mpc)',
            zaxis_title='Z (Mpc)',
            aspectmode='data'
        ),
        title="Universe with Rotation Axis at Center",
        showlegend=True
    )

    # Save the figure
    plot_file = "universe_rotation_visualization.html"
    fig.write_html(plot_file)
    print(f"[edd_data_analysis] 3D visualization saved as '{plot_file}'")
    return plot_file

def equatorial_to_cartesian(ra, dec, distance):
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = distance * np.cos(dec_rad) * np.cos(ra_rad)
    y = distance * np.cos(dec_rad) * np.sin(ra_rad)
    z = distance * np.sin(dec_rad)
    return np.array([x, y, z])

def estimate_angular_velocity(maser_data, earth_position, rotation_axis):
    """Estimate the angular velocity of the universe based on maser data"""
    angular_velocities = []

    for maser in maser_data:
        position = equatorial_to_cartesian(float(maser['ra']), float(maser['dec']), maser['maser_distance'])
        axis_distance = calculate_maser_axis_distance(position, earth_position, rotation_axis)
        observed_z = maser['redshift_velocity'] / SPEED_OF_LIGHT

        # Invert the redshift formula to solve for angular velocity
        v_t = SPEED_OF_LIGHT * (observed_z**2 + 2*observed_z) / (2 + 2*observed_z + observed_z**2)
        omega = v_t / (axis_distance * 3.086e22)  # Convert Mpc to meters

        angular_velocities.append({
            'identifier': maser['identifier'],
            'omega': omega,
            'maser_distance': maser['maser_distance'],
            'axis_distance': axis_distance,
            'observed_z': observed_z
        })

    return angular_velocities


import numpy as np
import plotly.graph_objs as go

def scale_positions(positions, scale_factor=1e-6):
    """
    Scale positions to bring them closer to origin while preserving relative distances.
    """
    return positions * scale_factor

def create_improved_3d_visualization(maser_data, rotation_axis, earth_position):
    fig = go.Figure()
    scale_factor = 1e-6

    scaled_earth_position = earth_position * scale_factor

    fig.add_trace(go.Scatter3d(
        x=[scaled_earth_position[0]],
        y=[scaled_earth_position[1]],
        z=[scaled_earth_position[2]],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=['Earth'],
        name='Earth'
    ))

    maser_x, maser_y, maser_z = [], [], []
    for maser in maser_data:
        position = equatorial_to_cartesian(float(maser['ra']), float(maser['dec']), maser['maser_distance'])
        adjusted_position = adjust_coordinates(position, earth_position, rotation_axis)
        scaled_position = adjusted_position * scale_factor
        maser_x.append(scaled_position[0])
        maser_y.append(scaled_position[1])
        maser_z.append(scaled_position[2])

    fig.add_trace(go.Scatter3d(
        x=maser_x, y=maser_y, z=maser_z,
        mode='markers',
        marker=dict(size=5, color='red'),
        text=[maser['identifier'] for maser in maser_data],
        name='Masers'
    ))

    max_range = max(max(abs(np.array(maser_x))), max(abs(np.array(maser_y))), max(abs(np.array(maser_z))))
    axis_length = max_range * 1.5

    fig.add_trace(go.Scatter3d(
        x=[-rotation_axis[0] * axis_length, rotation_axis[0] * axis_length],
        y=[-rotation_axis[1] * axis_length, rotation_axis[1] * axis_length],
        z=[-rotation_axis[2] * axis_length, rotation_axis[2] * axis_length],
        mode='lines',
        line=dict(color='green', width=5),
        name='Rotation Axis'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (Scaled Mpc)',
            yaxis_title='Y (Scaled Mpc)',
            zaxis_title='Z (Scaled Mpc)',
            aspectmode='data'
        ),
        title="Scaled Universe Visualization",
        showlegend=True
    )

    plot_file = "improved_universe_visualization.html"
    fig.write_html(plot_file)
    print(f"[edd_data_analysis] Improved 3D visualization saved as '{plot_file}'")
    return plot_file

# Additional visualization: 2D projection with relative distances preserved
def create_2d_projection(maser_data, rotation_axis, earth_position):
    fig = go.Figure()

    scale_factor = 1e-6

    # Calculate distances from axis and along axis for Earth
    earth_dist_from_axis = np.linalg.norm(earth_position - np.dot(earth_position, rotation_axis) * rotation_axis)
    earth_dist_along_axis = np.dot(earth_position, rotation_axis)

    fig.add_trace(go.Scatter(
        x=[earth_dist_from_axis * scale_factor],
        y=[earth_dist_along_axis * scale_factor],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=['Earth'],
        name='Earth'
    ))

    for maser in maser_data:
        position = equatorial_to_cartesian(float(maser['ra']), float(maser['dec']), maser['maser_distance'])
        adjusted_position = adjust_coordinates(position, earth_position, rotation_axis)
        dist_from_axis = np.linalg.norm(adjusted_position - np.dot(adjusted_position, rotation_axis) * rotation_axis)
        dist_along_axis = np.dot(adjusted_position, rotation_axis)

        fig.add_trace(go.Scatter(
            x=[dist_from_axis * scale_factor],
            y=[dist_along_axis * scale_factor],
            mode='markers+text',
            marker=dict(size=5, color='red'),
            text=[maser['identifier']],
            name=f'Maser {maser["identifier"]}'
        ))

    fig.update_layout(
        xaxis_title="Scaled Distance from Rotation Axis",
        yaxis_title="Scaled Distance along Rotation Axis",
        title=f"2D Projection of Cosmic Objects (Scale Factor: {scale_factor})"
    )

    plot_file = "2d_projection_visualization.html"
    fig.write_html(plot_file)
    print(f"[edd_data_analysis] 2D projection visualization saved as '{plot_file}'")
    return plot_file

def create_static_2d_distance_speed_graph(maser_data):
    distances = [maser['maser_distance'] for maser in maser_data]
    speeds = [maser['redshift_velocity'] for maser in maser_data]

    plt.figure(figsize=(10, 6))
    plt.scatter(distances, speeds)
    plt.xlabel('Distance from Earth (Mpc)')
    plt.ylabel('Rotational Speed (km/s)')
    plt.title('Distance vs Rotational Speed for Masers')

    # Save plot as PNG file
    plot_2d_file = "distance_vs_speed_graph.png"
    plt.savefig(plot_2d_file)
    plt.close()

    print(f"[edd_data_analysis] Static 2D graph saved as '{plot_2d_file}'")

    return plot_2d_file


def calculate_rotation_axis(optimal_ra, optimal_dec):
    """Calculate the rotation axis in Cartesian coordinates"""
    coord = SkyCoord(ra=optimal_ra*u.degree, dec=optimal_dec*u.degree, frame='icrs')
    galactic = coord.galactic
    return galactic_to_cartesian(galactic.l.degree, galactic.b.degree)

def calculate_frame_dragging_effect(mass, radius, distance_ly):
    """
    Calculate frame-dragging angular velocity based on mass, radius, and distance.
    mass: Mass of the object in kg
    radius: Radius of the object in meters
    distance_ly: Distance from the center of the universe in light years
    """
    distance_meters = distance_ly * 9.461e15  # Convert light years to meters

    # Calculate angular momentum J = (2/5) * M * R^2 * (v/R)
    angular_momentum = (2 / 5) * mass * radius**2 * (SPEED_OF_LIGHT / radius)

    # Frame dragging angular velocity Omega_LT = (2 * G * J) / (c^2 * r^3)
    frame_dragging_omega = (2 * GRAVITATIONAL_CONSTANT * angular_momentum) / (SPEED_OF_LIGHT**2 * distance_meters**3)

    return frame_dragging_omega

def calculate_maser_positions(maser_data, rotation_axis):
    maser_positions = []
    for maser in maser_data:
        ra = float(maser['ra'])
        dec = float(maser['dec'])
        distance = maser['maser_distance']

        # Convert spherical coordinates to Cartesian
        x = distance * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
        y = distance * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
        z = distance * np.sin(np.radians(dec))

        # Translate coordinates to make rotation axis the origin
        x -= rotation_axis[0] * UNIVERSE_RADIUS
        y -= rotation_axis[1] * UNIVERSE_RADIUS
        z -= rotation_axis[2] * UNIVERSE_RADIUS

        maser_positions.append((x, y, z))

    return maser_positions


def create_readme(maser_data, rotation_axis, median_angular_velocity, optimal_omega, optimal_axis_ra, optimal_axis_dec, cmb_dipole_axis, cmb_quadrupole_axis, earth_position):
    readme_content = f"""
# Rotating Universe Model Analysis

## Overview

This document summarizes the analysis of a hypothetical rotating universe model based on maser data and CMB observations.

## General Statistics

- **Total galaxies processed**: {total_galaxies}
- **Galaxies with maser distances**: {len(maser_data)}
- **Assumed distance to universe center**: {UNIVERSE_RADIUS:.2e} light years

## Rotation Parameters

- **Estimated median angular velocity**: {median_angular_velocity:.2e} rad/s
- **Optimal angular velocity**: {optimal_omega:.2e} rad/s
- **Optimal rotation axis**: RA = {np.degrees(optimal_axis_ra):.2f}°, Dec = {np.degrees(optimal_axis_dec):.2f}°

## CMB and Rotation Axis Analysis

| Axis | X | Y | Z |
|------|---|---|---|
| Rotation Axis | {rotation_axis[0]:.4f} | {rotation_axis[1]:.4f} | {rotation_axis[2]:.4f} |
| CMB Dipole Axis | {cmb_dipole_axis[0]:.4f} | {cmb_dipole_axis[1]:.4f} | {cmb_dipole_axis[2]:.4f} |
| CMB Quadrupole/Octupole Axis | {cmb_quadrupole_axis[0]:.4f} | {cmb_quadrupole_axis[1]:.4f} | {cmb_quadrupole_axis[2]:.4f} |

- **Angle between Rotation Axis and CMB Dipole**: {np.arccos(np.dot(rotation_axis, cmb_dipole_axis)) * 180 / np.pi:.2f} degrees
- **Angle between Rotation Axis and CMB Quadrupole/Octupole**: {np.arccos(np.dot(rotation_axis, cmb_quadrupole_axis)) * 180 / np.pi:.2f} degrees

## Earth and Maser Data

### Earth

- **Position**: ({earth_position[0]:.2f}, {earth_position[1]:.2f}, {earth_position[2]:.2f}) Mpc

### Maser Data

"""

    for maser in maser_data:
        position = equatorial_to_cartesian(float(maser['ra']), float(maser['dec']), maser['maser_distance'])
        adjusted_position = adjust_coordinates(position, earth_position, rotation_axis)
        axis_distance = calculate_maser_axis_distance(position, earth_position, rotation_axis)

        readme_content += f"""
#### Maser {maser['identifier']}
- **Distance from Earth**: {maser['maser_distance']:.2f} Mpc
- **Observed Redshift**: {maser['redshift_velocity'] / SPEED_OF_LIGHT:.6f}
- **Distance to rotation axis**: {axis_distance:.2f} Mpc
- **Original Coordinates**: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) Mpc
- **Adjusted Coordinates**: ({adjusted_position[0]:.2f}, {adjusted_position[1]:.2f}, {adjusted_position[2]:.2f}) Mpc
"""

    readme_content += """
## Notes

- The rotation axis is positioned at the center of the universe (0, 0, 0).
- Earth's position is calculated based on the CMB dipole and estimated rotation of the universe.
- Maser coordinates are adjusted relative to the central rotation axis.
- This model is highly speculative and should be interpreted with caution. It does not reflect the current scientific understanding of the universe's structure and dynamics.
- The visualization of this data can be found in the accompanying 3D plot file.
"""

    # Write README file
    with open('README.md', 'w') as readme_file:
        readme_file.write(readme_content)

    print("[edd_data_analysis] README.md file created successfully")
    

def create_distance_speed_graph(maser_data):
    distances = [maser['maser_distance'] for maser in maser_data]
    speeds = [maser['redshift_velocity'] for maser in maser_data]

    plt.figure(figsize=(10, 6))
    plt.scatter(distances, speeds)
    plt.xlabel('Distance from Earth (Mpc)')
    plt.ylabel('Rotational Speed (km/s)')
    plt.title('Distance vs Rotational Speed for Masers')

    # Save plot to a base64 encoded string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plot_data

def create_html_page(plot_3d_file, plot_2d_file):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Maser Visualization</title>
    </head>
    <body>
        <h1>Maser Visualization</h1>
        <h2>Interactive 3D Visualization of Maser Positions</h2>
        <p><a href="{plot_3d_file}" target="_blank">Click here to view the interactive 3D visualization</a></p>
        <h2>Distance vs Rotational Speed Graph</h2>
        <img src="{plot_2d_file}" alt="Distance vs Speed Graph">
    </body>
    </html>
    """

    with open('maser_visualization.html', 'w') as f:
        f.write(html_content)

    print("[edd_data_analysis] HTML page with links created: 'maser_visualization.html'")


def process_xml_file(file_path, namespace):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        print(f"[edd_data_analysis] {file_path} loaded successfully")

        table_data = root.find(".//votable:TABLEDATA", namespace)
        rows = table_data.findall("votable:TR", namespace) if table_data is not None else []

        if not rows:
            raise ValueError("No data rows found in XML file")

        print(f"[edd_data_analysis] Found {len(rows)} rows of data in {file_path}")
        return rows
    except Exception as e:
        print(f"[edd_data_analysis] Error processing {file_path}: {e}")
        return []

def process_galaxy(csv_writer, identifier, distance_mpc, redshift_velocity, distance_method, ra, dec, glon, glat):
    global galaxies_processed, galaxies_with_redshift, distance_min, distance_max

    galaxies_processed += 1
    if not np.isnan(distance_mpc):
        distance_min = min(distance_min, distance_mpc)
        distance_max = max(distance_max, distance_mpc)

    if not np.isnan(redshift_velocity):
        galaxies_with_redshift += 1

    # For maser galaxies, calculate frame-dragging effect
    if distance_method == 'DMmas':
        # Assuming mass of galaxy is a large value, let's use a dummy mass for illustration (1e40 kg)
        galaxy_mass = 1e40  # mass in kg
        galaxy_radius = 1e20  # radius in meters for simplification

        distance_ly = distance_mpc * 3.262e6  # Convert distance to light years
        frame_dragging_effect = calculate_frame_dragging_effect(galaxy_mass, galaxy_radius, distance_ly)
    else:
        frame_dragging_effect = np.nan

    csv_writer.writerow([identifier, distance_mpc, redshift_velocity, distance_method, ra, dec, glon, glat, frame_dragging_effect])

    # ONLY ADD TO MASER DATA IF THE DISTANCE METHOD IS 'DMmas'
    if distance_method == 'DMmas' and not np.isnan(redshift_velocity):
        maser_data.append({
            'identifier': identifier,
            'maser_distance': distance_mpc,
            'redshift_velocity': float(redshift_velocity),
            'ra': ra,
            'dec': dec,
            'glon': glon,
            'glat': glat,
            'frame_dragging_effect': frame_dragging_effect
        })
        print(f"[process_galaxy] Extracted maser data for {identifier}")

def create_maser_xml():
    print("[edd_data_analysis] Creating XML file for maser data")

    root = ET.Element("MaserData")

    for maser in maser_data:
        maser_element = ET.SubElement(root, "Maser")
        ET.SubElement(maser_element, "Identifier").text = maser['identifier']
        ET.SubElement(maser_element, "MaserDistanceMpc").text = f"{maser['maser_distance']:.2f}"
        ET.SubElement(maser_element, "RedshiftVelocity").text = f"{maser['redshift_velocity']:.2f}"
        ET.SubElement(maser_element, "RA").text = maser['ra']
        ET.SubElement(maser_element, "Dec").text = maser['dec']
        ET.SubElement(maser_element, "GalacticLongitude").text = maser['glon']
        ET.SubElement(maser_element, "GalacticLatitude").text = maser['glat']
        ET.SubElement(maser_element, "FrameDraggingEffect").text = f"{maser['frame_dragging_effect']:.6e}"

    tree = ET.ElementTree(root)
    xml_file_path = "maser_edd.xml"
    tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)
    print(f"[edd_data_analysis] XML file '{xml_file_path}' created successfully")

def calculate_rotational_redshift(position, omega, axis):
    """
    Calculate the redshift due to rotation in a hypothetical rotating universe.

    :param position: 3D position vector of the object in Mpc
    :param omega: Angular velocity vector of the universe in rad/s
    :param axis: Rotation axis unit vector
    :return: Calculated redshift
    """
    # Convert position to meters
    position_m = position * 3.086e22  # Convert Mpc to meters

    # Calculate tangential velocity
    v = np.cross(omega, position_m)
    v_t = np.linalg.norm(v)

    # Check if tangential velocity exceeds speed of light
    if v_t >= SPEED_OF_LIGHT:
        return np.inf  # Return infinity for superluminal velocities

    # Calculate redshift
    z = np.sqrt((1 + v_t / SPEED_OF_LIGHT) / (1 - v_t / SPEED_OF_LIGHT)) - 1

    return z



# Define the objective_function
def objective_function(params):
    omega_magnitude, axis_ra, axis_dec = params
    axis = equatorial_to_cartesian(axis_ra, axis_dec, 1)  # Unit vector
    omega = omega_magnitude * axis
    total_error = 0

    for maser in maser_data:
        position = equatorial_to_cartesian(float(maser['ra']), float(maser['dec']), maser['maser_distance'])
        observed_z = maser['redshift_velocity'] / SPEED_OF_LIGHT

        calculated_z = calculate_rotational_redshift(position, omega, axis)

        if np.isinf(calculated_z):
            total_error += 1e6  # Large penalty for superluminal velocities
        else:
            total_error += (observed_z - calculated_z)**2

    return total_error


def main():
    global total_galaxies, galaxies_processed, galaxies_with_redshift, distance_min, distance_max, maser_data

    # Load and process the EDD XML file
    edd_rows = process_xml_file("eddtable.xml", {'votable': 'http://www.ivoa.net/xml/VOTable/v1.2'})
    if not edd_rows:
        return

    # Output CSV file for processed galaxy data
    csv_file_path = "combined_galaxy_data.csv"

    print("[edd_data_analysis] Processing data and writing to CSV file")
    try:
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Source_Name", "Distance_Mpc", "Redshift_Velocity", "Distance_Method", "RA", "Dec", 
                                 "Gal_Lon", "Gal_Lat", "Frame_Dragging_Effect"])

            # Process EDD data
            for row in edd_rows:
                total_galaxies += 1
                columns = row.findall("votable:TD", {'votable': 'http://www.ivoa.net/xml/VOTable/v1.2'})
                if len(columns) < 28:
                    continue

                pgc_id = columns[0].text
                redshift_velocity = columns[3].text
                ra = columns[22].text
                dec = columns[23].text
                glon = columns[24].text
                glat = columns[25].text

                try:
                    redshift_velocity = float(redshift_velocity)
                except ValueError:
                    redshift_velocity = np.nan  # Assign NaN if conversion fails

                distance_mpc, distance_method = None, None
                for method, index in [('DMsnIa', 6), ('DMtf', 8), ('DMfp', 10), ('DMsbf', 12),
                                      ('DMsnII', 14), ('DMtrgb', 16), ('DMcep', 18), ('DMmas', 20)]:
                    dm = columns[index].text
                    if dm:
                        try:
                            distance = round(10 ** ((float(dm) + 5) / 5) / 1e6, 3)
                            if distance >= 10:
                                distance_mpc = distance
                                distance_method = method
                                distance_methods[method] += 1
                                break
                        except ValueError:
                            continue

                if distance_mpc:
                    process_galaxy(csv_writer, pgc_id, distance_mpc, redshift_velocity, distance_method, 
                                   ra, dec, glon, glat)

    except Exception as e:
        print(f"[edd_data_analysis] Error processing data: {e}")
        return

    print(f"[edd_data_analysis] CSV file '{csv_file_path}' created successfully")

    # Perform optimization to find best-fit parameters
    initial_guess = [1e-18, 0, 0]  # omega, axis_ra, axis_dec
    result = minimize(objective_function, initial_guess, method='Nelder-Mead')
    optimal_omega, optimal_axis_ra, optimal_axis_dec = result.x

    # Calculate universe rotation axis and CMB axes
    rotation_axis = calculate_universe_rotation_axis()
    cmb_dipole_axis, cmb_quadrupole_axis = calculate_cmb_axes()

    # Calculate Earth's position relative to the universe's rotation axis
    earth_position = calculate_earth_position(rotation_axis, cmb_dipole_axis)

    # Estimate angular velocity for each maser and log them
    maser_angular_velocities = estimate_angular_velocity(maser_data, earth_position, rotation_axis)
    median_angular_velocity = np.median([maser['omega'] for maser in maser_angular_velocities])
    print(f"Estimated median angular velocity: {median_angular_velocity:.2e} rad/s")

    # Calculate angles between the rotation axis and CMB axes
    angle_dipole = np.arccos(np.dot(rotation_axis, cmb_dipole_axis)) * 180 / np.pi
    angle_quadrupole = np.arccos(np.dot(rotation_axis, cmb_quadrupole_axis)) * 180 / np.pi

    # Create log file
    log_file_path = "maser_analysis_log.txt"
    with open(log_file_path, 'w') as log_file:
        log_file.write("Maser Analysis Log\n")
        log_file.write("===================\n\n")
        log_file.write(f"Total galaxies processed: {total_galaxies}\n")
        log_file.write(f"Galaxies with maser distances: {len(maser_data)}\n")
        log_file.write(f"Assumed distance to universe center: {UNIVERSE_RADIUS:.2e} Mpc\n\n")
        log_file.write(f"Estimated median angular velocity: {median_angular_velocity:.2e} rad/s\n")
        log_file.write(f"Optimal angular velocity: {optimal_omega:.2e} rad/s\n")
        log_file.write(f"Optimal rotation axis: RA = {np.degrees(optimal_axis_ra):.2f}°, "
                       f"Dec = {np.degrees(optimal_axis_dec):.2f}°\n\n")
        log_file.write("CMB and Rotation Axis Analysis:\n")
        log_file.write("===============================\n")
        log_file.write(f"Rotation Axis (Cartesian): [{rotation_axis[0]:.4f}, {rotation_axis[1]:.4f}, "
                       f"{rotation_axis[2]:.4f}]\n")
        log_file.write(f"CMB Dipole Axis (Cartesian): [{cmb_dipole_axis[0]:.4f}, {cmb_dipole_axis[1]:.4f}, "
                       f"{cmb_dipole_axis[2]:.4f}]\n")
        log_file.write(f"CMB Quadrupole Axis (Cartesian): [{cmb_quadrupole_axis[0]:.4f}, "
                       f"{cmb_quadrupole_axis[1]:.4f}, {cmb_quadrupole_axis[2]:.4f}]\n")
        log_file.write(f"Angle between Rotation Axis and CMB Dipole: {angle_dipole:.2f} degrees\n")
        log_file.write(f"Angle between Rotation Axis and CMB Quadrupole: {angle_quadrupole:.2f} degrees\n")

        # Log Earth and maser data
        log_file.write("\nEarth (Adjusted Position):\n")
        log_file.write(f"  Cartesian Coordinates: [{earth_position[0]:.2f}, {earth_position[1]:.2f}, "
                       f"{earth_position[2]:.2f}] (Mpc)\n\n")
        for i, maser in enumerate(maser_angular_velocities):
            maser_position = equatorial_to_cartesian(float(maser_data[i]['ra']), 
                                                     float(maser_data[i]['dec']), 
                                                     maser_data[i]['maser_distance'])
            log_file.write(f"Identifier: {maser['identifier']}\n")
            log_file.write(f"  Distance: {maser['maser_distance']:.2f} Mpc\n")
            log_file.write(f"  Observed Redshift: {maser['observed_z']:.6f}\n")
            log_file.write(f"  Distance to rotation axis: {maser['axis_distance']:.2f} Mpc\n")
            log_file.write(f"  Angular Velocity: {maser['omega']:.2e} rad/s\n")
            log_file.write(f"  Cartesian Coordinates: [{maser_position[0]:.2f}, "
                           f"{maser_position[1]:.2f}, {maser_position[2]:.2f}] (Mpc)\n\n")

    print(f"[edd_data_analysis] Log file '{log_file_path}' created successfully")

    # Create visualizations and README
    plot_3d_file = create_improved_3d_visualization(maser_data, rotation_axis, earth_position)
    plot_2d_file = create_2d_projection(maser_data, rotation_axis, earth_position)
    create_html_page(plot_3d_file, plot_2d_file)
    create_readme(maser_data, rotation_axis, median_angular_velocity, optimal_omega, 
                  optimal_axis_ra, optimal_axis_dec, cmb_dipole_axis, cmb_quadrupole_axis, earth_position)

    print("[edd_data_analysis] Analysis complete")
    create_maser_xml()

if __name__ == "__main__":
    main()
