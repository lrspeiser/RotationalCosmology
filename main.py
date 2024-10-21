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
galaxy_data = []
distance_methods = {
    'DMsnIa': 0, 'DMtf': 0, 'DMfp': 0, 'DMsbf': 0,
    'DMsnII': 0, 'DMtrgb': 0, 'DMcep': 0, 'DMmas': 0, 'Redshift': 0
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

def equatorial_to_cartesian(ra, dec, distance):
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = distance * np.cos(dec_rad) * np.cos(ra_rad)
    y = distance * np.cos(dec_rad) * np.sin(ra_rad)
    z = distance * np.sin(dec_rad)
    return np.array([x, y, z])

def calculate_distance_from_axis(position, rotation_axis):
    """
    Calculate the shortest distance from a point to the rotation axis.
    """
    projection_length = np.dot(position, rotation_axis)
    projection_vector = projection_length * rotation_axis
    perpendicular_vector = position - projection_vector
    distance_from_axis = np.linalg.norm(perpendicular_vector)
    return distance_from_axis

def calculate_tangential_velocity(distance_from_axis, angular_velocity):
    return angular_velocity * distance_from_axis * 3.086e22  # Convert Mpc to meters

def calculate_line_of_sight_unit_vector(position, earth_position):
    los_vector = position - earth_position
    return los_vector / np.linalg.norm(los_vector)

def calculate_tangential_velocity_vector(position, rotation_axis, tangential_speed):
    """
    Calculate the tangential velocity vector of a point due to rotation.
    """
    radius_vector = position - np.dot(position, rotation_axis) * rotation_axis
    radius_unit_vector = radius_vector / np.linalg.norm(radius_vector)
    tangential_velocity_vector = np.cross(rotation_axis, radius_unit_vector) * tangential_speed
    return tangential_velocity_vector

def calculate_rotational_redshift(earth_velocity_vector, galaxy_velocity_vector, los_unit_vector):
    relative_velocity = galaxy_velocity_vector - earth_velocity_vector
    velocity_along_los = np.dot(relative_velocity, los_unit_vector)
    return velocity_along_los / SPEED_OF_LIGHT

def redshift_to_distance(z_cosmological, H0=70):
    """
    Estimate distance from cosmological redshift using Hubble's Law.
    """
    velocity = z_cosmological * SPEED_OF_LIGHT  # km/s
    distance = velocity / H0  # Mpc
    return distance

def process_galaxy(csv_writer, identifier, distance_mpc, redshift_velocity, distance_method, ra, dec, glon, glat):
    global galaxies_processed, galaxies_with_redshift, distance_min, distance_max

    galaxies_processed += 1
    if not np.isnan(distance_mpc):
        distance_min = min(distance_min, distance_mpc)
        distance_max = max(distance_max, distance_mpc)

    if not np.isnan(redshift_velocity):
        galaxies_with_redshift += 1

    if distance_method == 'DMmas':
        frame_dragging_effect = calculate_frame_dragging_effect(1e40, 1e20, distance_mpc * 3.262e6)
    else:
        frame_dragging_effect = np.nan

    csv_writer.writerow([identifier, distance_mpc, redshift_velocity, distance_method, ra, dec, glon, glat, frame_dragging_effect])

    # Store galaxy data
    galaxy_data.append({
        'identifier': identifier,
        'distance_mpc': distance_mpc,
        'redshift_velocity': redshift_velocity,
        'distance_method': distance_method,
        'ra': ra,
        'dec': dec,
        'glon': glon,
        'glat': glat,
        'frame_dragging_effect': frame_dragging_effect
    })

    if distance_method == 'DMmas':
        maser_data.append({
            'identifier': identifier,
            'maser_distance': distance_mpc,
            'redshift_velocity': redshift_velocity,
            'ra': ra,
            'dec': dec,
            'glon': glon,
            'glat': glat,
            'frame_dragging_effect': frame_dragging_effect
        })

def create_visualizations(maser_data, galaxy_data, rotation_axis, earth_position):
    # Create improved 3D visualization
    plot_3d_file = create_improved_3d_visualization(maser_data, galaxy_data, rotation_axis, earth_position)
    # Create 2D projection
    plot_2d_file = create_2d_projection(maser_data, galaxy_data, rotation_axis, earth_position)
    # Create HTML page
    create_html_page(plot_3d_file, plot_2d_file)

def create_improved_3d_visualization(maser_data, galaxy_data, rotation_axis, earth_position):
    fig = go.Figure()
    scale_factor = 1e-6

    scaled_earth_position = earth_position * scale_factor

    # Plot Earth
    fig.add_trace(go.Scatter3d(
        x=[scaled_earth_position[0]],
        y=[scaled_earth_position[1]],
        z=[scaled_earth_position[2]],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=['Earth'],
        name='Earth'
    ))

    # Plot masers
    maser_x, maser_y, maser_z = [], [], []
    for maser in maser_data:
        adjusted_position = maser['adjusted_position'] * scale_factor
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

    # Plot other galaxies
    galaxy_x, galaxy_y, galaxy_z = [], [], []
    for galaxy in galaxy_data:
        if galaxy['distance_method'] == 'DMmas':
            continue  # Skip masers already plotted
        adjusted_position = galaxy['adjusted_position'] * scale_factor
        galaxy_x.append(adjusted_position[0])
        galaxy_y.append(adjusted_position[1])
        galaxy_z.append(adjusted_position[2])

    fig.add_trace(go.Scatter3d(
        x=galaxy_x, y=galaxy_y, z=galaxy_z,
        mode='markers',
        marker=dict(size=2, color='gray'),
        text=[galaxy['identifier'] for galaxy in galaxy_data if galaxy['distance_method'] != 'DMmas'],
        name='Galaxies'
    ))

    # Plot rotation axis
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

    # Update layout
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

    # Save the visualization
    plot_file = "improved_universe_visualization.html"
    fig.write_html(plot_file)
    print(f"[edd_data_analysis] Improved 3D visualization saved as '{plot_file}'")
    return plot_file

def create_2d_projection(maser_data, galaxy_data, rotation_axis, earth_position):
    fig = go.Figure()
    scale_factor = 1e-6

    # Earth's distances from and along the rotation axis
    earth_dist_from_axis = calculate_distance_from_axis(earth_position, rotation_axis)
    earth_dist_along_axis = np.dot(earth_position, rotation_axis)

    fig.add_trace(go.Scatter(
        x=[earth_dist_from_axis * scale_factor],
        y=[earth_dist_along_axis * scale_factor],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=['Earth'],
        name='Earth'
    ))

    # Plot masers
    for maser in maser_data:
        adjusted_position = maser['adjusted_position']
        dist_from_axis = calculate_distance_from_axis(adjusted_position, rotation_axis)
        dist_along_axis = np.dot(adjusted_position, rotation_axis)

        fig.add_trace(go.Scatter(
            x=[dist_from_axis * scale_factor],
            y=[dist_along_axis * scale_factor],
            mode='markers+text',
            marker=dict(size=5, color='red'),
            text=[maser['identifier']],
            name=f'Maser {maser["identifier"]}'
        ))

    # Plot other galaxies
    for galaxy in galaxy_data:
        if galaxy['distance_method'] == 'DMmas':
            continue
        adjusted_position = galaxy['adjusted_position']
        dist_from_axis = calculate_distance_from_axis(adjusted_position, rotation_axis)
        dist_along_axis = np.dot(adjusted_position, rotation_axis)

        fig.add_trace(go.Scatter(
            x=[dist_from_axis * scale_factor],
            y=[dist_along_axis * scale_factor],
            mode='markers',
            marker=dict(size=2, color='gray'),
            text=[galaxy['identifier']],
            name='Galaxy'
        ))

    # Update layout
    fig.update_layout(
        xaxis_title="Scaled Distance from Rotation Axis",
        yaxis_title="Scaled Distance along Rotation Axis",
        title=f"2D Projection of Cosmic Objects (Scale Factor: {scale_factor})"
    )

    # Save the visualization
    plot_file = "2d_projection_visualization.html"
    fig.write_html(plot_file)
    print(f"[edd_data_analysis] 2D projection visualization saved as '{plot_file}'")
    return plot_file

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
        <p><a href="{plot_2d_file}" target="_blank">Click here to view the 2D projection visualization</a></p>
    </body>
    </html>
    """

    with open('maser_visualization.html', 'w') as f:
        f.write(html_content)

    print("[edd_data_analysis] HTML page with links created: 'maser_visualization.html'")

def calculate_frame_dragging_effect(mass, radius, distance_ly):
    """
    Calculate frame-dragging angular velocity based on mass, radius, and distance.
    """
    distance_meters = distance_ly * 9.461e15  # Convert light years to meters

    # Calculate angular momentum J = (2/5) * M * R^2 * (v/R)
    angular_momentum = (2 / 5) * mass * radius**2 * (SPEED_OF_LIGHT / radius)

    # Frame dragging angular velocity Omega_LT = (2 * G * J) / (c^2 * r^3)
    frame_dragging_omega = (2 * GRAVITATIONAL_CONSTANT * angular_momentum) / (SPEED_OF_LIGHT**2 * distance_meters**3)

    return frame_dragging_omega

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

def export_eddstar_data_to_csv(galaxy_data, output_file="eddstar_data.csv"):
    """
    Export EDD star data with adjusted coordinates, rotational velocities, and redshifts to a CSV file.
    The data is sorted by rotational velocity from slowest to fastest.
    """
    # Filter out galaxies with missing tangential velocity
    valid_galaxies = [g for g in galaxy_data if g.get('tangential_velocity') is not None]

    # Sort galaxies by tangential velocity (slowest to fastest)
    sorted_galaxies = sorted(valid_galaxies, key=lambda g: g['tangential_velocity'])

    # Write sorted data to CSV
    with open(output_file, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow([
            "Identifier", "Distance_Mpc", "RA", "Dec", "Adjusted_X", "Adjusted_Y", "Adjusted_Z",
            "Tangential_Velocity (m/s)", "Observed_Redshift", "Rotational_Redshift", 
            "Cosmological_Redshift", "Corrected_Distance_Mpc"
        ])

        # Write data rows
        for galaxy in sorted_galaxies:
            adjusted_position = galaxy['adjusted_position']
            csv_writer.writerow([
                galaxy['identifier'],
                galaxy['distance_mpc'],
                galaxy['ra'],
                galaxy['dec'],
                f"{adjusted_position[0]:.6f}",
                f"{adjusted_position[1]:.6f}",
                f"{adjusted_position[2]:.6f}",
                f"{galaxy['tangential_velocity']:.6e}",
                f"{galaxy['z_observed']:.6f}",
                f"{galaxy['z_rotation']:.6f}",
                f"{galaxy['z_cosmological']:.6f}",
                f"{galaxy['distance_mpc_corrected']:.6f}"
            ])

    print(f"[edd_data_analysis] EDD star data exported to '{output_file}' successfully.")


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

def create_readme(maser_data, rotation_axis, median_angular_velocity, optimal_omega, optimal_axis_ra, optimal_axis_dec, cmb_dipole_axis, cmb_quadrupole_axis, earth_position):
    readme_content = f"""
# Rotating Universe Model Analysis

## Overview

This document summarizes the analysis of a hypothetical rotating universe model based on maser data and CMB observations.

## General Statistics

- **Total galaxies processed**: {total_galaxies}
- **Galaxies with maser distances**: {len(maser_data)}
- **Assumed distance to universe center**: {UNIVERSE_RADIUS:.2e} Mpc

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

## Reliability of Rotation Axis Estimation

Given that we only have data from **five maser galaxies**, the reliability of our rotation axis estimation is limited. With such a small sample size, it's challenging to accurately triangulate the exact location and orientation of the rotation axis. The estimation is more of an educated guess based on the available data and certain assumptions:

- **Assumptions Made**:
  - We assume that the rotation axis is aligned in some way with the Cosmic Microwave Background (CMB) dipole and quadrupole axes.
  - We consider Earth's position relative to these axes to estimate its location in the universe.
  - The maser galaxies are assumed to be representative of the larger cosmic structure, which may not be the case.

- **Limitations**:
  - **Sample Size**: Five data points are insufficient for precise triangulation in three-dimensional space.
  - **Measurement Errors**: Uncertainties in distance measurements and redshift velocities can significantly affect the estimation.
  - **Simplifications in the Model**: The model assumes a simple rotational motion without accounting for complex gravitational interactions and local motions.

Therefore, while the rotation axis estimation provides a starting point for our hypothetical model, it should not be considered precise. More data points and a more sophisticated model would be required for a reliable determination.

## Earth and Maser Data

### Earth

- **Position**: ({earth_position[0]:.2f}, {earth_position[1]:.2f}, {earth_position[2]:.2f}) Mpc

### Maser Data

"""

    for maser in maser_data:
        adjusted_position = maser['adjusted_position']
        axis_distance = calculate_distance_from_axis(adjusted_position, rotation_axis)

        readme_content += f"""
#### Maser {maser['identifier']}
- **Distance from Earth**: {maser['maser_distance']:.2f} Mpc
- **Observed Redshift**: {maser['redshift_velocity'] / SPEED_OF_LIGHT:.6f}
- **Distance to rotation axis**: {axis_distance:.2f} Mpc
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


def estimate_angular_velocity(maser_data, earth_position, rotation_axis):
    """
    Estimate the angular velocity of each maser based on its position relative to Earth's position and the rotation axis.
    """
    angular_velocities = []

    for maser in maser_data:
        # Convert RA/Dec to Cartesian coordinates
        position = equatorial_to_cartesian(
            float(maser['ra']), float(maser['dec']), maser['maser_distance']
        )

        # Adjust the position relative to Earth and rotation axis
        adjusted_position = adjust_coordinates(position, earth_position, rotation_axis)

        # Calculate the distance from the rotation axis
        axis_distance = calculate_distance_from_axis(adjusted_position, rotation_axis)

        # Calculate the observed redshift
        observed_z = maser['redshift_velocity'] / SPEED_OF_LIGHT

        # Calculate tangential velocity from redshift
        tangential_velocity = SPEED_OF_LIGHT * (observed_z**2 + 2 * observed_z) / (2 + 2 * observed_z + observed_z**2)

        # Calculate the angular velocity (omega = v_t / r)
        omega = tangential_velocity / (axis_distance * 3.086e22)  # Convert Mpc to meters

        angular_velocities.append({
            'identifier': maser['identifier'],
            'omega': omega,
            'maser_distance': maser['maser_distance'],
            'axis_distance': axis_distance,
            'observed_z': observed_z
        })

    return angular_velocities


def main():
    global total_galaxies, galaxies_processed, galaxies_with_redshift, distance_min, distance_max, maser_data, galaxy_data

    # Initialize the galaxy_data list
    galaxy_data = []

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
            csv_writer.writerow(["Source_Name", "Distance_Mpc", "Redshift_Velocity", "Distance_Method", 
                                 "RA", "Dec", "Gal_Lon", "Gal_Lat", "Frame_Dragging_Effect"])

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

                if distance_mpc is None and not np.isnan(redshift_velocity):
                    # Estimate distance using Hubble's Law
                    if redshift_velocity > 1000:  # Avoid peculiar velocities
                        distance_mpc = redshift_velocity / H0
                        distance_method = 'Redshift'
                        distance_methods['Redshift'] += 1

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

    # Earth's distance from rotation axis and tangential velocity
    earth_distance_from_axis = calculate_distance_from_axis(earth_position, rotation_axis)
    earth_tangential_velocity = calculate_tangential_velocity(earth_distance_from_axis, optimal_omega)
    earth_tangential_vector = calculate_tangential_velocity_vector(earth_position, rotation_axis, earth_tangential_velocity)

    # Process each galaxy
    for galaxy in galaxy_data:
        # Skip galaxies without redshift_velocity
        if galaxy['redshift_velocity'] is None or np.isnan(galaxy['redshift_velocity']):
            continue

        # Get position
        position = equatorial_to_cartesian(float(galaxy['ra']), float(galaxy['dec']), galaxy['distance_mpc'])
        adjusted_position = adjust_coordinates(position, earth_position, rotation_axis)
        galaxy['adjusted_position'] = adjusted_position

        # Galaxy's distance from rotation axis
        galaxy_distance_from_axis = calculate_distance_from_axis(adjusted_position, rotation_axis)
        galaxy['distance_from_axis'] = galaxy_distance_from_axis

        # Galaxy's tangential velocity
        galaxy_tangential_velocity = calculate_tangential_velocity(galaxy_distance_from_axis, optimal_omega)
        galaxy['tangential_velocity'] = galaxy_tangential_velocity

        # Line of sight unit vector
        los_unit_vector = calculate_line_of_sight_unit_vector(adjusted_position, earth_position)

        # Galaxy's tangential velocity vector
        galaxy_tangential_vector = calculate_tangential_velocity_vector(adjusted_position, rotation_axis, galaxy_tangential_velocity)

        # Rotational redshift component
        z_rotation = calculate_rotational_redshift(earth_tangential_vector, galaxy_tangential_vector, los_unit_vector)
        galaxy['z_rotation'] = z_rotation

        # Observed redshift
        z_observed = galaxy['redshift_velocity'] / SPEED_OF_LIGHT
        galaxy['z_observed'] = z_observed

        # Cosmological redshift
        z_cosmological = z_observed - z_rotation
        galaxy['z_cosmological'] = z_cosmological

        # Corrected distance
        galaxy['distance_mpc_corrected'] = redshift_to_distance(z_cosmological)

    # Similarly, process maser data
    for maser in maser_data:
        # Get position
        position = equatorial_to_cartesian(float(maser['ra']), float(maser['dec']), maser['maser_distance'])
        adjusted_position = adjust_coordinates(position, earth_position, rotation_axis)
        maser['adjusted_position'] = adjusted_position

        # Maser distance from rotation axis
        maser_distance_from_axis = calculate_distance_from_axis(adjusted_position, rotation_axis)
        maser['distance_from_axis'] = maser_distance_from_axis

        # Maser tangential velocity
        maser_tangential_velocity = calculate_tangential_velocity(maser_distance_from_axis, optimal_omega)
        maser['tangential_velocity'] = maser_tangential_velocity

        # Line of sight unit vector
        los_unit_vector = calculate_line_of_sight_unit_vector(adjusted_position, earth_position)

        # Maser tangential velocity vector
        maser_tangential_vector = calculate_tangential_velocity_vector(adjusted_position, rotation_axis, maser_tangential_velocity)

        # Rotational redshift component
        z_rotation = calculate_rotational_redshift(earth_tangential_vector, maser_tangential_vector, los_unit_vector)
        maser['z_rotation'] = z_rotation

        # Observed redshift
        z_observed = maser['redshift_velocity'] / SPEED_OF_LIGHT
        maser['z_observed'] = z_observed

        # Cosmological redshift
        z_cosmological = z_observed - z_rotation
        maser['z_cosmological'] = z_cosmological

        # Corrected distance
        maser['distance_mpc_corrected'] = redshift_to_distance(z_cosmological)

    # Create visualizations
    create_visualizations(maser_data, galaxy_data, rotation_axis, earth_position)

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

    # Create README
    create_readme(maser_data, rotation_axis, median_angular_velocity, optimal_omega, 
                  optimal_axis_ra, optimal_axis_dec, cmb_dipole_axis, cmb_quadrupole_axis, earth_position)

    print("[edd_data_analysis] Analysis complete")
    create_maser_xml()

    # Export EDD star data to CSV
    export_eddstar_data_to_csv(galaxy_data)

if __name__ == "__main__":
    main()
