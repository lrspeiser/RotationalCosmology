# Standard library imports
import xml.etree.ElementTree as ET
import csv
import math
import os
import sys
from datetime import datetime
from io import BytesIO

# Scientific computing imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Astronomy imports
from astropy.coordinates import SkyCoord
import astropy.units as u

# Visualization imports
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D

# Optional system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Other utilities
import base64
import markdown

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

# Initialize logging
def init_logging():
    """Initialize logging setup with proper error handling"""
    log_file = "analysis_log.txt"
    try:
        # Ensure log directory exists
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)

        # Create or clear log file
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Analysis Log Started at {datetime.now()} ===\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write("==================================\n\n")

        def log(message):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_message = f"[{timestamp}] {message}"
            print(full_message)
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(full_message + '\n')
            except Exception as e:
                print(f"Warning: Could not write to log file: {e}")

        return log
    except Exception as e:
        print(f"Warning: Could not initialize logging: {e}")
        return print  # Fallback to simple print if logging setup fails

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

# Initialize logging at module level
log = init_logging()
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



def equatorial_to_cartesian(ra, dec, distance):
    try:
        if ra is None or dec is None or distance is None:
            raise ValueError("RA, DEC, or Distance is None")

        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        x = distance * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance * np.cos(dec_rad) * np.sin(ra_rad)
        z = distance * np.sin(dec_rad)

        return np.array([x, y, z])

    except Exception as e:
        print(f"[equatorial_to_cartesian] Error: {e}")
        return np.array([None, None, None])

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

def calculate_frame_dragging_effect(mass, radius, distance_ly):
    """
    Calculate frame-dragging angular velocity based on mass, radius, and distance.

    Parameters:
    mass (float): Mass in kg
    radius (float): Radius in meters
    distance_ly (float): Distance in light years

    Returns:
    float: Frame dragging angular velocity in rad/s
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


def process_galaxy(csv_writer, identifier, distance_mpc, redshift_velocity, distance_method, ra, dec, glon, glat):
    global galaxies_processed, galaxies_with_redshift, distance_min, distance_max, rotation_axis, earth_position, earth_tangential_vector, optimal_omega

    galaxies_processed += 1
    # Convert distance_mpc to float safely
    try:
        distance_mpc = float(distance_mpc) if distance_mpc is not None else np.nan
    except (ValueError, TypeError):
        distance_mpc = np.nan

    # Convert redshift_velocity to float safely
    try:
        redshift_velocity = float(redshift_velocity) if redshift_velocity is not None else np.nan
    except (ValueError, TypeError):
        redshift_velocity = np.nan

    # Convert coordinates to float safely
    try:
        ra = float(ra) if ra is not None else None
        dec = float(dec) if dec is not None else None
        glon = float(glon) if glon is not None else None
        glat = float(glat) if glat is not None else None
    except (ValueError, TypeError):
        ra, dec, glon, glat = None, None, None, None

    # Update statistics only if we have valid numbers
    if not np.isnan(distance_mpc):
        distance_min = min(distance_min, distance_mpc)
        distance_max = max(distance_max, distance_mpc)

    if not np.isnan(redshift_velocity):
        galaxies_with_redshift += 1

    # Calculate frame dragging effect only for maser galaxies
    if distance_method == 'DMmas' and not np.isnan(distance_mpc):
        try:
            frame_dragging_effect = calculate_frame_dragging_effect(1e40, 1e20, distance_mpc * 3.262e6)
        except (ValueError, TypeError):
            frame_dragging_effect = np.nan
    else:
        frame_dragging_effect = np.nan

    # Format values for CSV - use "NA" for missing values
    csv_row = [
        str(identifier),
        f"{distance_mpc:.3f}" if not np.isnan(distance_mpc) else "NA",
        f"{redshift_velocity:.3f}" if not np.isnan(redshift_velocity) else "NA",
        str(distance_method) if distance_method else "NA",
        f"{ra:.6f}" if ra is not None else "NA",
        f"{dec:.6f}" if dec is not None else "NA",
        f"{glon:.6f}" if glon is not None else "NA",
        f"{glat:.6f}" if glat is not None else "NA",
        f"{frame_dragging_effect:.6e}" if not np.isnan(frame_dragging_effect) else "NA"
    ]
    csv_writer.writerow(csv_row)

    # Calculate position and velocities if coordinates are available
    adjusted_position = None
    if ra is not None and dec is not None and not np.isnan(distance_mpc):
        try:
            position = equatorial_to_cartesian(ra, dec, distance_mpc)
            adjusted_position = adjust_coordinates(position, earth_position, rotation_axis)
            distance_from_axis = calculate_distance_from_axis(adjusted_position, rotation_axis)
            tangential_velocity = calculate_tangential_velocity(distance_from_axis, optimal_omega)
        except Exception as e:
            print(f"[galaxy_processing] Error calculating position for {identifier}: {e}")
            adjusted_position = None
            distance_from_axis = None
            tangential_velocity = None
    else:
        distance_from_axis = None
        tangential_velocity = None

    # Store galaxy data with new fields
    galaxy_data_entry = {
        'identifier': str(identifier),
        'distance_mpc': float(distance_mpc) if not np.isnan(distance_mpc) else None,
        'redshift_velocity': float(redshift_velocity) if not np.isnan(redshift_velocity) else None,
        'distance_method': str(distance_method) if distance_method else None,
        'ra': ra,
        'dec': dec,
        'glon': glon,
        'glat': glat,
        'frame_dragging_effect': float(frame_dragging_effect) if not np.isnan(frame_dragging_effect) else None,
        'adjusted_position': adjusted_position,
        'distance_from_axis': distance_from_axis,
        'tangential_velocity': tangential_velocity
    }

    # Add redshift calculations if available
    if adjusted_position is not None and redshift_velocity is not None and not np.isnan(redshift_velocity):
        try:
            los_unit_vector = calculate_line_of_sight_unit_vector(adjusted_position, earth_position)
            galaxy_tangential_vector = calculate_tangential_velocity_vector(
                adjusted_position, rotation_axis, tangential_velocity
            )
            z_rotation = calculate_rotational_redshift(
                earth_tangential_vector, galaxy_tangential_vector, los_unit_vector
            )
            z_observed = redshift_velocity / SPEED_OF_LIGHT
            z_cosmological = z_observed - z_rotation
            distance_mpc_corrected = redshift_to_distance(z_cosmological)

            galaxy_data_entry.update({
                'z_rotation': z_rotation,
                'z_observed': z_observed,
                'z_cosmological': z_cosmological,
                'distance_mpc_corrected': distance_mpc_corrected
            })
        except Exception as e:
            print(f"[galaxy_processing] Error calculating redshifts for {identifier}: {e}")

    galaxy_data.append(galaxy_data_entry)

    # Store maser data with new fields
    if distance_method == 'DMmas' and not np.isnan(distance_mpc):
        maser_data_entry = {
            'identifier': str(identifier),
            'maser_distance': float(distance_mpc),
            'redshift_velocity': float(redshift_velocity) if not np.isnan(redshift_velocity) else None,
            'ra': ra,
            'dec': dec,
            'glon': glon,
            'glat': glat,
            'frame_dragging_effect': float(frame_dragging_effect) if not np.isnan(frame_dragging_effect) else None,
            'adjusted_position': adjusted_position,
            'distance_from_axis': distance_from_axis,
            'tangential_velocity': tangential_velocity
        }

        # Add redshift calculations if available
        if 'z_rotation' in galaxy_data_entry:
            maser_data_entry.update({
                'z_rotation': galaxy_data_entry['z_rotation'],
                'z_observed': galaxy_data_entry['z_observed'],
                'z_cosmological': galaxy_data_entry['z_cosmological'],
                'distance_mpc_corrected': galaxy_data_entry['distance_mpc_corrected']
            })

        maser_data.append(maser_data_entry)

def create_visualizations(maser_data, galaxy_data, rotation_axis, earth_position):
    # Create improved 3D visualization
    plot_3d_file = create_improved_3d_visualization(maser_data, galaxy_data, rotation_axis, earth_position)
    # Create 2D projection
    plot_2d_file = create_2d_projection(maser_data, galaxy_data, rotation_axis, earth_position)
    
def create_improved_3d_visualization(maser_data, galaxy_data, rotation_axis, earth_position):
    fig = go.Figure()
    scale_factor = 1e-6  # Scale Mpc values for better visualization

    # Plot Earth's position
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

    # Initialize lists for positions
    maser_positions = []
    galaxy_positions = []
    supernova_positions = []

    # Collect maser positions
    for maser in maser_data:
        if maser.get('adjusted_position') is not None:
            maser_positions.append(maser['adjusted_position'] * scale_factor)

    # Collect galaxy positions
    for galaxy in galaxy_data:
        if galaxy.get('adjusted_position') is not None:
            adjusted_position = galaxy['adjusted_position'] * scale_factor
            if galaxy.get('distance_method') in ['SNIa', 'SNII']:
                supernova_positions.append(adjusted_position)
            else:
                galaxy_positions.append(adjusted_position)

    # Plot masers if we have any
    if len(maser_positions) > 0:
        maser_positions = np.array(maser_positions)
        fig.add_trace(go.Scatter3d(
            x=maser_positions[:, 0],
            y=maser_positions[:, 1],
            z=maser_positions[:, 2],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Masers'
        ))

    # Plot supernovae if we have any
    if len(supernova_positions) > 0:
        supernova_positions = np.array(supernova_positions)
        fig.add_trace(go.Scatter3d(
            x=supernova_positions[:, 0],
            y=supernova_positions[:, 1],
            z=supernova_positions[:, 2],
            mode='markers',
            marker=dict(size=4, color='orange'),
            name='Supernovae'
        ))

    # Plot galaxies if we have any
    if len(galaxy_positions) > 0:
        galaxy_positions = np.array(galaxy_positions)
        fig.add_trace(go.Scatter3d(
            x=galaxy_positions[:, 0],
            y=galaxy_positions[:, 1],
            z=galaxy_positions[:, 2],
            mode='markers',
            marker=dict(size=2, color='gray'),
            name='Galaxies'
        ))

    # Calculate axis length based on available data
    all_max_ranges = [abs(scaled_earth_position).max()]  # Start with Earth's position

    if len(maser_positions) > 0:
        all_max_ranges.append(abs(maser_positions).max())
    if len(galaxy_positions) > 0:
        all_max_ranges.append(abs(galaxy_positions).max())
    if len(supernova_positions) > 0:
        all_max_ranges.append(abs(supernova_positions).max())

    max_range = max(all_max_ranges)
    axis_length = max_range * 1.5

    # Plot rotation axis
    fig.add_trace(go.Scatter3d(
        x=[-rotation_axis[0] * axis_length, rotation_axis[0] * axis_length],
        y=[-rotation_axis[1] * axis_length, rotation_axis[1] * axis_length],
        z=[-rotation_axis[2] * axis_length, rotation_axis[2] * axis_length],
        mode='lines',
        line=dict(color='green', width=5),
        name='Rotation Axis'
    ))

    # Configure layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (Scaled Mpc)',
            yaxis_title='Y (Scaled Mpc)',
            zaxis_title='Z (Scaled Mpc)',
            aspectmode='data'
        ),
        title="3D Visualization of Universe",
        showlegend=True
    )

    # Save visualization
    plot_file = "improved_universe_visualization.html"
    fig.write_html(plot_file)
    print(f"[visualization] 3D visualization saved as '{plot_file}'.")
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
        if maser.get('adjusted_position') is not None:
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
        if galaxy.get('distance_method') == 'DMmas':
            continue
        if galaxy.get('adjusted_position') is not None:
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

def extract_galaxy_data(columns):
    """Helper function to extract galaxy data from XML columns"""
    pgc_id = columns[0].text
    redshift_velocity = columns[3].text
    ra = columns[22].text
    dec = columns[23].text
    glon = columns[24].text
    glat = columns[25].text

    try:
        redshift_velocity = float(redshift_velocity)
    except (ValueError, TypeError):
        redshift_velocity = np.nan

    distance_mpc = None
    distance_method = None

    # Try to get distance from various methods
    for method, index in [
        ('DMsnIa', 6), ('DMtf', 8), ('DMfp', 10), ('DMsbf', 12),
        ('DMsnII', 14), ('DMtrgb', 16), ('DMcep', 18), ('DMmas', 20)
    ]:
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
        if redshift_velocity > 1000:
            distance_mpc = redshift_velocity / H0
            distance_method = 'Redshift'
            distance_methods['Redshift'] += 1

    return pgc_id, distance_mpc, redshift_velocity, distance_method, ra, dec, glon, glat


def create_maser_xml():
    """
    Create XML file for maser data with proper type handling and error checking.
    """
    log("[edd_data_analysis] Creating XML file for maser data")

    try:
        root = ET.Element("MaserData")

        # Add metadata
        metadata = ET.SubElement(root, "Metadata")
        ET.SubElement(metadata, "CreationDate").text = datetime.now().isoformat()
        ET.SubElement(metadata, "TotalMasers").text = str(len(maser_data))

        # Create masers container
        masers = ET.SubElement(root, "Masers")

        for maser in maser_data:
            try:
                maser_element = ET.SubElement(masers, "Maser")

                # Add all maser properties, converting each to string safely
                properties = {
                    "Identifier": str(maser['identifier']),
                    "MaserDistanceMpc": f"{float(maser['maser_distance']):.6f}",
                    "RedshiftVelocity": f"{float(maser['redshift_velocity']):.6f}" if maser['redshift_velocity'] is not None else "NA",
                    "RA": f"{float(maser['ra']):.6f}" if maser['ra'] is not None else "NA",
                    "Dec": f"{float(maser['dec']):.6f}" if maser['dec'] is not None else "NA",
                    "GalacticLongitude": f"{float(maser['glon']):.6f}" if maser['glon'] is not None else "NA",
                    "GalacticLatitude": f"{float(maser['glat']):.6f}" if maser['glat'] is not None else "NA",
                    "FrameDraggingEffect": f"{float(maser['frame_dragging_effect']):.6e}" if maser['frame_dragging_effect'] is not None else "NA"
                }

                # Optional properties if available
                if 'z_rotation' in maser:
                    properties.update({
                        "RotationalRedshift": f"{float(maser['z_rotation']):.6f}",
                        "ObservedRedshift": f"{float(maser['z_observed']):.6f}",
                        "CosmologicalRedshift": f"{float(maser['z_cosmological']):.6f}",
                        "CorrectedDistanceMpc": f"{float(maser['distance_mpc_corrected']):.6f}"
                    })

                # Add position data if available
                if maser.get('adjusted_position') is not None:
                    position = maser['adjusted_position']
                    properties.update({
                        "AdjustedX": f"{float(position[0]):.6f}",
                        "AdjustedY": f"{float(position[1]):.6f}",
                        "AdjustedZ": f"{float(position[2]):.6f}"
                    })

                # Create XML elements for each property
                for key, value in properties.items():
                    ET.SubElement(maser_element, key).text = value

            except (ValueError, TypeError, KeyError) as e:
                log(f"[create_maser_xml] Warning: Error processing maser {maser.get('identifier', 'unknown')}: {str(e)}")
                continue

        # Create XML string with proper formatting
        xml_str = ET.tostring(root, encoding='unicode', method='xml')

        # Write to file with proper XML declaration and encoding
        xml_file_path = "maser_edd.xml"
        with open(xml_file_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(xml_str)

        log(f"[create_maser_xml] Successfully created XML file: {xml_file_path}")
        log(f"[create_maser_xml] Wrote data for {len(maser_data)} masers")

    except Exception as e:
        log(f"[create_maser_xml] Error creating XML file: {str(e)}")
        raise

def validate_maser_xml(xml_file_path="maser_edd.xml"):
    """
    Validate the created XML file.

    Returns:
    --------
    bool
        True if validation successful, False otherwise
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Validate basic structure
        if root.tag != "MaserData":
            log("[validate_maser_xml] Error: Invalid root element")
            return False

        # Check if we have all masers
        masers = root.findall(".//Maser")
        if len(masers) != len(maser_data):
            log(f"[validate_maser_xml] Warning: Mismatch in maser count. XML: {len(masers)}, Data: {len(maser_data)}")
            return False

        log(f"[validate_maser_xml] XML validation successful: {xml_file_path}")
        return True

    except Exception as e:
        log(f"[validate_maser_xml] Error validating XML: {str(e)}")
        return False

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

def process_supernova_csv(file_path):
    """
    Process supernova data from CSV file with robust error handling and data validation.
    Returns a list of validated supernova data dictionaries.
    """
    supernova_data = []
    required_columns = {'Name', 'D (Mpc)', 'Method', 'RA', 'Dec'}

    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            # First pass: detect the dialect and check encoding
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)

            reader = csv.DictReader(csvfile, dialect=dialect)
            headers = reader.fieldnames

            if not headers:
                print(f"[supernova_data_processing] Error: No headers found in {file_path}")
                return []

            # Create column mapping for various possible header names
            header_mapping = {
                'Name': ['Name', 'GALAXY', 'ID', 'FRN', 'SN'],
                'D (Mpc)': ['D (Mpc)', 'DIST', 'DISTANCE', 'D', 'Distance'],
                'Method': ['Method', 'METH', 'TYPE', 'Distance Method'],
                'RA': ['RA', 'R.A.', 'RIGHT ASCENSION', 'Ra'],
                'Dec': ['Dec', 'DEC', 'DECLINATION', 'De']
            }

            # Find actual column names in file
            column_map = {}
            for required_col, possible_names in header_mapping.items():
                for header in headers:
                    if any(name.lower() in header.lower() for name in possible_names):
                        column_map[required_col] = header
                        break

            missing_columns = required_columns - set(column_map.keys())
            if missing_columns:
                print(f"[supernova_data_processing] Warning: Missing required columns: {missing_columns}")
                print(f"[supernova_data_processing] Available columns: {headers}")
                return []

            # Reset file pointer for data reading
            csvfile.seek(0)
            next(reader)  # Skip header row

            # Process each row with validation
            for row_num, row in enumerate(reader, start=2):
                try:
                    # Extract and validate each field
                    identifier = row[column_map['Name']].strip()
                    if not identifier:
                        continue

                    try:
                        distance_mpc = float(row[column_map['D (Mpc)']].strip())
                        if distance_mpc <= 0:
                            print(f"[supernova_data_processing] Warning: Invalid distance in row {row_num}")
                            continue
                    except (ValueError, AttributeError):
                        print(f"[supernova_data_processing] Warning: Invalid distance format in row {row_num}")
                        continue

                    try:
                        ra = float(row[column_map['RA']].strip())
                        dec = float(row[column_map['Dec']].strip())
                        if not (-360 <= ra <= 360 and -90 <= dec <= 90):
                            print(f"[supernova_data_processing] Warning: Invalid coordinates in row {row_num}")
                            continue
                    except (ValueError, AttributeError):
                        print(f"[supernova_data_processing] Warning: Invalid coordinate format in row {row_num}")
                        continue

                    distance_method = row[column_map['Method']].strip()
                    if not distance_method:
                        distance_method = 'SNIa'  # Default method for supernovae

                    # Create validated supernova entry
                    supernova_data.append({
                        'identifier': identifier,
                        'distance_mpc': distance_mpc,
                        'distance_method': distance_method,
                        'ra': ra,
                        'dec': dec,
                        'redshift_velocity': None,  # Initialize optional fields
                        'glon': None,
                        'glat': None,
                        'frame_dragging_effect': None,
                        'adjusted_position': None,
                        'distance_from_axis': None,
                        'tangential_velocity': None
                    })

                except Exception as e:
                    print(f"[supernova_data_processing] Error processing row {row_num}: {str(e)}")
                    continue

            print(f"[supernova_data_processing] Successfully processed {len(supernova_data)} valid supernova entries")
            return supernova_data

    except Exception as e:
        print(f"[supernova_data_processing] Error reading file {file_path}: {str(e)}")
        return []

def adjust_coordinates(position, earth_position, rotation_axis):
    """
    Adjust coordinates with proper error handling.
    """
    try:
        if position is None or earth_position is None or rotation_axis is None:
            raise ValueError("Invalid input: position, earth_position, or rotation_axis is None")

        relative_position = position - earth_position
        axial_component = np.dot(relative_position, rotation_axis) * rotation_axis
        perpendicular_component = relative_position - axial_component
        return perpendicular_component + axial_component

    except Exception as e:
        print(f"[adjust_coordinates] Error: {str(e)}")
        return None

def process_galaxy_positions(galaxy_data, earth_position, rotation_axis, optimal_omega):
    """
    Process galaxy positions with error handling and validation.
    """
    processed_galaxies = []

    for galaxy in galaxy_data:
        try:
            if galaxy['ra'] is None or galaxy['dec'] is None or galaxy['distance_mpc'] is None:
                continue

            position = equatorial_to_cartesian(float(galaxy['ra']), float(galaxy['dec']), galaxy['distance_mpc'])
            if position is None:
                continue

            adjusted_position = adjust_coordinates(position, earth_position, rotation_axis)
            if adjusted_position is None:
                continue

            galaxy['adjusted_position'] = adjusted_position
            galaxy_distance_from_axis = calculate_distance_from_axis(adjusted_position, rotation_axis)
            galaxy['distance_from_axis'] = galaxy_distance_from_axis

            galaxy_tangential_velocity = calculate_tangential_velocity(galaxy_distance_from_axis, optimal_omega)
            galaxy['tangential_velocity'] = galaxy_tangential_velocity

            if galaxy['redshift_velocity'] is not None:
                los_unit_vector = calculate_line_of_sight_unit_vector(adjusted_position, earth_position)
                galaxy_tangential_vector = calculate_tangential_velocity_vector(
                    adjusted_position, rotation_axis, galaxy_tangential_velocity
                )

                z_rotation = calculate_rotational_redshift(
                    earth_tangential_vector, galaxy_tangential_vector, los_unit_vector
                )
                galaxy['z_rotation'] = z_rotation

                z_observed = galaxy['redshift_velocity'] / SPEED_OF_LIGHT
                galaxy['z_observed'] = z_observed

                z_cosmological = z_observed - z_rotation
                galaxy['z_cosmological'] = z_cosmological

                galaxy['distance_mpc_corrected'] = redshift_to_distance(z_cosmological)
            else:
                galaxy['z_observed'] = None
                galaxy['z_rotation'] = None
                galaxy['z_cosmological'] = None
                galaxy['distance_mpc_corrected'] = galaxy['distance_mpc']

            processed_galaxies.append(galaxy)

        except Exception as e:
            print(f"[process_galaxy_positions] Error processing galaxy {galaxy.get('identifier', 'unknown')}: {str(e)}")
            continue

    return processed_galaxies


def setup_logging(log_file="analysis_log.txt"):
    """
    Set up logging to both file and console with proper timestamp formatting.

    Parameters:
    -----------
    log_file : str
        Path to the log file (default: "analysis_log.txt")

    Returns:
    --------
    callable
        Logging function that writes to both console and file
    """
    def log(message):
        """
        Log a message to both console and file with timestamp.

        Parameters:
        -----------
        message : str
            Message to be logged
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"

        # Print to console
        print(full_message)

        # Write to file
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(full_message + '\n')
        except Exception as e:
            print(f"Error writing to log file: {str(e)}")

    # Create or clear the log file with error handling
    try:
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Initialize the log file
        with open(log_file, 'w', encoding='utf-8') as f:
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"=== Analysis Log Started at {start_time} ===\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write("==================================\n\n")

    except Exception as e:
        print(f"Error initializing log file: {str(e)}")
        # Create a fallback logging function that only prints to console
        return print

    return log

def get_memory_usage():
    """
    Get current memory usage of the process.

    Returns:
    --------
    str
        Formatted string with memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.1f} MB"
    except ImportError:
        return "Memory usage unavailable"

def log_system_info(log_func):
    """
    Log system information at the start of the analysis.

    Parameters:
    -----------
    log_func : callable
        Logging function to use
    """
    try:
        import platform
        log_func("System Information:")
        log_func(f"  - OS: {platform.system()} {platform.release()}")
        log_func(f"  - Python: {sys.version.split()[0]}")
        log_func(f"  - Initial Memory Usage: {get_memory_usage()}")
        log_func("  - Working Directory: " + os.getcwd())
        log_func("----------------------------------")
    except Exception as e:
        log_func(f"Error getting system info: {str(e)}")

def load_desi_data(file_path='high_confidence_matches.csv', log_func=print):
    """
    Load and process DESI data with comprehensive error handling.

    Parameters:
    -----------
    file_path : str
        Path to the DESI data CSV file
    log_func : callable
        Function to use for logging (defaults to print)

    Returns:
    --------
    pandas.DataFrame or None
        Processed DESI data or None if loading fails
    """
    try:
        log_func(f"[load_desi_data] Loading {file_path}...")

        # Check if file exists
        if not os.path.exists(file_path):
            log_func(f"[load_desi_data] Error: File {file_path} not found")
            return pd.DataFrame()

        # Try reading the file
        try:
            desi_df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            log_func("[load_desi_data] Error: File is empty")
            return pd.DataFrame()
        except pd.errors.ParserError:
            log_func("[load_desi_data] Error: Unable to parse CSV file")
            return pd.DataFrame()

        # Check required columns
        required_columns = {'RA', 'DE', 'Vcmb'}
        missing_columns = required_columns - set(desi_df.columns)
        if missing_columns:
            log_func(f"[load_desi_data] Error: Missing required columns: {missing_columns}")
            log_func(f"[load_desi_data] Available columns: {list(desi_df.columns)}")
            return pd.DataFrame()

        # Remove rows with missing values in required columns
        initial_rows = len(desi_df)
        desi_df = desi_df.dropna(subset=list(required_columns))
        dropped_rows = initial_rows - len(desi_df)
        if dropped_rows > 0:
            log_func(f"[load_desi_data] Dropped {dropped_rows} rows with missing values")

        # Validate coordinate ranges
        valid_mask = (
            (desi_df['RA'].between(0, 360)) & 
            (desi_df['DE'].between(-90, 90)) &
            (desi_df['Vcmb'].notna())
        )
        desi_df = desi_df[valid_mask]

        # Calculate approximate distances
        H0 = 70  # Hubble constant (km/s/Mpc)
        desi_df['approx_distance'] = desi_df['Vcmb'] / H0

        # Convert coordinates to Cartesian
        def equatorial_to_cartesian_vectorized(ra, dec, distance):
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            x = distance * np.cos(dec_rad) * np.cos(ra_rad)
            y = distance * np.cos(dec_rad) * np.sin(ra_rad)
            z = distance * np.sin(dec_rad)
            return pd.DataFrame({
                'x': x,
                'y': y,
                'z': z
            })

        cartesian_coords = equatorial_to_cartesian_vectorized(
            desi_df['RA'], 
            desi_df['DE'], 
            desi_df['approx_distance']
        )

        # Add Cartesian coordinates to dataframe
        desi_df['x'] = cartesian_coords['x']
        desi_df['y'] = cartesian_coords['y']
        desi_df['z'] = cartesian_coords['z']

        log_func(f"[load_desi_data] Successfully loaded {len(desi_df)} DESI objects")
        return desi_df

    except Exception as e:
        log_func(f"[load_desi_data] Unexpected error: {str(e)}")
        return pd.DataFrame()

def export_adjusted_desi_data(desi_df, output_file='adjusted_desi_data.csv', log_func=print):
    """
    Export processed DESI data with error handling.
    """
    try:
        if desi_df.empty:
            log_func("[export_adjusted_desi_data] No data to export")
            return False

        desi_df.to_csv(output_file, index=False)
        log_func(f"[export_adjusted_desi_data] Successfully exported {len(desi_df)} objects to {output_file}")
        return True

    except Exception as e:
        log_func(f"[export_adjusted_desi_data] Error exporting data: {str(e)}")
        return False

def validate_and_process_coordinates(df, log_func=print):
    """
    Validate and process astronomical coordinates.
    """
    try:
        # Validate RA/Dec ranges
        invalid_coords = (
            (df['RA'] < 0) | (df['RA'] > 360) |
            (df['DE'] < -90) | (df['DE'] > 90)
        )

        if invalid_coords.any():
            n_invalid = invalid_coords.sum()
            log_func(f"[validate_coordinates] Found {n_invalid} invalid coordinate pairs")
            df = df[~invalid_coords]

        # Check for reasonable distances
        if 'approx_distance' in df.columns:
            unreasonable_dist = (df['approx_distance'] < 0) | (df['approx_distance'] > 15000)
            if unreasonable_dist.any():
                n_invalid_dist = unreasonable_dist.sum()
                log_func(f"[validate_coordinates] Found {n_invalid_dist} unreasonable distances")
                df = df[~unreasonable_dist]

        return df

    except Exception as e:
        log_func(f"[validate_coordinates] Error during validation: {str(e)}")
        return df


def create_readme(maser_data, rotation_axis, median_angular_velocity, optimal_omega, 
                  optimal_axis_ra, optimal_axis_dec, cmb_dipole_axis, 
                  cmb_quadrupole_axis, earth_position):
    log("[create_readme] Starting README generation...")
    try:
        readme_content = f"""
# Rotating Universe Model Analysis

## Overview

This document summarizes the analysis of a hypothetical rotating universe model based on maser data and CMB observations. Instead of assuming the universe is expanding since the big bang, it provides a hypothetical alternative where the universe can be significantly older, and the illusion of expansion comes from the frame-dragging gravity effects of a Godel-inspired rotating universe. In addition, we explore the possibility of both light and gravity circumnavigating the universe repeatedly, giving an alternative explanation to CMB uniformity and the extra gravity we detect but can't attribute to observable objects. 

To start, we use the axis of evil calculations indicating a direction to the CMB. We then apply known celestial object distances (that don't rely on redshift) to determine how much frame dragging is necessary to generate the observed redshift. We also attempt to estimate our distance from a central axis and chart what this rotating universe looks like. Additional data from sets like DESI will be used to overlay large-scale structures in hopes of explaining their formation through this model.

## Analysis Process

1. **Data Collection and Processing**
   - Processed {total_galaxies} galaxies from the EDD database
   - Identified {len(maser_data)} maser galaxies
   - Processed supernova data for additional reference points
   - Created initial combined galaxy dataset

2. **Parameter Calculation**
   - Calculated CMB axes from dipole and quadrupole measurements
   - Determined universe rotation axis from CMB data
   - Computed Earth's position relative to the rotation axis
   - Optimized angular velocity parameters

3. **Velocity Analysis**
   - Calculated tangential velocities for all objects
   - Computed rotational redshifts
   - Adjusted cosmological redshifts for rotation effects
   - Estimated corrected distances

4. **Output Generation**
   - Created comprehensive galaxy CSV dataset
   - Generated maser-specific XML data
   - Produced interactive 2D and 3D visualizations
   - Created rotating visualization model

## General Statistics

- **Total galaxies processed**: {total_galaxies}
- **Galaxies with maser distances**: {len(maser_data)}
- **Galaxies with supernova distances**: {len(supernova_data)}
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

Given that we only have data from **five maser galaxies**, the reliability of our rotation axis estimation is limited. With such a small sample size, it's challenging to accurately triangulate the exact location and orientation of the rotation axis. The estimation is an educated guess based on the available data and certain assumptions:

- **Assumptions Made**:
  - We assume that the rotation axis aligns with the Cosmic Microwave Background (CMB) dipole and quadrupole axes.
  - We consider Earth's position relative to these axes to estimate its location in the universe.
  - The maser galaxies are assumed to represent the larger cosmic structure, which may not be the case.

- **Limitations**:
  - **Sample Size**: Five data points are insufficient for precise triangulation.
  - **Measurement Errors**: Uncertainties in distance measurements and redshift velocities affect the estimation.
  - **Simplifications in the Model**: The model assumes simple rotational motion, ignoring complex gravitational interactions.

With **supernova data**, we now have more data points for axis estimation. Supernovae provide independent distance measurements not reliant on redshift.

- **Improvements**:
  - **Increased Data Points**: Adding supernovae increases the dataset size.
  - **Independent Distance Measurements**: Supernova distances reduce reliance on redshift.

- **Remaining Limitations**:
  - **Data Quality**: Supernova measurements may have biases.
  - **Model Assumptions**: The model assumes simple rotational motion.

## Earth and Maser Data

### Earth

- **Position**: ({earth_position[0]:.2f}, {earth_position[1]:.2f}, {earth_position[2]:.2f}) Mpc

### Maser Data
"""

        for maser in maser_data:
            adjusted_position = maser['adjusted_position']
            axis_distance = calculate_distance_from_axis(adjusted_position, rotation_axis)
            print(f"[maser_data_processing] Assigned adjusted_position for maser {maser['identifier']}.")
            readme_content += f"""
#### Maser {maser['identifier']}
- **Distance from Earth**: {maser['maser_distance']:.2f} Mpc
- **Observed Redshift**: {maser['redshift_velocity'] / SPEED_OF_LIGHT:.6f}
- **Distance to rotation axis**: {axis_distance:.2f} Mpc
- **Adjusted Coordinates**: ({adjusted_position[0]:.2f}, {adjusted_position[1]:.2f}, {adjusted_position[2]:.2f}) Mpc
"""

        readme_content += """
### Supernova Data
"""

        for supernova in supernova_data:
            adjusted_position = supernova['adjusted_position']
            axis_distance = calculate_distance_from_axis(adjusted_position, rotation_axis)
            readme_content += f"""
#### Supernova {supernova['identifier']}
- **Distance from Earth**: {supernova['distance_mpc']:.2f} Mpc
- **Distance to rotation axis**: {axis_distance:.2f} Mpc
- **Adjusted Coordinates**: ({adjusted_position[0]:.2f}, {adjusted_position[1]:.2f}, {adjusted_position[2]:.2f}) Mpc
"""

        readme_content += """
## Notes

- The rotation axis is at the center of the universe (0, 0, 0).
- Earth's position is based on the CMB dipole and estimated universe rotation.
- Maser coordinates are adjusted relative to the rotation axis.
- This model is speculative and not reflective of current scientific understanding.
"""

        with open('README.md', 'w') as readme_file:
            readme_file.write(readme_content)

        log(f"[create_readme] README.md created with {len(readme_content)} bytes")
        return True
    except Exception as e:
        log(f"[create_readme] Error creating README: {str(e)}")
        return False


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


def export_eddstar_data_to_csv(galaxy_data, output_file="eddstar_data.csv"):
    """
    Export EDD star data with adjusted coordinates and velocities.
    """
    try:
        # Filter out galaxies with missing data
        valid_galaxies = [g for g in galaxy_data if g.get('adjusted_position') is not None]

        with open(output_file, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write header
            csv_writer.writerow([
                "Identifier", "Distance_Mpc", "RA", "Dec", 
                "Position_X", "Position_Y", "Position_Z",
                "Tangential_Velocity", "Observed_Redshift", 
                "Rotational_Redshift", "Cosmological_Redshift", 
                "Corrected_Distance_Mpc"
            ])

            # Write data rows
            for galaxy in valid_galaxies:
                try:
                    pos = galaxy['adjusted_position']
                    csv_writer.writerow([
                        galaxy['identifier'],
                        f"{galaxy['distance_mpc']:.6f}",
                        f"{galaxy['ra']:.6f}" if galaxy['ra'] is not None else "NA",
                        f"{galaxy['dec']:.6f}" if galaxy['dec'] is not None else "NA",
                        f"{pos[0]:.6f}",
                        f"{pos[1]:.6f}",
                        f"{pos[2]:.6f}",
                        f"{galaxy.get('tangential_velocity', 'NA')}",
                        f"{galaxy.get('z_observed', 'NA')}",
                        f"{galaxy.get('z_rotation', 'NA')}",
                        f"{galaxy.get('z_cosmological', 'NA')}",
                        f"{galaxy.get('distance_mpc_corrected', 'NA')}"
                    ])
                except Exception as e:
                    print(f"[export_eddstar_data] Error writing galaxy {galaxy.get('identifier', 'unknown')}: {e}")
                    continue

        print(f"[export_eddstar_data] Successfully exported {len(valid_galaxies)} galaxies to {output_file}")
        return True

    except Exception as e:
        print(f"[export_eddstar_data] Error: {str(e)}")
        return False

def create_visualizations_with_desi(maser_data, galaxy_data, desi_df, rotation_axis, earth_position):
    """Create visualizations with overlaid DESI data."""
    fig = go.Figure()
    scale_factor = 1e-6  # Scale Mpc values for better visualization

    # Plot DESI objects using the correct column names
    if not desi_df.empty and all(col in desi_df.columns for col in ['x', 'y', 'z']):
        fig.add_trace(go.Scatter3d(
            x=desi_df['x'] * scale_factor,
            y=desi_df['y'] * scale_factor,
            z=desi_df['z'] * scale_factor,
            mode='markers',
            marker=dict(size=3, color='purple', opacity=0.6),
            name='DESI Objects'
        ))

    # Plot Earth's position
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

    # Plot masers
    maser_positions = [m['adjusted_position'] * scale_factor for m in maser_data if m.get('adjusted_position') is not None]
    if maser_positions:
        positions = np.array(maser_positions)
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Masers'
        ))

    # Plot rotation axis
    axis_length = np.max(np.abs(scaled_earth_position)) * 2
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
        title="3D Visualization with DESI Overlay",
        scene=dict(
            xaxis_title='X (Scaled Mpc)',
            yaxis_title='Y (Scaled Mpc)',
            zaxis_title='Z (Scaled Mpc)',
            aspectmode='data'
        ),
        showlegend=True
    )

    # Save visualization
    try:
        fig.write_html("universe_with_desi_overlay.html")
        print("[visualization] Saved 3D visualization with DESI overlay.")
    except Exception as e:
        print(f"[visualization] Error saving DESI visualization: {str(e)}")



def main():
    global total_galaxies, galaxies_processed, galaxies_with_redshift, distance_min, distance_max
    global maser_data, galaxy_data, rotation_axis, earth_position, earth_tangential_vector, optimal_omega
    global supernova_data  

    # Set up logging
    log = setup_logging()
    log("[main] Starting universe rotation analysis")
    log_system_info(log)

    # Initialize data lists and statistics
    galaxy_data = []
    maser_data = []
    total_galaxies = 0
    galaxies_processed = 0
    galaxies_with_redshift = 0
    distance_min = float('inf')
    distance_max = float('-inf')

    # Calculate universal parameters
    log("[main] Calculating universal parameters...")
    rotation_axis = calculate_universe_rotation_axis()
    cmb_dipole_axis, cmb_quadrupole_axis = calculate_cmb_axes()
    earth_position = calculate_earth_position(rotation_axis, cmb_dipole_axis)
    log(f"[main] Rotation axis calculated: [{rotation_axis[0]:.6f}, {rotation_axis[1]:.6f}, {rotation_axis[2]:.6f}]")
    log(f"[main] Earth position determined: [{earth_position[0]:.6f}, {earth_position[1]:.6f}, {earth_position[2]:.6f}]")

    # Initialize optimization parameters
    log("[main] Starting optimization for angular velocity...")
    initial_guess = [1e-18, 0, 0]
    result = minimize(objective_function, initial_guess, method='Nelder-Mead')
    optimal_omega, optimal_axis_ra, optimal_axis_dec = result.x
    log(f"[main] Optimization complete - Optimal angular velocity: {optimal_omega:.2e} rad/s")

    # Calculate Earth's parameters
    earth_distance_from_axis = calculate_distance_from_axis(earth_position, rotation_axis)
    earth_tangential_velocity = calculate_tangential_velocity(earth_distance_from_axis, optimal_omega)
    earth_tangential_vector = calculate_tangential_velocity_vector(
        earth_position, rotation_axis, earth_tangential_velocity
    )
    log(f"[main] Earth parameters calculated:")
    log(f"  - Distance from rotation axis: {earth_distance_from_axis:.2f} Mpc")
    log(f"  - Tangential velocity: {earth_tangential_velocity:.2e} m/s")

    # Process the EDD data
    log("[main] Processing EDD data...")
    edd_rows = process_xml_file("eddtable.xml", {'votable': 'http://www.ivoa.net/xml/VOTable/v1.2'})
    if not edd_rows:
        log("[main] Error: No EDD data found")
        return

    csv_file_path = "combined_galaxy_data.csv"
    log("[main] Writing galaxy data to CSV file...")
    try:
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Source_Name", "Distance_Mpc", "Redshift_Velocity", "Distance_Method", 
                               "RA", "Dec", "Gal_Lon", "Gal_Lat", "Frame_Dragging_Effect"])
            for row in edd_rows:
                total_galaxies += 1
                columns = row.findall("votable:TD", {'votable': 'http://www.ivoa.net/xml/VOTable/v1.2'})
                if len(columns) < 28:
                    continue
                process_galaxy(csv_writer, *extract_galaxy_data(columns))

                if total_galaxies % 1000 == 0:
                    log(f"[main] Processed {total_galaxies} galaxies...")
    except Exception as e:
        log(f"[main] Error processing EDD data: {e}")
        return

    log(f"[main] Galaxy statistics:")
    log(f"  - Total galaxies processed: {total_galaxies}")
    log(f"  - Galaxies with redshift: {galaxies_with_redshift}")
    log(f"  - Distance range: {distance_min:.2f} to {distance_max:.2f} Mpc")

    # Process Supernova data
    log("[main] Processing Supernova data...")
    supernova_data = process_supernova_csv("nedd.csv")
    if supernova_data:
        total_galaxies += len(supernova_data)
        galaxy_data.extend(supernova_data)
        log(f"[main] Added {len(supernova_data)} supernova objects")
    else:
        # Initialize empty supernova_data if none was loaded
        supernova_data = []

    # Process galaxy positions
    log("[main] Processing galaxy positions...")
    processed_galaxies = process_galaxy_positions(
        galaxy_data, earth_position, rotation_axis, optimal_omega
    )
    galaxy_data = processed_galaxies
    log(f"[main] Successfully processed {len(processed_galaxies)} galaxy positions")

    # Calculate velocities and redshifts
    log("[main] Calculating velocities and redshifts...")
    galaxies_with_complete_data = 0
    for galaxy in galaxy_data:
        if galaxy['ra'] is not None and galaxy['dec'] is not None:
            position = equatorial_to_cartesian(float(galaxy['ra']), float(galaxy['dec']), galaxy['distance_mpc'])
            adjusted_position = adjust_coordinates(position, earth_position, rotation_axis)
            galaxy['adjusted_position'] = adjusted_position

            if adjusted_position is not None:
                galaxies_with_complete_data += 1
                galaxy_distance_from_axis = calculate_distance_from_axis(adjusted_position, rotation_axis)
                galaxy['distance_from_axis'] = galaxy_distance_from_axis
                galaxy_tangential_velocity = calculate_tangential_velocity(galaxy_distance_from_axis, optimal_omega)
                galaxy['tangential_velocity'] = galaxy_tangential_velocity

                if galaxy['redshift_velocity'] is not None:
                    los_unit_vector = calculate_line_of_sight_unit_vector(adjusted_position, earth_position)
                    galaxy_tangential_vector = calculate_tangential_velocity_vector(
                        adjusted_position, rotation_axis, galaxy_tangential_velocity
                    )
                    z_rotation = calculate_rotational_redshift(
                        earth_tangential_vector, galaxy_tangential_vector, los_unit_vector
                    )
                    galaxy['z_rotation'] = z_rotation
                    z_observed = galaxy['redshift_velocity'] / SPEED_OF_LIGHT
                    galaxy['z_observed'] = z_observed
                    z_cosmological = z_observed - z_rotation
                    galaxy['z_cosmological'] = z_cosmological
                    galaxy['distance_mpc_corrected'] = redshift_to_distance(z_cosmological)
                else:
                    galaxy['z_observed'] = None
                    galaxy['z_rotation'] = None
                    galaxy['z_cosmological'] = None
                    galaxy['distance_mpc_corrected'] = galaxy['distance_mpc']

    log(f"[main] Velocity calculations complete:")
    log(f"  - Galaxies with complete data: {galaxies_with_complete_data}")

    # Calculate median angular velocity
    log("[main] Calculating angular velocities...")
    angular_velocities = estimate_angular_velocity(maser_data, earth_position, rotation_axis)
    median_angular_velocity = np.median([v['omega'] for v in angular_velocities])
    log(f"[main] Median angular velocity: {median_angular_velocity:.2e} rad/s")

    # Load and process DESI data
    log("[main] Processing DESI data...")
    desi_df = load_desi_data(log_func=log)
    if not desi_df.empty:
        export_adjusted_desi_data(desi_df, log_func=log)
        log(f"[main] Processed {len(desi_df)} DESI objects")


    
    log("[main] Generating output files...")
    try:
        # Create XML files
        log("[main] Creating maser XML file...")
        create_maser_xml()
        validate_maser_xml()

        # Export galaxy data
        log("[main] Exporting galaxy data to CSV...")
        if export_eddstar_data_to_csv(galaxy_data, output_file="eddstar_data.csv"):
            log("[main] Successfully exported galaxy data to CSV")
        else:
            log("[main] Warning: Error exporting galaxy data to CSV")

        # Create README
        log("[main] Generating README file...")
        supernova_data = []  # Initialize if not already defined
        create_readme(
            maser_data=maser_data, 
            rotation_axis=rotation_axis, 
            median_angular_velocity=median_angular_velocity,
            optimal_omega=optimal_omega, 
            optimal_axis_ra=optimal_axis_ra, 
            optimal_axis_dec=optimal_axis_dec,
            cmb_dipole_axis=cmb_dipole_axis, 
            cmb_quadrupole_axis=cmb_quadrupole_axis, 
            earth_position=earth_position
        )
        # Verify README was created
        if os.path.exists('README.md'):
            with open('README.md', 'r') as f:
                content = f.read()
                log(f"[main] README.md created successfully ({len(content)} bytes)")
        else:
            log("[main] Warning: README.md file was not created")

        log("[main] Output files created successfully")
    except Exception as e:
        log(f"[main] Error creating output files: {str(e)}")
        # Print full error details for debugging
        import traceback
        log(f"[main] Full error details:\n{traceback.format_exc()}")

    # Create visualizations
    log("[main] Generating visualizations...")
    try:
        create_visualizations(maser_data, galaxy_data, rotation_axis, earth_position)
        create_visualizations_with_desi(maser_data, galaxy_data, desi_df, rotation_axis, earth_position)
        log("[main] Visualizations created successfully")
    except Exception as e:
        log(f"[main] Error creating visualizations: {str(e)}")

    log("[main] Analysis complete")
    log(f"[main] Final statistics:")
    log(f"  - Total galaxies: {total_galaxies}")
    log(f"  - Maser galaxies: {len(maser_data)}")
    log(f"  - DESI objects: {len(desi_df) if not desi_df.empty else 0}")
    log(f"  - Galaxies with complete positional data: {galaxies_with_complete_data}")

if __name__ == "__main__":
    main()