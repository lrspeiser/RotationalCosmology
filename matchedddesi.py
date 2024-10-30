import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import xml.etree.ElementTree as ET

def log(message):
    print(f"[matchedddesi.py] {message}")

def equatorial_to_cartesian(ra, dec, distance):
    """Convert RA/DEC coordinates to Cartesian coordinates."""
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    x = distance * np.cos(dec) * np.cos(ra)
    y = distance * np.cos(dec) * np.sin(ra)
    z = distance * np.sin(dec)

    return np.array([x, y, z])

def create_3d_visualization(merged_df):
    """Create 3D visualization of matched objects."""
    try:
        import plotly.graph_objs as go

        log("[create_3d_visualization] Starting 3D visualization creation.")

        fig = go.Figure()

        # Calculate approximate distances based on redshift velocity
        H0 = 70  # km/s/Mpc
        merged_df['approx_distance'] = merged_df['Vcmb'] / H0

        log("[create_3d_visualization] Calculated distances.")

        # Convert coordinates to Cartesian
        positions = [
            equatorial_to_cartesian(row['RA'], row['DE'], row['approx_distance']) 
            for _, row in merged_df.iterrows() if not np.isnan(row['Vcmb'])
        ]

        if positions:
            positions = np.array(positions)
            x, y, z = positions.T

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=5, color='red', opacity=0.8),
                name='Large Scale Structure'
            ))

            log("[create_3d_visualization] Plotted large-scale structure.")

            # Add text labels for significant structures
            significant_structures = merged_df[merged_df['Vcmb'].notna()].nlargest(10, 'Vcmb')
            for _, struct in significant_structures.iterrows():
                pos = equatorial_to_cartesian(struct['RA'], struct['DE'], struct['approx_distance'])
                fig.add_trace(go.Scatter3d(
                    x=[pos[0]], y=[pos[1]], z=[pos[2]],
                    mode='markers+text',
                    marker=dict(size=8, color='yellow'),
                    text=[f"PGC {struct['PGC_ID']}"],
                    name=f"PGC {struct['PGC_ID']}"
                ))

            log("[create_3d_visualization] Plotted significant structures.")

        fig.update_layout(
            title="Large Scale Structure from DESI-EDD Matches",
            scene=dict(
                xaxis_title="X (Mpc)",
                yaxis_title="Y (Mpc)",
                zaxis_title="Z (Mpc)",
                aspectmode='cube'
            ),
            showlegend=True
        )

        output_file = "large_scale_structure.html"
        fig.write_html(output_file)
        log(f"[create_3d_visualization] 3D visualization saved to '{output_file}'")

    except Exception as e:
        log(f"[create_3d_visualization] Error creating visualization: {str(e)}")

def print_coord_stats(df, ra_col, dec_col, name):
    log(f"\n{name} Coordinate Statistics:")
    log(f"RA range: {df[ra_col].min():.4f} to {df[ra_col].max():.4f}")
    log(f"DEC range: {df[dec_col].min():.4f} to {df[dec_col].max():.4f}")
    log(f"Number of unique RA values: {df[ra_col].nunique()}")
    log(f"Number of unique DEC values: {df[dec_col].nunique()}")

try:
    log("Opening DESI FITS file: 'survey-bricks.fits'")
    with fits.open("survey-bricks.fits") as hdul:
        bintable = hdul[1]
        data_dict = {
            'BRICKNAME': [name.strip() for name in bintable.data['BRICKNAME']],
            'BRICKID': bintable.data['BRICKID'].astype(np.int64),
            'RA': bintable.data['RA'].astype(np.float64),
            'DEC': bintable.data['DEC'].astype(np.float64)
        }
        desi_df = pd.DataFrame(data_dict)
        log(f"Loaded DESI data with {len(desi_df)} entries.")

    log("Reading EDD XML data from 'eddtable.xml'")
    tree = ET.parse('eddtable.xml')
    root = tree.getroot()
    ns = {'vo': 'http://www.ivoa.net/xml/VOTable/v1.2'}
    columns = [field.get('name') for field in root.findall('.//vo:FIELD', ns)]

    data = [
        [float(td.text) if td.text else np.nan for td in tr.findall('vo:TD', ns)]
        for tr in root.findall('.//vo:TR', ns)
    ]
    edd_df = pd.DataFrame(data, columns=columns)

    desi_df.dropna(subset=['RA', 'DEC'], inplace=True)
    edd_df.dropna(subset=['RA', 'DE'], inplace=True)

    print_coord_stats(desi_df, 'RA', 'DEC', 'DESI')
    print_coord_stats(edd_df, 'RA', 'DE', 'EDD')

    desi_coords = SkyCoord(ra=desi_df['RA'].values * u.deg, dec=desi_df['DEC'].values * u.deg, frame='icrs')
    edd_coords = SkyCoord(ra=edd_df['RA'].values * u.deg, dec=edd_df['DE'].values * u.deg, frame='icrs')

    idx, sep2d, _ = desi_coords.match_to_catalog_sky(edd_coords)
    matches = sep2d < 1 * u.arcmin

    if matches.sum() > 0:
        matched_desi = desi_df[matches].copy()
        matched_edd = edd_df.iloc[idx[matches]].copy()

        matched_desi_coords = SkyCoord(ra=matched_desi['RA'].values * u.deg, dec=matched_desi['DEC'].values * u.deg, frame='icrs')
        matched_desi_gal = matched_desi_coords.galactic

        final_desi = pd.DataFrame({
            'DESI_RA': matched_desi['RA'],
            'DESI_DEC': matched_desi['DEC'],
            'DESI_GLON': matched_desi_gal.l.deg,
            'DESI_GLAT': matched_desi_gal.b.deg
        })

        merged_df = pd.concat([final_desi.reset_index(drop=True), matched_edd.reset_index(drop=True)], axis=1)
        merged_df['separation_arcsec'] = sep2d[matches].to(u.arcsec).value

        log("Creating 3D visualization of large-scale structure...")
        create_3d_visualization(merged_df)

    else:
        log("No matches found within the specified tolerance.")

except Exception as e:
    log(f"An error occurred: {str(e)}")
    import traceback
    log(f"Traceback: {traceback.format_exc()}")
