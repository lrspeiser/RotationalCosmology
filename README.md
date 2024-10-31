
# Rotating Universe Model Analysis

## Overview

This document summarizes the analysis of a hypothetical rotating universe model based on maser data and CMB observations. Instead of assuming the universe is expanding since the big bang, it provides a hypothetical alternative where the universe can be significantly older, and the illusion of expansion comes from the frame-dragging gravity effects of a Godel-inspired rotating universe. In addition, we explore the possibility of both light and gravity circumnavigating the universe repeatedly, giving an alternative explanation to CMB uniformity and the extra gravity we detect but can't attribute to observable objects. 

To start, we use the axis of evil calculations indicating a direction to the CMB. We then apply known celestial object distances (that don't rely on redshift) to determine how much frame dragging is necessary to generate the observed redshift. We also attempt to estimate our distance from a central axis and chart what this rotating universe looks like. Additional data from sets like DESI will be used to overlay large-scale structures in hopes of explaining their formation through this model.

## Analysis Process

1. **Data Collection and Processing**
   - Processed 55874 galaxies from the EDD database
   - Identified 5 maser galaxies
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

- **Total galaxies processed**: 55874
- **Galaxies with maser distances**: 5
- **Galaxies with supernova distances**: 0
- **Assumed distance to universe center**: 4.29e+03 Mpc

## Rotation Parameters

- **Estimated median angular velocity**: 4.03e-21 rad/s
- **Optimal angular velocity**: 1.00e-18 rad/s
- **Optimal rotation axis**: RA = 0.00°, Dec = 0.00°

## CMB and Rotation Axis Analysis

| Axis | X | Y | Z |
|------|---|---|---|
| Rotation Axis | -0.8038 | -0.4059 | -0.4350 |
| CMB Dipole Axis | -0.0694 | -0.6622 | 0.7461 |
| CMB Quadrupole/Octupole Axis | -0.2500 | -0.4330 | 0.8660 |

- **Angle between Rotation Axis and CMB Dipole**: 90.00 degrees
- **Angle between Rotation Axis and CMB Quadrupole/Octupole**: 90.00 degrees

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

- **Position**: (-2.98, -28.44, 32.04) Mpc

### Maser Data

#### Maser 20679
- **Distance from Earth**: 51.52 Mpc
- **Observed Redshift**: 0.011311
- **Distance to rotation axis**: 50.20 Mpc
- **Adjusted Coordinates**: (-5.95, 53.14, 12.29) Mpc

#### Maser 50048
- **Distance from Earth**: 87.50 Mpc
- **Observed Redshift**: 0.021742
- **Distance to rotation axis**: 22.95 Mpc
- **Adjusted Coordinates**: (-71.29, -15.78, -18.43) Mpc

#### Maser 53012
- **Distance from Earth**: 112.20 Mpc
- **Observed Redshift**: 0.028606
- **Distance to rotation axis**: 21.23 Mpc
- **Adjusted Coordinates**: (-79.13, -47.37, -22.04) Mpc

#### Maser 59306
- **Distance from Earth**: 131.83 Mpc
- **Observed Redshift**: 0.034581
- **Distance to rotation axis**: 82.23 Mpc
- **Adjusted Coordinates**: (-28.53, -83.78, 29.54) Mpc

#### Maser 59868
- **Distance from Earth**: 109.14 Mpc
- **Observed Redshift**: 0.027616
- **Distance to rotation axis**: 65.62 Mpc
- **Adjusted Coordinates**: (-12.97, -48.73, 43.48) Mpc

### Supernova Data

## Notes

- The rotation axis is at the center of the universe (0, 0, 0).
- Earth's position is based on the CMB dipole and estimated universe rotation.
- Maser coordinates are adjusted relative to the rotation axis.
- This model is speculative and not reflective of current scientific understanding.
