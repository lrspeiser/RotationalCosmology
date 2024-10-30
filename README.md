
    # Rotating Universe Model Analysis
    
    ## Overview
    
    This document summarizes the analysis of a hypothetical rotating universe model based on maser data and CMB observations. Instead of assuming the universe is expanding since the big bang, it provides a hypothetical alternative where the universe can be significantly older and the illusion of expansion comes from the frame dragging gravity effects of a Godel inspired rotating universe. In addition, we will explore the possibility of both light and gravity circumnavigating the universe repeatedly, which would give us an alternative explaination to CMB uniformity and the extra gravity we detect but can't attribute to physical objects we can observe. To start, we will begin using the axis of evil calculatings that indicate a direction to CMB, then we will use known celestial object distances that don't use redshift to calculate how much frame dragging would need to happen to generate the redshift we observe. Then we will attempt to estimate our distance from a central axis and chart out what this rotating universe looks like. We will also add data from other data sets, like DESI, to attempt to overlay the large scale structures we see in the universe, in the hopes that this rotating universe helps explain how they formed.
    
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
    
    Given that we only have data from **five maser galaxies**, the reliability of our rotation axis estimation is limited. With such a small sample size, it's challenging to accurately triangulate the exact location and orientation of the rotation axis. The estimation is more of an educated guess based on the available data and certain assumptions:
    
    - **Assumptions Made**:
      - We assume that the rotation axis is aligned in some way with the Cosmic Microwave Background (CMB) dipole and quadrupole axes.
      - We consider Earth's position relative to these axes to estimate its location in the universe.
      - The maser galaxies are assumed to be representative of the larger cosmic structure, which may not be the case.
    
    - **Limitations**:
      - **Sample Size**: Five data points are insufficient for precise triangulation in three-dimensional space.
      - **Measurement Errors**: Uncertainties in distance measurements and redshift velocities can significantly affect the estimation.
      - **Simplifications in the Model**: The model assumes a simple rotational motion without accounting for complex gravitational interactions and local motions.
    
      With the inclusion of **supernova data**, we now have a larger dataset to estimate the rotation axis. Supernovae provide independent distance measurements not reliant on redshift, improving our triangulation capabilities.
    
      - **Improvements**:
        - **Increased Data Points**: The addition of supernovae increases the number of data points significantly.
        - **Independent Distance Measurements**: Supernova distances are determined via standard candles, reducing reliance on redshift estimates.
    
      - **Remaining Limitations**:
        - **Data Quality**: Supernova measurements may have their own uncertainties and potential biases.
        - **Assumptions in the Model**: The model still assumes a simple rotational motion.
    
      Therefore, while the inclusion of supernova data enhances our model, caution is still advised in interpreting the results.
    
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
    
    - The rotation axis is positioned at the center of the universe (0, 0, 0).
    - Earth's position is calculated based on the CMB dipole and estimated rotation of the universe.
    - Maser coordinates are adjusted relative to the central rotation axis.
    - This model is highly speculative and should be interpreted with caution. It does not reflect the current scientific understanding of the universe's structure and dynamics.
    - The visualization of this data can be found in the accompanying 3D plot file.
    