This repository offers a complete library of files used to generate results for Quantifying Subglacial Water Flow Using Glaciohydraulic Tremor: A Multi-glacier Empirical Model
by Keeya S. Beausoleil, Timothy C. Bartholomaus, Alison S. Criscitiello, Elowyn M. Yager 

Step 1: Calculate raw flow accumulation time series for degree-day modelled melt and precipitation at each site and downstream gauge location. 
Each "raw_flow_acc_GLACIER_NAME" file consists of a similar workflow.
They differ in specified raw files (& formatting), parameters (sampling frequency, degree-day factor, lapse rate), duration of interest, and file saving procedure. 

This process will save CSV files containing a time series of accumulated melt and accumulated precipitation at individual proglacial gauge & glacier seismic site locations. 

Step 2: Estimate source discharge at glacier seismic sites using the smoothed ratio method. 
Each "source_discharge_GLACIER_NAME" file consists of a nearly identical workflow. They differ in specified raw files and duration of interest, developed individually for ease of troubleshooting and flexibility in development. 

This file will generate a variety of intermediary plots comparing discharges at the gauge (observation, raw flow acc, LSQ model), up-glacier discharge source (raw flow acc sum, area-scaled, ratio-smoothed scaling), etc. 
It will also generate power-law models (Gimbert theoretical 5/8 exponent and best-fit result) for each site (or site groupings at Lemon Creek) and plot hysteresis time series for individual sites. 

