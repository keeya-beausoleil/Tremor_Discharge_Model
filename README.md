This repository offers a complete library of files used to generate results for Quantifying Subglacial Water Flow Using Glaciohydraulic Tremor: A Multi-glacier Empirical Model
by Keeya S. Beausoleil, Timothy C. Bartholomaus, Alison S. Criscitiello, Elowyn M. Yager 

Part 2: Hydrological Modelling 

Step 1: Calculate raw flow accumulation time series for degree-day modelled melt and precipitation at each site and downstream gauge location. 
Each "raw_flow_acc_GLACIER_NAME" file consists of a similar workflow.
They differ in specified raw files (& formatting), parameters (sampling frequency, degree-day factor, lapse rate), duration of interest, and file saving procedure. 

Data required: meteorological observations (air temp & precipitation records), seasonal snowline evolution (modified GLACEE generated), period of interest (matching seismic observations), DEMS (raw and trough-modified), local melt parameters (degree-day factor, lapse rate, weather station elevation), location of gauge & seismic sites

This process will save CSV files containing a time series of accumulated melt and accumulated precipitation at individual proglacial gauge & glacier seismic site locations. 

Step 2: Estimate source discharge at glacier seismic sites using the smoothed ratio method. 
Each "source_discharge_GLACIER_NAME" file consists of a nearly identical workflow. They differ in specified raw files and duration of interest, developed individually for ease of troubleshooting and flexibility in development. 

Data required: Raw flow accumulation data from Step 1 at gauge and seismic sites, corrected tremor amplitude (PSD amplitude 1.5-10 Hz and corrected for attenuation), stream gauge records (USGS water data), adjustable parameters - proportional upstream contributing area for each site (% to total contributing area at proglacial gauge) & flow path (seismic site to gauge) for lag

This file will generate a variety of intermediary plots comparing discharges at the gauge (observation, raw flow acc, LSQ model), up-glacier discharge source (raw flow acc sum, area-scaled, ratio-smoothed scaling), etc. It will also generate power-law models (Gimbert theoretical 5/8 exponent and best-fit result) for each site (or site groupings at Lemon Creek) and plot hysteresis time series for individual sites. 

It will create CSV files containing time series of final corrected tremor amplitude and source subglacial discharge estimated at each site. 

Step 3: Create multi-glacier tremor-discharge model with Monte Carlo sampling procedure (& assess model uncertainty)



