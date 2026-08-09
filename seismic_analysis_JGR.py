
#%% 
# import packages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import xarray as xr
import pickle
from scipy import signal
from matplotlib.ticker import MultipleLocator
import richdem as rd
import sys
from osgeo import gdal
import rasterio
from oggm import utils
from pyproj import CRS
from shapely.geometry import mapping
from rasterio import features
import geopandas as gpd
from affine import Affine
from rasterio import features
from shapely.ops import nearest_points
import math
from openpyxl import load_workbook
import contextily as cx
from shapely.ops import unary_union
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit


#%%
# define function to read PSD tremor data and raw stream gauge observations
class BBseis:
    def __init__(self, name, t, GHT_powdB):
        self.name = name
        self.t = t
        self.powdB = GHT_powdB    # dB rel. 1 (m/s)^2
        self.pdB_LF = np.array([0]) # low frequency filtered version of dB
        self.T_amp = np.array([0]) # m/s : Tremor amplitude
        self.Ta_LF = np.array([0]) # m/s : low frequency filtered version of Tremor amplitude
    
def tremor_data(glacier,stn):
    if stn[0] == "W": 
        filename = '/data/stor/proj/keeya_thesis/Wolverine_Glacier/medspec/output_logs/[1.5, 10]'+stn +'all.pickle'
    if glacier[0] == "L": 
        filename = '/data/stor/proj/keeya_thesis/LC_Glacier/medspec/output_logs/[1.5, 10]'+stn +'all.pickle'
    if glacier[0] == "G": 
        filename = '/data/stor/proj/keeya_thesis/Gulkana_Glacier/analysis/med_spec/output/[1.5, 10]'+stn +'all.pickle'
    if glacier[0] == "M": 
        #filename = '/data/stor/proj/keeya_thesis/Gulkana_Glacier/analysis/med_spec/output/[1.5, 10]'+stn +'all.pickle'
        filename = '/data/stor/proj/keeya_thesis/Mendenhall_Glacier/analysis/med_spec/output/[1.5, 10]'+stn +'all.pickle'
    if glacier[0] == "A": 
        filename = '/data/stor/proj/keeya_thesis/Argentiere_Glacier/med_spec/output/[1.5, 10]'+stn +'all.pickle'
    BB = dict()
    with open(filename, 'rb') as f:  
        BB, fGHT, t_interp = pickle.load(f)
    return BB

def create_dataframe(glacier, stn):
    temp_dict = tremor_data(glacier, stn)
    df=pd.DataFrame({'date_time': temp_dict[stn].t})
    df["T_Amp"]= temp_dict[stn].T_amp
    df["T_Amp"] = df["T_Amp"].replace(0.0, np.nan)
    df["date_time"] = pd.to_datetime(df["date_time"])
    print(temp_dict)
    start_date = pd.Timestamp(df["date_time"].iloc[0]).tz_localize('UTC')
    end_date = pd.Timestamp(df["date_time"].iloc[-1]).tz_localize('UTC')
    df.set_index("date_time", inplace=True)
    df.index = df.index.tz_localize('UTC')
    if glacier[0] == "W": 
        sg_filename = '/data/stor/proj/keeya_thesis/Wolverine_Glacier/data/raw_data/wolv_streamgauge_02.10.2025.txt'
        sg_code = "1424_00060"
    if glacier[0] == "A": 
        sg_filename = '/data/stor/proj/keeya_thesis/Wolverine_Glacier/data/raw_data/wolv_streamgauge_02.10.2025.txt'
        sg_code = "1424_00060"
        weather_data = pd.read_csv('/data/stor/proj/keeya_thesis/Wolverine_Glacier/data/raw_data/wolverine990_15min_LVL2_2022.csv')
    if glacier[0] == "L": 
        sg_filename = '/data/stor/proj/keeya_thesis/LC_Glacier/data/raw_data/lemoncreek_streamgauge_02.10.2025.txt'
        sg_code = "1294_00060"
        weather_data = pd.read_csv('/data/stor/proj/keeya_thesis/LC_Glacier/data/raw_data/C17_met_data.txt')
    if glacier[0] == "G": 
        sg_filename = '/data/stor/proj/keeya_thesis/Gulkana_Glacier/data/raw_data/gulkana-phelan_streamgauge_02.10.2025.txt'
        sg_code = "1754_00060"
    if glacier[0] == "M": 
        sg_filename = '/data/stor/proj/keeya_thesis/Mendenhall_Glacier/raw_data/mendenhall_streamgauge_02.10.2025.txt'
        sg_code = "1303_00060"
    
    sg_df = pd.read_csv(sg_filename, sep='\t',comment='#')
    sg_df = sg_df.iloc[1:]
    sg_df['datetime'] = pd.to_datetime(sg_df['datetime'], errors='coerce')
    sg_df['datetime'] = sg_df['datetime'].dt.tz_localize('America/Anchorage', ambiguous="NaT").dt.tz_convert('UTC') # stream gauge observations are in AKDT (in file)
    filtered_df = sg_df[(sg_df['datetime'] >= start_date) & (sg_df['datetime'] <= end_date)]
    filtered_df.set_index('datetime', inplace=True)
    filtered_df[sg_code] = pd.to_numeric(filtered_df[sg_code], errors='coerce')
    filtered_df = filtered_df[[sg_code]]  # Remove any non-numeric columns
    filtered_df = filtered_df.resample('30min').mean() # tremor amplitude is 30 min resolution 
    filtered_df[sg_code] = filtered_df[sg_code]/(3.2804**3)
    df = df.join(filtered_df[[sg_code]])
    df.rename(columns={sg_code: "Q_cms"}, inplace=True)
    df.reset_index(inplace=True)
    return df 

#%%
# Get Lemon Creek data
BBEL_df = create_dataframe("LC","BBEL")
BBEU_df = create_dataframe("LC","BBEU")
BBWL_df = create_dataframe("LC","BBWL")
BBWU_df = create_dataframe("LC","BBWU")
BBGL_df = create_dataframe("LC","BBGL")
BBGU_df = create_dataframe("LC","BBGU")

#%% Trim site seismic observations to remove flatlines + pre & end of season 

### NOTES ###
# cut off any signal for ALL sites *before* 07-01-2017 
# cut off any signal for on ice sites after 08-15-2017
# cut off signal from off ice ites at 09-01-2017

trim_start = pd.to_datetime("2017-07-08 00:00:00").tz_localize('UTC')
trim_end1 = pd.to_datetime("2017-08-15 00:00:00").tz_localize('UTC')
trim_end2 = pd.to_datetime("2017-08-31 23:30:00").tz_localize('UTC')
BBEL_df = BBEL_df[BBEL_df["date_time"] >= trim_start]
BBEU_df = BBEU_df[BBEU_df["date_time"] >= trim_start]
BBWL_df = BBWL_df[BBWL_df["date_time"] >= trim_start]
BBWU_df = BBWU_df[BBWU_df["date_time"] >= trim_start]
BBGL_df = BBGL_df[BBGL_df["date_time"] >= trim_start]
BBGU_df = BBGU_df[BBGU_df["date_time"] >= trim_start]

BBEL_df = BBEL_df[BBEL_df["date_time"] <= trim_end2]
BBEU_df = BBEU_df[BBEU_df["date_time"] <= trim_end2]
BBWL_df = BBWL_df[BBWL_df["date_time"] <= trim_end2]
BBWU_df = BBWU_df[BBWU_df["date_time"] <= trim_end2]
BBGL_df = BBGL_df[BBGL_df["date_time"] <= trim_end1]
BBGU_df = BBGU_df[BBGU_df["date_time"] <= trim_end1]

# mid time series
mask = (BBGU_df["date_time"] >= pd.to_datetime("2017-08-07 03:00:00").tz_localize('UTC')) & (BBGU_df["date_time"] <= pd.to_datetime("2017-08-07 06:00:00").tz_localize('UTC'))
BBGU_df.loc[mask, "T_Amp"] = float("nan")

#%%
LC_df = pd.DataFrame()
dt_index = pd.date_range(start="2017-07-08 00:00", end="2017-08-31 23:30", freq="30T")
LC_df["date_time"] = dt_index

# %%
glacier_id_LC = "RGI2000-v7.0-G-01-19406"
#centerline_file = "/data/stor/proj/keeya_thesis/AK_data/RGI2000-v7.0-L-01_alaska/RGI2000-v7.0-L-01_alaska.shp" # RGI Center Line
BB_site_file = "/data/stor/proj/keeya_thesis/AK_data/BB_SiteLocations.xlsx"
LC_SiteLoc = pd.read_excel(BB_site_file, sheet_name="LC_Glacier") 

sites_gdf = gpd.GeoDataFrame( LC_SiteLoc,geometry=gpd.points_from_xy(LC_SiteLoc["Long"], LC_SiteLoc["Lat"]), crs="EPSG:4326")

cl_LC = gpd.read_file("/data/stor/proj/keeya_thesis/LC_Glacier/data/raw_data/dem_files/lemon_creek_trough.shp") # created 
cl_LC= cl_LC.set_crs(epsg=32606)

## to add offset (for perturbation testing)
#offset_LC = +300 # positive = left, negative = right. ## 325: high sites | 300: low sites 
#cl_LC_offset = cl_LC.copy()
#cl_LC_offset["geometry"] = cl_LC_offset.geometry.apply(lambda geom: geom.parallel_offset(offset_LC) if offset_LC != 0 and geom.geom_type == 'LineString' else geom)
#cl_LC= cl_LC.set_crs(epsg=4326)
cl_geom_LC = unary_union(cl_LC.geometry)
#cl_geom_LC = unary_union(cl_LC_offset.geometry) ##ADDD FOR OFFSET

sites_utm_LC = sites_gdf.to_crs(cl_LC.crs)
sites_utm_LC["lat_dist_m"] = sites_utm_LC.geometry.distance(cl_geom_LC)

sites_utm_LC["ice_thickness"] = [280,270, 300,360,270,240] # estimated from stress model, active seismics, and Milan model 
# adding error to thickneses
#sites_utm_LC["ice_thickness_err"] = sites_utm_LC["ice_thickness"] * 0.10 (uncertinaty for perturbation testing)
#sites_utm_LC["ice_thickness"] = sites_utm_LC["ice_thickness"] + sites_utm_LC["ice_thickness_err"]

# Find closest lateral point on centre line
sites_utm_LC["closest_pt"] = sites_utm_LC.geometry.apply( lambda p: nearest_points(p, cl_geom_LC)[1])
closest_pts_ll = gpd.GeoSeries(sites_utm_LC["closest_pt"],crs=sites_utm_LC.crs).to_crs(epsg=4326)
sites_utm_LC["closest_lon"] = closest_pts_ll.x
sites_utm_LC["closest_lat"] = closest_pts_ll.y
#sites_utm_LC["lat_dist_m"] = sites_utm_LC.geometry.apply(lambda p: p.distance(cl_geom_LC))
sites_utm_LC["tot_dist_m"] = np.sqrt(sites_utm_LC["lat_dist_m"]**2 + sites_utm_LC["ice_thickness"]**2)


fig, ax = plt.subplots(figsize=(9, 8))
gpd.GeoSeries([cl_geom_LC], crs="EPSG:32608").plot(ax=ax, color="cyan", linewidth=1, label="Centerline") # center line 
sites_utm_LC.plot(ax=ax, color="red", markersize=40, label="Sites") # sites
gpd.GeoSeries(sites_utm_LC["closest_pt"], crs=sites_utm_LC.crs).plot(ax=ax, color="orange", markersize=30, marker ="*",label="Closest point")
cx.add_basemap(ax,source=cx.providers.Esri.WorldImagery,crs=sites_utm_LC.crs.to_string())  # EPSG:32608)

ax.set_title("Sites and Lemon Creek Glacier Centerline on Satellite Imagery")
ax.legend()
ax.set_axis_off()


print(sites_utm_LC[["Site", "ice_thickness", "lat_dist_m", "tot_dist_m","closest_lon","closest_lat"]])
sites_utm_LC.to_csv("sites_data.csv", index=False)  # index=False avoids saving the row numbers

#%%

# Amplitude correction for rayleigh waves
freq = (1.5+10)/2 # should this be the average frequency we witness in the observations or average of range? 
Q = 71 
v = 1800 
alpha = (math.pi*freq)/(Q*v)
dist_BBEL = (sites_utm_LC.loc[sites_utm_LC["Site"] == "BBEL", "tot_dist_m"].values[0])
BBEL_df["T_Amp_corr"] = (BBEL_df["T_Amp"]*(dist_BBEL))*math.exp(alpha*(dist_BBEL))
dist_BBEU = (sites_utm_LC.loc[sites_utm_LC["Site"] == "BBEU", "tot_dist_m"].values[0])
BBEU_df["T_Amp_corr"] = BBEU_df["T_Amp"]*(dist_BBEU)*math.exp(alpha*(dist_BBEU))
dist_BBWL = (sites_utm_LC.loc[sites_utm_LC["Site"] == "BBWL", "tot_dist_m"].values[0])
BBWL_df["T_Amp_corr"] = BBWL_df["T_Amp"]*(dist_BBWL)* math.exp(alpha*(dist_BBWL))
dist_BBWU = (sites_utm_LC.loc[sites_utm_LC["Site"] == "BBWU", "tot_dist_m"].values[0])#+100
BBWU_df["T_Amp_corr"] = BBWU_df["T_Amp"]*(dist_BBWU)*math.exp(alpha*(dist_BBWU))
dist_BBGL = (sites_utm_LC.loc[sites_utm_LC["Site"] == "BBGL", "tot_dist_m"].values[0])
BBGL_df["T_Amp_corr"] = BBGL_df["T_Amp"]*(dist_BBGL)*math.exp(alpha*(dist_BBGL))
dist_BBGU = (sites_utm_LC.loc[sites_utm_LC["Site"] == "BBGU", "tot_dist_m"].values[0])#-100
BBGU_df["T_Amp_corr"] = BBGU_df["T_Amp"]*(dist_BBGU)*math.exp(alpha*(dist_BBGU))

#%%
# save corrected tremor to CSV file 
df_bbgl =  (BBGL_df.set_index('date_time')['T_Amp_corr'].resample('1h').mean().reset_index() )
df_bbgl.to_csv("bbgl_tremor.csv", index=False)

df_bbgu =  (BBGU_df.set_index('date_time')['T_Amp_corr'].resample('1h').mean().reset_index() )
df_bbgu.to_csv("bbgu_tremor.csv", index=False)

df_bbwl =  (BBWL_df.set_index('date_time')['T_Amp_corr'].resample('1h').mean().reset_index() )
df_bbwl.to_csv("bbwl_tremor.csv", index=False)

df_bbwu =  (BBWU_df.set_index('date_time')['T_Amp_corr'].resample('1h').mean().reset_index() )
df_bbwu.to_csv("bbwu_tremor.csv", index=False)

df_bbel =  (BBEL_df.set_index('date_time')['T_Amp_corr'].resample('1h').mean().reset_index() )
df_bbel.to_csv("bbel_tremor.csv", index=False)

df_bbeu =  ( BBEU_df.set_index('date_time')['T_Amp_corr'].resample('1h').mean().reset_index() )
df_bbeu.to_csv("bbeu_tremor.csv", index=False)


# %%
# get Wolverine files & clean data
WOLC_df = create_dataframe("Wolverine", "WOLC")
WOLN_df = create_dataframe("Wolverine", "WOLN")

WOLC_df["date_time"] = pd.to_datetime(WOLC_df["date_time"], utc=True)
WOLN_df["date_time"] = pd.to_datetime(WOLN_df["date_time"], utc=True)

trim_start_wolv = pd.to_datetime("2022-05-03 00:00:00").tz_localize('UTC')
trim_start_wolv = pd.to_datetime("2022-06-15 00:00:00").tz_localize('UTC')

WOLC_df = WOLC_df[WOLC_df["date_time"] >= trim_start_wolv]
WOLN_df = WOLN_df[WOLN_df["date_time"] >= trim_start_wolv]
mask = (WOLC_df["date_time"] >= pd.to_datetime("2022-05-08 22:00:00").tz_localize('UTC')) & (WOLC_df["date_time"] <= pd.to_datetime("2022-05-10 00:00:00").tz_localize('UTC'))
WOLC_df.loc[mask, "T_Amp"] = float("nan")

# %%
glacier_id_WOLV = "RGI2000-v7.0-G-01-11350"
centerline_file = "/data/stor/proj/keeya_thesis/AK_data/RGI2000-v7.0-L-01_alaska/RGI2000-v7.0-L-01_alaska.shp" # RGI Center Line

BB_site_file = "/data/stor/proj/keeya_thesis/AK_data/BB_SiteLocations.xlsx"
WOLV_SiteLoc = pd.read_excel(BB_site_file, sheet_name="WOLV_Glacier") 

sites_gdf = gpd.GeoDataFrame(WOLV_SiteLoc,geometry=gpd.points_from_xy(WOLV_SiteLoc["Long"], WOLV_SiteLoc["Lat"]), crs="EPSG:4326")

cl_WOLV = gpd.read_file("/data/stor/proj/keeya_thesis/Wolverine_Glacier/data/raw_data/wolv_trench_final14.shp")
cl_WOLV = cl_WOLV.to_crs(epsg=32606)

#offset_WOLV = +355 # positive = right, negative = left. ## WOLN =355  # WOLC = 1100
#cl_WOLV_offset = cl_WOLV.copy()
#cl_WOLV_offset["geometry"] = cl_WOLV_offset.geometry.apply(lambda geom: geom.parallel_offset(offset_WOLV) if offset_WOLV != 0 and geom.geom_type == 'LineString' else geom)

cl_geom_WOLV = unary_union(cl_WOLV.geometry)
#cl_geom_WOLV = unary_union(cl_WOLV_offset.geometry) ## add for offset
sites_utm_WOLV = sites_gdf.to_crs(cl_WOLV.crs)

sites_utm_WOLV["closest_pt"] = sites_utm_WOLV.geometry.apply(lambda p: nearest_points(p, cl_geom_WOLV)[1])

closest_pts_w = gpd.GeoSeries(sites_utm_WOLV["closest_pt"],crs=sites_utm_WOLV.crs).to_crs(epsg=4326)
sites_utm_WOLV["closest_lon"] = closest_pts_w.x
sites_utm_WOLV["closest_lat"] = closest_pts_w.y


sites_utm_WOLV["lat_dist_m"] = sites_utm_WOLV.geometry.apply(lambda p: p.distance(cl_geom_WOLV))
sites_utm_WOLV["lat_dist_m"][0] = sites_utm_WOLV["lat_dist_m"][0]#-950
sites_utm_WOLV["ice_thickness"] = np.array([337.0236511,315.1152344]) # from Milan's model
#sites_utm_WOLV["ice_thickness"] = np.array([337.0236511+101,315.1152344+101]) # from Milan's model + uncertainty
#sites_utm_WOLV["ice_thickness"] = np.array([337.0236511-101,315.1152344-101]) # from Milan's model - uncertainty
sites_utm_WOLV["tot_dist_m"] = np.sqrt(sites_utm_WOLV["lat_dist_m"]**2 + sites_utm_WOLV["ice_thickness"]**2)
print(sites_utm_WOLV[["Site", "ice_thickness", "lat_dist_m", "tot_dist_m","closest_lon","closest_lat"]])

# %%
# correction for attenuation of Rayleigh waves
freq = (1.5+10)/2 
Q = 71 
v = 1800 
alpha = (math.pi*freq)/(Q*v)
dist_WOLC = (sites_utm_WOLV.loc[sites_utm_WOLV["Site"] == "WOLC", "tot_dist_m"].values[0])
WOLC_df["T_Amp_corr"] = (WOLC_df["T_Amp"]*(dist_WOLC))*math.exp(alpha*(dist_WOLC))

dist_WOLN = (sites_utm_WOLV.loc[sites_utm_WOLV["Site"] == "WOLN", "tot_dist_m"].values[0])
WOLN_df["T_Amp_corr"] = (WOLN_df["T_Amp"]*(dist_WOLN))*math.exp(alpha*(dist_WOLN))
#%%
# save corrected tremor time series at CSV
df_wolc =  (WOLC_df.set_index('date_time')['T_Amp_corr'].resample('30min').mean().reset_index() )
df_wolc.to_csv("wolc_tremor.csv", index=False)

df_woln =  (WOLN_df.set_index('date_time')['T_Amp_corr'].resample('30min').mean() .reset_index() )
df_woln.to_csv("woln_tremor.csv", index=False)

#%%
glacier_id_MEN = "RGI2000-v7.0-G-01-19425"
MEND_df = create_dataframe("Mendenhall", "AMBR")

MEND_df["date_time"] = pd.to_datetime(MEND_df["date_time"], utc=True)
trim_start_mend = pd.to_datetime("2012-06-05 00:00:00").tz_localize('UTC')
trim_start_mend = pd.to_datetime("2012-06-30 00:00:00").tz_localize('UTC')
trim_end_mend = pd.to_datetime("2012-08-17 23:00:00").tz_localize('UTC')

MEND_df = MEND_df[(MEND_df["date_time"] >= trim_start_mend) & (MEND_df["date_time"] <= trim_end_mend)]

#mask = (MEND_df["date_time"] >= pd.to_datetime("2012-07-01 03:00:00").tz_localize('UTC')) & (MEND_df["date_time"] <= pd.to_datetime("2012-07-10 23:00:00").tz_localize('UTC'))
#MEND_df.loc[mask, "T_Amp"] = float("nan")
#plt.plot(MEND_df["T_Amp"])

BB_site_file = "/data/stor/proj/keeya_thesis/AK_data/BB_SiteLocations.xlsx"
MEND_SiteLoc = pd.read_excel(BB_site_file, sheet_name="MEND_Glacier") 

sites_gdf = gpd.GeoDataFrame(MEND_SiteLoc,geometry=gpd.points_from_xy(MEND_SiteLoc["Long"], MEND_SiteLoc["Lat"]), crs="EPSG:4326")

sites_gdf = sites_gdf[sites_gdf["Site"] == "AMBR"]

#%%
cl_MEND = gpd.read_file("/data/stor/proj/keeya_thesis/Mendenhall_Glacier/raw_data/mend_trench.shp")
cl_MEND= cl_MEND.to_crs(epsg=32608)

#offset_MEND = -345 # positive = right, negative = left
#cl_MEND_offset = cl_MEND.copy()
#cl_MEND_offset["geometry"] = cl_MEND_offset.geometry.apply(lambda geom: geom.parallel_offset(offset_MEND) if offset_MEND != 0 and geom.geom_type == 'LineString' else geom)

cl_geom_MEND = unary_union(cl_MEND.geometry)
#cl_geom_MEND = unary_union(cl_MEND_offset.geometry)

sites_utm_MEND = sites_gdf.to_crs(epsg=32608)
sites_utm_MEND["closest_pt"] = sites_utm_MEND.geometry.apply( lambda p: nearest_points(p, cl_geom_MEND)[1])

closest_pts_m = gpd.GeoSeries(sites_utm_MEND["closest_pt"],crs=sites_utm_MEND.crs).to_crs(epsg=4326)
sites_utm_MEND["closest_lon"] = closest_pts_m.x
sites_utm_MEND["closest_lat"] = closest_pts_m.y


sites_utm_MEND["lat_dist_m"] = sites_utm_MEND.geometry.apply(lambda p: p.distance(cl_geom_MEND))
sites_utm_MEND["ice_thickness"] = np.array([340]) #milan - AMBR
#sites_utm_MEND["ice_thickness"] = np.array([441])
#sites_utm_MEND["ice_thickness"] = np.array([340-101])
sites_utm_MEND["tot_dist_m"] = np.sqrt(sites_utm_MEND["lat_dist_m"]**2 + sites_utm_MEND["ice_thickness"]**2)

fig, ax = plt.subplots(figsize=(9, 8))
gpd.GeoSeries([cl_geom_MEND], crs="EPSG:32608").plot(ax=ax, color="cyan", linewidth=2, label="Centerline") # center line 
sites_utm_MEND.plot(ax=ax, color="red", markersize=40, label="Sites") # sites

# Add satellite basemap
cx.add_basemap( ax,source=cx.providers.Esri.WorldImagery,crs=sites_utm_MEND.crs.to_string()  )# EPSG:32608
plt.show()

print(sites_utm_MEND[["Site", "ice_thickness", "lat_dist_m", "tot_dist_m","closest_lon","closest_lat"]])

freq = (1.5+10)/2 
Q = 71 
v = 1800 
alpha = (math.pi*freq)/(Q*v)
dist_MEND = (sites_utm_MEND.loc[sites_utm_MEND["Site"] == "AMBR", "tot_dist_m"].values[0])
MEND_df["T_Amp_corr"] = (MEND_df["T_Amp"]*(dist_MEND))*math.exp(alpha*(dist_MEND))
plt.plot(MEND_df["T_Amp_corr"])

# save as CSV
df_mend =  (MEND_df.set_index('date_time')['T_Amp_corr'].resample('1h').mean().reset_index() )
df_mend.to_csv("ambr_tremor.csv", index=False)

#%%
# Q = k * V^(8/5), fit in log-Q space
def log_power_law_Q(logV, log_k):
    return log_k + (8/5) * logV

def sci_latex(x):
    s = f"{x:.2e}"
    coef, exp = s.split("e")
    return f"{coef} \\times 10^{{{int(exp)}}}"

# plot tremor against proglacial stream gauging, compare tremor pre/post corrections & fit to Gimbert theory
fig, axes = plt.subplots(ncols=2, figsize=(20, 8))

color_WOLC = "#C75803"
color_WOLN = "#2370DD"
color_ice_up = "#9476BD"
color_ice_low = "#411D71"
color_off_up = "#EBB400"
color_off_low = "#926C03"
color_MEND = "#B11997"

WOLN_Q = pd.concat([WOLN_df["Q_cms"]], ignore_index=True)
WOLN_T = pd.concat([WOLN_df["T_Amp"]], ignore_index=True)

WOLC_Q = pd.concat([WOLC_df["Q_cms"]], ignore_index=True)
WOLC_T = pd.concat([WOLC_df["T_Amp"]], ignore_index=True)

MEND_Q = pd.concat([MEND_df["Q_cms"]], ignore_index=True)
MEND_T = pd.concat([MEND_df["T_Amp"]], ignore_index=True)

BBGU_Q = pd.concat([BBGU_df["Q_cms"]], ignore_index=True)
BBGU_T = pd.concat([BBGU_df["T_Amp"]], ignore_index=True)

BBGL_Q = pd.concat([BBGL_df["Q_cms"]], ignore_index=True)
BBGL_T = pd.concat([BBGL_df["T_Amp"]], ignore_index=True)

BBU_Q = pd.concat([BBEU_df["Q_cms"], BBWU_df["Q_cms"]], ignore_index=True)
BBU_T = pd.concat([BBEU_df["T_Amp"], BBWU_df["T_Amp"]], ignore_index=True)

BBL_Q = pd.concat([BBEL_df["Q_cms"], BBWL_df["Q_cms"]], ignore_index=True)
BBL_T = pd.concat([BBEL_df["T_Amp"], BBWL_df["T_Amp"]], ignore_index=True)

sns.kdeplot(x=np.log10(WOLC_T), y=np.log10(WOLC_Q), ax=axes[0], fill=False, color=color_WOLC, levels=15, linewidths=1.5, alpha=0.5)
sns.kdeplot(x=np.log10(WOLN_T), y=np.log10(WOLN_Q), ax=axes[0], fill=False, color=color_WOLN, levels=15, linewidths=1.5, alpha=0.5)
sns.kdeplot(x=np.log10(MEND_T), y=np.log10(MEND_Q), ax=axes[0], fill=False, color=color_MEND, levels=15, linewidths=1.5, alpha=0.5)
sns.kdeplot(x=np.log10(BBGU_T), y=np.log10(BBGU_Q), ax=axes[0], fill=False, color=color_ice_up, levels=15, linewidths=1.5, alpha=0.5)
sns.kdeplot(x=np.log10(BBGL_T), y=np.log10(BBGL_Q), ax=axes[0], fill=False, color=color_ice_low, levels=15, linewidths=1.5, alpha=0.5)
sns.kdeplot(x=np.log10(BBU_T),  y=np.log10(BBU_Q),  ax=axes[0], fill=False, color=color_off_up, levels=15, linewidths=1.5, alpha=0.5)
sns.kdeplot(x=np.log10(BBL_T),  y=np.log10(BBL_Q),  ax=axes[0], fill=False, color=color_off_low, levels=15, linewidths=1.5, alpha=0.5)

axes[0].set_xlabel("Log$_{10}$ Raw Tremor Amplitude, A(r), [m/s]", fontsize=21,labelpad=6)
axes[0].set_ylabel("Log$_{10}$ Gauged Streamflow [m³/s]", fontsize=21)

handles = [
    Line2D([0], [0], color=color_WOLC, lw=3.5, label="WOLV Upper"),
    Line2D([0], [0], color=color_WOLN, lw=3.5, label="WOLV Lower"),
    Line2D([0], [0], color=color_ice_up, lw=3.5, label="LC On-Ice Upper"),
    Line2D([0], [0], color=color_ice_low, lw=3.5, label="LC On-Ice Lower"),
    Line2D([0], [0], color=color_off_up, lw=3.5, label="LC Off-Ice Upper"),
    Line2D([0], [0], color=color_off_low, lw=3.5, label="LC Off-Ice Lower"),
    Line2D([0], [0], color=color_MEND, lw=3.5, label="Mendenhall")
]
axes[0].legend(handles=handles, fontsize=18, loc="upper left")
axes[0].set_xlim([-8.75, -7])
axes[0].set_ylim([0.25, 2.5])
axes[0].tick_params(axis='x', labelsize=18)
axes[0].tick_params(axis='y', labelsize=18)
axes[0].yaxis.get_offset_text().set_fontsize(18)

WOLC_Tc = pd.concat([WOLC_df["T_Amp_corr"]], ignore_index=True)
WOLN_Tc = pd.concat([WOLN_df["T_Amp_corr"]], ignore_index=True)
MEND_Tc = pd.concat([MEND_df["T_Amp_corr"]], ignore_index=True)
BBGU_Tc = pd.concat([BBGU_df["T_Amp_corr"]], ignore_index=True)
BBGL_Tc = pd.concat([BBGL_df["T_Amp_corr"]], ignore_index=True)
BBU_Tc = pd.concat([BBEU_df["T_Amp_corr"], BBWU_df["T_Amp_corr"]], ignore_index=True)
BBL_Tc = pd.concat([BBEL_df["T_Amp_corr"], BBWL_df["T_Amp_corr"]], ignore_index=True)

sns.kdeplot(x=np.log10(WOLN_Tc), y=np.log10(WOLN_Q), ax=axes[1], fill=False, color=color_WOLN, levels=15, linewidths=1.5, alpha=0.5)
sns.kdeplot(x=np.log10(WOLC_Tc), y=np.log10(WOLC_Q), ax=axes[1], fill=False, color=color_WOLC, levels=15, linewidths=1.5, alpha=0.5)
sns.kdeplot(x=np.log10(MEND_Tc), y=np.log10(MEND_Q), ax=axes[1], fill=False, color=color_MEND, levels=15, linewidths=1.5, alpha=0.5)
sns.kdeplot(x=np.log10(BBGU_Tc), y=np.log10(BBGU_Q), ax=axes[1], fill=False, color=color_ice_up, levels=15, linewidths=1.5, alpha=0.5)
sns.kdeplot(x=np.log10(BBGL_Tc), y=np.log10(BBGL_Q), ax=axes[1], fill=False, color=color_ice_low, levels=15, linewidths=1.5, alpha=0.5)
sns.kdeplot(x=np.log10(BBU_Tc),  y=np.log10(BBU_Q),  ax=axes[1], fill=False, color=color_off_up, levels=15, linewidths=1.5, alpha=0.5)
sns.kdeplot(x=np.log10(BBL_Tc),  y=np.log10(BBL_Q),  ax=axes[1], fill=False, color=color_off_low, levels=15, linewidths=1.5, alpha=0.5)

all_Q = pd.concat([WOLC_Q, WOLN_Q, BBGU_Q, BBGL_Q, BBU_Q, BBL_Q, MEND_Q], ignore_index=True)
all_Tc = pd.concat([WOLC_Tc, WOLN_Tc, BBGU_Tc, BBGL_Tc, BBU_Tc, BBL_Tc, MEND_Tc], ignore_index=True)

mask = np.isfinite(all_Q) & np.isfinite(all_Tc) & (all_Q > 0) & (all_Tc > 0)
popt, _ = curve_fit(log_power_law_Q, np.log10(all_Tc[mask]), np.log10(all_Q[mask]), p0=[9.0])
k_fit = 10**popt[0]

log_resid = np.log10(all_Q[mask]) - np.log10(k_fit * all_Tc[mask]**(8/5))
ss_res_log = np.sum(log_resid**2)
ss_tot_log = np.sum((np.log10(all_Q[mask]) - np.mean(np.log10(all_Q[mask])))**2)
r_squared_log = 1 - (ss_res_log / ss_tot_log)

V_fit = np.linspace(all_Tc[mask].min(), all_Tc[mask].max(), 600)

axes[1].plot(np.log10(V_fit), np.log10(k_fit * V_fit**(8/5)), color="black", lw=3, linestyle="--",
             label=f"Q = ${sci_latex(k_fit)}$ · V$^{{8/5}}$ | R$^2_{{\\log}}$={r_squared_log:.2f}")

axes[1].set_xlabel("Log$_{10}$ Corrected Tremor Amplitude, A$_0$, [m/s]", fontsize=21)
#axes[1].set_ylabel("Log$_{10}$ Gauged Streamflow [m³/s]", fontsize=21)
axes[1].set_ylabel("")
axes[1].set_xlim([-5.75, -4])
axes[1].set_ylim([0.25, 2.5])
axes[1].tick_params(axis='x', labelsize=18)
axes[1].tick_params(axis='y', labelsize=18)
axes[1].yaxis.get_offset_text().set_fontsize(18)

handles2 = [Line2D([0], [0], color="black", lw=3, linestyle="--",label=f"Q = ${sci_latex(k_fit)}$ · V$^{{8/5}}$ | R$^2_{{\\log}}$={r_squared_log:.2f}")]
axes[1].legend(handles=handles2, fontsize=20, loc="lower right")

plt.tight_layout()

plt.savefig( "/data/stor/proj/keeya_thesis/LC_Glacier/analysis/output_figs/fig2_tremor_corr_comp.png", dpi=300,transparent=True, bbox_inches="tight")
plt.show()
# %%
