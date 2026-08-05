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
#%%

#%% time aggragate functions 
def sumtime(t, x, t_new):
    deltat = t_new[1] - t_new[0]
    bins = np.append(t_new, t_new[-1]+deltat)
    x_new = np.full(t_new.shape, np.nan)
    count_new = np.full(t_new.shape, np.nan)
    for i in range(bins.shape[0]-1):
        ind = np.where( (t>bins[i]) & (t<=bins[i+1]) )[0] # Following time step
        if ind.shape[0] != 0:
            x_new[i] = np.nansum(x[ind])
            count_new[i] = np.sum( np.invert( np.isnan(x[ind]) ) )
    
    return x_new, count_new

def mean_func(t, x, t_new):
    t = t.astype("datetime64[s]").astype(float)
    t_new = t_new.astype("datetime64[s]").astype(float)

    deltat = t_new[1] - t_new[0]
    bins = np.append(t_new, t_new[-1]+deltat)
    #maxv = np.full(t_new.shape, np.nan)
    #minv = np.full(t_new.shape, np.nan)
    meanv = np.full(t_new.shape, np.nan)

    for i in range(bins.shape[0]-1):
        print(type(bins[i]))
        ind = np.where( (t>bins[i]) & (t<=bins[i+1]) )[0]
        if (ind.size != 1) & (ind.shape[0] != 0):
            #maxv[i] = np.nanmax(x[ind])
            #minv[i] = np.nanmin(x[ind])
            meanv[i] = np.nanmean(x[ind])
        elif ind.size == 1:
            meanv[i] = x[ind[0]]
    return meanv 

#%%
# make weather data record subset 
def make_subset(w_filename, s_filename, start_date, end_date, sampling_freq):
    weather_data =  pd.read_csv(w_filename) 
    weather_data['UTC_time'] = pd.to_datetime(weather_data['UTC_time'],utc=True)
    subset_weather = weather_data[(weather_data['UTC_time'] >= start_date) & (weather_data['UTC_time'] <= end_date)]
    w_subset = pd.DataFrame({'date_time' : pd.date_range(start_date, end_date, freq=sampling_freq )})
    subset_weather["UTC_time"] = subset_weather["UTC_time"].dt.tz_localize(None)
    w_subset["date_time"] = w_subset["date_time"].dt.tz_localize(None)
    temp_subset = mean_func(subset_weather["UTC_time"].values, subset_weather["site_temp"].values, w_subset["date_time"].values)
    rain = sumtime(subset_weather["UTC_time"].values, subset_weather["Precip_Weighing_Incremental"].values, w_subset["date_time"].values)[0] # given in mm we.
    precip_subset = rain/1000
    w_subset["Air_Temp"] = temp_subset
    w_subset["Accum_Rain"] = precip_subset
    df_snow = pd.read_excel(s_filename, sheet_name="WOLV_Glacier")
    df_snow.columns = ["date", "snow_elv"]
    #df_snow = df_snow.drop(columns=['system:index', 'SLA_lower_bound_m', 'SLA_m', 'glacier_area_m2', 'ice_area_m2', 'percent_AOI_coverage', 'rock_area_m2', 'snow_area_m2', 'source', 'spatial_scale_m', 'transient_AAR', 'water_area_m2']) # previous raw glacee format
    weekly_snow = pd.to_datetime(df_snow["date"])
    df_snow['date'] = pd.to_datetime(df_snow['date'])
    df_snow.set_index('date', inplace=True)
    df_snow['snow_elv'] = df_snow['snow_elv']
    snowline = df_snow['snow_elv'].to_numpy()
    hourly_index = pd.date_range(start=df_snow.index.min(), end=df_snow.index.max(), freq='30min')
    hourly_series = df_snow.reindex(hourly_index)
    hourly_series = hourly_series.interpolate(method='time')
    df_hourly = hourly_series.reset_index().rename(columns={'index': 'date_time', 'snow_elv': 'hourly_upper_SLA'})
    df_hourly['date_time'] = (pd.to_datetime(df_hourly['date_time']).dt.tz_localize('UTC'))
    subset_snow = df_hourly[(df_hourly['date_time'] >= start_date) & (df_hourly['date_time'] <= end_date)]
    temp_sla_elv = np.array(subset_snow["hourly_upper_SLA"])
    plt.plot(temp_sla_elv)
    w_subset["SLA"] = temp_sla_elv

    return w_subset


#%%
# define observation period and make subset from raw data files
start_date_WOLV = pd.Timestamp("2022/05/12 22:00:00").tz_localize('UTC') # start of rain dataset (smallest time constraint)
end_date_WOLV = pd.Timestamp("2022/09/30 22:00:00").tz_localize('UTC') # end of rain dataset (smallest time constraint)
sampling_freq_WOLV = '30min' # weather station data is the lowest resolution 

w_filename_WOLV = '/data/stor/proj/keeya_thesis/Wolverine_Glacier/data/raw_data/wolverine990_15min_LVL2_2022.csv'# USGS benchmark weather dataset
s_filename_WOLV = '/data/stor/proj/keeya_thesis/LC_Glacier/data/raw_data/snow_line_final.xlsx' # glacee snowline elevation file

weather_subset_WOLV = make_subset(w_filename_WOLV , s_filename_WOLV, start_date_WOLV, end_date_WOLV, sampling_freq_WOLV)

#%%
# get dem files
dem_path = '/data/stor/proj/keeya_thesis/Wolverine_Glacier/data/raw_data/wolv_dem_clip_final14.tif'
dem = gdal.Open( dem_path )
xres,yres = dem.GetGeoTransform()[1::4]
pix_area = -xres*yres 
GT_dem = dem.GetGeoTransform()
dem_data = dem.GetRasterBand(1).ReadAsArray()
dem_data[dem_data==-9999] = np.nan

WS_elv = 990
elev_diff = dem_data - WS_elv

x0, y0 = GT_dem[0], GT_dem[3]
x1 = x0 + dem.RasterXSize * xres
y1 = y0 + dem.RasterYSize * yres
xmin, xmax = sorted([x0, x1])
ymin, ymax = sorted([y0, y1])
extent = (xmin, xmax, ymin, ymax)
extent = (xmin, xmax, ymax, ymin)
#al = utils.get_rgi_glacier_entities(['RGI60-01.11350'], version='62')
outline_path = "/data/stor/proj/keeya_thesis/Wolverine_Glacier/data/raw_data/wolv_outline.shp"
al = gpd.read_file(outline_path)
# Reproject outline to DEM CRS if needed
dem_wkt = dem.GetProjection()
dem_crs = CRS.from_wkt(dem_wkt)
al = al.to_crs(dem_crs)

trough_path =  '/data/stor/proj/keeya_thesis/Wolverine_Glacier/data/raw_data/wolv_dem_trench_final_clip14.tif'
trough = gdal.Open( trough_path )
GT_trough= trough.GetGeoTransform()
trough_data = trough.GetRasterBand(1).ReadAsArray()
rd_trough = rd.rdarray(trough_data, no_data=-9999)
rd_trough.geotransform = GT_trough
trough_filled = rd.FillDepressions(rd_trough,epsilon=True, in_place=False)

# Mask to glacier outline
ny, nx = dem_data.shape
al = al.to_crs(dem_crs)
shapes = [(mapping(geom), 1) for geom in al.geometry]
transform_affine = Affine.from_gdal(*GT_dem)
mask = features.rasterize(shapes, out_shape=(ny, nx), transform=transform_affine, fill=0, all_touched=False, dtype='uint8')
masked_dem = np.where(mask == 1, dem_data, np.nan)


#%%
# Define coordinates of sites on trough DEM 
wolc_point = (270,260)
woln_point = (311,512)
gauge_point =  (403,825)

#%%
# investigate points on flow accumulation map
def plot_trough_flow(trough, point):
    flow_acc = rd.FlowAccumulation(trough_filled, method='D8')
    plt.figure(figsize=(6, 6))
    plt.imshow(np.log10(flow_acc), origin='upper')
    #plt.scatter(point[0],point[1], color="red")
    plt.colorbar(label='log10(flow accumulation)')
    plt.title('Flow Accumulation (D8)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    #plt.xlim([1200,1400])
    #plt.ylim([1300,1600])
    plt.tight_layout()
    plt.show()

plot_trough_flow(trough_filled,gauge_point)

#%%
# run flow accumulation for degree-day melt and precipitation @ sites and gauge

ddf = 4.7  # in mm/day/C, adjust based on glacier properties
#ddf = 3.1 # snow degree day factor 
lapse_rate = -0.0065 # C/m https://pubs.usgs.gov/sir/2010/5247/pdf/sir20105247.pdf 

gauge_discharge_flux = np.zeros(len(weather_subset_WOLV['date_time']))
gauge_precip_flux = np.zeros(len(weather_subset_WOLV['date_time']))

wolc_discharge_flux = np.zeros(len(weather_subset_WOLV['date_time']))
wolc_precip_flux = np.zeros(len(weather_subset_WOLV['date_time']))

woln_discharge_flux = np.zeros(len(weather_subset_WOLV['date_time']))
woln_precip_flux = np.zeros(len(weather_subset_WOLV['date_time']))


for i in np.arange(len(weather_subset_WOLV['date_time'])):
    temp_day = weather_subset_WOLV["Air_Temp"][i]
    temp_pixels = temp_day + lapse_rate *elev_diff
    melt = np.where( ((temp_pixels > 0) & (mask == 1))| ((temp_pixels > 0) & (mask == 0)  & (dem_data > weather_subset_WOLV["SLA"][i])), ddf * (temp_pixels), 0)
    melt_dt = ((melt/1000)/43200).astype(np.float64) # 30 min resolution  m/s 
    base = np.ones_like(melt_dt)
    runoff_melt = rd.FlowAccumulation(trough_filled, method='D8', weights=melt_dt) # m/s per pixel
    runoff_base = rd.FlowAccumulation(trough_filled, method='D8', weights=base)
    runoff_precip = runoff_base* weather_subset_WOLV["Accum_Rain"][i] # pix *mm 
    melt_discharge = runoff_melt * pix_area
    precip_flux = (runoff_precip*pix_area)/1800 # m^3/s
    gauge_discharge_flux[i] = melt_discharge[gauge_point[1], gauge_point[0]]
    gauge_precip_flux[i] = precip_flux[gauge_point[1], gauge_point[0]]

    wolc_discharge_flux[i] = melt_discharge[wolc_point[1], wolc_point[0]]
    wolc_precip_flux[i] = precip_flux[wolc_point[1], wolc_point[0]]
    woln_discharge_flux[i] = melt_discharge[woln_point[1], woln_point[0]]
    woln_precip_flux[i] = precip_flux[woln_point[1], woln_point[0]]

#%%
# save data to csv files (for each site and the downstream gauge)
df_gauge = pd.DataFrame({
    "date_time": weather_subset_WOLV["date_time"],
    "gauge_hourly_discharge_flux": gauge_discharge_flux,
    "gauge_hourly_precip_flux": gauge_precip_flux
})

df_gauge.to_csv("wolv_gauge_flow_acc.csv", index=False)


df_wolc = pd.DataFrame({
    "date_time": weather_subset_WOLV["date_time"],
    "site_hourly_discharge_flux": wolc_discharge_flux,
    "site_hourly_precip_flux": woln_precip_flux
})

df_wolc.to_csv("wolc_flow_acc.csv", index=False)

df_woln = pd.DataFrame({
    "date_time": weather_subset_WOLV["date_time"],
    "site_hourly_discharge_flux": woln_discharge_flux,
    "site_hourly_precip_flux": woln_precip_flux
})

df_woln.to_csv("woln_flow_acc.csv", index=False)

#%%