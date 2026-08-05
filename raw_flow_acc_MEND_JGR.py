#%%
#import packages
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

def make_subset(w_filename, s_filename, start_date, end_date, sampling_freq):
    weather_data =  pd.read_csv(w_filename) 
    weather_data['time_UTC'] = pd.to_datetime(weather_data['time_UTC'],utc=True)
    subset_weather = weather_data[(weather_data['time_UTC'] >= start_date) & (weather_data['time_UTC'] <= end_date)]
    w_subset = pd.DataFrame({'date_time' : pd.date_range(start_date, end_date, freq=sampling_freq )})
    subset_weather["time_UTC"] = subset_weather["time_UTC"].dt.tz_localize(None)
    w_subset["date_time"] = w_subset["date_time"].dt.tz_localize(None)
    temp_subset = mean_func(subset_weather["time_UTC"].values, subset_weather["temp"].values, w_subset["date_time"].values)
    rain = sumtime(subset_weather["time_UTC"].values, subset_weather["precip_hourly"].values, w_subset["date_time"].values)[0] # given in mm we.
    precip_subset = rain/1000
    w_subset["Air_Temp"] = temp_subset
    w_subset["Accum_Rain"] = precip_subset

    df_snow = pd.read_excel(s_filename, sheet_name="MEND_Glacier")
    print(df_snow)
    df_snow.columns = ["date", "snow_elv"]
    #df_snow = df_snow.drop(columns=['system:index', 'SLA_lower_bound_m', 'SLA_m', 'glacier_area_m2', 'ice_area_m2', 'percent_AOI_coverage', 'rock_area_m2', 'snow_area_m2', 'source', 'spatial_scale_m', 'transient_AAR', 'water_area_m2'])
    weekly_snow = pd.to_datetime(df_snow["date"])
    df_snow['date'] = pd.to_datetime(df_snow['date'])
    df_snow.set_index('date', inplace=True)
    #snowline = df_snow['SLA_upper_bound_m'].to_numpy()
    df_snow['snow_elv']=df_snow['snow_elv']#+100
    snowline = df_snow['snow_elv'].to_numpy()
    hourly_index = pd.date_range(start=df_snow.index.min(), end=df_snow.index.max(), freq='1h')
    hourly_series = df_snow.reindex(hourly_index)
    hourly_series = hourly_series.interpolate(method='time')
    #df_hourly = hourly_series.reset_index().rename(columns={'index': 'date_time', 'SLA_upper_bound_m': 'hourly_upper_SLA'})
    df_hourly = hourly_series.reset_index().rename(columns={'index': 'date_time', 'snow_elv': 'hourly_upper_SLA'})
    df_hourly['date_time'] = (pd.to_datetime(df_hourly['date_time']).dt.tz_localize('UTC'))
    subset_snow = df_hourly[(df_hourly['date_time'] >= start_date) & (df_hourly['date_time'] <= end_date)]
    temp_sla_elv = np.array(subset_snow["hourly_upper_SLA"])
    plt.plot(temp_sla_elv)
    w_subset["SLA"] = temp_sla_elv

    return w_subset


#%%
start_date = pd.Timestamp("2012-06-02 00:00:00").tz_localize('UTC') # actualy at 05.01
end_date = pd.Timestamp("2012-09-30 23:00:00").tz_localize('UTC')
sampling_freq_MEND = '1h' # weather station data is the lowest resolution (no other w.s. sources have < 1 hr)

w_filename_MEND = '/data/stor/proj/keeya_thesis/Mendenhall_Glacier/raw_data/airport_2006_2019.csv' # Juneau airport weather (with precip)
s_filename_MEND = '/data/stor/proj/keeya_thesis/LC_Glacier/data/raw_data/snow_line_final.xlsx' # glacee snowline elevation file

weather_subset_MEND = make_subset(w_filename_MEND , s_filename_MEND, start_date, end_date, sampling_freq_MEND)

#%%
dem_path = '/data/stor/proj/keeya_thesis/Mendenhall_Glacier/raw_data/mend_dem_5_final_02_26.tif'
dem = gdal.Open( dem_path )
xres,yres = dem.GetGeoTransform()[1::4]
pix_area = -xres*yres 
GT_dem = dem.GetGeoTransform()
dem_data = dem.GetRasterBand(1).ReadAsArray().astype(float)
dem_data[dem_data==-9999] = np.nan

WS_elv = 20
elev_diff = dem_data - WS_elv

x0, y0 = GT_dem[0], GT_dem[3]
x1 = x0 + dem.RasterXSize * xres
y1 = y0 + dem.RasterYSize * yres
xmin, xmax = sorted([x0, x1])
ymin, ymax = sorted([y0, y1])
extent = (xmin, xmax, ymin, ymax)
extent = (xmin, xmax, ymax, ymin)

#al = utils.get_rgi_glacier_entities(['RGI60-01.11350'], version='62')
outline_path = "/data/stor/proj/keeya_thesis/Mendenhall_Glacier/raw_data/mendenhall_outline.shp"
al = gpd.read_file(outline_path)
# Reproject outline to DEM CRS if needed
dem_wkt = dem.GetProjection()
dem_crs = CRS.from_wkt(dem_wkt)
al = al.to_crs(dem_crs)

trough_path =  '/data/stor/proj/keeya_thesis/Mendenhall_Glacier/raw_data/mend_trenched_dem_02_26.tif'
trough_path =  '/data/stor/proj/keeya_thesis/Mendenhall_Glacier/raw_data/mendenhall_trenched_dem_final_5_02_26.tif'
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
# Define coordinates of site and gauge on trough DEM 
ambr_point = (1460,2900)
gauge_point =  (1037,4535)
#%%
# investigate points on flow accumulation map

def plot_trough_flow(trough, point):
    flow_acc = rd.FlowAccumulation(trough_filled, method='D8')
    plt.figure(figsize=(6, 6))
    plt.imshow(np.log10(flow_acc), origin='upper')
    plt.scatter(point[0],point[1], color="red")
    plt.colorbar(label='log10(flow accumulation)')
    plt.title('Flow Accumulation (D8)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.xlim([1450,1500])
    plt.ylim([2950,2850])
    plt.tight_layout()
    plt.show()

plot_trough_flow(trough_filled,ambr_point)
#%%
# calculate flow accumulation for degree-day melt and precipitation

ddf = 5.7 #https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.00137/full #5.9  # in mm/day/C, adjust based on glacier properties
#ddf = 4.6
lapse_rate = -0.006 # C/m 
gauge_discharge_flux = np.zeros(len(weather_subset_MEND['date_time']))
gauge_precip_flux = np.zeros(len(weather_subset_MEND['date_time']))

ambr_discharge_flux = np.zeros(len(weather_subset_MEND['date_time']))
ambr_precip_flux = np.zeros(len(weather_subset_MEND['date_time']))


for i in np.arange(len(weather_subset_MEND['date_time'])):
    temp_day = weather_subset_MEND["Air_Temp"][i]
    temp_pixels = temp_day + lapse_rate *elev_diff
    melt = np.where( ((temp_pixels > 0) & (mask == 1))| ((temp_pixels > 0) & (mask == 0)  & (dem_data > weather_subset_MEND["SLA"][i])), ddf * (temp_pixels), 0)
    melt_dt = ((melt/1000)/86400).astype(np.float64)
    base = np.ones_like(melt_dt)
    runoff_melt = rd.FlowAccumulation(trough_filled, method='D8', weights=melt_dt) # m/s per pixel
    runoff_base = rd.FlowAccumulation(trough_filled, method='D8', weights=base)
    runoff_precip = runoff_base* weather_subset_MEND["Accum_Rain"][i] # pix *mm 
    melt_discharge = runoff_melt * pix_area
    #precip_flux = (runoff_precip*pix_area)/3600 # m^3/s - should be dt = 3600s 
    precip_flux = (runoff_precip*pix_area)/1800
    gauge_discharge_flux[i] = melt_discharge[gauge_point[1], gauge_point[0]]
    gauge_precip_flux[i] = precip_flux[gauge_point[1], gauge_point[0]]

    ambr_discharge_flux[i] = melt_discharge[ambr_point[1], ambr_point[0]]
    ambr_precip_flux[i] = precip_flux[ambr_point[1], ambr_point[0]]

#%%
# save data to csv files
df_gauge = pd.DataFrame({
    "date_time": weather_subset_MEND["date_time"],
    "gauge_hourly_discharge_flux": gauge_discharge_flux,
    "gauge_hourly_precip_flux": gauge_precip_flux
})

df_gauge.to_csv("mend_gauge_flow_acc.csv", index=False)


df_ambr = pd.DataFrame({
    "date_time": weather_subset_MEND["date_time"],
    "site_hourly_discharge_flux": ambr_discharge_flux,
    "site_hourly_precip_flux": ambr_precip_flux
})

df_ambr.to_csv("ambr_model_flow_acc.csv", index=False)
#%%
