#%% import packages
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
    deltat = t_new[1] - t_new[0]
    bins = np.append(t_new, t_new[-1]+deltat)
    #maxv = np.full(t_new.shape, np.nan)
    #minv = np.full(t_new.shape, np.nan)
    meanv = np.full(t_new.shape, np.nan)

    for i in range(bins.shape[0]-1):
        print(type(bins[i]))
        ind = np.where( (t>bins[i]) & (t<=bins[i+1]) )[0]
        if ind.size != 1 & ind.shape[0] != 0:
            #maxv[i] = np.nanmax(x[ind])
            #minv[i] = np.nanmin(x[ind])
            meanv[i] = np.nanmean(x[ind])
        elif ind.size == 1:
            meanv[i] = x[i]

    return meanv # 

#%%

#%%
# make dataframe (add weather and precip data)
def make_subset_LC(w_filename, r_filename, s_filename, start_date, end_date, sampling_freq):
    weather_data =  pd.read_csv(w_filename, delimiter='\t',header=0, parse_dates=[0]) 
    weather_data = weather_data.drop(['WindSp', 'WindDir', 'WindSp.1', 'Time', 'WindSP', 'WindDir.1', 'WindDir.2'], axis=1)
    weather_data = weather_data.rename(columns={"Date/Time": "date_time"})
    weather_data = weather_data.iloc[2:].reset_index(drop=True)
    weather_data['date_time'] = pd.to_datetime(weather_data['date_time']).dt.tz_localize('US/Alaska').dt.tz_convert('UTC')
    subset_weather = weather_data[(weather_data['date_time'] >= start_date) & (weather_data['date_time'] <= end_date)]
    w_subset = pd.DataFrame({'date_time' : pd.date_range(start_date, end_date, freq=sampling_freq )})
    #dt = np.unique(np.diff(w_subset['date_time'].values.astype('datetime64[s]')))[0]
    temp_subset = mean_func(subset_weather["date_time"].values, subset_weather["AirTemp"].values, w_subset["date_time"].values)
    w_subset["Air_Temp"] = temp_subset

    rain_data = pd.read_csv(r_filename, delimiter='\t',header=0, parse_dates=["Time"])
    rain_data = rain_data.iloc[1:].reset_index(drop=True)
    rain_data = rain_data.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    rain_data = rain_data.rename(index=str, columns={"Time": "date_time"})
    rain_data['date_time'] = pd.to_datetime(rain_data.date_time).dt.tz_localize('UTC')
    subset_rain = rain_data[(rain_data['date_time'] >= start_date) & (rain_data['date_time'] <= end_date)]
    tips = sumtime(subset_rain["date_time"].values, subset_rain["Tips"].values, w_subset["date_time"].values)[0]
    precip_subset = (tips *0.2)/1000 
    w_subset["Accum_Rain"] = precip_subset

    df_snow = pd.read_excel(s_filename, sheet_name="LC_Glacier")
    df_snow.columns = ["date", "snow_elv"]
    #df_snow = pd.read_excel(s_filename, sheet_name="LemonCreek_SR")
    #df_snow = df_snow.drop(columns=['system:index', 'SLA_lower_bound_m', 'SLA_m', 'glacier_area_m2', 'ice_area_m2', 'percent_AOI_coverage', 'rock_area_m2', 'snow_area_m2', 'source', 'spatial_scale_m', 'transient_AAR', 'water_area_m2'])
    weekly_snow = pd.to_datetime(df_snow["date"])
    df_snow['date'] = pd.to_datetime(df_snow['date'])
    df_snow.set_index('date', inplace=True)
    #snowline = df_snow['SLA_upper_bound_m'].to_numpy()
    df_snow['snow_elv'] = df_snow['snow_elv'] #- 100 
    snowline = df_snow['snow_elv'].to_numpy()
    hourly_index = pd.date_range(start=df_snow.index.min(), end=df_snow.index.max(), freq='h')
    hourly_series = df_snow.reindex(hourly_index)
    hourly_series = hourly_series.interpolate(method='time')
    print(hourly_series)
    #df_hourly = hourly_series.reset_index().rename(columns={'index': 'date_time', 'SLA_upper_bound_m': 'hourly_upper_SLA'})
    df_hourly = hourly_series.reset_index().rename(columns={'index': 'date_time', 'snow_elv': 'hourly_upper_SLA'})
    df_hourly['date_time'] = (pd.to_datetime(df_hourly['date_time']).dt.tz_localize('UTC'))
    subset_snow = df_hourly[(df_hourly['date_time'] >= start_date) & (df_hourly['date_time'] <= end_date)]
    temp_sla_elv = np.array(subset_snow["hourly_upper_SLA"])
    plt.plot(temp_sla_elv)
    w_subset["SLA"] = temp_sla_elv

    return w_subset


#%%
start_date_LC = pd.Timestamp("2017/06/15 00:00:00").tz_localize('UTC') # start of rain dataset (smallest time constraint)
end_date_LC = pd.Timestamp("2017/09/25 23:00:00").tz_localize('UTC') # end of rain dataset (smallest time constraint)
sampling_freq_LC = '1h'  # weather station data is the lowest resolution (no other w.s. sources have < 1 hr)


w_filename_LC = '/data/stor/proj/keeya_thesis/LC_Glacier/data/raw_data/C17_met_data.txt'
r_filename_LC = '/data/stor/proj/keeya_thesis/LC_Glacier/data/raw_data/C17Rain_cleaned.txt'
s_filename_LC = '/data/stor/proj/keeya_thesis/LC_Glacier/data/raw_data/SLA_glasee.xlsx'
s_filename_LC = '/data/stor/proj/keeya_thesis/LC_Glacier/data/raw_data/snow_line_final.xlsx'

weather_subset_LC = make_subset_LC(w_filename_LC ,r_filename_LC, s_filename_LC, start_date_LC, end_date_LC, sampling_freq_LC)
#%%
# get dem (troughed and normal)
dem_path= '/data/stor/proj/keeya_thesis/LC_Glacier/data/raw_data/lemon_down.tif'
dem = gdal.Open( dem_path )
xres,yres = dem.GetGeoTransform()[1::4]
pix_area = -xres*yres 
GT_dem = dem.GetGeoTransform()
dem_data = dem.GetRasterBand(1).ReadAsArray()
dem_data[dem_data==-9999] = np.nan

WS_elv = 1280
elev_diff = dem_data - WS_elv

x0, y0 = GT_dem[0], GT_dem[3]
x1 = x0 + dem.RasterXSize * xres
y1 = y0 + dem.RasterYSize * yres
xmin, xmax = sorted([x0, x1])
ymin, ymax = sorted([y0, y1])
extent = (xmin, xmax, ymin, ymax)

al = utils.get_rgi_glacier_entities(['RGI60-01.01104'], version='62')
dem_wkt = dem.GetProjection()
dem_crs = CRS.from_wkt(dem_wkt)
al = al.to_crs(dem_crs)

trough_path =  '/data/stor/proj/keeya_thesis/LC_Glacier/data/raw_data/trough_down.tif'
trough = gdal.Open(trough_path)
GT_trough = trough.GetGeoTransform()
trough_data = trough.GetRasterBand(1).ReadAsArray()
rd_trough = rd.rdarray(trough_data, no_data=-9999)
rd_trough.geotransform = GT_trough
trough_filled = rd.FillDepressions(rd_trough, epsilon=True, in_place=False)

# Mask to glacier outline
ny, nx = dem_data.shape
al = al.to_crs(dem_crs)
shapes = [(mapping(geom), 1) for geom in al.geometry]
transform_affine = Affine.from_gdal(*GT_dem)
mask = features.rasterize(shapes, out_shape=(ny, nx), transform=transform_affine, fill=0, all_touched=False, dtype='uint8')

masked_dem = np.where(mask == 1, dem_data, np.nan)


#%%
# Define coordinates of sites on trough DEM 
gauge_point =  (410,1037)

bbgu_point = (1276,1457)
bbgl_point = (1239,1126)

bbeu_point = (1274,1401)
bbel_point = (1240,1130)

bbwu_point = (1272,1482)
bbwl_point = (1213,1033)

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

#plot_trough_flow(trough_filled,bbgu_point)

#%%
# run flow accumulation (for both melt and precipitation) for each site and gauge location 

ddf = 4.5  # in mm/day/C, adjust based on glacier properties
#ddf = 3.6 # snow
lapse_rate = -0.005 # C/m https://www.cambridge.org/core/journals/journal-of-glaciology/article/explaining-mass-balance-and-retreat-dichotomies-at-taku-and-lemon-creek-glaciers-alaska/0E97879B913E3961BED852708237AB95

gauge_discharge_flux = np.zeros(len(weather_subset_LC['date_time']))
gauge_precip_flux = np.zeros(len(weather_subset_LC['date_time']))

bbgl_discharge_flux = np.zeros(len(weather_subset_LC['date_time']))
bbgl_precip_flux = np.zeros(len(weather_subset_LC['date_time']))

bbgu_discharge_flux = np.zeros(len(weather_subset_LC['date_time']))
bbgu_precip_flux = np.zeros(len(weather_subset_LC['date_time']))

bbel_discharge_flux = np.zeros(len(weather_subset_LC['date_time']))
bbel_precip_flux = np.zeros(len(weather_subset_LC['date_time']))

bbeu_discharge_flux = np.zeros(len(weather_subset_LC['date_time']))
bbeu_precip_flux = np.zeros(len(weather_subset_LC['date_time']))

bbwl_discharge_flux = np.zeros(len(weather_subset_LC['date_time']))
bbwl_precip_flux = np.zeros(len(weather_subset_LC['date_time']))

bbwu_discharge_flux = np.zeros(len(weather_subset_LC['date_time']))
bbwu_precip_flux = np.zeros(len(weather_subset_LC['date_time']))


for i in np.arange(len(weather_subset_LC['date_time'])):
    print(i)
    temp_day = weather_subset_LC["Air_Temp"][i]
    temp_pixels = temp_day + lapse_rate *elev_diff
    melt = np.where( ((temp_pixels > 0) & (mask == 1))| ((temp_pixels > 0) & (mask == 0)  & (dem_data > weather_subset_LC["SLA"][i])), ddf * (temp_pixels), 0)

    melt_dt = (melt/1000)/86400 # m/s
    base = np.ones_like(melt_dt)
    runoff_melt = rd.FlowAccumulation(trough_filled, method='D8', weights=melt_dt) # m/s per pixel
    runoff_base = rd.FlowAccumulation(trough_filled, method='D8', weights=base)
    runoff_precip = runoff_base* weather_subset_LC["Accum_Rain"][i] # pix *mm 
    melt_discharge = runoff_melt * pix_area
    precip_flux = (runoff_precip*pix_area)/3600 # m^3/s - should be dt = 3600s 

    gauge_discharge_flux[i] = melt_discharge[gauge_point[1], gauge_point[0]]
    gauge_precip_flux[i] = precip_flux[gauge_point[1], gauge_point[0]]

    bbgl_discharge_flux[i] = melt_discharge[bbgl_point[1], bbgl_point[0]]
    bbgl_precip_flux[i] = precip_flux[bbgl_point[1], bbgl_point[0]]
    bbgu_discharge_flux[i] = melt_discharge[bbgu_point[1], bbgu_point[0]]
    bbgu_precip_flux[i] = precip_flux[bbgu_point[1], bbgu_point[0]]

    bbel_discharge_flux[i] = melt_discharge[bbel_point[1], bbel_point[0]]
    bbel_precip_flux[i] = precip_flux[bbel_point[1], bbel_point[0]]
    bbeu_discharge_flux[i] = melt_discharge[bbeu_point[1], bbeu_point[0]]
    bbeu_precip_flux[i] = precip_flux[bbeu_point[1], bbeu_point[0]]

    bbwl_discharge_flux[i] = melt_discharge[bbwl_point[1], bbwl_point[0]]
    bbwl_precip_flux[i] = precip_flux[bbwl_point[1], bbwl_point[0]]
    bbwu_discharge_flux[i] = melt_discharge[bbwu_point[1], bbwu_point[0]]
    bbwu_precip_flux[i] = precip_flux[bbwu_point[1], bbwu_point[0]]

#%%
# save datasets
df_gauge = pd.DataFrame({
    "date_time": weather_subset_LC["date_time"],
    "gauge_hourly_discharge_flux": gauge_discharge_flux,
    "gauge_hourly_precip_flux": gauge_precip_flux
})

df_gauge.to_csv("gauge_flowacc_07_24_2026.csv", index=False)


df_bbgu = pd.DataFrame({
    "date_time": weather_subset_LC["date_time"],
    "site_hourly_discharge_flux": bbgu_discharge_flux,
    "site_hourly_precip_flux": bbgu_precip_flux
})

df_bbgu.to_csv("bbgu_flowacc_07_24_2026.csv", index=False)

df_bbgl = pd.DataFrame({
    "date_time": weather_subset_LC["date_time"],
    "site_hourly_discharge_flux": bbgl_discharge_flux,
    "site_hourly_precip_flux": bbgl_precip_flux
})

df_bbgl.to_csv("bbgl_flowacc_07_24_2026.csv", index=False)

df_bbeu = pd.DataFrame({
    "date_time": weather_subset_LC["date_time"],
    "site_hourly_discharge_flux": bbeu_discharge_flux,
    "site_hourly_precip_flux": bbeu_precip_flux
})

df_bbeu.to_csv("bbeu_flowacc_07_24_2026.csv", index=False)

df_bbel = pd.DataFrame({
    "date_time": weather_subset_LC["date_time"],
    "site_hourly_discharge_flux": bbel_discharge_flux,
    "site_hourly_precip_flux": bbel_precip_flux
})

df_bbel.to_csv("bbel_flowacc_07_24_2026.csv", index=False)


df_bbwu = pd.DataFrame({
    "date_time": weather_subset_LC["date_time"],
    "site_hourly_discharge_flux": bbwu_discharge_flux,
    "site_hourly_precip_flux": bbwu_precip_flux
})

df_bbwu.to_csv("bbwu_flowacc_07_24_2026.csv", index=False)

df_bbwl = pd.DataFrame({
    "date_time": weather_subset_LC["date_time"],
    "site_hourly_discharge_flux": bbwl_discharge_flux,
    "site_hourly_precip_flux": bbwl_precip_flux
})

df_bbwl.to_csv("bbwl_flowacc_07_24_2026.csv", index=False)

#%%