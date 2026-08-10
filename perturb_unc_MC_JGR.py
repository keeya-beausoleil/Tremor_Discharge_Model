#%%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import xarray as xr
from scipy import signal
from matplotlib.ticker import MultipleLocator
import sys
import rasterio
from pyproj import CRS
from shapely.geometry import mapping
from rasterio import features
import geopandas as gpd
from affine import Affine
from rasterio import features
import math
from scipy.io import loadmat
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.colors as cols
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist
from itertools import product
#%%

#%%
# These functions define the complete subglacial source discharge workflows from each glacier to easily re-run different flow accumulation scenarios. 

# Warning: This testing was an addition after the workflows existed and built into one another. I recognize there is a more efficient process to construct external functions that can be used amongst many files. 
# This will be considered and implemented for future work on the project, however, with the time constraints on my thesis... whoops, a great learning experience!
def load_file(file_name, file_type):
    temp_file = pd.read_csv(file_name)
    if file_type == "tremor":
        temp_tremor = temp_file["T_Amp_corr"].to_numpy()
        temp_time = pd.to_datetime(temp_file["date_time"])
        tremor_df = pd.DataFrame({"date_time": temp_time,"T_Amp_corr": temp_tremor})
        return temp_tremor, tremor_df
    if file_type == "gauge_flowacc":
        temp_file["gauge_hourly_discharge_flux"].to_numpy()
        melt_flux = temp_file["gauge_hourly_discharge_flux"].to_numpy()
        precip_flux = temp_file["gauge_hourly_precip_flux"].to_numpy()
        temp_time = pd.to_datetime(temp_file["date_time"])
        return melt_flux, precip_flux, temp_time
    if file_type == "site_flowacc":
        melt_flux = temp_file["site_hourly_discharge_flux"].to_numpy()
        precip_flux = temp_file["site_hourly_precip_flux"].to_numpy()
        temp_time = pd.to_datetime(temp_file["date_time"])
        return melt_flux, precip_flux, temp_time

def get_gauge_wolv(file_name, start_date, end_date, sg_code):
    sg_df = pd.read_csv(file_name, sep='\t',comment='#')
    sg_df = sg_df.iloc[1:]
    sg_df['datetime'] = pd.to_datetime(sg_df['datetime'], errors='coerce')
    sg_df['datetime'] = sg_df['datetime'].dt.tz_localize('America/Anchorage').dt.tz_convert('UTC') # stream gauge observations are in AKDT (in file)
    filtered_df = sg_df[(sg_df['datetime'] >= start_date) & (sg_df['datetime'] <= end_date)]
    filtered_df.set_index('datetime', inplace=True)
    filtered_df[sg_code] = pd.to_numeric(filtered_df[sg_code], errors='coerce')
    filtered_df = filtered_df[[sg_code]]  # Remove any non-numeric columns
    filtered_df = filtered_df.resample('30min').mean()
    filtered_df["Q"] = filtered_df[sg_code]/(3.2804**3)
    Q = np.array(filtered_df["Q"])
    Q_df = pd.DataFrame({"date_time": filtered_df.index,"Q": Q})
    return Q, Q_df

def get_gauge_lc(file_name, start_date, end_date, sg_code):
    sg_df = pd.read_csv(file_name)
    sg_df=sg_df.drop(columns=['x', 'y', 'id', 'time_series_id', 'monitoring_location_id','parameter_code', 'statistic_id', 'unit_of_measure','approval_status', 'qualifier', 'last_modified'])
    sg_df["time"] = pd.to_datetime(sg_df["time"], utc=True, errors="coerce")
    sg_df = sg_df.sort_values("time").reset_index(drop=True)
    sg_df = sg_df.dropna(subset=["time"])
    
    sg_df = sg_df[(sg_df['time'] >= start_date) & (sg_df['time'] <= end_date)]
    sg_df["value"] = pd.to_numeric(sg_df["value"], errors="coerce")
    sg_df = sg_df.set_index("time").resample("1h").mean()

    Q = sg_df["value"].to_numpy()
    QQ = Q/(3.2804**3)

    QQ_df = pd.DataFrame({"date_time": sg_df.index,"Q": QQ})
    return QQ,QQ_df

def get_gauge_mend(start_date, end_date):
    stage_mat = loadmat("/data/stor/proj/keeya_thesis/Mendenhall_Glacier/raw_data/MendenhallLake.stage2012.mat", squeeze_me=True, struct_as_record=False)
    discharge_mat = loadmat("/data/stor/proj/keeya_thesis/Mendenhall_Glacier/raw_data/MendenhallLake.discharge2012.mat", squeeze_me=True, struct_as_record=False)
    stage = stage_mat["stage"]
    discharge = discharge_mat["discharge"]

    def matlab_datenum_to_datetime(dn):
        dn = np.asarray(dn)
        return np.array([
            datetime.fromordinal(int(d))
            + timedelta(days=d % 1)
            - timedelta(days=366)
            for d in dn ])

    def get(obj, field):
        return getattr(obj, field)
    def set_field(obj, field, value):
        setattr(obj, field, value)
        return obj

    stage_data = np.asarray(get(stage, "data"), dtype=float)
    window = 5
    kernel = np.ones(window) / window
    stage_smooth = np.convolve(stage_data, kernel, mode="same")
    dt = 1 / stage.Fs
    dhdt = np.diff(stage_smooth)/ dt
    area = 4.225e6
    dis_data = np.asarray(get(discharge, "data"), dtype=float)
    set_field(discharge, "data", dis_data[:-1])
    dis_data = np.asarray(get(discharge, "data"))
    dis_data = dis_data[:-1]
    min_len = min(len(dhdt), len(dis_data))
    dhdt = dhdt[:min_len]
    dis_data = dis_data[:min_len]

    Qin = dhdt * area + dis_data
    start_time = get(discharge, "start")
    freq = discharge.Fs
    samp_int = 1 / freq
    Qin_time = start_time + np.arange(len(Qin)) * (samp_int / 86400)
    Qin_time = Qin_time + 8/24
    Qin_dt = matlab_datenum_to_datetime(Qin_time)
    discharge_date_time = pd.Series(pd.to_datetime(Qin_dt[100:-100]).round("15min"),name="date_time")
    discharge_Q = Qin[100:-100]
    df = pd.DataFrame({ "Q": discharge_Q}, index=discharge_date_time)

    discharge_df = df.resample("1h").mean()
    discharge_subset = discharge_df.loc[start_date:end_date].reset_index()
    Q = discharge_subset["Q"].values
    return Q, discharge_subset

def get_smooth(data, tau, time_array):
    exp_factor = -1 / tau
    kern_len = tau * 5
    t_kern = np.arange(0, kern_len, 1)
    kernel = np.exp(exp_factor * t_kern)
    kernel = kernel / np.sum(kernel)
    return pd.Series(np.convolve(data, kernel, 'full')[:len(time_array)])

def gauge_LSQ(melt_smooth, precip_smooth, gauge_obs, time):
    model = []
    ss_tot = np.sum((gauge_obs - np.mean(gauge_obs))**2)
    A = np.vstack([melt_smooth,precip_smooth]).T
    gauge_obs_T  = gauge_obs.T
    beta, res, rank, s = np.linalg.lstsq(A, gauge_obs_T, rcond=None)
    pred = A @ beta
    residuals = gauge_obs - pred
    ss_res = np.sum(residuals**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    result = {"r2": r2, "rmse": rmse, "beta": beta}
    model_df = pd.DataFrame({"date_time": time,"model": pred})
    return result, model_df

def downscale_tau(best_tau,perc_area):
    tau = best_tau*perc_area 
    exp = -1/(tau)
    kern = (tau *5) 
    t = np.arange(0, kern, 1)
    kernel = np.exp(exp*t)
    kernel = kernel/np.sum(kernel)
    return kernel

def site_model(m_flux,kernel_m,p_flux, kernel_p,time,betas):
    smooth_melt = np.convolve(m_flux, kernel_m, 'full')[:len(time)]
    smooth_precip = np.convolve(p_flux, kernel_p, 'full')[:len(time)]
    model = smooth_melt*betas[0] + smooth_precip*betas[1]
    model_df = pd.DataFrame({"date_time": time,"model": model})
    return model_df

def site_ratio(gauge_model, site_model, window_len, discharge,pts_day): 
    start_temp = max(site_model["date_time"].min(), gauge_model["date_time"].min())
    end_temp = min(site_model["date_time"].max(), gauge_model["date_time"].max())
    site_model_subset = site_model[(site_model['date_time'] >= start_temp) & (site_model['date_time'] <= end_temp)]
    gauge_model_subset = gauge_model[(gauge_model['date_time'] >= start_temp) & (gauge_model['date_time'] <= end_temp)]
    discharge_subset = discharge[(discharge['date_time'] >= start_temp) & (discharge['date_time'] <= end_temp)]
    site_model_subset = site_model_subset.set_index("date_time")
    gauge_model_subset = gauge_model_subset.set_index("date_time")
    discharge_subset = discharge_subset.set_index("date_time")
    site_ratio_temp = site_model_subset["model"]/gauge_model_subset["model"]
    final_ratio = site_ratio_temp.rolling(window_len*pts_day, center=True, min_periods=math.ceil(window_len)*pts_day).mean()
    site_est =  discharge_subset["Q"]*final_ratio 
    site_est_df = (pd.DataFrame({"final_site_est": site_est,"Q_obs": discharge_subset["Q"]}).rename_axis("date_time").reset_index())
    return site_est_df

def align_V_Q(site_est, tremor,lag):
    site_est = site_est.copy()
    tremor = tremor.copy()
    site_est["date_time"] = pd.to_datetime(site_est["date_time"]).dt.tz_localize(None)
    tremor["date_time"] = pd.to_datetime(tremor["date_time"]).dt.tz_localize(None)
    if lag == 0:
        site_V_Q_final = pd.merge(site_est, tremor, on="date_time", how="inner")
    else:
        site_V_Q_final = pd.merge(site_est[lag:], tremor[:-lag], on="date_time",how="inner" )
    return site_V_Q_final

def runoff_model_lc(tremor_data, gauge_data, site_data, tau_m, tau_p, ratio_window):
    start_date = pd.Timestamp("2017/06/15 00:00:00").tz_localize('UTC') # start of rain dataset (smallest time constraint)
    end_date = pd.Timestamp("2017/09/25 23:00:00").tz_localize('UTC') # end of rain dataset (smallest time constraint)
    #end_date = pd.Timestamp("2017/09/05 11:00:00").tz_localize('UTC')
    sg_code = "1294_00060"
    
    sites       = ["bbgl", "bbgu"]
    lower_sites = ["bbgl"]
    upper_sites = ["bbgu"]


    tremor_df = {}
    for site, fp in zip(sites, tremor_data):
        _, tremor_df[site] = load_file(fp, "tremor")

    gauge_melt_flux,gauge_precip_flux,gauge_time = load_file(gauge_data[0], "gauge_flowacc")
    
    site_df = {}
    for site, fp in zip(sites, site_data):
        melt, precip, time = load_file(fp, "site_flowacc")
        site_df[site] = {"melt": melt, "precip": precip, "time": time}

    discharge, discharge_df = get_gauge_lc('/data/stor/proj/keeya_thesis/LC_Glacier/data/raw_data/lemon_creek_discharge_02.15.2026.csv',start_date,end_date,sg_code)
    valid_mask = (
    ~np.isnan(discharge) &
    ~np.isnan(gauge_melt_flux) &
    ~np.isnan(gauge_precip_flux))

    discharge_final = discharge[valid_mask]
    gauge_melt_flux_final = gauge_melt_flux[valid_mask]
    gauge_precip_flux_final = gauge_precip_flux[valid_mask]
    gauge_time_final = gauge_time[valid_mask]

    gauge_melt_smooth = get_smooth(gauge_melt_flux_final,tau_m,gauge_time_final)
    gauge_precip_smooth= get_smooth(gauge_precip_flux_final,tau_p,gauge_time_final)

    gauge_LSQ_best, gauge_model = gauge_LSQ(gauge_melt_smooth, gauge_precip_smooth, discharge_final,gauge_time_final)
    print("LC pipeline beta:", gauge_LSQ_best["beta"])
    print("LC pipeline r2:", gauge_LSQ_best["r2"])
    kernel_melt_l = downscale_tau(tau_m,0.181)
    kernel_melt_u = downscale_tau(tau_m,0.1116)
    kernel_precip_l = downscale_tau(tau_p,0.181)
    kernel_precip_u = downscale_tau(tau_p,0.1116)

    site_models = {}
    for site in lower_sites:
        d = site_df[site]
        site_models[site] = site_model(d["melt"], kernel_melt_l, d["precip"], kernel_precip_l, d["time"], gauge_LSQ_best["beta"])
    for site in upper_sites:
        d = site_df[site]
        site_models[site] = site_model(d["melt"], kernel_melt_u, d["precip"], kernel_precip_u, d["time"], gauge_LSQ_best["beta"])

    site_est_dfs = {site: site_ratio(gauge_model, site_models[site], ratio_window, discharge_df, 24) for site in sites}
    
    lag_u = round(7800/3600)
    lag_l = round(6300/3600)

    V_Q_finals = {}
    for site in lower_sites:
        V_Q_finals[site] = align_V_Q(site_est_dfs[site], tremor_df[site], lag_l)
    for site in upper_sites:
        V_Q_finals[site] = align_V_Q(site_est_dfs[site], tremor_df[site], lag_u)

    return V_Q_finals

def runoff_model_wolv(tremor_data, gauge_data, site_data, tau_m, tau_p, ratio_window):
    start_date = pd.Timestamp("2022/05/12 22:00:00").tz_localize('UTC') # start of rain dataset (smallest time constraint)
    end_date = pd.Timestamp("2022/09/30 22:00:00").tz_localize('UTC') # end of rain dataset (smallest time constraint)
    sg_code = "1424_00060"

    sites       = ["wolc", "woln"]
    upper_sites = ["wolc"]
    lower_sites = ["woln"]


    tremor_df = {}
    for site, fp in zip(sites, tremor_data):
        _, tremor_df[site] = load_file(fp, "tremor")

    gauge_melt_flux,gauge_precip_flux,gauge_time = load_file(gauge_data[0], "gauge_flowacc")
    
    site_df = {}
    for site, fp in zip(sites, site_data):
        melt, precip, time = load_file(fp, "site_flowacc")
        site_df[site] = {"melt": melt, "precip": precip, "time": time}

    discharge, discharge_df= get_gauge_wolv("/data/stor/proj/keeya_thesis/Wolverine_Glacier/data/raw_data/wolv_streamgauge_02.10.2025.txt",start_date,end_date,sg_code)
    discharge_df["date_time"] = discharge_df["date_time"].dt.tz_localize(None)
    valid_mask = (
    ~np.isnan(discharge) &
    ~np.isnan(gauge_melt_flux) &
    ~np.isnan(gauge_precip_flux))

    discharge_final = discharge[valid_mask]
    gauge_melt_flux_final = gauge_melt_flux[valid_mask]
    gauge_precip_flux_final = gauge_precip_flux[valid_mask]
    gauge_time_final = gauge_time[valid_mask]

    gauge_melt_smooth = get_smooth(gauge_melt_flux_final,tau_m,gauge_time_final)
    gauge_precip_smooth= get_smooth(gauge_precip_flux_final,tau_p,gauge_time_final)
    
    gauge_LSQ_best, gauge_model = gauge_LSQ(gauge_melt_smooth, gauge_precip_smooth, discharge_final,gauge_time_final)
    print("WOLV pipeline beta:", gauge_LSQ_best["beta"])
    print("WOLV pipeline r2:", gauge_LSQ_best["r2"])
    kernel_melt_n = downscale_tau(tau_m,0.55)
    kernel_melt_c = downscale_tau(tau_m,0.314)
    kernel_precip_n = downscale_tau(tau_p,0.55)
    kernel_precip_c = downscale_tau(tau_p,0.314)
    
    site_models = {}
    for site in lower_sites:
        d = site_df[site]
        site_models[site] = site_model(d["melt"], kernel_melt_n, d["precip"], kernel_precip_n, d["time"], gauge_LSQ_best["beta"])
    for site in upper_sites:
        d = site_df[site]
        site_models[site] = site_model(d["melt"], kernel_melt_c, d["precip"], kernel_precip_c, d["time"], gauge_LSQ_best["beta"])

    site_est_dfs = {site: site_ratio(gauge_model, site_models[site], ratio_window, discharge_df,48) for site in sites}

    lag_c = round(6700/3600)
    lag_n = round(4000/3600)

    V_Q_finals = {}
    for site in lower_sites:
        V_Q_finals[site] = align_V_Q(site_est_dfs[site], tremor_df[site], lag_n)
    for site in upper_sites:
        V_Q_finals[site] = align_V_Q(site_est_dfs[site], tremor_df[site], lag_c)

    return V_Q_finals

def runoff_model_mend(tremor_data, gauge_data, site_data, tau_m, tau_p, ratio_window):
    start_date = pd.Timestamp("2012/06/02 00:00:00")#.tz_localize('UTC') # start of rain dataset (smallest time constraint)
    end_date = pd.Timestamp("2012/09/30 23:00:00")#.tz_localize('UTC') # end of rain dataset (smallest time constraint)

    sites       = ["ambr"]

    tremor_df = {}
    for site, fp in zip(sites, tremor_data):
        _, tremor_df[site] = load_file(fp, "tremor")

    gauge_melt_flux,gauge_precip_flux,gauge_time = load_file(gauge_data[0], "gauge_flowacc")
    
    site_df = {}
    for site, fp in zip(sites, site_data):
        melt, precip, time = load_file(fp, "site_flowacc")
        site_df[site] = {"melt": melt, "precip": precip, "time": time}

    discharge, discharge_df= get_gauge_mend(start_date,end_date)
    discharge_df["date_time"] = discharge_df["date_time"].dt.tz_localize(None)

    gauge_melt_flux= gauge_melt_flux[:-17]
    gauge_precip_flux = gauge_precip_flux[:-17]
    gauge_time = gauge_time[:-17]
 
    valid_mask = (
    ~np.isnan(discharge) &
    ~np.isnan(gauge_melt_flux) &
    ~np.isnan(gauge_precip_flux))

    discharge_final = discharge[valid_mask]
    gauge_melt_flux_final = gauge_melt_flux[valid_mask]
    gauge_precip_flux_final = gauge_precip_flux[valid_mask]
    gauge_time_final = gauge_time[valid_mask]

    gauge_melt_smooth = get_smooth(gauge_melt_flux_final,tau_m,gauge_time_final)
    gauge_precip_smooth= get_smooth(gauge_precip_flux_final,tau_p,gauge_time_final)
    
    gauge_LSQ_best, gauge_model = gauge_LSQ(gauge_melt_smooth, gauge_precip_smooth, discharge_final,gauge_time_final)
    
    kernel_melt = downscale_tau(tau_m,0.5)
    kernel_precip = downscale_tau(tau_p,0.5)
    
    site_models = {}
    for site in sites:
        d = site_df[site]
        site_models[site] = site_model(d["melt"], kernel_melt, d["precip"], kernel_precip, d["time"], gauge_LSQ_best["beta"])

    site_est_dfs = {site: site_ratio(gauge_model, site_models[site], ratio_window, discharge_df,24) for site in sites}

    lag = round(2000/3600)

    V_Q_finals = {}
    for site in sites:
        V_Q_finals[site] = align_V_Q(site_est_dfs[site], tremor_df[site], lag)

    return V_Q_finals

#%%
def iterate_runoff_model(glaciers): 

    all_parameters= ["Ice Thickness", "Conduit Pathway", "Degree Day Factor", "Snowline Elevation", "Exp. Smoothing Length", "Ratio Window Length"]
    window_array = [7,14,21]
    all_results = {}

    if "Lemon_Creek"in glaciers:
        glacier = "Lemon_Creek"
        all_results[glacier] = {}
        tremor_files_IT = {
            "best_path_best_IT":["bbgl_tremor.csv","bbgu_tremor.csv"],
            "best_path_high_IT":["bbgl_tremor_high_IT.csv","bbgu_tremor_high_IT.csv"],
            "best_path_low_IT":["bbgl_tremor_low_IT.csv","bbgu_tremor_low_IT.csv"]} 
        tremor_files_CP = {
            "best_path_best_IT":["bbgl_tremor.csv","bbgu_tremor.csv"],
            "left_path_best_IT":["bbgl_tremor_left.csv","bbgu_tremor_left.csv"],
            "right_path_best_IT":["bbgl_tremor_right.csv","bbgu_tremor_right.csv"]}
        gauge_files_LR = { 
            "ice_LR_best_SL": ['gauge_flowacc_07_24_2026.csv'], 
            "snow_LR_best_SL": ['gauge_flowacc_snow_LR.csv']}
        gauge_files_SL = { 
            "ice_LR_best_SL": ['gauge_flowacc_07_24_2026.csv'],
            "ice_LR_low_SL": ['gauge_flowacc_low_snowline.csv'],
            "ice_LR_high_SL": ['gauge_flowacc_high_snowline.csv']}
        site_files_LR = { 
            "snow_LR_best_SL": ['bbgl_flowacc_snow_LR.csv',"bbgu_flowacc_snow_LR.csv"],
            "ice_LR_best_SL": ['bbgl_flowacc_07_24_2026.csv',"bbgu_flowacc_07_24_2026.csv"]} 
        site_files_SL = { 
            "ice_LR_best_SL": ['bbgl_flowacc_07_24_2026.csv',"bbgu_flowacc_07_24_2026.csv"],
            "ice_LR_low_SL": ["bbgl_flowacc_low_snowline.csv","bbgu_flowacc_low_snowline.csv"],
            "ice_LR_high_SL": ["bbgl_flowacc_high_snowline.csv","bbgu_flowacc_high_snowline.csv"] } 
        tau_m_array = [4,3] # full time series & July/August are same, two weeks in August? 
        tau_p_array = [25,13] 

        for p in all_parameters: 
            all_results[glacier][p] = {}
            opt_tremor = tremor_files_IT["best_path_best_IT"]
            opt_gauge = gauge_files_LR["ice_LR_best_SL"]
            opt_site = site_files_LR["ice_LR_best_SL"]
            opt_window = window_array[1]
            opt_tau_m = tau_m_array[0]
            opt_tau_p = tau_p_array[0] 
            if p == "Ice Thickness": 
                for scenario in tremor_files_IT: 
                    V_Q_final = runoff_model_lc(tremor_files_IT[scenario], opt_gauge, opt_site,opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Conduit Pathway": 
                for scenario in tremor_files_CP: 
                    V_Q_final = runoff_model_lc(tremor_files_CP[scenario], opt_gauge, opt_site,opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Degree Day Factor": 
                for scenario in gauge_files_LR: 
                    V_Q_final = runoff_model_lc(opt_tremor, gauge_files_LR[scenario], site_files_LR[scenario],opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Snowline Elevation": 
                for scenario in gauge_files_SL: 
                    V_Q_final = runoff_model_lc(opt_tremor, gauge_files_SL[scenario], site_files_SL[scenario],opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Exp. Smoothing Length": 
                for i in range(len(tau_m_array)): 
                    V_Q_final = runoff_model_lc(opt_tremor, opt_gauge, opt_site,tau_m_array[i], tau_p_array[i],opt_window)
                    all_results[glacier][p][i] = {"V_Q": V_Q_final}
            if p == "Ratio Window Length": 
                for i in window_array: 
                    V_Q_final = runoff_model_lc(opt_tremor, opt_gauge, opt_site,opt_tau_m, opt_tau_p,i)
                    all_results[glacier][p][i] = {"V_Q": V_Q_final}
    
    if "Wolverine" in glaciers :
        glacier = "Wolverine" 
        all_results[glacier] = {}
        start_date = pd.Timestamp("2022/05/12 22:00:00").tz_localize('UTC') # start of rain dataset (smallest time constraint)
        end_date = pd.Timestamp("2022/09/30 22:00:00").tz_localize('UTC') # end of rain dataset (smallest time constraint)
        sg_code = "1424_00060"
        tremor_files_IT = {
            "best_path_best_IT":["wolc_tremor.csv","woln_tremor.csv"],
            "best_path_high_IT":["wolc_tremor_high_IT.csv","woln_tremor_high_IT.csv"],
            "best_path_low_IT":["wolc_tremor_low_IT.csv","woln_tremor_low_IT.csv"]}
        tremor_files_CP = {
            "best_path_best_IT":["wolc_tremor.csv","woln_tremor.csv"],
            "right_path_best_IT":["wolc_tremor_right.csv","woln_tremor_right.csv"],
            "left_path_best_IT":["wolc_tremor_left.csv","woln_tremor_left.csv"]}
        gauge_files_LR = { 
            "snow_LR_best_SL": ["wolv_gauge_flow_acc_snow_LR.csv"],
            "ice_LR_best_SL": ["wolv_gauge_model_melt_and_precip.csv"]}
        gauge_files_SL = { 
            "ice_LR_best_SL": ["wolv_gauge_model_melt_and_precip.csv"],
            "ice_LR_low_SL": ["wolv_gauge_flow_acc_low_SL.csv"],
            "ice_LR_high_SL": ["wolv_gauge_flow_acc_high_SL.csv"] }
        site_files_LR = { 
            "snow_LR_best_SL": ["wolc_flow_acc_snow_LR.csv","woln_flow_acc_snow_LR.csv"],
            "ice_LR_best_SL": ["wolc_model_melt_and_precip.csv","woln_model_melt_and_precip.csv"]}
        site_files_SL = { 
            "ice_LR_best_SL": ["wolc_model_melt_and_precip.csv","woln_model_melt_and_precip.csv"],
            "ice_LR_low_SL": ["wolc_flow_acc_low_SL.csv","woln_flow_acc_low_SL.csv"],
            "ice_LR_high_SL": ["wolc_flow_acc_high_SL.csv","woln_flow_acc_high_SL.csv"] } 
        tau_m_array = [120,15,10] # full time series, July/August, two weeks in August? 
        tau_p_array = [187,85,25]

        for p in all_parameters: 
            all_results[glacier][p] = {}
            opt_tremor = tremor_files_IT["best_path_best_IT"]
            opt_gauge = gauge_files_LR["ice_LR_best_SL"]
            opt_site = site_files_LR["ice_LR_best_SL"]
            opt_window = window_array[1]
            opt_tau_m = tau_m_array[1]
            opt_tau_p = tau_p_array[1]
            if p == "Ice Thickness": 
                for scenario in tremor_files_IT: 
                    V_Q_final = runoff_model_wolv(tremor_files_IT[scenario], opt_gauge, opt_site,opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Conduit Pathway": 
                for scenario in tremor_files_CP: 
                    V_Q_final = runoff_model_wolv(tremor_files_CP[scenario], opt_gauge, opt_site,opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Degree Day Factor": 
                for scenario in gauge_files_LR: 
                    V_Q_final = runoff_model_wolv(opt_tremor, gauge_files_LR[scenario], site_files_LR[scenario],opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Snowline Elevation": 
                for scenario in gauge_files_SL: 
                    V_Q_final = runoff_model_wolv(opt_tremor, gauge_files_SL[scenario], site_files_SL[scenario],opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Exp. Smoothing Length": 
                for i in range(len(tau_m_array)): 
                    V_Q_final = runoff_model_wolv(opt_tremor, opt_gauge, opt_site,tau_m_array[i], tau_p_array[i],opt_window)
                    all_results[glacier][p][i] = {"V_Q": V_Q_final}
            if p == "Ratio Window Length": 
                for i in window_array: 
                    V_Q_final = runoff_model_wolv(opt_tremor, opt_gauge, opt_site,opt_tau_m, opt_tau_p,i)
                    all_results[glacier][p][i] = {"V_Q": V_Q_final}

    if "Mendenhall" in glaciers :
        glacier = "Mendenhall" 
        all_results[glacier] = {}
        start_date = pd.Timestamp("2012/06/02 00:00:00")#.tz_localize('UTC') # start of rain dataset (smallest time constraint)
        end_date = pd.Timestamp("2012/09/30 23:00:00")#.tz_localize('UTC') # end of rain dataset (smallest time constraint)
        tremor_files_IT = {
            "best_path_best_IT":["ambr_tremor.csv"],
            "best_path_high_IT":["ambr_tremor_high_IT.csv"],
            "best_path_low_IT":["ambr_tremor_low_IT.csv"]}
        tremor_files_CP = {
            "best_path_best_IT":["ambr_tremor.csv"],
            "left_path_best_IT":["ambr_tremor_left.csv"],
            "right_path_best_IT":["ambr_tremor_right.csv"] }
        gauge_files_LR = { 
            "snow_LR_best_SL": ["mend_gauge_flow_acc_snow_LR.csv"],
            "ice_LR_best_SL": ["mend_gauge_model_melt_and_precip.csv"]}
        gauge_files_SL = { 
            "ice_LR_best_SL": ["mend_gauge_model_melt_and_precip.csv"],
            "ice_LR_low_SL": ["mend_gauge_flow_acc_low_SL.csv"],
            "ice_LR_high_SL": ["mend_gauge_flow_acc_high_SL.csv"] }
        site_files_LR = { 
            "snow_LR_best_SL": ["ambr_model_flow_acc_snow_LR.csv"],
            "ice_LR_best_SL": ["ambr_model_melt_and_precip.csv"]}
        site_files_SL = { 
            "ice_LR_best_SL": ["ambr_model_melt_and_precip.csv"],
            "ice_LR_low_SL": ["ambr_model_flow_acc_low_SL.csv"],
            "ice_LR_high_SL": ["ambr_model_flow_acc_high_SL.csv"] } 
        tau_m_array = [451,19,15] # full time series, July/August, two weeks in August? 
        tau_p_array = [31,49,25]

        for p in all_parameters: 
            all_results[glacier][p] = {}
            opt_tremor = tremor_files_IT["best_path_best_IT"]
            opt_gauge = gauge_files_LR["ice_LR_best_SL"]
            opt_site = site_files_LR["ice_LR_best_SL"]
            opt_window = window_array[1]
            opt_tau_m = tau_m_array[1]
            opt_tau_p = tau_p_array[1]
            if p == "Ice Thickness": 
                for scenario in tremor_files_IT: 
                    V_Q_final = runoff_model_mend(tremor_files_IT[scenario], opt_gauge, opt_site,opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Conduit Pathway": 
                for scenario in tremor_files_CP: 
                    V_Q_final = runoff_model_mend(tremor_files_CP[scenario], opt_gauge, opt_site,opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Degree Day Factor": 
                for scenario in gauge_files_LR: 
                    V_Q_final = runoff_model_mend(opt_tremor, gauge_files_LR[scenario], site_files_LR[scenario],opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Snowline Elevation": 
                for scenario in gauge_files_SL: 
                    V_Q_final = runoff_model_mend(opt_tremor, gauge_files_SL[scenario], site_files_SL[scenario],opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Exp. Smoothing Length": 
                for i in range(len(tau_m_array)): 
                    V_Q_final = runoff_model_mend(opt_tremor, opt_gauge, opt_site,tau_m_array[i], tau_p_array[i],opt_window)
                    all_results[glacier][p][i] = {"V_Q": V_Q_final}
            if p == "Ratio Window Length": 
                for i in window_array: 
                    V_Q_final = runoff_model_mend(opt_tremor, opt_gauge, opt_site,opt_tau_m, opt_tau_p,i)
                    all_results[glacier][p][i] = {"V_Q": V_Q_final}

    print("all_results keys sample:", list(all_results.keys()))
    min_len = min(
    len(df)
    for glacier in all_results
    for p in all_results[glacier]
    for scenario in all_results[glacier][p]
    for site, df in all_results[glacier][p][scenario]["V_Q"].items())
    print("min rows:", min_len)
    return all_results
all_results = iterate_runoff_model(["Lemon_Creek", "Wolverine", "Mendenhall"])

#%%
# this adapted function replicates the empirical model... good test to esnure perturbations are running properly (calling correct files etc.) & matches final results
'''
def iterate_runoff_model_opt(glaciers): 

    all_parameters= ["Ice Thickness", "Conduit Pathway", "Degree Day Factor", "Snowline Elevation", "Exp. Smoothing Length", "Ratio Window Length"]
    window_array = [14]
    all_results = {}

    if "Lemon_Creek"in glaciers:
        glacier = "Lemon_Creek"
        all_results[glacier] = {}
        tremor_files_IT = {
            "best_path_best_IT":["bbgl_tremor.csv","bbgu_tremor.csv"],}
        tremor_files_CP = {
            "best_path_best_IT":["bbgl_tremor.csv","bbgu_tremor.csv"],}
        gauge_files_LR = { 
            "ice_LR_best_SL": ['gauge_flowacc_07_24_2026.csv'], }
        gauge_files_SL = { 
            "ice_LR_best_SL": ['gauge_flowacc_07_24_2026.csv'],}
        site_files_LR = { 
            "ice_LR_best_SL": ['bbgl_flowacc_07_24_2026.csv',"bbgu_flowacc_07_24_2026.csv"]} 
        site_files_SL = { 
            "ice_LR_best_SL": ['bbgl_flowacc_07_24_2026.csv',"bbgu_flowacc_07_24_2026.csv"], } 
        tau_m_array = [2] 
        tau_p_array = [25] 

        for p in all_parameters: 
            all_results[glacier][p] = {}
            opt_tremor = tremor_files_IT["best_path_best_IT"]
            opt_gauge = gauge_files_LR["ice_LR_best_SL"]
            opt_site = site_files_LR["ice_LR_best_SL"]
            opt_window = window_array[0]
            opt_tau_m = tau_m_array[0]
            opt_tau_p = tau_p_array[0] 
            if p == "Ice Thickness": 
                for scenario in tremor_files_IT: 
                    V_Q_final = runoff_model_lc(tremor_files_IT[scenario], opt_gauge, opt_site,opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Conduit Pathway": 
                for scenario in tremor_files_CP: 
                    V_Q_final = runoff_model_lc(tremor_files_CP[scenario], opt_gauge, opt_site,opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Degree Day Factor": 
                for scenario in gauge_files_LR: 
                    V_Q_final = runoff_model_lc(opt_tremor, gauge_files_LR[scenario], site_files_LR[scenario],opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Snowline Elevation": 
                for scenario in gauge_files_SL: 
                    V_Q_final = runoff_model_lc(opt_tremor, gauge_files_SL[scenario], site_files_SL[scenario],opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Exp. Smoothing Length": 
                for i in range(len(tau_m_array)): 
                    V_Q_final = runoff_model_lc(opt_tremor, opt_gauge, opt_site,tau_m_array[i], tau_p_array[i],opt_window)
                    all_results[glacier][p][i] = {"V_Q": V_Q_final}
            if p == "Ratio Window Length": 
                for i in window_array: 
                    V_Q_final = runoff_model_lc(opt_tremor, opt_gauge, opt_site,opt_tau_m, opt_tau_p,i)
                    all_results[glacier][p][i] = {"V_Q": V_Q_final}
    
    if "Wolverine" in glaciers :
        glacier = "Wolverine" 
        all_results[glacier] = {}
        start_date = pd.Timestamp("2022/05/12 22:00:00").tz_localize('UTC') # start of rain dataset (smallest time constraint)
        end_date = pd.Timestamp("2022/09/30 22:00:00").tz_localize('UTC') # end of rain dataset (smallest time constraint)
        sg_code = "1424_00060"
        tremor_files_IT = {
            "best_path_best_IT":["wolc_tremor.csv","woln_tremor.csv"],}
        tremor_files_CP = {
            "best_path_best_IT":["wolc_tremor.csv","woln_tremor.csv"]}
        gauge_files_LR = { 
            "ice_LR_best_SL": ["wolv_gauge_model_melt_and_precip.csv"]}
        gauge_files_SL = { 
            "ice_LR_best_SL": ["wolv_gauge_model_melt_and_precip.csv"]}
        site_files_LR = { 
            "ice_LR_best_SL": ["wolc_model_melt_and_precip.csv","woln_model_melt_and_precip.csv"]}
        site_files_SL = { 
            "ice_LR_best_SL": ["wolc_model_melt_and_precip.csv","woln_model_melt_and_precip.csv"],} 
        tau_m_array = [15] # full time series, July/August, two weeks in August? 
        tau_p_array = [85]

        for p in all_parameters: 
            all_results[glacier][p] = {}
            opt_tremor = tremor_files_IT["best_path_best_IT"]
            opt_gauge = gauge_files_LR["ice_LR_best_SL"]
            opt_site = site_files_LR["ice_LR_best_SL"]
            opt_window = window_array[0]
            opt_tau_m = tau_m_array[0]
            opt_tau_p = tau_p_array[0]
            if p == "Ice Thickness": 
                for scenario in tremor_files_IT: 
                    V_Q_final = runoff_model_wolv(tremor_files_IT[scenario], opt_gauge, opt_site,opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Conduit Pathway": 
                for scenario in tremor_files_CP: 
                    V_Q_final = runoff_model_wolv(tremor_files_CP[scenario], opt_gauge, opt_site,opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Degree Day Factor": 
                for scenario in gauge_files_LR: 
                    V_Q_final = runoff_model_wolv(opt_tremor, gauge_files_LR[scenario], site_files_LR[scenario],opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Snowline Elevation": 
                for scenario in gauge_files_SL: 
                    V_Q_final = runoff_model_wolv(opt_tremor, gauge_files_SL[scenario], site_files_SL[scenario],opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Exp. Smoothing Length": 
                for i in range(len(tau_m_array)): 
                    V_Q_final = runoff_model_wolv(opt_tremor, opt_gauge, opt_site,tau_m_array[i], tau_p_array[i],opt_window)
                    all_results[glacier][p][i] = {"V_Q": V_Q_final}
            if p == "Ratio Window Length": 
                for i in window_array: 
                    V_Q_final = runoff_model_wolv(opt_tremor, opt_gauge, opt_site,opt_tau_m, opt_tau_p,i)
                    all_results[glacier][p][i] = {"V_Q": V_Q_final}

    if "Mendenhall" in glaciers :
        glacier = "Mendenhall" 
        all_results[glacier] = {}
        start_date = pd.Timestamp("2012/06/02 00:00:00")#.tz_localize('UTC') # start of rain dataset (smallest time constraint)
        end_date = pd.Timestamp("2012/09/30 23:00:00")#.tz_localize('UTC') # end of rain dataset (smallest time constraint)
        tremor_files_IT = {
            "best_path_best_IT":["ambr_tremor.csv"]}
        tremor_files_CP = {
            "best_path_best_IT":["ambr_tremor.csv"],}
        gauge_files_LR = { 
            "ice_LR_best_SL": ["mend_gauge_model_melt_and_precip.csv"]}
        gauge_files_SL = { 
            "ice_LR_best_SL": ["mend_gauge_model_melt_and_precip.csv"],}
        site_files_LR = { 
            "ice_LR_best_SL": ["ambr_model_melt_and_precip.csv"]}
        site_files_SL = { 
            "ice_LR_best_SL": ["ambr_model_melt_and_precip.csv"], } 
        tau_m_array = [19] # full time series, July/August, two weeks in August? 
        tau_p_array = [49]

        for p in all_parameters: 
            all_results[glacier][p] = {}
            opt_tremor = tremor_files_IT["best_path_best_IT"]
            opt_gauge = gauge_files_LR["ice_LR_best_SL"]
            opt_site = site_files_LR["ice_LR_best_SL"]
            opt_window = window_array[0]
            opt_tau_m = tau_m_array[0]
            opt_tau_p = tau_p_array[0]
            if p == "Ice Thickness": 
                for scenario in tremor_files_IT: 
                    V_Q_final = runoff_model_mend(tremor_files_IT[scenario], opt_gauge, opt_site,opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Conduit Pathway": 
                for scenario in tremor_files_CP: 
                    V_Q_final = runoff_model_mend(tremor_files_CP[scenario], opt_gauge, opt_site,opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Degree Day Factor": 
                for scenario in gauge_files_LR: 
                    V_Q_final = runoff_model_mend(opt_tremor, gauge_files_LR[scenario], site_files_LR[scenario],opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Snowline Elevation": 
                for scenario in gauge_files_SL: 
                    V_Q_final = runoff_model_mend(opt_tremor, gauge_files_SL[scenario], site_files_SL[scenario],opt_tau_m, opt_tau_p,opt_window)
                    all_results[glacier][p][scenario] = {"V_Q": V_Q_final}
            if p == "Exp. Smoothing Length": 
                for i in range(len(tau_m_array)): 
                    V_Q_final = runoff_model_mend(opt_tremor, opt_gauge, opt_site,tau_m_array[i], tau_p_array[i],opt_window)
                    all_results[glacier][p][i] = {"V_Q": V_Q_final}
            if p == "Ratio Window Length": 
                for i in window_array: 
                    V_Q_final = runoff_model_mend(opt_tremor, opt_gauge, opt_site,opt_tau_m, opt_tau_p,i)
                    all_results[glacier][p][i] = {"V_Q": V_Q_final}

    print("all_results keys sample:", list(all_results.keys()))
    for glacier in all_results:
        for p in all_results[glacier]:
            for scenario in all_results[glacier][p]:
                V_Q = all_results[glacier][p][scenario]["V_Q"]
                for site, df in V_Q.items():
                    print(f"{glacier} | {p} | {scenario} | {site}: {len(df)} rows")
    return all_results
all_results_opt = iterate_runoff_model_opt(["Lemon_Creek", "Wolverine", "Mendenhall"])
'''

#%%
# define Monte-Carlo bootstrapping parameters
n_iterations = 1000 
n_samples = 750 
b_gimbert_inv = 8/5 

def log_powerlaw_theor(logV, log_k):
    return log_k + b_gimbert_inv * logV
    
def log_powerlaw_best(logV, log_k, beta):
    return log_k + beta * logV

def fit_power_law(T,Q):
    mask = np.isfinite(T) & np.isfinite(Q) & (T > 0) & (Q > 0)
    T, Q = T[mask], Q[mask]

    if len(T) == 0: return None
    logT = np.log10(T);logQ = np.log10(Q)

    popt_t, _ = curve_fit(log_powerlaw_theor, logT, logQ, p0=[1.0])
    k_theor = 10 ** popt_t[0]

    popt_b, _ = curve_fit(log_powerlaw_best, logT, logQ, p0=[1.0, 1.0])
    k_best = 10 ** popt_b[0]
    b_best = popt_b[1]

    return k_theor, k_best, b_best

def make_all_sites(all_results, p, glaciers, scenario):

    all_sites = []
    for glacier in glaciers:
        if scenario not in all_results[glacier][p]:
            continue 
        V_Q = all_results[glacier][p][scenario]["V_Q"]

        for site, df in V_Q.items():
            T = df["T_Amp_corr"].to_numpy()
            Q = df["final_site_est"].to_numpy()
            all_sites.append((T, Q))

    return all_sites

def MC_iterations(all_sites, n_iter, n_samp):
    k_theor_arr = np.full(n_iter, np.nan)
    k_best_arr  = np.full(n_iter, np.nan)
    b_best_arr  = np.full(n_iter, np.nan)
    filtered_sites = []
    for T, Q in all_sites:
        mask = np.isfinite(T) & np.isfinite(Q) & (T > 0) & (Q > 0)
        T_f, Q_f = T[mask], Q[mask]
        if len(T_f) > 0:
            filtered_sites.append((T_f, Q_f))

    if len(filtered_sites) == 0:
        return {"k_theor": k_theor_arr, "k_best": k_best_arr, "b_best": b_best_arr}

    site_lens = [len(T_f) for T_f, _ in filtered_sites]
    min_len = min(site_lens)

    for i in range(n_iter):
        rng_i = np.random.default_rng(i)

        T_pool, Q_pool = [], []
        for T_f, Q_f in filtered_sites:
            idx = rng_i.choice(len(T_f), size=n_samp, replace=False)
            T_pool.append(T_f[idx])
            Q_pool.append(Q_f[idx])

        T_all = np.concatenate(T_pool)
        Q_all = np.concatenate(Q_pool)

        result = fit_power_law(T_all, Q_all)
        if result is not None:
            k_theor_arr[i], k_best_arr[i], b_best_arr[i] = result

    return { "k_theor": k_theor_arr, "k_best": k_best_arr,"b_best": b_best_arr}

all_parameters= ["Ice Thickness", "Conduit Pathway", "Degree Day Factor", "Snowline Elevation", "Exp. Smoothing Length", "Ratio Window Length"]

def make_sites_for_combination(all_results, p, glaciers, combo):
    all_sites = []
    for glacier, scenario in combo.items():
        if scenario not in all_results[glacier][p]:
            continue
        V_Q = all_results[glacier][p][scenario]["V_Q"]
        for site, df in V_Q.items():
            T = df["T_Amp_corr"].to_numpy()
            Q = df["final_site_est"].to_numpy()
            all_sites.append((T, Q))
    return all_sites

MC_results = {}
glaciers = ["Lemon_Creek", "Wolverine", "Mendenhall"]

for p in all_parameters:
    k_theor_all = []
    k_best_all  = []
    b_best_all  = []

    scenarios_per_glacier = {g: list(all_results[g][p].keys()) for g in glaciers}
    glacier_names = list(scenarios_per_glacier.keys())
    scenario_lists = [scenarios_per_glacier[g] for g in glacier_names]

    for combo_tuple in product(*scenario_lists):
        combo = dict(zip(glacier_names, combo_tuple))
        all_sites = make_sites_for_combination(all_results, p, glaciers, combo)

        if len(all_sites) == 0:
            continue

        mc_result = MC_iterations(all_sites, n_iterations, n_samples)
        k_theor_all.append(mc_result["k_theor"])
        k_best_all.append(mc_result["k_best"])
        b_best_all.append(mc_result["b_best"])

    MC_results[p] = {"k_theor": np.concatenate(k_theor_all),
        "k_best":  np.concatenate(k_best_all),
        "b_best":  np.concatenate(b_best_all)}
def sci_latex(x):
    s = f"{x:.2e}"
    coef, exp = s.split("e")
    return f"{coef} \\times 10^{{{int(exp)}}}"

rows = []
for p in MC_results:
    for metric in ["k_theor", "k_best", "b_best"]:
        for v in MC_results[p][metric]:
            rows.append({"parameter": p, "metric": metric, "value": v})

df = pd.DataFrame(rows)
parameters = list(all_parameters)

param_labels = {"Ice Thickness":       "Ice\nThickness",
    "Conduit Pathway":     "Conduit\nPathway",
    "Degree Day Factor":   "Degree Day\nFactor",
    "Snowline Elevation":  "Snowline\nElevation",
    "Exp. Smoothing Length": "Smoothing\nLength",
    "Ratio Window Length": "Ratio\nWindow"}

df["parameter"] = df["parameter"].map(param_labels)
param_order = list(param_labels.values())

metric_titles = {"k_theor": r"$a) \ \lambda_{\mathrm{t}}$",
    "k_best":  r"$b) \ \lambda_{\mathrm{b}}$",
    "b_best":    r"$c) \ x_{\mathrm{b}}$",}
metrics = ["k_theor", "k_best", "b_best"]

# Define Empirical Model Results 

k_theor_ref = 5.629e8
k_best_ref  = 4.717e10
b_best_ref  = 1.982

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))

for j, metric in enumerate(metrics):
    ax = axes[j]
    df_plot = df[df["metric"] == metric]
    mask = (df['parameter'] == 'Conduit\nPathway') & (df['metric'] == 'k_best')
    sns.boxplot(data=df_plot, x="parameter", y="value", order=param_order, ax=ax,width=0.5,flierprops=dict(marker='.', markerfacecolor="#898989",markeredgecolor="#656565", markersize=10, alpha=0.3),boxprops=dict(facecolor="#257793", alpha=0.85))
    ax.set_title(metric)
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=45, labelsize=8)

    ref_val = {"k_theor": k_theor_ref, "k_best": k_best_ref, "b_best": b_best_ref}[metric]
    if metric == "k_theor":
        ax.set_ylim(1.5e8, 9e8)
        ref_label = f"$\\lambda_t = 5.629 \\times 10^8$"
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e8:.0f}×10⁸"))
    elif metric == "k_best":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter( lambda x, _: f"{x/1e10:.0f}×10¹⁰"))
        ax.set_ylim(1e3,  3e13)
        ref_label = f"$\\lambda_b = 4.717 \\times 10^{{10}}$"
        ax.set_yscale('log')
    elif metric == "b_best":
        ax.set_ylim(0.5, 2.6)
        ref_label = f"$x = 1.98$"

    ax.axhline(ref_val, linestyle="--", linewidth=1.8, color="#ab0000",label=f"Empirical Model Value \n {ref_label}", zorder=5)

    ax.set_title(metric_titles[metric], fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("" if j > 0 else "Fitted Value", fontsize=10)
    ax.tick_params(axis='x', rotation=35, labelsize=8.5)
    ax.tick_params(axis='y', labelsize=9)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.7, color="white", zorder=0)
    ax.set_axisbelow(True)

    ax.legend(fontsize=8, framealpha=0.7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("sensitivity_boxplot_JGR.png", bbox_inches="tight", dpi=300)
plt.show()


# %%
