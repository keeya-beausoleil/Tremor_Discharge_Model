#%%
# import packages
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
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.colors as cols
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist

#%%
# define functions to import data and format corrected tremor, stream gauge, and flow accumulation files
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

def get_gauge(file_name, start_date, end_date, sg_code):
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


#%%

start_date_WOLV = pd.Timestamp("2022/05/12 22:00:00").tz_localize('UTC') # start of rain dataset (smallest time constraint)
end_date_WOLV = pd.Timestamp("2022/09/30 22:00:00").tz_localize('UTC') # end of rain dataset (smallest time constraint)
sg_code_WOLV = "1424_00060"

# tremor files
wolc_tremor, wolc_tremor_df = load_file("wolc_tremor.csv","tremor")
woln_tremor, woln_tremor_df = load_file("woln_tremor.csv","tremor")

# all flow accumulation files
gauge_melt_flux,gauge_precip_flux,gauge_time = load_file("wolv_gauge_model_melt_and_precip.csv","gauge_flowacc")
woln_melt_flux,woln_precip_flux,woln_time= load_file("woln_model_melt_and_precip.csv","site_flowacc")
wolc_melt_flux,wolc_precip_flux,wolc_time = load_file("wolc_model_melt_and_precip.csv","site_flowacc")

# stream gauge 
discharge, discharge_df= get_gauge("/data/stor/proj/keeya_thesis/Wolverine_Glacier/data/raw_data/wolv_streamgauge_02.10.2025.txt",start_date_WOLV,end_date_WOLV,sg_code_WOLV)
#%%

# testing different dataset durations 
len_test=0
#len_test = 4600 # august - sept
#len_test = 2356  # subset final (09.03) # 2356 july - sept  #48*30
discharge = discharge[len_test:]
gauge_melt_flux= gauge_melt_flux[len_test:]
gauge_precip_flux = gauge_precip_flux[len_test:]
gauge_time = gauge_time[len_test:]
#%%
# mask Nan data from any datasets
valid_mask = (~np.isnan(discharge) &~np.isnan(gauge_melt_flux) & ~np.isnan(gauge_precip_flux))

discharge_final = discharge[valid_mask]
gauge_melt_flux_final = gauge_melt_flux[valid_mask]
gauge_precip_flux_final = gauge_precip_flux[valid_mask]
gauge_time_final = gauge_time[valid_mask]

#%%
# run convolution with tau smoothing window at plausible range for melt and precipitation
def try_smooths(data,tau_e_max,inc,x2):
    tau_e = np.linspace(1,tau_e_max,tau_e_max)
    og_data = pd.Series(data).copy()
    smooth_df = pd.DataFrame()
    smooth_df[f"OG modelled"] = data
    for tau_e in range(1,tau_e_max+1,inc):
        exp_factor = -1/(tau_e)
        kern_len = (tau_e  *5)
        t_kern = np.arange(0, kern_len, 1)
        kernel = np.exp(exp_factor*t_kern)
        kernel = kernel/np.sum(kernel)
        smooth_df[f"smooth: {tau_e} tau "] = np.convolve(data, kernel, 'full')[:len(x2)]
    return smooth_df

gauge_melt_smooths = try_smooths(gauge_melt_flux_final,24, 1,gauge_time_final)
#gauge_melt_smooths = try_smooths(gauge_melt_flux_final,14, 1,gauge_time_final)

gauge_precip_smooths= try_smooths(gauge_precip_flux_final,24*40, 6,gauge_time_final)
#gauge_precip_smooths= try_smooths(gauge_precip_flux_final,24*3+12, 6,gauge_time_final) -- Best Fit 14 // 14 (79)
#%%
# find best-fit smoothing windows and LSQ scaling model 
def gauge_LSQ(melt_smooths, precip_smooths, gauge_obs, time):
    model = []
    #gauge_pred = []
    ss_tot = np.sum((gauge_obs - np.mean(gauge_obs))**2)
    best = {"r2": -np.inf,"beta": None,"smooths":None}
    for i in range(0,len(melt_smooths.columns),1):
        melt_temp = melt_smooths.iloc[:, i]
        for j in range(0,len(precip_smooths.columns),1):
            precip_temp = precip_smooths.iloc[:, j]
            A = np.vstack([melt_temp,precip_temp]).T
            gauge_obs_T  = gauge_obs.T
            beta, res, rank, s = np.linalg.lstsq(A, gauge_obs_T, rcond=None)
            pred = A @ beta
            residuals = gauge_obs - pred
            ss_res = np.sum(residuals**2)
            r2 = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean(residuals**2))
            if r2 > best["r2"]:
                best.update(r2=r2, beta=beta, smooths=(i, j))
                model = pred
                model_df = pd.DataFrame({"date_time": time,"model": model})
    return best, model_df
gauge_LSQ_best, gauge_model = gauge_LSQ(gauge_melt_smooths, gauge_precip_smooths, discharge_final,gauge_time_final)

# %%
# Intermediate check: compare raw flow accumulation, best LSQ model, and observations at gauge
def plot_subset_LSQ():
    fig, ax = plt.subplots(figsize=(13,6))
    ax.plot(gauge_time_final, gauge_melt_flux_final, "--", alpha = 0.4,color="#828586", label = "Raw Melt Contribution")
    ax.plot(gauge_time_final, gauge_precip_flux_final, "--", alpha = 0.4,color="#86C7D7", label = "Raw Precip Contribution")
    ax.plot(gauge_time_final, discharge_final, color="#1C275F",alpha = 0.9,linewidth = 1.5, label = "Gauge Observation")
    ax.plot(gauge_time_final, gauge_model["model"], color="#FF9B05",alpha = 0.95,linewidth = 1.75,label=f"LSQ-Fit | $R^2$={gauge_LSQ_best['r2']:.2f}")
    
    ax.set_ylim(0, 30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Discharge [$m^3$/s]", fontsize=18)
    plt.title("Lemon Creek Modelled and Observed Discharge @ Gauge", fontsize=22)
    ax.set_xlim(gauge_time_final[2356].replace(day=1), max(gauge_time_final))
    #ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.legend(loc="upper left",fontsize=22)

def plot_entire_LSQ():
    fig, ax = plt.subplots(figsize=(13,6))
    ax.plot(gauge_time_final, gauge_melt_flux_final, "--", alpha = 0.4,color="#828586", label = "Raw Melt Contribution")
    ax.plot(gauge_time_final, gauge_precip_flux_final, "--", alpha = 0.4,color="#86C7D7", label = "Raw Precip Contribution")
    ax.plot(gauge_time_final, gauge_melt_flux_final+gauge_precip_flux_final, "--", alpha = 0.4,color="#828586", label = f"Unfit Model")
    ax.plot(gauge_time_final, discharge_final, color="#1C275F",alpha = 0.8,linewidth = 1.5, label = "Gauge Observation")
    ax.plot(gauge_time_final, gauge_model["model"], color="#FF9B05",alpha = 0.95,linewidth = 1.75,label=f"LSQ-Fit Model | $R^2$={gauge_LSQ_best['r2']:.2f}")
    
    ax.set_ylim(0, 65)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlabel("Date (2022)", fontsize=20,labelpad=5)
    plt.ylabel("Discharge [$m^3$/s]", fontsize=20)
    plt.title("c) Wolverine", fontsize=22)
    ax.set_xlim(min(gauge_time_final), max(gauge_time_final))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    plt.legend(loc='upper right',fontsize=16)
plot_entire_LSQ()
plt.savefig("thesis_figs/WOLV_gauge_comp.png", dpi=300, transparent=True)

#%%
# plot cummulated discharge (model & raw versus observations)
def plot_contributions():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [2.25, 1]})
    plt.subplots_adjust(wspace=0.215)
    ax1.plot(gauge_time_final, gauge_melt_flux_final, "--", alpha = 0.4,color="#1D566D", label = f"Raw Melt Contribution")
    ax1.plot(gauge_time_final, gauge_precip_flux_final, "--", alpha = 0.4,color="#86C7D7", label = "Raw Precip Contribution")
    ax1.plot(gauge_time_final, discharge_final, color="#1C275F",alpha = 0.9,linewidth = 1.5, label = "Gauge Observation")
    ax1.plot(gauge_time_final, gauge_model["model"], color="#FF9B05",alpha = 0.95,linewidth = 1.75,label=f"LSQ-Fit Model")
    
    ax1.set_ylim(0, 65)
    
    ax1.set_xlabel("Date (2022)", fontsize=20,labelpad=5)
    ax1.set_ylabel("Discharge [$m^3$/s]", fontsize=20)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_xlim(min(gauge_time_final), max(gauge_time_final))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.legend(loc="upper left",fontsize=16)

    cum_obs = np.cumsum(discharge_final)
    cum_raw = np.cumsum(gauge_melt_flux_final + gauge_precip_flux_final)
    cum_lsq = np.cumsum(gauge_model["model"].values)

    t = mdates.date2num(gauge_time_final)
    t_norm = (t - t.min()) / (t.max() - t.min())

    min_val = min(cum_obs.min(), cum_raw.min(), cum_lsq.min())
    max_val = max(cum_obs.max(), cum_raw.max(), cum_lsq.max())

    for i in range(len(cum_obs) - 1):
        c = plt.cm.viridis(t_norm[i])
        ax2.plot(cum_obs[i:i+2], cum_raw[i:i+2],color=c, linewidth=3, linestyle="--", alpha=0.7)
        ax2.plot(cum_obs[i:i+2], cum_lsq[i:i+2],color=c, linewidth=5, alpha=0.95)

    ax2.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1.5, linestyle="--")

    ax2.plot([], [], color="grey", linewidth=1.6, alpha = 0.4,label=f"Raw Runoff")
    ax2.plot([], [], color="grey", linewidth=2, alpha =1, label=f"LSQ Model")
    ax2.set_xlim(0, 80000)
    ax2.set_ylim(0, 80000)
    ax2.set_aspect('equal')
    tick_locs = np.arange(0, 80001, 30000)
    ax2.set_xticks(tick_locs)
    ax2.set_yticks(tick_locs)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=t.min(), vmax=t.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, pad=0.02)
    cbar.set_label("Date", fontsize=20)
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    cbar.ax.tick_params(labelsize=15)
    #tick_locs = np.linspace(min_val, max_val, 4)
    #ax2.set_xticks(tick_locs)
    #ax2.set_yticks(tick_locs)
    ax2.set_xlabel("$\Sigma$ Obs. Discharge [m³/s]", fontsize=20, labelpad=5)
    ax2.set_ylabel("$\Sigma$ Modelled Discharge [m³/s]", fontsize=20)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.legend(loc="lower right", fontsize=16)
    plt.suptitle("c) Wolverine", fontsize=20, y=0.936)
    cum_raw = np.cumsum(gauge_melt_flux_final + gauge_precip_flux_final)
    cum_lsq = np.cumsum(gauge_model["model"].values)  # .values converts to numpy array

    bias_raw = np.mean(cum_raw - cum_obs)
    bias_lsq = np.mean(cum_lsq - cum_obs)
    rmse_raw = np.sqrt(np.mean((cum_raw - cum_obs)**2))
    rmse_lsq = np.sqrt(np.mean((cum_lsq - cum_obs)**2))
    final_error_raw = cum_raw[-1] - cum_obs[-1]
    final_error_lsq = cum_lsq[-1] - cum_obs[-1]

    plt.savefig("thesis_figs/WOLV_cumm_gauge.png", dpi=300, transparent=True)
plot_contributions()

# %%
# multiply tau smoothing window by proportional area of upstream contribution above sites
def downscale_tau(best_tau,perc_area):
    tau = best_tau*perc_area  # e folding time scale
    exp = -1/(tau)
    kern = (tau *5) 
    t = np.arange(0, kern, 1)
    kernel = np.exp(exp*t)
    kernel = kernel/np.sum(kernel)
    return kernel


kernel_melt_c = downscale_tau(gauge_LSQ_best["smooths"][0],0.314)
kernel_melt_n = downscale_tau(gauge_LSQ_best["smooths"][0],0.55)
kernel_precip_c = downscale_tau(((gauge_LSQ_best["smooths"][1]-1)*6)+1,0.314)
kernel_precip_n = downscale_tau(((gauge_LSQ_best["smooths"][1]-1)*6)+1,0.55)

#%%

# run LSQ model for site location: convolution with scaled tau length and LSQ parameters fit from gauge 

def site_model(m_flux,kernel_m,p_flux, kernel_p,time,betas):
    smooth_melt = np.convolve(m_flux, kernel_m, 'full')[:len(time)]
    smooth_precip = np.convolve(p_flux, kernel_p, 'full')[:len(time)]
    model = smooth_melt*betas[0] + smooth_precip*betas[1]
    #model = smooth_melt*0.47859973 + smooth_precip*2.04760289
    model_df = pd.DataFrame({"date_time": time,"model": model})
    return model_df

wolc_model = site_model(wolc_melt_flux,kernel_melt_c,wolc_precip_flux,kernel_precip_c,wolc_time,gauge_LSQ_best["beta"])
woln_model = site_model(woln_melt_flux,kernel_melt_n,woln_precip_flux,kernel_precip_n,woln_time,gauge_LSQ_best["beta"])
# %%

# multiply gauge discharge observations by smoothed ratio between gauge model and upstream site model & plot results

def plot_ratio(gauge_model, site_model,raw_ratio,smoothed_ratio, site_df,site_name):
    fig, ax1 = plt.subplots(figsize=(13,6))
    ax1.plot(gauge_model["date_time"], gauge_model["model"], "--", alpha = 0.6,linewidth = 1.25,color="#0B3B4A", label = "Gauge Model")
    ax1.plot(site_model["date_time"], site_model["model"], "--", alpha = 0.7,linewidth = 1.25,color="#2BB1D3", label = "Site Model")
    ax1.set_ylabel("Discharge [$m^3$/s]", fontsize=20, color ="#125E75")
    ax1.set_ylim(0, 50)
    ax2 = ax1.twinx()
    ax2.plot(site_df["date_time"], raw_ratio, color="#E209A5",alpha = 0.4,linewidth = 1, label = "Raw Ratio")
    ax2.plot(site_df["date_time"], smoothed_ratio, color="#7C0889",alpha = 0.95,linewidth = 1.5,label=f"14 day Smoothed Ratio")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Ratio", fontsize=20, color ="#4A0E4F")
    plt.title(f"Wolverine: {site_name}", fontsize=20)
    ax1.set_xlim(min(site_model["date_time"]), max(site_model["date_time"]))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax1.tick_params(axis='both', labelsize=16,length=6)
    ax1.tick_params(axis='both', labelsize=16,length=6)
    ax2.tick_params(axis='y', labelsize=16)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.set_xlabel("Date (2022)", fontsize=20,labelpad=5)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc = "upper right",fontsize=16)
    plt.savefig(f"thesis_figs/{site_name}_ratio_plot.png",dpi=300, transparent=True)

def site_ratio(gauge_model, site_model, window_len, discharge,site): 
    site_model["date_time"] = pd.to_datetime(site_model["date_time"]).dt.tz_localize(None)
    gauge_model["date_time"] = pd.to_datetime(gauge_model["date_time"]).dt.tz_localize(None)
    discharge["date_time"] = pd.to_datetime(discharge["date_time"]).dt.tz_localize(None)
    start_temp = max(site_model["date_time"].min(), gauge_model["date_time"].min())
    end_temp = min(site_model["date_time"].max(), gauge_model["date_time"].max())
    site_model_subset = site_model[(site_model['date_time'] >= start_temp) & (site_model['date_time'] <= end_temp)]
    gauge_model_subset = gauge_model[(gauge_model['date_time'] >= start_temp) & (gauge_model['date_time'] <= end_temp)]
    discharge_subset = discharge[(discharge['date_time'] >= start_temp) & (discharge['date_time'] <= end_temp)]
    site_model_subset = site_model_subset.set_index("date_time")
    gauge_model_subset = gauge_model_subset.set_index("date_time")
    discharge_subset = discharge_subset.set_index("date_time")
    site_ratio_temp = site_model_subset["model"]/gauge_model_subset["model"]
    final_ratio = site_ratio_temp.rolling(window_len*48, center=True, min_periods=math.ceil(window_len)*48).mean()
    site_est =  discharge_subset["Q"]*final_ratio 
    site_est_df = (pd.DataFrame({"final_site_est": site_est,"Q_obs": discharge_subset["Q"]}).rename_axis("date_time").reset_index())
    site_model_subset = site_model_subset.reset_index()
    gauge_model_subset = gauge_model_subset.reset_index()
    discharge_subset = discharge_subset.reset_index()
    plot_ratio(gauge_model_subset, site_model_subset,site_ratio_temp,final_ratio,site_est_df,site)
    return site_est_df

#%%
window = 14 # days (ratio smoothing window)
wolc_est_df = site_ratio(gauge_model, wolc_model, window,discharge_df,"WOLC")
woln_est_df = site_ratio(gauge_model, woln_model, window,discharge_df,"WOLN")
#%%
# plot comparison source discharges (area scaled, smoothed ratio scaled, and raw LSQ model) 
def plot_site_est(site_name, site_est,area_prop,site_model):
    fig, ax1 = plt.subplots(figsize=(13,6))
    ax1.plot(site_model["date_time"], site_model["model"], "--", alpha = 0.6,linewidth = 1,color="#2F96AF", label = "LSQ-Model @ Site")
    ax1.plot(site_est["date_time"], site_est["Q_obs"]*area_prop, alpha = 0.7,linewidth = 1.25,color="#616862", label = "Drainage Area Ratio")
    ax1.plot(site_est["date_time"], site_est["final_site_est"], alpha = 0.7,linewidth = 1.25,color="#E30031", label = "Smoothed Modeled Discharge Ratio")
    ax1.set_ylabel("Discharge [$m^3$/s]", fontsize=20,labelpad=5)
    ax1.set_xlim(min(site_est["date_time"]), max(site_est["date_time"]))
    ax1.set_xlim(pd.Timestamp("2022-06-15"), pd.Timestamp("2022-09-15"))
    ax1.set_ylim(0, 15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax1.tick_params(axis='x', length=8)
    plt.xlabel("Date (2022)", fontsize=20,labelpad=5)
    plt.title(f"Wolverine: {site_name}", fontsize=20)
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.legend(loc="upper right",fontsize=16)
    plt.savefig(f"thesis_figs/{site_name}_discharge_plot.png",dpi=300, transparent=True)

# %%
# plot timeseries of corrected tremor amplitude alongside site discharge estimations (ratio scaled result)
def plot_v_q_timeseries(site_name, q_v_df):
    corr = q_v_df["final_site_est"].corr(q_v_df["T_Amp_corr"])
    fig, ax1 = plt.subplots(figsize=(13,6))
    ax1.plot(q_v_df["date_time"], q_v_df["final_site_est"],  alpha = 0.8,linewidth = 1.7,color="#2636C9", label = "Estimated Site Discharge")
    ax1.set_ylabel("Discharge [$m^3$/s]", fontsize=20,color="#2636C9", labelpad=5)
    ax1.tick_params(axis='y', colors='#2636C9',labelsize=16)
    ax1.set_ylim(0, 14)
    ax2 = ax1.twinx()
    ax2.plot(q_v_df["date_time"], q_v_df["T_Amp_corr"], alpha = 0.8,linewidth = 1.5,color="#C35807", label = "Corrected Tremor Amplitude")
    ax2.set_ylabel("Tremor Amplitude [m/s]", fontsize=20,color="#C35807", labelpad=15)
    ax2.tick_params(axis='y', colors='#C35807',labelsize=16)
    ax1.set_xlim(min(q_v_df["date_time"]), max(q_v_df["date_time"]))
    ax1.set_xlim(pd.Timestamp("2022-06-15"), pd.Timestamp("2022-09-15"))
    ax2.set_ylim(0, 1.8e-5)
    ax1.tick_params(axis='x', labelsize=16,length=6)
    ax1.set_xlabel("Date (2022)", fontsize=20,labelpad=5)
    plt.title(f"Wolverine: {site_name} | corr = {corr:.2f}", fontsize=20)
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    #ax1.legend(h1 + h2, l1 + l2,loc="upper right")
    plt.savefig(f"thesis_figs/{site_name}_v_q_timeseries_plot.png",dpi=300, transparent=True)

#%%
# align tremor and discharge time series with transport time lag
def align_V_Q(site_name,site_est, tremor,lag,area,raw_model):
    
    site_est = site_est.copy()
    tremor = tremor.copy()
    site_est["date_time"] = pd.to_datetime(site_est["date_time"]).dt.tz_localize(None)
    tremor["date_time"] = pd.to_datetime(tremor["date_time"]).dt.tz_localize(None)

    site_V_Q_final = pd.merge(site_est[lag:], tremor[:-lag], on="date_time",how="inner" )
    site_V_Q_final = site_V_Q_final.dropna(subset=["final_site_est"])
    print(site_V_Q_final)
    plot_site_est(site_name, site_V_Q_final,area,raw_model)
    plot_v_q_timeseries(site_name, site_V_Q_final)
    return site_V_Q_final

lag_c = round(6700/3600)
lag_n = round(4000/3600) # @ 1m/s rate

wolc_V_Q_final = align_V_Q("WOLC",wolc_est_df, wolc_tremor_df,lag_c,0.314,wolc_model)
wolc_V_Q_final.to_csv("wolc_V_Q_final.csv", index=False)
woln_V_Q_final = align_V_Q("WOLN",woln_est_df, woln_tremor_df,lag_n,0.55,woln_model)
woln_V_Q_final.to_csv("woln_V_Q_final.csv", index=False)
# %%

# fit tremor versus source discharge to Gimbert's theoretical power-law and LSQ best-fit power-law relationship
# includes both scatter and density plots for each site 
y_up = np.concatenate([wolc_V_Q_final["final_site_est"]])
y_low = np.concatenate([woln_V_Q_final["final_site_est"]])

x_up = np.concatenate([wolc_V_Q_final["T_Amp_corr"]])
x_low = np.concatenate([woln_V_Q_final["T_Amp_corr"]])

mask_up = (x_up > 0) & (y_up > 0); mask_low = (x_low > 0) & (y_low > 0)
x_up = x_up[mask_up]; x_low = x_low[mask_low]
y_up = y_up[mask_up]; y_low = y_low[mask_low]

b_gimbert_inv = 8/5

def log_powerlaw_theor(logV, log_k):
    return log_k + b_gimbert_inv * logV
        
def powerlaw_theor(V, k,b):
    return k * V**b
    
def log_powerlaw_best(logV, log_k, beta):
    return log_k + beta * logV

def sci_latex(x):
    s = f"{x:.2e}"
    coef, exp = s.split("e")
    return f"{coef} \\times 10^{{{int(exp)}}}"

def r2_log(x_obs, y_obs, a, b):
    Q_pred = a * x_obs**b
    log_Q_obs = np.log10(y_obs)
    log_Q_pred = np.log10(Q_pred)
    ss_res = np.sum((log_Q_obs - log_Q_pred)**2)
    ss_tot = np.sum((log_Q_obs - log_Q_obs.mean())**2)
    return 1 - ss_res / ss_tot
    
def find_fits(V_all,Q_all):
    log_V = np.log10(V_all)
    log_Q = np.log10(Q_all)
    p0_t = [1.0]
    popt_t, pcov_t = curve_fit(log_powerlaw_theor, log_V, log_Q, p0_t)
    k_theor = 10**popt_t[0]

    p0_b = [1.0, 1.0]
    popt_b, pcov_b = curve_fit(log_powerlaw_best, log_V, log_Q, p0_b)
    k_best = 10**popt_b[0]
    b_best = popt_b[1]

    r2_log_theor = r2_log(V_all, Q_all, k_theor, b_gimbert_inv)
    Q_pred_theor = k_theor * V_all**(b_gimbert_inv)
    log_resid_t = log_Q - np.log10(Q_pred_theor)
    std_resid_t = np.std(log_resid_t)
    rmse_log_theor = np.sqrt(np.mean((log_resid_t)**2))
    print(f"Theoretical: Q = {k_theor:.2e} · V^(8/5)  |  R² = {r2_log_theor:.3f} | log-RMSE={rmse_log_theor:.3f}  |  ×{10**rmse_log_theor:.2f}")

    r2_log_best = r2_log(V_all, Q_all, k_best, b_best)
    Q_pred_best = k_best * V_all**(b_best)
    log_resid_b = log_Q - np.log10(Q_pred_best)
    std_resid_b = np.std(log_resid_b)
    rmse_log_best = np.sqrt(np.mean((log_resid_b)**2))
    print(f"Best fit:    Q = {k_best:.2e} · V^{b_best:.3f} | R² = {r2_log_best:.3f} | log-RMSE={rmse_log_best:.3f} | ×{10**rmse_log_best:.2f}")

    n = len(V_all)
    alpha = 0.05
    
    p_t = len(popt_t)
    dof_t = n-p_t
    perr_t = np.sqrt(np.diag(pcov_t))
    k_theor_ci_log = (t_dist.ppf(1.0 - alpha / 2, dof_t))*perr_t[0]
    k_theor_ci = k_theor * np.log(10) * k_theor_ci_log
    print(f"Optimal k_theor: {k_theor:.2e} | standard errors: {perr_t[0]:.2e} | 95% C.I. +/- {k_theor_ci:.2e}")

    p_b = len(popt_b)
    dof_b = n-p_b
    perr_b = np.sqrt(np.diag(pcov_b))
    k_best_ci_log = t_dist.ppf(1.0 - alpha / 2, dof_b)*perr_b[0]
    k_best_ci = k_best * np.log(10) * k_best_ci_log
    b_best_ci = t_dist.ppf(1.0 - alpha / 2, dof_b)*perr_b[1]
    print(f"Optimal k_best: {k_best:.2e} | standard errors: {perr_b[0]:.2e} | 95% C.I. +/- {k_best_ci:.2e}")
    print(f"Optimal b_best: {popt_b[1]:.2f} | standard errors: {perr_b[1]:.2e} | 95% C.I. +/- {b_best_ci:.2e}")

    print(f"Q prediction interval (±1σ_t): ×{10**(std_resid_t):.2f}")
    print(f"Q prediction interval (±1σ_b): ×{10**(std_resid_b):.2f}")

    return k_theor, k_best, b_best, r2_log_theor, r2_log_best

k_theor_up, k_best_up, b_best_up, r2_log_theor_up, r2_log_best_up = find_fits(x_up,y_up)
k_theor_low, k_best_low, b_best_low, r2_log_theor_low, r2_log_best_low = find_fits(x_low,y_low)

xx_up = np.linspace(x_up.min(), x_up.max(), 1000)
xx_low = np.linspace(x_low.min(), x_low.max(), 1000)

cmap_up  = LinearSegmentedColormap.from_list("up",  ["#F7C6AE", "#AC3006"])
cmap_low = LinearSegmentedColormap.from_list("low", ["#B6CCFC", "#214EAF"])

my_cmap = cmap_up(np.arange(cmap_up.N))
nn = np.linspace(0, 1, cmap_up.N)
my_cmap[:,-1] = nn ** (0.5)
my_cmap_up = cols.ListedColormap(my_cmap)

my_cmap2 = cmap_low(np.arange(cmap_low.N))
nn2 = np.linspace(0, 1, cmap_low.N)
my_cmap2[:,-1] = nn2 ** (0.5)
my_cmap_low = cols.ListedColormap(my_cmap2)

#sns.kdeplot(x=x_up, y=y_up,  levels=7, thresh=0.05, cmap = my_cmap_up, fill=True, label="Upper: WOLC", ax=ax)
#sns.kdeplot(x=x_low, y=y_low, levels=7, thresh=0.05, cmap =my_cmap_low, fill=True, label="Lower: WOLN", ax=ax)
fig, ax = plt.subplots(figsize=(12,12))


up_data = plt.scatter(x_up, y_up, s=9.5, facecolors='none', edgecolor="#AC3006", linewidth = 1.5, alpha=0.25, label="WOLC",zorder=1)
low_data = plt.scatter(x_low, y_low, s=9.5, facecolors='none', edgecolor="#214EAF", linewidth = 1.5, alpha=0.25, label="WOLN",zorder=2)

fit_theor_up, = ax.plot(xx_up,k_theor_up * xx_up**(b_gimbert_inv), 'k--', alpha = 0.45,lw=2, label=f'WOLC Theoretical:\n' f'$Q = {sci_latex(k_theor_up)} ∙ V^{{8/5}}   \;|\; R^2_{{\\log}}={r2_log_theor_up:.2f}$', zorder=3)
fit_best_up,  = ax.plot(xx_up,k_best_up * xx_up**(b_best_up), linewidth = 2.5, color="#872200", alpha=0.7, label=f'Best Fit:\n' f'$Q = {sci_latex(k_best_up)} ∙ V^{{{b_best_up:.2f}}}  \;|\; R^2_{{\\log}}={r2_log_best_up:.2f}$', zorder=4)

fit_theor_low, = ax.plot(xx_low,k_theor_low * xx_low**(b_gimbert_inv), 'k--', alpha = 0.45,lw=2, label=f'WOLN Theoretical:\n' f'$Q = {sci_latex(k_theor_low)} ∙ V^{{8/5}}   \;|\; R^2_{{\\log}}={r2_log_theor_low:.2f}$', zorder=3)
fit_best_low,  = ax.plot(xx_low,k_best_low * xx_low**(b_best_low), linewidth = 2.5, color="#032670", alpha=0.8, label=f'Best Fit:\n' f'$Q = {sci_latex(k_best_low)} ∙ V^{{{b_best_low:.2f}}}  \;|\; R^2_{{\\log}}={r2_log_best_low:.2f}$', zorder=4)


ax.set_xscale("log")
ax.set_yscale("log")
ax.set_aspect('equal')

ax.set_xlim([min(min(x_low),min(x_up)) * 0.85, max(max(x_low),max(x_up)) * 3.3])
ax.set_ylim([min(min(y_low),min(y_up)) * 0.6, max(max(y_low),max(y_up)) * 1.5])

  
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.tick_params(which='minor', length=5)
ax.tick_params(which='major', length=7)
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
#up_patch = mpatches.Patch(color="#AC3006", alpha=0.5, label="Upper sites")
#low_patch = mpatches.Patch(color="#214EAF", alpha=0.5, label="Lower sites")
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles=[up_patch, low_patch] + handles, labels=["Upper: WOLC", "Lower: WOLN"] + labels, fontsize=16, loc="upper left")
#ax.legend(fontsize=16, loc="upper left")
legend1 = ax.legend(handles=[low_data, up_data], fontsize=18, loc='upper left', framealpha=0.5)
ax.add_artist(legend1)
legend2 = ax.legend(handles=[fit_theor_low, fit_best_low, fit_theor_up, fit_best_up], fontsize=18, loc='lower right', framealpha=0.5)

plt.ylabel("Discharge, Q [$m^3$/s]", fontsize=20)
plt.xlabel("Tremor Amplitude, V [m/s]", fontsize=20)
plt.title("c) Wolverine", fontsize=22)

#plt.savefig("thesis_figs/WOLV_V_vs_Q_density_plot.png", dpi=300, transparent=True,bbox_inches='tight')
plt.savefig("thesis_figs/WOLV_V_vs_Q_scatterplot.png", dpi=300, transparent=True,bbox_inches='tight')

plt.show()

#%%

# %%
# plot hysteresis plot for each site (colored data points represent timeseries) 
t_up = np.concatenate([wolc_V_Q_final["date_time"].values])
t_up = t_up[mask_up]

t_low = np.concatenate([woln_V_Q_final["date_time"].values])
t_low = t_low[mask_low]

def hysteresis(x,y,t, xx,k_theor,k_best, b_best,cc, site_area):
    t = pd.to_datetime(t)
    t_md = t.map(lambda d: d.replace(year=2000))
    t_num = mdates.date2num(t_md)

    fig, ax = plt.subplots(figsize=(8, 6)) 
    b_gimbert_inv1 = 8/5
    b_gimbert_inv2 = 6/14

    x_lo, x_hi = min(x) * 0.85, max(x) * 1.1
    y_lo, y_hi = min(y) * 0.85, max(y) * 1.1
    xx_ext = np.geomspace(x_lo * 0.01, x_hi * 100, 500)

    n_lines = 14
    y_min= min(y)
    y_max = max(y)
    x_mid = np.sqrt(min(xx) * max(xx)) 

    k_min1 = y_lo / x_hi**b_gimbert_inv1  # line through bottom-right corner
    k_max1 = y_hi / x_lo**b_gimbert_inv1  # line through top-left corner
    k_offsets1 = np.geomspace(k_min1 / 4, k_max1 * 4, n_lines)  # overshoot both ends

    k_min2 = y_lo / x_hi**b_gimbert_inv2
    k_max2 = y_hi / x_lo**b_gimbert_inv2
    k_offsets2 = np.geomspace(k_min2 / 4, k_max2 * 4, n_lines)

    for i, (k1, k2) in enumerate(zip(k_offsets1, k_offsets2)):
        label1 = "V ∝ Q$^{5/8}$"
        label2 = "V ∝ Q$^{14/6}$"
        ax.plot(k1 * xx**b_gimbert_inv1, xx, color = "#575757", alpha=0.35, lw=2, label=label1, zorder=1)
        ax.plot(k2 * xx**b_gimbert_inv2, xx, color='red', alpha=0.35, lw=2, label=label2, zorder=1)

    #sc = ax.scatter(x, y, alpha = 0.65,c=t_num, s=6, cmap="viridis")
    #sc = ax.scatter(y, x, alpha = 0.65,c=t_num, s=17, cmap="viridis", linewidths=0)
    vmin = mdates.date2num(pd.Timestamp('2000-05-15'))
    vmax = mdates.date2num(pd.Timestamp('2000-09-15'))

    sc = ax.scatter(y, x, alpha=0.65, c=t_num, s=17, cmap="viridis",linewidths=0, vmin=vmin, vmax=vmax)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylim([min(x) * 0.8, max(x) * 1.3])
    ax.set_xlim([min(y) * 0.85, max(y) * 1.1])

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.tick_params(which='minor', length=5)
    ax.tick_params(which='major', length=7)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.legend( fontsize=16, loc="upper left")

    plt.xlabel("Discharge, Q [$m^3$/s]", fontsize=20)
    plt.ylabel("Tremor Amplitude, V [m/s]", fontsize=20)

    if site_area == "Upper":
        plt.title("Wolverine: WOLC", fontsize=20)
    if site_area == "Lower":
        plt.title("Wolverine: WOLN", fontsize=20)

    cbar = plt.colorbar(sc)

    cbar.ax.yaxis.set_major_locator(mdates.MonthLocator(interval=1))
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    cbar.set_label("Date (2022)", fontsize=20,labelpad=7)
    cbar.ax.tick_params(labelsize=15)
    plt.savefig(f"thesis_figs/WOLV_{site_area}_hyst_plot.png", dpi=300, transparent=True,bbox_inches='tight')
    plt.show()

hysteresis(x_up,y_up,t_up, xx_up,k_theor_up,k_best_up, b_best_up,"#872200", "Upper")
hysteresis(x_low,y_low,t_low, xx_low,k_theor_low,k_best_low, b_best_low,"#214EAF", "Lower")
