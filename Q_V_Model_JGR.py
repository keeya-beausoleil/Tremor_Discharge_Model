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
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from matplotlib.patches import Patch
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.colors as cols
from scipy.stats import t as t_dist
from scipy.stats import skew, kurtosis
from itertools import combinations
from scipy import stats
from scipy.stats import sem
import warnings
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

#%%
# Lemon Creek corrected tremor/source discharge files
BBGL_df = pd.read_csv("bbgl_V_Q_final.csv")
BBGU_df = pd.read_csv("bbgu_V_Q_final.csv")
BBEL_df = pd.read_csv("bbel_V_Q_final.csv")
BBEU_df = pd.read_csv("bbeu_V_Q_final.csv")
BBWL_df = pd.read_csv("bbwl_V_Q_final.csv")
BBWU_df = pd.read_csv("bbwu_V_Q_final.csv")

# Wolverine corrected tremor/source discharge files
WOLC_df = pd.read_csv("wolc_V_Q_final.csv")  
WOLN_df = pd.read_csv("woln_V_Q_final.csv")  

# Mendenhall corrected tremor/source discharge files
MEND_df = pd.read_csv("ambr_V_Q_final.csv")  

#%%

b_gimbert_inv = 8/5 
# define various funcions used in plots (colour map generation and text formating) and generating models
def sci_latex(x):
    s = f"{x:.2e}"
    coef, exp = s.split("e")
    return f"{coef} \\times 10^{{{int(exp)}}}"

def make_cmap(hex_color, light_color):
    cmap = LinearSegmentedColormap.from_list("", [light_color, hex_color])
    cmap_array = cmap(np.arange(cmap.N))
    nn = np.linspace(0, 1, cmap.N)
    cmap_array[:, -1] = nn ** 0.5
    return cols.ListedColormap(cmap_array)

def log_powerlaw_theor(logV, log_k):
    return log_k + b_gimbert_inv * logV

def log_powerlaw_best(logV, log_k, beta):
    return log_k + beta * logV

def r2_log(V_obs, Q_obs, k, b):
        log_Q_obs = np.log10(Q_obs)
        log_Q_pred = np.log10(k * V_obs**b)
        ss_res = np.sum((log_Q_obs - log_Q_pred)**2)
        ss_tot = np.sum((log_Q_obs - log_Q_obs.mean())**2)
        return 1 - ss_res / ss_tot

def create_model(sites):
    site_obs = []
    site_labels = []
    colors = []
    light_colors = []

    if "Lemon_Creek" in sites: 
        offset = len(site_obs)
        site_obs += [BBGL_df, BBGU_df]
        site_labels +=["BBGL","BBGU"]
        colors += ["#622F82","#E6AD00"]
        legend_labels = {"#622F82": "Lemon Creek: BBGL", "#E6AD00": "Lemon Creek: BBGU"}
        light_colors += ["#CDB2DD","#F7DE92"]

    if "Wolverine" in sites: 
        offset = len(site_obs)
        site_obs += [WOLN_df, WOLC_df]
        site_labels += ["WOLN","WOLC"]
        colors += ["#214EAF","#AC6106"]
        legend_labels.update({"#214EAF": "Wolverine: WOLN", "#AC6106": "Wolverine: WOLC"})
        light_colors += ["#B6CCFC","#F7D6AE"]
    if "Mendenhall" in sites: 
        offset = len(site_obs)
        site_obs +=[MEND_df]
        site_labels +=["AMBR"]
        colors += ["#B91C8D"]
        legend_labels.update({"#B91C8D": "Mendenhall: AMBR"})
        light_colors += ["#E294CC"]

    b_gimbert_inv = 8/5
    
    site_data = []
    for i, (est, tremor, label) in enumerate(zip(site_obs, site_obs, site_labels)):
        V = tremor["T_Amp_corr"].values
        Q = est["final_site_est"].values
        t = pd.to_datetime(tremor["date_time"]).dt.tz_localize(None)
        mask = (V > 0) & (Q > 0)
        print(f"{label}: raw n={len(V)}, masked n={mask.sum()}")
        site_data.append((V[mask], Q[mask], t[mask]))
    
    V_all = np.concatenate([V for V, Q, t in site_data])
    Q_all = np.concatenate([Q for V, Q, t in site_data])
    log_V = np.log10(V_all)
    log_Q = np.log10(Q_all)

    n_mc = 1000
    k_theor_boot = []; k_best_boot = []
    b_best_boot = []
    r2_theor_boot = []; r2_best_boot = []
    std_resid_t_boot = []; std_resid_b_boot = []
    rmsle_t_boot = []; rmsle_b_boot = []

    for mc_iter in range(n_mc):
        rng = np.random.default_rng(mc_iter)
        V_parts, Q_parts = [], []
        for V_s, Q_s, _ in site_data:
            idx = rng.choice(len(V_s), size=850, replace=False)
            V_parts.append(V_s[idx])
            Q_parts.append(Q_s[idx])

        log_VV = np.log10(np.concatenate(V_parts))
        log_QQ = np.log10(np.concatenate(Q_parts))
        V_mc = np.concatenate(V_parts) 
        Q_mc = np.concatenate(Q_parts)

        pt, _ = curve_fit(log_powerlaw_theor, log_VV, log_QQ, p0=[1.0])
        pb, _ = curve_fit(log_powerlaw_best,  log_VV, log_QQ, p0=[1.0, 1.0])

        k_theor=10**pt[0]
        k_best = 10**pb[0]
        b_best = pb[1]

        log_resid_t = log_QQ - (np.log10(k_theor) + b_gimbert_inv * log_VV)
        log_resid_b = log_QQ - (np.log10(k_best)  + b_best * log_VV)

        k_theor_boot.append(k_theor)
        k_best_boot.append(k_best)
        b_best_boot.append(b_best)
        r2_theor_boot.append(r2_log(V_mc, Q_mc, k_theor, b_gimbert_inv))
        r2_best_boot.append(r2_log(V_mc, Q_mc, k_best, b_best))
        std_resid_t_boot.append(np.std(log_resid_t))
        std_resid_b_boot.append(np.std(log_resid_b))
        rmsle_t_boot.append(np.sqrt(np.mean(log_resid_t**2)))
        rmsle_b_boot.append(np.sqrt(np.mean(log_resid_b**2)))


    k_theor_boot = np.array(k_theor_boot); k_best_boot = np.array(k_best_boot)
    b_best_boot = np.array(b_best_boot)
    r2_theor_boot = np.array(r2_theor_boot);r2_best_boot = np.array(r2_best_boot)
    std_resid_t_boot = np.array(std_resid_t_boot); std_resid_b_boot = np.array(std_resid_b_boot)
    rmsle_t_boot = np.array(rmsle_t_boot); rmsle_b_boot = np.array(rmsle_b_boot)
 

    k_theor_final = np.median(k_theor_boot)
    k_best_final=np.median(k_best_boot); b_best_final=np.median(b_best_boot)
    r2_theor_final = np.median(r2_theor_boot); r2_best_final =  np.median(r2_best_boot)
    std_resid_t = np.median(std_resid_t_boot); std_resid_b = np.median(std_resid_b_boot)
    rmsle_t = np.median(rmsle_t_boot); rmsle_b = np.median(rmsle_b_boot)

    n_mc_sample = len(site_data) * 850
    tval_t = t_dist.ppf(0.975, n_mc_sample - 1); tval_b = t_dist.ppf(0.975, n_mc_sample - 2)

    fig, ax = plt.subplots(figsize=(12, 8))
    xx = np.logspace(np.log10(V_all.min() * 0.7), np.log10(V_all.max() * 1.5), 1000)

    Q_t_upper = k_theor_final * xx**b_gimbert_inv * 10**( tval_t * std_resid_t); Q_t_lower = k_theor_final * xx**b_gimbert_inv * 10**(-tval_t * std_resid_t)
    Q_b_upper = k_best_final  * xx**b_best_final  * 10**( tval_b * std_resid_b); Q_b_lower = k_best_final  * xx**b_best_final  * 10**(-tval_b * std_resid_b)

    ax.fill_between(xx, Q_t_lower, Q_t_upper, color="#AAAAAA", alpha=0.1, zorder=1)
    ax.fill_between(xx, Q_b_lower, Q_b_upper, color="#D48383", alpha=0.1, zorder=1)

    ax.plot(xx, Q_t_upper, color='dimgray',  lw=1, ls=':', zorder=2)
    ax.plot(xx, Q_t_lower, color='dimgray',  lw=1, ls=':', zorder=2)
    ax.plot(xx, Q_b_upper, color='#c0504d',  lw=1, ls=':', zorder=2)
    ax.plot(xx, Q_b_lower, color='#c0504d',  lw=1, ls=':', zorder=2)

    fit_theor, = ax.plot(xx, k_theor_final * xx**b_gimbert_inv, '-', color = "#000000", alpha=0.45, lw=2, label=f'Theoretical: $Q = {sci_latex(k_theor_final)} ∙ V^{{8/5}}  \;|\;  R^2_{{\\log}}={r2_theor_final:.2f}$', zorder=5)
    fit_best,  = ax.plot(xx, k_best_final  * xx**b_best_final,  '-', color = "#910202", alpha=0.5, lw=2, label=f'Best fit: $Q = {sci_latex(k_best_final)} ∙ V^{{{b_best_final:.2f}}}  \;|\; R^2_{{\\log}}={r2_best_final:.2f}$', zorder=5)

    band_theor = Patch(color="#AAAAAA", alpha=0.3, label=f'Theoretical 95% PI')
    band_best  = Patch(color="#D48383", alpha=0.3, label=f'Best-fit 95% PI ')
    
    for i, (x, y, _) in enumerate(site_data):
        cmap_i = make_cmap(colors[i], light_colors[i])
        sns.kdeplot(x=x, y=y, levels=7, thresh=0.05, cmap=cmap_i, fill=True, ax=ax, zorder=4)

    patch_handles = [mpatches.Patch(color=c, alpha=0.6, label=lbl) for c, lbl in legend_labels.items()]

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([V_all.min() * 0.7, V_all.max() * 1.6])
    ax.set_ylim([Q_all.min() * 0.7, Q_all.max() * 1.6])
    ax.set_ylabel("Discharge, Q [m³/s]", fontsize=20)
    ax.set_xlabel("Tremor Amplitude, V [m/s]", fontsize=20)
    ax.tick_params(labelsize=16)

    legend1 = ax.legend(handles=patch_handles, fontsize=16, loc='upper left')
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=[fit_theor, fit_best, band_theor, band_best],fontsize=16, loc='lower right')
    plt.tight_layout()
    plt.savefig("thesis_figs/EmpiricalModel_MC_final.png", dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

    print(f"Theoretical predictive interval (±t·σ_t): ×{10**(tval_t*std_resid_t):.2f}")
    print(f"Best-fit predictive interval (±t·σ_b): ×{10**(tval_b*std_resid_b):.2f}")

    print(f"MC Parameter Estimates  (n_iter={n_mc}, 95% CI)")
    for arrayy, label in zip([k_theor_boot, k_best_boot, b_best_boot],["k_theor",    "k_best",    "β_best"]):
        median_val= np.median(arrayy)
        low_perc= np.percentile(arrayy, 2.5)
        high_perc   = np.percentile(arrayy, 97.5)
        print(f"  {label:<10}: {median_val:.3e}  +{high_perc-median_val:.3e} / -{median_val-low_perc:.3e}")
    print(f"R²_log theor:{r2_theor_final:.4f} (median over MC iteration)")
    print(f"R²_log best :{r2_best_final:.4f} (median over MC iterations)")
    print(f"RMSLE theor:{rmsle_t:.4f} -- ×{10**rmsle_t:.2f} ")
    print(f"RMSLE best :{rmsle_b:.4f} -- ×{10**rmsle_b:.2f} ")

    return k_theor_final, k_best_final, b_best_final

#%%
k_theor_EM, k_best_EM, b_best_EM= create_model(["Lemon_Creek", "Wolverine","Mendenhall"])

# %%

def fit_each_site(V, Q):
    alpha = 0.05

    mask = (V > 0) & (Q > 0)
    V, Q = V[mask], Q[mask]
    log_V = np.log10(V); log_Q = np.log10(Q)
    n = len(V)
    
    popt_t, pcov_t = curve_fit(log_powerlaw_theor, log_V, log_Q, p0=[1.0])
    k_theor= 10**popt_t[0]
    perr_t = np.sqrt(np.diag(pcov_t))
    dof_t= n - 1
    k_theor_ci_log = t_dist.ppf(1 - alpha/2, dof_t) * perr_t[0]  # in log10 units

    popt_b, pcov_b = curve_fit(log_powerlaw_best, log_V, log_Q, p0=[1.0, 1.0])
    k_best= 10**popt_b[0]
    b_best= popt_b[1]
    perr_b= np.sqrt(np.diag(pcov_b))
    dof_b= n - 2

    k_best_ci_log = t_dist.ppf(1 - alpha/2, dof_b) * perr_b[0] 
    b_best_ci = t_dist.ppf(1 - alpha/2, dof_b) * perr_b[1]  

    return { "k_theor":k_theor,"k_theor_ci_log":k_theor_ci_log,"k_best":k_best,"k_best_ci_log":k_best_ci_log, "b_best": b_best,"b_best_ci":b_best_ci,
        "r2_theor":r2_log(V, Q, k_theor, b_gimbert_inv),"r2_best": r2_log(V, Q, k_best,b_best),"n": n,}

individual_sites = {
    "BBGL": (BBGL_df, "#622F82"),"BBEL": (BBEL_df, "#622F82"),"BBWL": (BBWL_df, "#622F82"),
    "BBGU": (BBGU_df, "#E6AD00"),"BBEU": (BBEU_df, "#E6AD00"),"BBWU": (BBWU_df, "#E6AD00"),
    "AMBR":  (MEND_df,  "#B91C8D"),
    "WOLN":  (WOLN_df,  "#214EAF"),"WOLC":  (WOLC_df,  "#AC6106")}

results = {}
for label, (df, color) in individual_sites.items():
    V = df["T_Amp_corr"].values
    Q = df["final_site_est"].values
    results[label] = {**fit_each_site(V, Q), "color": color}

labels = list(results.keys())
colors = [results[s]["color"] for s in labels]
x = np.arange(len(labels))

fig, axes = plt.subplots(1, 2, figsize=(18, 6)) # horizontal orientation 

ax = axes[0]
for i, s in enumerate(labels):
    r = results[s]
    kt, kt_err = r["k_theor"], r["k_theor_ci_log"]
    kb, kb_err = r["k_best"],  r["k_best_ci_log"]
    ax.errorbar(i, kt, yerr=[[kt - kt / 10**kt_err], [kt * 10**kt_err - kt]], fmt='o', color="#AAAAAA", markersize=10, capsize=5,elinewidth=1.5, ecolor="#AAAAAA", zorder=3,markeredgecolor="white", markeredgewidth=0.8)
    ax.errorbar(i, kb,yerr=[[kb - kb / 10**kb_err], [kb * 10**kb_err - kb]],fmt='o', color=results[s]["color"], markersize=10, capsize=5,elinewidth=1.5, ecolor=results[s]["color"], zorder=3,markeredgecolor="white", markeredgewidth=0.8)

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_ylim([5*(10**3), 6*(10**10)])
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.tick_params(axis='both', labelsize=18)
ax.set_ylabel("Scaling Parameter, $\\lambda$", fontsize=22)

ax.axhline(k_theor_EM, color="#5B5B5B", lw=2.2, ls="--", zorder=2, label=f"${sci_latex(k_theor_EM)}$") # empirical model theoretical scaling parameter 
ax.axhline(k_best_EM, color="#771112", lw=2.2, zorder=2, label=f"${sci_latex(k_best_EM)}$") # empirical model best-fit scaling parameter 

ax.yaxis.set_major_formatter( mticker.FuncFormatter(lambda v, _: f"$10^{{{np.log10(v):.0f}}}$"))
legend_handles = [mpatches.Patch(color="#797979", label="$\\lambda_{\\mathrm{t}}$"),mpatches.Patch(color="#940000", label="$\\lambda_{\\mathrm{b}}$"),]
ax.legend(handles=legend_handles, fontsize=16, loc ="lower left")


ax = axes[1]
for i, s in enumerate(labels):
    r = results[s]
    ax.errorbar(i, r["b_best"], yerr=r["b_best_ci"], fmt='o', color=r["color"], markersize=10, capsize=5,elinewidth=1.5, ecolor=r["color"], zorder=3,markeredgecolor="white", markeredgewidth=0.8)

ax.axhline(8/5, color="#000000", ls="--", lw=2.1, zorder=2, label="8/5")
ax.axhline(b_best_EM, color="#A20404", lw=2, ls="-", zorder=2, label=f"{(b_best_EM):.2f}")
ax.axhline(6/14, color="grey", lw=1.5, ls="--", zorder=2, label="6/14")

ax.set_ylim([0.25, 2.25])
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.tick_params(axis='both', labelsize=18)
ax.set_ylabel("Best-fit Exponent, $x$", fontsize=22, labelpad=7)
ax.legend(fontsize=18, loc ="lower left")

plt.tight_layout()
plt.subplots_adjust(wspace=0.2)
plt.savefig(f"thesis_figs/site_model_param_comparison.png", dpi=300, transparent=True,bbox_inches='tight')


#%%

def r_squared(y_obs, Q_pred):
    coeff_matrix = np.corrcoef(y_obs, Q_pred)
    r = coeff_matrix[0, 1]
    r_square = r**2
    return r_square


def nse(y_obs, Q_pred):
    ss_res = np.sum((y_obs - Q_pred)**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    nse = (1-ss_res)/ss_tot
    return nse


def perc_bias(y_obs, Q_pred): #  + = over-prediction, - = under-prediction
    p_bias = (np.sum(Q_pred - y_obs) / np.sum(y_obs))*100
    return p_bias


def mapd(y_obs, Q_pred):
    mapd = mapd = np.mean(np.abs((Q_pred - y_obs) / y_obs)) * 100
    return 100 * np.mean(np.abs((Q_pred - y_obs) / y_obs))


def factor2(y_obs, Q_pred):
    ratio = Q_pred / y_obs
    perc_f2 = (np.mean((ratio >= 0.5) & (ratio <= 2))) *100
    return perc_f2


def kge(y_obs, Q_pred): # Kling-Gupta Efficiency (Gupta et al., 2009)
    r_kge = np.corrcoef(y_obs, Q_pred)[0, 1]
    alpha_kge = np.std(Q_pred)/np.std(y_obs)
    beta_kge = np.mean(Q_pred)/np.mean(y_obs)
    kge_value = 1 - np.sqrt((r_kge - 1)**2 + (alpha_kge - 1)**2 + (beta_kge - 1)**2)
    return kge_value, r_kge, alpha_kge, beta_kge


def metrics(V_obs, Q_actual, Q_pred):
    r2_val = r_squared(Q_actual, Q_pred)
    nse_val = nse(Q_actual, Q_pred)
    pbias_val = perc_bias(Q_actual, Q_pred)
    mapd_val = mapd(Q_actual, Q_pred)
    f2_val = factor2(Q_actual, Q_pred)

    kge_val, r, alpha, beta = kge(Q_actual, Q_pred)
    return kge_val, r, alpha, beta, r2_val, nse_val, pbias_val, mapd_val, f2_val

def validate_model(sites):
    site_registry = {
        "Lemon Creek": {"train_dfs": [BBGL_df, BBGU_df],
            "test_dfs": [BBGL_df, BBGU_df, BBEL_df, BBEU_df, BBWL_df, BBWU_df],
            "labels": ["BBGL", "BBGU", "BBEL", "BBEU", "BBWL", "BBWU"],
            "color": "#D4B000"},
        "Wolverine": {"train_dfs": [WOLN_df, WOLC_df],
            "test_dfs": [WOLN_df, WOLC_df],
            "labels": ["WOLN", "WOLC"],
            "color": "#006DC6"},
        "Mendenhall": {"train_dfs": [MEND_df],
            "test_dfs": [MEND_df],
            "labels": ["AMBR"],
            "color": "#AB009D" }   }

    def get_train_data(glacier_name):
            stations = []
            for df in site_registry[glacier_name]["train_dfs"]:
                x, y = df["T_Amp_corr"].values, df["final_site_est"].values
                mask = (x > 0) & (y > 0)
                stations.append((x[mask], y[mask]))
            return stations
    
    def get_test_data(glacier_name):
        xs, ys = [], []
        for df in site_registry[glacier_name]["test_dfs"]:
            x, y = df["T_Amp_corr"].values, df["final_site_est"].values
            mask = (x > 0) & (y > 0)
            xs.append(x[mask]); ys.append(y[mask])
        return np.concatenate(xs), np.concatenate(ys)

    b_gimbert_inv = 8/5

    results = {}

    fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

    for ax_idx, left_out in enumerate(sites):
        training_glaciers = [s for s in sites if s != left_out]

        train_site_data = []
        for g in training_glaciers:
            train_site_data.extend(get_train_data(g))

        n_mc = 1000
        k_theor_boot, k_best_boot, b_best_boot = [], [], []

        for mc_iter in range(n_mc):
            rng = np.random.default_rng(mc_iter)
            V_parts, Q_parts = [], []
            for x_g, y_g in train_site_data:
                idx = rng.choice(len(x_g), size=850, replace=False)
                V_parts.append(x_g[idx])
                Q_parts.append(y_g[idx])
            lv = np.log10(np.concatenate(V_parts))
            lq = np.log10(np.concatenate(Q_parts))
            pt, _ = curve_fit(log_powerlaw_theor, lv, lq, p0=[1.0])
            pb, _ = curve_fit(log_powerlaw_best,  lv, lq, p0=[1.0, 1.6])
            k_theor_boot.append(10**pt[0])
            k_best_boot.append(10**pb[0])
            b_best_boot.append(pb[1])

        k_theor_boot = np.array(k_theor_boot)
        k_best_boot = np.array(k_best_boot)
        b_best_boot = np.array(b_best_boot)

        k_theor = np.median(k_theor_boot)
        k_best = np.median(k_best_boot)
        b_best = np.median(b_best_boot)

        print(f"Theoretical:k = {k_theor:.4e} [{np.percentile(k_theor_boot,2.5):.3e}, {np.percentile(k_theor_boot,97.5):.3e}]")
        print(f"Best fit:k = {k_best:.4e} [{np.percentile(k_best_boot,2.5):.3e}, {np.percentile(k_best_boot,97.5):.3e}]")
        print(f"β = {b_best:.3f} [{np.percentile(b_best_boot,2.5):.4f}, {np.percentile(b_best_boot,97.5):.4f}]")

        x_train = np.concatenate([x_g for x_g, _ in train_site_data])
        y_train = np.concatenate([y_g for _, y_g in train_site_data])

        x_held, y_held = get_test_data(left_out)

        Q_pred_theor = k_theor * x_held**b_gimbert_inv
        Q_pred_best  = k_best  * x_held**b_best

        kge_t, r_t, alpha_t, beta_t, r2_t, nse_t, pbias_t, mapd_t, f2_t = metrics(x_held, y_held, Q_pred_theor)
        kge_b, r_b, alpha_b, beta_b, r2_b, nse_b, pbias_b, mapd_b, f2_b = metrics(x_held, y_held, Q_pred_best)

        print(f"  Theoretical → R2={r2_t:.3f}  NSE={nse_t:.3f}  PBIAS={pbias_t:+.1f}%  MAPD={mapd_t:.1f}%  F2={f2_t:.1%}")
        print(f"  Best fit    → R2={r2_b:.3f}  NSE={nse_b:.3f}  PBIAS={pbias_b:+.1f}%  MAPD={mapd_b:.1f}%  F2={f2_b:.1%}")
        print(f"  Theoretical → KGE={kge_t:.3f} (r={r_t:.3f}, α={alpha_t:.3f}, β={beta_t:.3f})")
        print(f"  Best fit    → KGE={kge_b:.3f} (r={r_b:.3f}, α={alpha_b:.3f}, β={beta_b:.3f})")

        results[left_out] = dict(
            kge_theor=kge_t, r_theor=r_t, alpha_theor=alpha_t, beta_theor=beta_t,
            r2_theor=r2_t, nse_theor=nse_t, pbias_theor=pbias_t, mapd_theor=mapd_t, f2_theor=f2_t,
            kge_best=kge_b, r_best=r_b, alpha_best=alpha_b, beta_best=beta_b,
            r2_best=r2_b, nse_best=nse_b, pbias_best=pbias_b, mapd_best=mapd_b, f2_best=f2_b,
            k_theor=k_theor, k_best=k_best, b_best=b_best,
            Q_actual=y_held, Q_pred_theor=Q_pred_theor, Q_pred_best=Q_pred_best)

        color = site_registry[left_out]["color"]
        ax = axes[ax_idx]

        ax.scatter(x_train, y_train, c='none', edgecolor="#858585", linewidths=1.5, s=15, alpha=0.1, label="Training Glaciers")
        ax.scatter(x_held, y_held, c='none', edgecolor=color, linewidths=1.5, s=20, alpha=0.2, label=f"{left_out}")

        xx = np.logspace(np.log10(min(x_train.min(), x_held.min())*0.8), np.log10(max(x_train.max(), x_held.max())*1.2), 200)

        ax.plot(xx, k_theor * xx**b_gimbert_inv, 'k--', lw=2, label=f'Theoretical: Q = ${sci_latex(k_theor)} \\cdot V^{{8/5}}$ | $KGE$={kge_t:.2f}')
        ax.plot(xx, k_best  * xx**b_best, 'r--', lw=2, label=f'Best fit: Q = ${sci_latex(k_best)} \\cdot V^{{{b_best:.2f}}}$ | $KGE$={kge_b:.2f}')
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis='both', labelsize=15)
        ax.set_ylabel("Discharge, Q [m³/s]", fontsize=20)
        if ax_idx == len(sites) - 1: 
            ax.set_xlabel("Tremor Amplitude, V [m/s]", fontsize=20)
        ax.legend(fontsize=15, markerscale=3)

        ax.set_xlim([min(x_train.min(), x_held.min())*0.7, max(x_train.max(), x_held.max())*1.7])
        ax.set_ylim([min(y_train.min(), y_held.min())*0.6, max(y_train.max(), y_held.max())*30])

        ax.set_title(f"Testing: {left_out}", fontsize=18)
        plt.savefig(f"thesis_figs/all_model_validation_plot_08_09.png", dpi=300, transparent=True, bbox_inches='tight')

    plt.show()

    print(f"{'Left Out Glacier':<15} {'R2_t':>6} {'R2_b':>6} {'NSE_t':>7} {'NSE_b':>7} " f"{'PBIAS_t':>9} {'PBIAS_b':>9} {'MAPD_t':>8} {'MAPD_b':>8} {'F2_t':>7} {'F2_b':>7}")
    for name, r in results.items():
        print(f"{name:<15} {r['r2_theor']:>6.3f} {r['r2_best']:>6.3f} "
              f"{r['nse_theor']:>7.3f} {r['nse_best']:>7.3f} "
              f"{r['pbias_theor']:>+9.1f} {r['pbias_best']:>+9.1f} "
              f"{r['mapd_theor']:>8.1f} {r['mapd_best']:>8.1f} "
              f"{r['f2_theor']:>7.1%} {r['f2_best']:>7.1%}")

    print(f"{'Left Out Glacier':<15} {'KGE_t':>7} {'KGE_b':>7} {'r_t':>6} {'r_b':>6} "f"{'α_t':>6} {'α_b':>6} {'β_t':>6} {'β_b':>6}")
    for name, r in results.items():
        print(f"{name:<15} {r['kge_theor']:>7.3f} {r['kge_best']:>7.3f} "
              f"{r['r_theor']:>6.3f} {r['r_best']:>6.3f} "f"{r['alpha_theor']:>6.3f} {r['alpha_best']:>6.3f} "
              f"{r['beta_theor']:>6.3f} {r['beta_best']:>6.3f}")

    return results

cv_results = validate_model(["Lemon Creek", "Wolverine", "Mendenhall"])
#%%
R_W_G_cmp = LinearSegmentedColormap.from_list("RWG", ["#690a05", "#ffffff", "#135731"])
B_W_B_cmp = LinearSegmentedColormap.from_list("BWB", ["#074995", "#ffffff", "#074995"])
W_G_cmp = LinearSegmentedColormap.from_list("WG", ["#e1e5e0", "#26642C"])
W_P_cmp = LinearSegmentedColormap.from_list("WP", ["#e6c5e3", "#801367"])

def plot_metrics_comp(results):
    site_order = list(results.keys())
    site_labels = {s: s for s in site_order}
    n = 3
    y_pos = {s: n - 1 - i for i, s in enumerate(site_order)}
    y_lims = (-1.3, n - 1 + 0.6)

    # name, theor_val, bestfit_val, vmin, vmax, cmap, criteria, diverging axis
    panels = [
        ("a) NSE","nse_theor","nse_best",-4,1,R_W_G_cmp, (0.5,1),True),
        ("b) KGE","kge_theor","kge_best",-1,1,R_W_G_cmp,(0.5,1),True),
        ("c) Pbias (%)","pbias_theor", "pbias_best",-80,80,B_W_B_cmp,(-25,25),True),
        #("MAPD (%)","mapd_theor", "mapd_best",0,70,W_P_cmp,(0,25),False),
        ("d) FAC2 (%)", "f2_theor","f2_best",0,100,W_G_cmp,(80,100),False),]
    
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4.6))
    for ax, (title, key_t, key_b, vmin, vmax, cmap, crit_range, diverging) in zip(axes, panels):
        if isinstance(cmap, str):
            cmap = mpl.colormaps[cmap]
        if diverging : 
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) 
        else:
            norm = mpl.colors.Normalize(vmin, vmax)

        gradient_axis = np.linspace(vmin, vmax, 256).reshape(1, -1)
        ax.imshow(gradient_axis, aspect="auto", cmap=cmap, norm=norm, extent=[vmin, vmax, -0.95, -0.55], zorder=1)

        if crit_range is not None:
            cr_min, cr_max = max(crit_range[0], vmin), min(crit_range[1], vmax)
            ax.axvspan(cr_min, cr_max, color="#888888", alpha=0.05, zorder=0.5)

        for site in site_order:
            y = y_pos[site]
            v_t = np.clip(results[site][key_t], vmin, vmax)
            v_b = np.clip(results[site][key_b], vmin, vmax)
            ax.scatter(v_t, y + 0.13, s=190, marker="o", facecolor=cmap(norm(v_t)),edgecolor=cmap(norm(vmax)), linewidth=0.2, zorder=3)
            ax.scatter(v_b, y - 0.13, s=190, marker="s", facecolor=cmap(norm(v_b)), edgecolor=cmap(norm(vmax)), linewidth=0.2, zorder=3)

        ticks = np.linspace(vmin, vmax, 5)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(*y_lims)
        ax.set_ylim(-0.95, y_lims[1])
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0" if t == 0 else "1" if t == 1 else "-4" if t == -4 else f"{t:.1f}" if abs(t) < 10 else f"{t:.0f}" for t in ticks], fontsize=18)
        ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_title(title, fontsize=17, y=-0.22)

    axes[0].set_yticks([y_pos[s] for s in site_order])
    axes[0].set_yticklabels([site_labels[s] for s in site_order], fontsize=20)
    axes[0].tick_params(axis="y", length=0)

    legend_elements = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888", markersize=11, label="Theoretical"), 
                       plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#888888", markersize=11, label="Best-fit")]
    fig.legend(handles=legend_elements, loc="center right", bbox_to_anchor=(0.89, 0.82),frameon=True, fontsize=18, edgecolor="#888888")

    plt.subplots_adjust(wspace=0.12, right=0.93, bottom=0.22, top=0.92)
    plt.savefig(f"thesis_figs/crossvalidation_performance.png", dpi=300, transparent=True, bbox_inches='tight')

    plt.show()
    return 
    
plot_metrics_comp(cv_results)

#%%
