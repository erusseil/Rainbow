from typing import Dict, List, Union
from pathlib import Path
from argparse import ArgumentParser
import glob
from itertools import chain
from functools import lru_cache
from astropy.table import vstack, Table
import astropy.constants as c
import astropy.units as u
import os
import sys
import pickle
import numpy as np
import pandas as pd
import sncosmo
import subprocess
import shlex
from iminuit import Minuit
from iminuit.cost import LeastSquares
import kernel as kern
import time
import shutil
import random
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson


def Am_to_Hz(wl):
    """
    Convert Ångström to Hertz

    Parameters
    ----------
    wl : array
        Wave length array

    Returns
    -------
        Array converted to frequency
    """
    return c.c.value / (wl * 1e-10)


def generate_plasticc_lcs(object_class, field):
    """
    PLAsTiCC lightcurve generator to the correct format.
    Apply MWEBV correction to flux and fluxerr
    
    Parameters
    ----------
    object_class: str
        Name of the PLAsTiCC class to compute
    field: str
        LSST cadence to use : 'wfd' or 'ddf'
        
    Yields
    ------
    table : astropytable
        Formatted lightcurve table. Ensures that the minimum
        number of points from kernel.py is respected
    """
    
    meta = pd.read_csv(f'{kern.plasticc_path}/plasticc_test_metadata.csv.gz', index_col='object_id')
    meta = meta[meta['true_target'] == object_class]
    
    if field == 'ddf':
        file_indices = ['01']
    elif field == 'wfd':
        file_indices = [f'{i:02d}' for i in range(2, 12)]
    elif field == 'all':
        file_indices = [f'{i:02d}' for i in range(1, 12)]
    
    for file_index in file_indices:
        file_name = f'{kern.plasticc_path}/plasticc_test_lightcurves_{file_index}.csv.gz'
        df = pd.read_csv(file_name, index_col='object_id')
        for object_id, table in df.groupby('object_id'):
            if object_id not in meta.index:
                continue
            
            assert np.all(np.diff(table['mjd']) >= 0), 'Light curve must be time-sorted'
            
            table = Table.from_pandas(table)
            table.rename_columns(['mjd', 'flux', 'flux_err'], ['MJD', 'FLUXCAL', 'FLUXCALERR'])
            table['BAND'] = kern.PASSBANDS[table['passband']]
            
            table.meta = meta.loc[object_id].to_dict()
            table.meta['SIM_PEAKMJD'] = table.meta['true_peakmjd']
            table.meta['SNID'] = object_id
            
            detections = table[table['detected_bool'] == 1]
            det_per_band = dict(zip(*np.unique(detections["BAND"], return_counts=True)))

            if any(det_per_band.get(band, 0) < min_det for band, min_det in kern.min_det_per_band.items()):
                continue
            
            if (table.meta['SIM_PEAKMJD'] < detections['MJD'][0]
                 or table.meta['SIM_PEAKMJD'] > detections['MJD'][-1]):
                continue
            
            yield table

def generate_plasticc_YSE(object_class, method):
    """
    YSE lightcurve generator to the correct format
    
    Parameters
    ----------
    object_class: str
        Name of the YSE class to compute
    method : str
        Specify the future method of feature extraction to be
        used on the data. Impact the minimum number of point per band.
        
    Yields
    ------
    table : astropytable
        Formatted lightcurve table. Ensures that the minimum
        number of points from kernel.py is respected

    """
    
    meta = pd.read_csv(f'{kern.YSE_path}/YSE_baseline_plasticc_meta.csv', index_col='object_id')
    meta = meta[meta['true_target'] == object_class]
    
    file_name = f'{kern.YSE_path}/YSE_baseline_plasticc_data.csv'
    df = pd.read_csv(file_name, index_col='object_id')

    for object_id, table in df.groupby('object_id'):
        if object_id not in meta.index:
            continue

        assert np.all(np.diff(table['mjd']) >= 0), 'Light curve must be time-sorted'

        table = Table.from_pandas(table)
        table.rename_columns(['mjd', 'flux', 'flux_err'], ['MJD', 'FLUXCAL', 'FLUXCALERR'])


        table.meta = meta.loc[object_id].to_dict()
        table.meta['SIM_PEAKMJD'] = table.meta['true_peakmjd']
        table.meta['SNID'] = object_id

        table2 = table.copy()
        for col in ['BAND', 'passband']:
            for band in ['g', 'r']:
                table2[col][table2[col]==f'PS{band}'] = band + ' '

        table2 = table2[table2['BAND']!= 'PSi']

        if method == 'bazin':
            table = table2

        detections = table2[table2['detected_bool'] == 1]
        det_per_band = dict(zip(*np.unique(detections["BAND"], return_counts=True)))

        if any(det_per_band.get(band, 0) < min_det for band, min_det in kern.min_det_per_band_YSE.items()):
            continue

        if (table2.meta['SIM_PEAKMJD'] < detections['MJD'][0]
             or table2.meta['SIM_PEAKMJD'] > detections['MJD'][-1]):
            continue

        yield table
        

def format_plasticc(object_class, max_n, field):
    """
    Use plasticc generator and aggregate
    the light curves
    
    Parameters
    ----------
    object_class: str
        Name of the PLAsTiCC class to compute
    max_n: int
        Maximum number of light curves to aggregate.
    field: str
        LSST cadence to use : 'wfd' or 'ddf'
        
    Returns
    -------
    lcs : list
        List of preprocessed and formatted astropy tables.
    """
        
    lcs = []
    for idx, lc in zip(range(max_n), generate_plasticc_lcs(object_class, field)):
        lcs.append(lc)
        
    # Save preprocessed data as pkl for later use
    if not os.path.exists("data_plasticc"):
        os.mkdir("data_plasticc")

    if not os.path.exists("data_plasticc/formatted"):
        os.mkdir("data_plasticc/formatted")
        
    file = f"data_plasticc/formatted/{kern.PLASTICC_TARGET_INV.get(object_class)}_{field}.pkl"

    with open(file, "wb") as handle:
        pickle.dump(lcs, handle)


def format_YSE(object_class, max_n, field):
    """
    Use YSE generator and aggregate
    the light curves
    
    Parameters
    ----------
    object_class: str
        Name of the PLAsTiCC class to compute
    max_n: int
        Maximum number of light curves to aggregate.
    field: str
        fake LSST cadence for file naming only.
        
    Returns
    -------
    lcs : list
        List of preprocessed and formatted astropy tables.
    """

    # Save preprocessed data as pkl for later use
    if not os.path.exists("data_YSE"):
        os.mkdir("data_YSE")

    if not os.path.exists("data_YSE/formatted"):
        os.mkdir("data_YSE/formatted")
            
    lcs_rainbow, lcs_bazin = [], []

    for idx, lc in zip(range(max_n), generate_plasticc_YSE(object_class, 'rainbow')):
        lcs_rainbow.append(lc)

    for idx, lc in zip(range(max_n), generate_plasticc_YSE(object_class, 'bazin')):
        lcs_bazin.append(lc)

    for fraction in kern.YSE_fractions:

        sub_lcs_rainbow, sub_lcs_bazin  = [], []
        
        for idx in range(len(lcs_rainbow)):
            table_rainbow, table_bazin = lcs_rainbow[idx], lcs_bazin[idx]
            sub_table_rainbow, sub_table_bazin = table_rainbow[4:], table_bazin[4:],

            row_index = np.sort(np.random.choice(sub_table_rainbow['Unnamed: 0'], size = round(len(sub_table_rainbow) * fraction), replace=False))

            mask_rainbow = [True]*4 + [i in row_index for i in sub_table_rainbow['Unnamed: 0']]
            mask_bazin = [True]*4 + [i in row_index for i in sub_table_bazin['Unnamed: 0']]

            detections = table_bazin[mask_bazin][table_bazin[mask_bazin]['detected_bool'] == 1]
            det_per_band = dict(zip(*np.unique(detections["BAND"], return_counts=True)))
            reject = any(det_per_band.get(band, 0) < min_det for band, min_det in kern.min_det_per_band_YSE.items())

            if not reject:
                sub_lcs_rainbow.append(table_rainbow[mask_rainbow])
                sub_lcs_bazin.append(table_bazin[mask_bazin])
        
        file_rainbow = f"data_YSE/formatted/{kern.PLASTICC_TARGET_INV.get(object_class)}_{field}_f{str(round(100*fraction))}_rainbow.pkl"
        file_bazin = f"data_YSE/formatted/{kern.PLASTICC_TARGET_INV.get(object_class)}_{field}_f{str(round(100*fraction))}_bazin.pkl"

        with open(file_rainbow, "wb") as handle:
            pickle.dump(sub_lcs_rainbow, handle)
            
        with open(file_bazin, "wb") as handle:
            pickle.dump(sub_lcs_bazin, handle)


def preprocess(dataset, object_class, field):
    """
    Normalize each light curve of the dataset.
    Saves a pickle file of the preprocessed data.
    
    Parameters
    ----------
    database: str
        'plasticc' or 'YSE'
    object_class: str
        Name of the PLAsTiCC class to compute
    field: str
        LSST cadence to use : 'wfd' or 'ddf' (always 'wfd' for YSE)

    Returns
    -------
    list
    List of normalized astropy tables
    """

    path_to_use = kern.main_path + f'/data_{dataset}/formatted/{object_class}_{field}.pkl'

    with open(path_to_use, "rb") as handle:
        lcs = pickle.load(handle)

    new_lcs = [normalize(lc) for lc in lcs]
                  
    # Save preprocessed data as pkl for later use
    if not os.path.exists(f"data_{dataset}"):
        os.mkdir(f"data_{dataset}")

    if not os.path.exists(f"data_{dataset}/preprocessed"):
        os.mkdir(f"data_{dataset}/preprocessed")
        
    file = f"data_{dataset}/preprocessed/{object_class}_{field}"

    with open(f'{file}.pkl' , "wb") as handle:
        pickle.dump(new_lcs, handle)

    return new_lcs


def normalize(lc):
    """
    Apply normalization transformation to
    an astropy table light curve
    
    Parameters
    ----------
    lc: astropy table
        Light curve

    Returns
    -------
    lc : astropy table
        Normalized light curve with :
        - Fluxes divided by max r band flux
        - Time 0 shifted by time of max r band flux
        - Added columns before normalization : max_flux, max_flux_time
    """

    normband = lc[(lc["BAND"] == "r ") | (lc["BAND"] == "PSr")]
    maxi = normband["FLUXCAL"].max()

    lc["max_flux"] = maxi
    t_maxi = normband["MJD"][normband["FLUXCAL"].argmax()]
    lc["max_flux_time"] = t_maxi
    lc["FLUXCAL"] = lc["FLUXCAL"] / maxi
    lc["FLUXCALERR"] = lc["FLUXCALERR"] / maxi
    lc["MJD"] = lc["MJD"] - t_maxi

    return lc

def Fbaz(t, a, t0, tfall, trise):
    """
    Compute flux using Bazin with
    baseline fixed to 0.

    Parameters
    ----------
    t: array
        Time value of data points.
    a: float
        Amplitude.
    t0: float
        Time phase (related to peak time by : tpeak = t0 + trise × ln(tfall/trise − 1))
    tfall: float
        Falling factor
    trise: float
        Rising factor

    Returns:
    --------
        Computed flux at each time t.
    """

    return a * np.exp(-(t - t0) / tfall) / (1 + np.exp((t - t0) / trise))


def Tsig(t, Tmin, dT, ksig, t0):
    """
    Compute temperature using sigmoid.

    Parameters
    ----------
    t: array
        Time value of data points.
    Tmin: float
        Minimum temperature to reach at the end of the sigmoid
    dT: float
        Difference between beginning and end temperature of the sigmoid.
    ksig: float
        Slope parameter of the sigmoid.
    t0: float
        Bazin time value related to time of maximum flux.

    Returns:
    --------
        Computed temperature at each time t.
    """

    return Tmin + dT / (1 + np.exp((t - t0) / ksig))


def planck_nu(nu, T):
    """
    Compute blackbody intensity from temperature and frequency.

    Parameters
    ----------
    nu: array
        Frequency for which to compute intensity.
    T: array
        Temperature values at different times.

    Returns:
    --------
        Computed spectral radiance.
    """

    return (2 * c.h.value / c.c.value**2) * nu**3 / np.expm1(c.h.value * nu / (c.k_B.value * T))


@lru_cache(maxsize=64)
def planck_passband_spline(passband, T_min=1e2, T_max=1e6, T_steps=10_000):
    """
    Compute spline of passband intensity for the range of temperatures.

    Parameters
    ----------
    passband: string
        Name of the passband
    T_min: float
        Minimum temperature to consider
    T_max: float
        Maximum temperature to consider
    T_steps: int
        Step between temperatures to consider

    Returns
    -------
    scipy.iterpolate.UnivariateSpline
        Spline of the passband intensity versus temperature
    """
    passband = sncosmo.get_bandpass(passband)
    nu = (c.c / (passband.wave * u.AA)).to_value(u.Hz)
    trans = passband.trans  # photon counter transmission

    T = np.logspace(np.log10(T_min), np.log10(T_max), T_steps)
    black_body_integral = simpson(x=nu, y=planck_nu(nu, T[:, None]) * trans / nu, axis=-1)
    transmission_integral = simpson(x=nu, y=trans / nu)

    return UnivariateSpline(x=T, y=black_body_integral / transmission_integral, s=0, k=3, ext='raise')


def planck_passband(passband, T):
    """
    Compute spectral radiance from temperature and passband.

    Parameters
    ----------
    passband: array of strings
        Names of passband for which to compute intensity.
    T: array of floats
        Temperature values at different times.

    Returns:
    --------
    array of floats
        Computed intensity.
    """
    passband, T = np.broadcast_arrays(passband, T)

    # Speed up computation if only a single value is passed
    if passband.size == 1:
        return planck_passband_spline(passband.item())(T)

    unique_passbandes, indices = np.unique(passband, return_inverse=True)
    output = np.zeros_like(passband, dtype=float)
    for index, passband in enumerate(unique_passbandes):
        T_passband = T[indices == index]
        spline = planck_passband_spline(passband)
        output[indices == index] = spline(T_passband)

    return output


def planck(nu, T):
    """
    Compute intensity from temperature and frequency / passband.

    Parameters
    ----------
    nu: array of floats or strings
        Frequencies or passband names for which to compute intensity.
    T: array of floats
        Temperature values at different times.

    Returns:
    --------
    array of floats
        Computed intensity.
    """
    integrate = False
    if np.any(nu < 0):
        nu = np.vectorize(sncosmoband_convert.get)(nu)
        integrate = True
    nu = np.asarray(nu)
    
    if integrate:
        return planck_passband(nu, T)
   
    return planck_nu(nu, T)



''' OLD Function used taking into account redshift z
#######################################################
# Flux of lightcurves at any time at any frequency
def Fnuz(x, a, t0, tfall, trise, Tmin, dT, ksig, z):
    t, nu = x
    T = Tsig(t, Tmin, dT, ksig, t0) * (1+z)
    Fbol = Fbaz(t, a, t0, tfall, trise)
    amplitude = 1e15

    return np.pi / c.sigma_sb.value * Fbol * planckz(nu, T, z) / T**4 * amplitude

def planckz(nu, T, z):
    integrate = False
    if np.any(nu < 0):
        nu = np.vectorize(sncosmoband_convert.get)(nu)
        integrate = True
    nu = np.asarray(nu)
    return planck_passbandz(nu, T, z)

def planck_passbandz(passband, T, z):
    passband, T = np.broadcast_arrays(passband, T)

    # Speed up computation if only a single value is passed
    if passband.size == 1:
        return planck_passband_splinez(passband.item(), z)(T)

    unique_passbandes, indices = np.unique(passband, return_inverse=True)
    output = np.zeros_like(passband, dtype=float)
    for index, passband in enumerate(unique_passbandes):
        T_passband = T[indices == index]
        spline = planck_passband_splinez(passband, z)
        output[indices == index] = spline(T_passband)

    return output

def planck_passband_splinez(passband, z, T_min=1e2, T_max=1e6, T_steps=10_000):
    passband = sncosmo.get_bandpass(passband)
    nu = (c.c / (passband.wave * u.AA)).to_value(u.Hz)
    nu = nu * (1+z)
    trans = passband.trans  # photon counter transmission

    T = np.logspace(np.log10(T_min), np.log10(T_max), T_steps)
    black_body_integral = simpson(x=nu, y=planck_nu(nu, T[:, None]) * trans / nu, axis=-1)
    transmission_integral = simpson(x=nu, y=trans / nu)

    return UnivariateSpline(x=T, y=black_body_integral / transmission_integral, s=0, k=3, ext='raise')
##################################################################################
'''

# Flux of lightcurves at any time at any frequency
def Fnu(x, a, t0, tfall, trise, Tmin, dT, ksig):
    """
    Complete fitting function. Used to compute flux at any
    frequency at any time (scaled by an arbitrary amplitude term).

    Parameters
    ----------
    x: ndarray
        Array of pair time and frequency from which to compute flux.
    a: float
        Bazin amplitude.
    t0: float
        Bazin time value related to time of maximum flux.
    tfall: float
        Bazin value related to length of the falling part of the slope.
    trise: float
        Bazin value related to length of the rising part slope.
    Tmin: float
        Minimum temperature to reach at the end of the sigmoid
    dT: float
        Difference between beginning and end temperature of the sigmoid.
    ksig: float
        Slope parameter of the sigmoid.

    Returns:
    --------
        Computed flux at any frequency and at any time (scaled by
        an arbitrary amplitude term).
    """
    t, nu = x
    
    T = Tsig(t, Tmin, dT, ksig, t0)
    Fbol = Fbaz(t, a, t0, tfall, trise)
    amplitude = 1e15

    return np.pi / c.sigma_sb.value * Fbol * planck(nu, T) / T**4 * amplitude


def perform_fit_rainbow(obj):
    """
    Find best fit parameters for rainbow method using iminuit.
    Adds additionnal values useful for later ML.

    Parameters
    ----------
    obj: astropy table
        Single object table

    Returns
    -------
    Dict of best fitted values.
    And list of values:
        fit_error: float
            Error between fit and true data
        max_flux: float
            Maximum observed flux before normalization
        max_flux_time: float
            Time of maximum observed flux
        peak: float
            Meta data of ELASTiCC, true time of maximum flux
        nb_points: int
            Total number of measurements in all bands
    """

    global_flux = obj["FLUXCAL"]
    global_fluxerr = obj["FLUXCALERR"]
    global_nu = obj["NU"]
    global_mjd = obj["MJD"]
    
    parameters_dict = {
        "a": global_flux.max(),
        "t0": global_mjd[np.argmax(global_flux)],
        "tfall": 30,
        "trise": -5,
        "Tmin": 4000,
        "dT": 7000,
        "ksig": 4,
    }
    
    least_squares = LeastSquares(
        np.array([global_mjd, global_nu]), global_flux, global_fluxerr, Fnu
    )

    fit = Minuit(
        least_squares,
        **parameters_dict,
    )

    fit.limits['Tmin'] = (100, 100000)
    fit.limits['dT'] = (0, 100000)
    fit.limits['t0'] = (-1000, 1000)
    fit.limits['a'] = (0.1, 1000)
    fit.limits['ksig'] = (1.5, 300)
    fit.limits['trise'] = (-100, -0.5)
    fit.limits['tfall'] = (0.5, 500)
    fit.migrad()

    max_flux = obj["max_flux"][0]
    max_time = obj["max_flux_time"][0]
    fit_error = fit.fval
    peak = obj.meta["SIM_PEAKMJD"]
    nb_points = len(global_mjd)
    objid = obj.meta["SNID"]

    additionnal = [fit_error, max_flux, max_time, peak, nb_points, objid]

    return fit.values, additionnal


def perform_fit_bazin(obj, bands):
    """
    Find best fit parameters for Bazin method using iminuit.
    Adds additionnal values useful for later ML.

    Parameters
    ----------
    obj: astropy table
        Single object table

    Returns
    -------
    list of the form  [[param_g, extra_g], [param_r, extra_r], [param_i, extra_i]]
    With param_* : best fitted value for each passband.
    And extra_* : list of additionnal values for each passband:
        fit_error: float
            Error between fit and true data
        max_flux: float
            Maximum observed flux before normalization
        max_flux_time: float
            Time of maximum observed flux
        peak: float
            Meta data of ELASTiCC, true time of maximum flux
        nb_points: int
            Number of measurements
    """
    all_parameters = []

    for band in bands:

        lc = obj[obj["BAND"] == band]
        global_flux = lc["FLUXCAL"]
        global_fluxerr = lc["FLUXCALERR"]
        global_mjd = lc["MJD"]

        parameters_dict = {
            "a": global_flux.max(),
            "t0": global_mjd[np.argmax(global_flux)],
            "tfall": 30,
            "trise": -5,
        }

        least_squares = LeastSquares(global_mjd, global_flux, global_fluxerr, Fbaz)
        fit = Minuit(
            least_squares,
            **parameters_dict,
        )
        
        fit.limits['t0'] = (-1000, 1000)
        fit.limits['a'] = (0.1, 1000)
        fit.limits['trise'] = (-100, -0.5)
        fit.limits['tfall'] = (0.5, 500)

        fit.migrad()

        max_flux = lc["max_flux"][0]
        max_time = lc["max_flux_time"][0]
        fit_error = fit.fval
        peak = lc.meta["SIM_PEAKMJD"]
        nb_points = len(global_mjd)
        objid = lc.meta['SNID']

        additionnal = [fit_error, nb_points, max_flux, max_time, peak, objid]

        all_parameters.append([fit.values, additionnal])

    return all_parameters


def extract_rainbow(lcs, object_class, split, database, band_wavelength):
    """
    Apply perform_fit_rainbow to each object.
    Build and save a parameter dataframe.

    Parameters
    ----------
    lcs: list
        List of astropy table lightcurves
    object_class: str
        Name of the ELASTiCC class to compute
    split: int
        Number of the core computing the extraction
    band_wavelength: str
        'effective' to use only the effective
        wavelength of the filter. Uses "freq_dic" defined below
        
        'integrate' to use the filter wavelength
        profile as provided in sncosmo. Uses "sncosmoband_convert_temporary"
        and "sncosmoband_convert" because iminuit can only use numerical
        values as input.
        
    """

    all_param = []
    for obj in lcs:
        if band_wavelength=='effective':
            obj["NU"] = np.vectorize(freq_dic.get)(list(obj["BAND"]))
            
        elif band_wavelength=='integrate':
            obj["NU"] = np.vectorize(sncosmoband_convert_temporary.get)(obj["BAND"])
        
        extraction = perform_fit_rainbow(obj)
        all_param.append(list(extraction[0].to_dict().values()) + extraction[1])

    features = pd.DataFrame(
        columns=Fnu.__code__.co_varnames[1:8] + ("error", "max_flux", "max_time", "true_peak", "nb_points", "object_id"),
        data=all_param,
    )

    features.to_parquet(f"data_{database}/features/rainbow_{band_wavelength}/{object_class}/features_{object_class}_{split}.parquet")


def extract_bazin(lcs, object_class, split, database):
    """
    Apply perform_fit_bazin to each object.
    Build and save a parameter dataframe.

    Parameters
    ----------
    lcs: list
        List of astropy table lightcurves
    object_class: str
        Name of the ELASTiCC class to compute
    split: int
        Number of the core computing the extraction
    """
   
    bands = ["g ", "r "]
    
    if database != "YSE" :
        bands += "i "

    all_param = []
    name_param = []

    for idx, obj in enumerate(lcs):

        param = perform_fit_bazin(obj, bands)

        obj_param = []
        for idx_band, band in enumerate([i[0] for i in bands]):

            if idx == 0:
                baz_name = [x + f"_{band}" for x in list(Fbaz.__code__.co_varnames[1:]) + ["error", "nb_points"]]
                name_param.append(baz_name)

            obj_param.append(
                list(param[idx_band][0].to_dict().values()) + [param[idx_band][1][0]] + [param[idx_band][1][1]]
            )

        flat_obj_param = [x for xs in obj_param for x in xs] + param[0][1][
            2:
        ]
        all_param.append(flat_obj_param)

    flat_name_param = [x for xs in name_param for x in xs] + [
        "max_flux",
        "max_time",
        "true_peak",
        "object_id"
    ]

    features = pd.DataFrame(columns=flat_name_param, data=all_param)
    features.to_parquet(f"data_{database}/features/bazin/{object_class}/features_{object_class}_{split}.parquet")

def train_test_cutting_generator_YSE_rainbow(sub_formatted, formatted_test):
    """
    Function used once test dataset has been generated for YSE dataset
    Use every point non selected in test, including PS-i band !!

    Parameters
    ----------
    sub_formatted: list
        List of astropy table lightcurves
    formatted_test: list
        List of testing sample astropy table
    
    Yields
    ------
    train, test
        astropy table, astropy table
    """

    for idx in range(len(sub_formatted)):
        obj, bazin_test = sub_formatted[idx], formatted_test[idx]
        obj_train = obj[[not i in (bazin_test['Unnamed: 0']) for i in obj['Unnamed: 0']]]
        obj_test = obj[[i in (bazin_test['Unnamed: 0']) for i in obj['Unnamed: 0']]]

        yield obj_train, obj_test


    
def train_test_cutting_generator(formatted, bands = ['g ', 'r ', 'i '], mdpb=kern.min_det_per_band,\
                                 mtot=sum(list(kern.min_det_per_band.values())), seed=kern.default_seed):
    """
    Remove random windows of detected points.
    Use removed points to create testing dataset.
    Use remaining points to create training dataset
    Ensures that the training datasets has enough points.

    Parameters
    ----------
    formatted: list
        List of astropy table lightcurves
    bands: list
        Name of the passbands used in the dataset
        Default is ['g ', 'r ', 'i ']
    mdpb: dict
        Minimum detection per passband cut condition.
        Default value is defined in kernel
    mtot: int
        Total number of points required throught the passbands
        Defaut is the sum of the minimum number of points
        for each band as defined kernel
    seed: int
        Seed used to randomly generate the bins 
    
    Yields
    ------
    train, test
        astropy table, astropy table
    """
    
    random.seed(seed)

    for obj in formatted: 
            
        all_bands_train, all_bands_test = [], []

        mask = [False] * len(obj)
        for band in bands:
            mask = mask | (obj['BAND'] == band)

        sub_obj = obj[mask]

        bins = np.arange(sub_obj['MJD'].min(), sub_obj['MJD'].max(), kern.point_cut_window)

        valid = []

        for idx in range(len(bins)-1):
            if ((sub_obj['MJD'] >= bins[idx]) &\
                (sub_obj['MJD'] <= bins[idx + 1]) &\
                (sub_obj['detected_bool'] == 1)).sum()>0:

                valid.append(idx)

        if len(valid)<=2:
            continue

        chosen = random.sample(valid, 2)
        mask = (((sub_obj['MJD']>=bins[chosen[0]]) &\
                 (sub_obj['MJD']<bins[chosen[0] + 1])) |\
                ((sub_obj['MJD']>=bins[chosen[1]]) &\
                 (sub_obj['MJD']<bins[chosen[1] + 1])))

        invalid = False
        for band in bands:
            pointperband = len(sub_obj[(~mask) & (sub_obj['BAND'] == band)])

            if pointperband < mdpb.get(band):
                invalid = True
                
            if len(sub_obj[~mask]) < mtot:
                invalid = True
                
        if invalid:     
            continue

        yield sub_obj[~mask], sub_obj[mask]

def train_test_rising_cutting_generator(prepro, bands = ['g ', 'r ', 'i '], mdpb=kern.min_det_per_band):
    """
    Remove all points after true peak.
    Use removed points to create testing dataset.
    Use remaining points to create training dataset
    Ensures that the training datasets has enough points.

    Parameters
    ----------
    prepro: list
        List of astropy table lightcurves
    bands: list
        Name of the passbands used in the dataset
        Default is ['g ', 'r ', 'i ']
    mdpb: dict
        Minimum detection per passband cut condition.
        Default value is defined in kernel

    Yields
    ------
    train, test
        astropy table, astropy table
    """

    for obj in prepro:
        
        band_mask = [False] * len(obj)
        for band in bands:
            band_mask = band_mask | (obj['BAND'] == band)
            
        obj = obj[band_mask]
        invalid = False
        all_bands_train, all_bands_test = [], []

        threshold = obj.meta['true_peakmjd']
        
        detec_before = (obj[obj['MJD'] < threshold]['detected_bool']).sum()
        if detec_before == 0:
            continue

        for band in bands:

            if invalid:
                continue
                
            sub_obj = obj[obj['BAND'] == band]
            mask = sub_obj['MJD'] > threshold

            if ((~mask).sum() < mdpb.get(band)) |\
               ((mask).sum() < 1):
                invalid = True
                continue

            all_bands_train.append(sub_obj[~mask])
            all_bands_test.append(sub_obj[mask])

        if invalid:
            continue
            
        yield vstack(all_bands_train), vstack(all_bands_test)
        

def check_inputs(database, field):
    
    """
    Verify that inputs correspond to possible inputs

    Parameters
    ----------
    database: str
        'elasticc', 'plasticc' or 'YSE'
    field: str
        'ddf' or 'wfd' (always 'wfd' for YSE)
    """
    
    if (database != 'elasticc') & (database != 'plasticc') & (database != 'YSE'):
        raise ValueError('Function must be elasticc or plasticc or YSE')

    if (database == 'plasticc') & (field != 'ddf') & (field != 'wfd') & (field != 'all'):
        raise ValueError('Field must be ddf, wfd or all')
        
    if not os.path.exists(f"data_{database}/"):
            os.mkdir(f"data_{database}/")
    
    
def format_target(object_class, n_max, database, field):
    """
    Format a given class to the correct data shape.
    Saves a pkl file
    
    Parameters
    ----------
    object_class: str
        Type of object to format
    n_max: int
        Maximum number of object to use for
        future analysis.
    database: str
        'elasticc', 'plasticc' or 'YSE'
    field: str
        'ddf' or 'wfd' (always 'wfd' for YSE)
    """
    
    check_inputs(database, field)
        
    if database == 'plasticc':
        format_plasticc(kern.PLASTICC_TARGET.get(object_class), n_max, field)

    if database == 'YSE':
        format_YSE(kern.PLASTICC_TARGET.get(object_class), n_max, field)


def preprocess_target(object_class, database, field):
    """
    Preprocess a given class from a formated pkl file.
    Saves a pkl file
    
    Parameters
    ----------
    object_class: str
        Type of object to format
    database: str
        'plasticc' or 'YSE'
    field: str
        'ddf' or 'wfd' (always 'wfd' for YSE)
    """
        
    check_inputs(database, field)
    preprocess(database, object_class, field)
        

def feature_extract_target(object_class, fex_function, cores, database, field, band_wavelength):
    """
    Extract features of a given class from a preprocessed pkl file.
    Saves a parquet file
    
    Parameters
    ----------
    object_class: str
        Type of object to format
    n_max: int
        Maximum number of object to use for
        future analysis.
    database: str
        'elasticc', 'plasticc' or 'YSE'
    field: str
        'ddf' or 'wfd' (always 'wfd' for YSE)
    band_wavelength: str
        'effective' to use only the effective
        wavelength of the filter. Uses "freq_dic" defined below
        
        'integrate' to use the filter wavelength
        profile as provided in sncosmo. Uses "sncosmoband_convert_temporary"
        and "sncosmoband_convert" because iminuit can only use numerical
        values as input.
    """

    start_time = time.time()
    show_band_wavelength = "_" + band_wavelength if fex_function == 'rainbow' else ''


    if not os.path.exists(f"data_{database}/features/"):
            os.mkdir(f"data_{database}/features/")
    
    if not os.path.exists(f"data_{database}/features/{fex_function}{show_band_wavelength}/"):
            os.mkdir(f"data_{database}/features/{fex_function}{show_band_wavelength}/")

    check_inputs(database, field)
    
    if (fex_function != 'rainbow') & (fex_function != 'bazin'):
        raise ValueError('Function must be rainbow or bazin')
        
    if (band_wavelength != 'effective') & (band_wavelength != 'integrate'):
        raise ValueError('Band_wavelength must be effective or integrate')
            

    if not os.path.exists(f"data_{database}/features/{fex_function}{show_band_wavelength}/{object_class}_{field}"):
        os.mkdir(f"data_{database}/features/{fex_function}{show_band_wavelength}/{object_class}_{field}")

    subprocess.call(
        shlex.split(f"sh feature_extraction.sh {cores} {object_class}_{field} {fex_function} {database} {band_wavelength}")
    )

    temp_path = f"data_{database}/features/{fex_function}{show_band_wavelength}/{object_class}_{field}/"
        
        
    n_computed_files = len(
        [
            entry
            for entry in os.listdir(temp_path)
            if os.path.isfile(os.path.join(temp_path, entry))
        ]
    )

    while (int(n_computed_files) != int(cores)):
        if (time.time() - start_time)<(3600*kern.max_computation_time):
            time.sleep(0.5)
            n_computed_files = len(
                [
                    entry
                    for entry in os.listdir(temp_path)
                    if os.path.isfile(os.path.join(temp_path, entry))
                ]
            )

        else:
            raise RuntimeError('Computation time reached maximimum allowed time. Main job has been killed')

    all_filenames = np.sort(np.array(os.listdir(temp_path)))

    isfeatures = [".parquet" in f for f in all_filenames]
    features = pd.concat(
        [pd.read_parquet(temp_path + f) for f in all_filenames[isfeatures]],
        ignore_index=True,
    )
    
    total_time = time.time() - start_time

    features.to_parquet(f"data_{database}/features/{fex_function}{show_band_wavelength}/{object_class}_{field}_features.parquet")

    with open(f"data_{database}/features/{fex_function}{show_band_wavelength}/{object_class}_{field}_info.txt", "w") as f:
        f.write(f"Feature extraction took {total_time} sec, over {cores} cores")
        f.write("\n")
        f.write(
            f'The features table take {os.path.getsize(f"data_{database}/features/{fex_function}{show_band_wavelength}/{object_class}_{field}_features.parquet")} bytes of space'
        )
        f.write("\n")
        f.write((features.head()).to_string())

    shutil.rmtree(temp_path)
        
        
def train_test_bins_target(object_class, database, field):
    """
    Cut data into a train and a test sample
    using the random bins removal method
    Saves a pkl file
    
    Parameters
    ----------
    object_class: str
        Type of object to format
    database: str
        'elasticc', 'plasticc' or 'YSE'
    field: str
        In the case of plasticc and YSE.
        'ddf' or 'wfd' (always 'wfd' for YSE)
    """
    
    path = f'data_{database}/formatted/'
        
    if database == 'YSE':
        seed = int(field[field.find('_v')+2:])
        bands = ['g ', 'r ']
        mdpb = kern.min_det_per_band_YSE
        mtot = sum(list(kern.min_det_per_band.values()))
        unversion_field = field[:field.find('_v')]
        
        with open(f'{path}{object_class}_{unversion_field}.pkl', "rb") as handle:
            formatted = pickle.load(handle)

    else:
        seed = kern.default_seed
        bands = ['g ', 'r ', 'i ']
        mdpb = kern.min_det_per_band
        mtot = sum(list(kern.min_det_per_band.values()))
        
        with open(f'{path}{object_class}_{field}.pkl', "rb") as handle:
            formatted = pickle.load(handle)

    generator = train_test_cutting_generator(formatted, bands=bands, mdpb=mdpb, mtot=mtot, seed=seed)
    
    # Case of YSE, we utilize the train_test performed for bazin and we add PSi
    if '_rainbow' in field:
        bazin_field = field.replace("_rainbow", "_bazin")
        
        with open(f'{path}{object_class}_test_w{kern.point_cut_window}_{bazin_field}.pkl', "rb") as handle:
            formatted_test = pickle.load(handle)
            
        sub_formatted = [i for i in formatted if (i.meta['SNID'] in ([j.meta['SNID'] for j in formatted_test]))]
        generator = train_test_cutting_generator_YSE_rainbow(sub_formatted, formatted_test)
    
    train_test = list(generator)

    with open(f'{path}{object_class}_train_w{kern.point_cut_window}_{field}.pkl', "wb") as handle:
        pickle.dump([x[0] for x in train_test], handle)

    with open(f'{path}{object_class}_test_w{kern.point_cut_window}_{field}.pkl', "wb") as handle:
        pickle.dump([x[1] for x in train_test], handle)

    
def train_test_rising_target(object_class, database, field):
    """
    Cut data into a train and a test sample
    using the after rising removal method
    Saves a pkl file
    
    Parameters
    ----------
    object_class: str
        Type of object to format
    database: str
        'elasticc', 'plasticc' or 'YSE'
    field: str
        In the case of plasticc and YSE.
        'ddf' or 'wfd' (always 'wfd' for YSE)
    """
    
    bands = ['g ', 'r ', 'i ']
    mdpb = kern.min_det_per_band
    
    if database == 'YSE':
        mdpb = kern.min_det_per_band_YSE
        bands = ['g ', 'r ', 'PSg', 'PSr', 'PSi']
        
    path = f'data_{database}/formatted/'

    if database == 'elasticc':
        with open(f'{path}{object_class}.pkl', "rb") as handle:
            formatted = pickle.load(handle)

        generator = train_test_rising_cutting_generator(formatted, bands=bands, mdpb=mdpb)
        train_test = list(generator)

        with open(f'{path}{object_class}_train_rising.pkl', "wb") as handle:
            pickle.dump([x[0] for x in train_test], handle)

        with open(f'{path}{object_class}_test_rising.pkl', "wb") as handle:
            pickle.dump([x[1] for x in train_test], handle)
            
    else:
        with open(f'{path}{object_class}_{field}.pkl', "rb") as handle:
            formatted = pickle.load(handle)

        generator = train_test_rising_cutting_generator(formatted, bands=bands, mdpb=mdpb)
        train_test = list(generator)

        with open(f'{path}{object_class}_train_rising_{field}.pkl', "wb") as handle:
            pickle.dump([x[0] for x in train_test], handle)

        with open(f'{path}{object_class}_test_rising_{field}.pkl', "wb") as handle:
            pickle.dump([x[1] for x in train_test], handle)
        
        
# __________________________USEFUL VALUES________________________

SATURATION_FLUX = 1e5

# Source of values used for the filters : http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=LSST&asttype=
nu_u = Am_to_Hz(3751)
nu_g = Am_to_Hz(4742)
nu_r = Am_to_Hz(6173)
nu_i = Am_to_Hz(7502)
nu_z = Am_to_Hz(8679)
nu_Y = Am_to_Hz(9711)
nu_PSg = Am_to_Hz(4811)
nu_PSr = Am_to_Hz(6156)
nu_PSi = Am_to_Hz(7504)
nu_ZTFg = Am_to_Hz(4723)
nu_ZTFr = Am_to_Hz(6340)
nu_ZTFi = Am_to_Hz(7886)
nu_B = Am_to_Hz(4450)


freq_dic = {"u ": nu_u, "g ": nu_g, "r ": nu_r, "i ": nu_i, "z ": nu_z, "Y ": nu_Y, "ZTFg": nu_ZTFg, "ZTFr": nu_ZTFr, "ZTFi": nu_ZTFi, "PSg":nu_PSg, "PSr":nu_PSr, "PSi":nu_PSi, "B":nu_B}

sncosmoband_convert_temporary = {"u ": -1, "g ": -2, "r ": -3, "i ": -4, "z ": -5, "Y ": -6,\
                                "ZTFg":-7, "ZTFr":-8, "ZTFi":-9, "PSg":-10, "PSr":-11, "PSi":-12, "B":-13}

sncosmoband_convert = {-1: "lsstu", -2: "lsstg", -3: "lsstr",\
                       -4: "lssti", -5: "lsstz", -6: "lssty",\
                       -7: "ztfg", -8: "ztfr", -9: "ztfi",\
                       -10: "ps1::g", -11: "ps1::r",\
                       -12: "ps1::i", -13: "standard::b"}

# ________________________________________________________________

if __name__ == "__main__":
    """
    Format, preprocess and feature extract a given class with both bazin and rainbow.
    Take 5 arguments :
        1 : str, Name of the class
        2 : int, Maximum number of objects to consider
        3 : int, Number of cores to use
        4 : str, elasticc or plasticc data
        5 : str, in plasticc case, what field to consider ddf or wfd ?
    """

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--target', required=True, help='Object type')
    arg_parser.add_argument('--nmax', default=100, type=int, help='Maximum number of objects to feature extract')
    arg_parser.add_argument('--cores', default=1, help='Number of cores to use')
    arg_parser.add_argument('--database', required=True, help='plasticc or elasticc')
    arg_parser.add_argument('--field', default='ddf', help='for plasticc : ddf, wfd or all')
    arg_parser.add_argument('--band_wavelength', default='effective', help='Use effective passband wavelength or integrate ?')
    
    args = arg_parser.parse_args()

    object_class = args.target
    n_max = args.nmax
    cores = args.cores
    database = args.database
    field = args.field
    band_wavelength = args.band_wavelength
    
    
    fex_functions = ['bazin', 'rainbow']
    fractions = ['']
    
    # PROCESS TO A COMPLETE FEATURE EXTRACTION :

    # Format original data
    format_target(object_class, n_max, database, field)
    print(f'{object_class} formatted')
  
    if database == 'YSE':
        fractions = [f"{field}_f{str(round(100*fraction))}" for fraction in kern.YSE_fractions]
        fields = []
        for fraction in fractions:
            fields.append(f"{fraction}_bazin")
            fields.append(f"{fraction}_rainbow")
        
    else:
        fields = [field]
        
    for fie in fields :
        
        if database != 'YSE':
            
            # Preprocess entire light curves
            preprocess_target(object_class, database, fie)
            print(f'{object_class} preprocessed for {fie}')

            # Remove bins of points and preprocess the training data
            train_test_bins_target(object_class, database, fie)

            bin_class = object_class + f'_train_w{kern.point_cut_window}'
            print(f'{bin_class} for {fie} created')
            preprocess_target(bin_class, database, fie)
            print(f'{bin_class} for {fie} preprocessed')
        
            # Remove points after rising part and preprocess training data
            train_test_rising_target(object_class, database, fie)
            rising_class = object_class + '_train_rising'
            print(f'{rising_class} created')
            preprocess_target(rising_class, database, fie)
            print(f'{rising_class} preprocessed')
            
        else:
            for version in kern.seeds:
                version_field = fie + f"_v{version}"
                # Remove bins of points and preprocess the training data
                train_test_bins_target(object_class, database, version_field)
                bin_class = object_class + f'_train_w{kern.point_cut_window}'
                preprocess_target(bin_class, database, version_field)
            print(f'{bin_class} for {fie} preprocessed')

    for fex_function in fex_functions:
        for fraction in fractions:
            if database == 'YSE':
                for version in kern.seeds:
                    fie = fraction + '_' + fex_function + "_v" + str(version)

                    feature_extract_target(bin_class, fex_function, cores, database, fie, band_wavelength)
                print(f'{object_class} {fie} feature extracted with {fex_function}')
                
            else:
                fie = field
                # Feature extract the 3 preprocessed database
                feature_extract_target(object_class, fex_function, cores, database, fie, band_wavelength)
                print(f'{object_class} feature extracted with {fex_function}')
                
                feature_extract_target(rising_class, fex_function, cores, database, fie, band_wavelength)
                print(f'{rising_class} feature extracted with {fex_function}')
                
                feature_extract_target(bin_class, fex_function, cores, database, fie, band_wavelength)
                print(f'{object_class} {fie} feature extracted with {fex_function}')

    print(f'{object_class} COMPLETED FEATURE EXTRACTED')

    