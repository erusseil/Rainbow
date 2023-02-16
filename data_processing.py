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

def generate_plasticc_YSE(object_class):
    """
    YSE lightcurve generator to the correct format
    
    Parameters
    ----------
    object_class: str
        Name of the YSE class to compute
        
    Yields
    ------
    table : astropytable
        Formatted lightcurve table. Ensures that the minimum
        number of points from kernel.py is respected
    """
    
    
    meta = pd.read_csv(f'{kern.YSE_path}/YSE_plasticc_meta.csv', index_col='object_id')
    meta = meta[meta['true_target'] == object_class]
    
    file_name = f'{kern.YSE_path}/YSE_plasticc_data.csv'
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

        detections = table[table['detected_bool'] == 1]
        det_per_band = dict(zip(*np.unique(detections["BAND"], return_counts=True)))
        
        if any(det_per_band.get(band, 0) < min_det for band, min_det in kern.min_det_per_band_YSE.items()):
            continue

        if (table.meta['SIM_PEAKMJD'] < detections['MJD'][0]
             or table.meta['SIM_PEAKMJD'] > detections['MJD'][-1]):
            continue

        yield table
        

def plasticc_reddening_correction(lc):
    """
    Correction the flux/fluxerr from the Milky Way 
    reddening using mwebv from plasticc
    
    Parameters
    ----------
    lc: astropy.Table
        Light curve
    """
    
    mwebv = lc.meta['mwebv']
    
    for band, red in enumerate([red_u, red_g, red_r, red_i, red_z, red_Y]):

        A = red * mwebv
        correction = 10**(-A/2.5)
        
        lc['FLUXCAL'][lc['passband']==band] = lc['FLUXCAL'][lc['passband']==band] * correction
        lc['FLUXCALERR'][lc['passband']==band] = lc['FLUXCALERR'][lc['passband']==band] * correction


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
        plasticc_reddening_correction(lc)
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

    lcs = []
    for idx, lc in zip(range(max_n), generate_plasticc_YSE(object_class)):
        lcs.append(lc)

    # Save preprocessed data as pkl for later use
    if not os.path.exists("data_plasticc"):
        os.mkdir("data_plasticc")

    if not os.path.exists("data_plasticc/formatted"):
        os.mkdir("data_plasticc/formatted")

    file = f"data_YSE/formatted/{kern.PLASTICC_TARGET_INV.get(object_class)}_{field}.pkl"

    with open(file, "wb") as handle:
        pickle.dump(lcs, handle)


def format_elasticc(object_class, max_n):

    """
    Preprocess raw ELASTiCC training sample. Apply cuts on
    passbands and number of points. Apply transformations to light curves.

    Parameters
    ----------
    object_class: str
        Name of the ELASTiCC class to compute
    max_n: int
        Maximum number of object to compute
        
    Returns
    -------
    lcs : list
        list of astropy tables. Each table is one light curve.
    """
    
    if kern.global_seed != None:
        random.seed(kern.global_seed)

    heads = sorted(glob.glob(f"{kern.ELASTiCC_path}{object_class}/*_HEAD.FITS.gz"))
    phots = sorted(glob.glob(f"{kern.ELASTiCC_path}{object_class}/*_PHOT.FITS.gz"))
    assert len(heads) != 0, "no *_HEAD_FITS.gz are found"
    assert len(heads) == len(phots), "there are different number of HEAD and PHOT files"

    min_det_per_band = kern.min_det_per_band

    lcs = []
    
    for head, phot in zip(heads, phots):
        if len(lcs)<max_n:
            lcs += list(parse_fits_snana(head, phot, min_det_per_band=min_det_per_band))

    n_objects = int(np.where(max_n<=len(lcs), max_n, len(lcs)))
    lcs = random.sample(lcs, n_objects, )

    # Save preprocessed data as pkl for later use
    if not os.path.exists("data_elasticc"):
        os.mkdir("data_elasticc")

    if not os.path.exists("data_elasticc/formatted"):
        os.mkdir("data_elasticc/formatted")
        
    file = f"data_elasticc/formatted/{object_class}.pkl"

    with open(file, "wb") as handle:
        pickle.dump(lcs, handle)


def preprocess(dataset, object_class, field=''):

    path_to_use = kern.main_path + f'/data_{dataset}/formatted/{object_class}_{field}.pkl'
                          
    with open(path_to_use, "rb") as handle:
        lcs = pickle.load(handle)

    new_lcs = [normalize(lc) for lc in lcs]
                  
    # Save preprocessed data as pkl for later use
    if not os.path.exists(f"data_{dataset}"):
        os.mkdir(f"data_{dataset}")

    if not os.path.exists(f"data_{dataset}/preprocessed"):
        os.mkdir(f"data_{dataset}/preprocessed")
        
    if dataset == 'elasticc':
        file = f"data_{dataset}/preprocessed/{object_class}"
        
    else:
        file = f"data_{dataset}/preprocessed/{object_class}_{field}"

    with open(f'{file}.pkl' , "wb") as handle:
        pickle.dump(lcs, handle)

    return lcs


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
        - Fluxes divided by max g band flux
        - Time 0 shifted by time of max g band flux
        - Added columns before normalization : max_flux, max_flux_time
    """

    gband = lc[lc["BAND"] == "g "]
    maxi = gband["FLUXCAL"].max()

    lc["max_flux"] = maxi
    t_maxi = gband["MJD"][gband["FLUXCAL"].argmax()]
    lc["max_flux_time"] = t_maxi
    lc["FLUXCAL"] = lc["FLUXCAL"] / maxi
    lc["FLUXCALERR"] = lc["FLUXCALERR"] / maxi
    lc["MJD"] = lc["MJD"] - t_maxi

    return lc

def parse_fits_snana(
    head: Union[str, Path], phot: Union[str, Path], *, min_det_per_band: Dict[str, int]
) -> List[Table]:

    """
    Reads ELASTiCC training data. Returns it after applying cuts
    on passbands, number of points and saturation flux. 
    Also requires at least one point before and after peak.

    Parameters
    ----------
    head: Union[str, Path]
        Paths to ELASTiCC head files

    phot: Union[str, Path]
        Paths to ELASTiCC phot files

    min_det_per_band: dict
        Dict of filter names along with the minimum
        number of point requiered for each filter

    Returns
    -------
    lcs : list
        list of astropy tables. Each table is one light curve.
    """

    i = head.find("_HEAD.FITS.gz")
    assert head[:i] == phot[:i], f"HEAD and PHOT files name mismatch: {head}, {phot}"

    bands = np.array(list(min_det_per_band), dtype=bytes)

    lcs = []
    for lc in sncosmo.read_snana_fits(head, phot):

        # Keep passbands we need
        lc = lc[np.isin(lc["BAND"], bands)]

        # Remove saturated observations
        lc = lc[lc["FLUXCAL"] <= SATURATION_FLUX]

        # we use this variable for cuts only, while putting the full light curve into dataset
        detections = lc[(lc["PHOTFLAG"] != 0)]
        det_per_band = dict(zip(*np.unique(detections["BAND"], return_counts=True)))

        # Not enough number of detections in some passband
        not_enough = False
        for band, min_det in min_det_per_band.items():
            if det_per_band.get(band, 0) < min_det:
                not_enough = True
                
        if not_enough:
            continue
                
        # We requiere to have observation before and after peak
        if (lc.meta['SIM_PEAKMJD'] < detections['MJD'][0]
                 or lc.meta['SIM_PEAKMJD'] > detections['MJD'][-1]):
            continue

        lcs.append(lc)

    return lcs


def Fbaz(t, a, t0, tfall, trise):
    """
    Compute flux using Bazin with
    baseline fixed to 0.

    Parameters
    ----------
    t: array
        Time value of data points.
    a: float
        Bazin amplitude.
    t0: float
        Time value related to time of maximum flux.
    tfall: float
        Value related to length of the falling part of the slope.
    trise: float
        Value related to length of the rising part slope.

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
    nu = np.asarray(nu)
    if np.issubdtype(nu.dtype, np.number):
        return planck_nu(nu, T)
    elif np.issubdtype(nu.dtype, np.string):
        return planck_passband(nu, T)
    raise ValueError(f'Invalid type for nu: {nu.dtype}')


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
    obj["NU"] = np.vectorize(freq_dic.get)(list(obj["BAND"]))

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
    fit.limits['t0'] = (-200, 100)
    fit.limits['a'] = (0.1, 100)
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


def perform_fit_bazin(obj):
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

    for band in ["g ", "r ", "i "]:

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
        
        fit.limits['t0'] = (-200, 100)
        fit.limits['a'] = (0.1, 100)
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


def extract_rainbow(lcs, object_class, split, database):
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
    """

    all_param = []
    for obj in lcs:
        extraction = perform_fit_rainbow(obj)
        all_param.append(list(extraction[0].to_dict().values()) + extraction[1])

    features = pd.DataFrame(
        columns=Fnu.__code__.co_varnames[1:8] + ("error", "max_flux", "max_time", "true_peak", "nb_points", "object_id"),
        data=all_param,
    )

    features.to_parquet(f"data_{database}/features/rainbow/{object_class}/features_{object_class}_{split}.parquet")


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

    all_param = []
    name_param = []

    for idx, obj in enumerate(lcs):

        param = perform_fit_bazin(obj)

        obj_param = []
        for idx_band, band in enumerate(["g", "r", "i"]):

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

        
def train_test_cutting_generator(prepro, perband=False, bands = ['g ', 'r ', 'i '], mdpb=kern.min_det_per_band):
    """
    Remove random windows of detected points.
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
    
    Yields
    ------
    train, test
        astropy table, astropy table
    """
    
    if kern.global_seed != None:
        random.seed(kern.global_seed)
    
    for obj in prepro:
        
        all_bands_train, all_bands_test = [], []
        
        if perband:
            
            invalid_band = False
            for band in bands:

                if invalid_band:
                    continue

                sub_obj = obj[obj['BAND'] == band]
                bins = np.arange(sub_obj['MJD'].min(), sub_obj['MJD'].max(), kern.point_cut_window)

                valid = []
                for idx in range(len(bins)-1):
                    if ((sub_obj['MJD'] >= bins[idx]) &\
                        (sub_obj['MJD'] <= bins[idx + 1]) &\
                        (sub_obj['detected_bool'] == 1)).sum()>0:

                        valid.append(idx)

                if len(valid)<=2:
                    invalid_band = True
                    continue

                chosen = random.sample(valid, 2)
                mask = (((sub_obj['MJD']>=bins[chosen[0]]) &\
                         (sub_obj['MJD']<bins[chosen[0] + 1])) |\
                        ((sub_obj['MJD']>=bins[chosen[1]]) &\
                         (sub_obj['MJD']<bins[chosen[1] + 1])))

                if (~mask).sum() < mdpb.get(band):
                    invalid_band = True
                    continue

                all_bands_train.append(sub_obj[~mask])
                all_bands_test.append(sub_obj[mask])

            if invalid_band:
                continue

            yield vstack(all_bands_train), vstack(all_bands_test)


        else:

            mask = [True] * len(obj)
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
            
            invalid_band = False
            for band in bands:
                pointperband = len(sub_obj[(~mask) & (sub_obj['BAND'] == band)])
                
                if pointperband < mdpb.get(band):
                    invalid_band = True
                    
            if invalid_band:
                continue
                    
            yield sub_obj[~mask], sub_obj[mask]
        
        
def train_test_rising_cutting_generator(prepro, bands = ['g ', 'r ', 'i '], mdpb=kern.min_det_per_band):
    """
    Choose a random point during rising phase, remove all points after this.
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
    
    Yields
    ------
    train, test
        astropy table, astropy table
    """
    if kern.global_seed != None:
        random.seed(kern.global_seed)
        
    for obj in prepro:
        invalid_band = False
        all_bands_train, all_bands_test = [], []
        
        mini = (obj[obj['detected_bool'] == 1]['MJD']).min()
        maxi = obj.meta['true_peakmjd']
        threshold = random.uniform(mini, maxi)
            
        for band in bands:

            if invalid_band:
                continue
                
            sub_obj = obj[obj['BAND'] == band]
            mask = sub_obj['MJD'] > threshold

            if ((~mask).sum() < mdpb.get(band)) |\
               ((mask).sum() < 1):
                invalid_band = True
                continue

            all_bands_train.append(sub_obj[~mask])
            all_bands_test.append(sub_obj[mask])

        if invalid_band:
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
        In the case of plasticc and YSE.
        'ddf' or 'wfd'
    """
    
    if (database != 'elasticc') & (database != 'plasticc') & (database != 'YSE'):
        raise ValueError('Function must be elasticc or plasticc or YSE')

    if (database == 'plasticc') & (field != 'ddf') & (field != 'wfd') & (field != 'all'):
        raise ValueError('Field must be ddf, wfd or all')
        
    if not os.path.exists(f"data_{database}/"):
            os.mkdir(f"data_{database}/")
    
    
def format_target(object_class, n_max, database, field=''):
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
        In the case of plasticc and YSE.
        'ddf' or 'wfd' (always 'wfd' for YSE)
    """
    
    check_inputs(database, field)

    if database == 'elasticc':
        format_elasticc(object_class, n_max)
            
    elif (database == 'plasticc') or (database == 'YSE'):
        
        if database == 'plasticc':
            format_plasticc(kern.PLASTICC_TARGET.get(object_class), n_max, field)

        if database == 'YSE':
            format_YSE(kern.PLASTICC_TARGET.get(object_class), n_max, field)
            

def preprocess_target(object_class, database, field=''):
    """
    Preprocess a given class from a formated pkl file.
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
        
    check_inputs(database, field)
    preprocess(database, object_class, field)
        

def feature_extract_target(object_class, fex_function, cores, database, field=''):
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
        In the case of plasticc and YSE.
        'ddf' or 'wfd' (always 'wfd' for YSE)
    """

    start_time = time.time()

    if not os.path.exists(f"data_{database}/features/"):
            os.mkdir(f"data_{database}/features/")
    
    if not os.path.exists(f"data_{database}/features/{fex_function}/"):
            os.mkdir(f"data_{database}/features/{fex_function}/")

    check_inputs(database, field)
    
    if (fex_function != 'rainbow') & (fex_function != 'bazin'):
        raise ValueError('Function must be rainbow or bazin')
        
            
    if database == 'elasticc':
        
        if not os.path.exists(f"data_elasticc/features/{fex_function}/{object_class}"):
            os.mkdir(f"data_elasticc/features/{fex_function}/{object_class}")
        
        subprocess.call(
            shlex.split(f"sh feature_extraction.sh {cores} {object_class} {fex_function} {database}")
        )

        temp_path = f"data_elasticc/features/{fex_function}/{object_class}/"
        
        
    if (database == 'plasticc') or (database == 'YSE'):

        if not os.path.exists(f"data_{database}/features/{fex_function}/{object_class}_{field}"):
            os.mkdir(f"data_{database}/features/{fex_function}/{object_class}_{field}")
            
        subprocess.call(
            shlex.split(f"sh feature_extraction.sh {cores} {object_class}_{field} {fex_function} {database}")
        )

        temp_path = f"data_{database}/features/{fex_function}/{object_class}_{field}/"
        
        
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
    
    if database == 'elasticc':
        
        features.to_parquet(f"data_{database}/features/{fex_function}/{object_class}_features.parquet")

        with open(f"data_{database}/features/{fex_function}/{object_class}_info.txt", "w") as f:
            f.write(f"Feature extraction took {total_time} sec, over {cores} cores")
            f.write("\n")
            f.write(
                f'The features table take {os.path.getsize(f"data_{database}/features/{fex_function}/{object_class}_features.parquet")} bytes of space'
            )
            f.write("\n")
            f.write((features.head()).to_string())

        shutil.rmtree(temp_path)

        print(f"{object_class} features have been computed succesfully")


    elif (database == 'plasticc') or (database == 'YSE'):
        
        features.to_parquet(f"data_{database}/features/{fex_function}/{object_class}_{field}_features.parquet")

        with open(f"data_{database}/features/{fex_function}/{object_class}_{field}_info.txt", "w") as f:
            f.write(f"Feature extraction took {total_time} sec, over {cores} cores")
            f.write("\n")
            f.write(
                f'The features table take {os.path.getsize(f"data_{database}/features/{fex_function}/{object_class}_{field}_features.parquet")} bytes of space'
            )
            f.write("\n")
            f.write((features.head()).to_string())

        shutil.rmtree(temp_path)

        print(f"{object_class}_{field} features have been computed succesfully")
        
        
def train_test_bins_target(object_class, database, field=''):
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
    
    bands = ['g ', 'r ', 'i ']
    mdpb = kern.min_det_per_band

    if database == 'YSE':
        mdpb = kern.min_det_per_band_YSE
        bands = ['g ', 'r ', 'PSg', 'PSr', 'PSi']

    path = f'data_{database}/formatted/'

    with open(f'{path}{object_class}_{field}.pkl', "rb") as handle:
        formatted = pickle.load(handle)
        
    generator = train_test_cutting_generator(formatted, bands=bands, mdpb=mdpb)
    train_test = list(generator)

    with open(f'{path}{object_class}_train_w{kern.point_cut_window}_{field}.pkl', "wb") as handle:
        pickle.dump([x[0] for x in train_test], handle)
        
    with open(f'{path}{object_class}_test_w{kern.point_cut_window}_{field}.pkl', "wb") as handle:
        pickle.dump([x[1] for x in train_test], handle)
        
    
def train_test_rising_target(object_class, database, field=''):
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

# F99 Reddening with Rv=3.1 as given in Table 6 of : https://iopscience.iop.org/article/10.1088/0004-637X/737/2/103/pdf
red_u = 4.145
red_g = 3.237
red_r = 2.273
red_i = 1.684
red_z = 1.323
red_Y = 1.088

freq_dic = {"u ": nu_u, "g ": nu_g, "r ": nu_r, "i ": nu_i, "z ": nu_z, "Y ": nu_Y, "PSg": nu_PSg, "PSr": nu_PSr, "PSi": nu_PSi}

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
    
    args = arg_parser.parse_args()

    object_class = args.target
    n_max = args.nmax
    cores = args.cores
    database = args.database
    field = args.field
    
    fex_functions = ['rainbow']
    if database != 'YSE':
        fex_functions += ['bazin']
    
    # PROCESS TO A COMPLETE FEATURE EXTRACTION :
    
    # Format original data
    format_target(object_class, n_max, database, field)
    print(f'{object_class} formatted')
    
    # Preprocess entire light curves
    preprocess_target(object_class, database, field)
    print(f'{object_class} preprocessed')
    
    # Remove bins of points and preprocess the training data
    train_test_bins_target(object_class, database, field)
    bin_class = object_class + f'_train_w{kern.point_cut_window}'
    print(f'{bin_class} created')
    preprocess_target(bin_class, database, field)
    print(f'{bin_class} preprocessed')

    # Remove points after rising part and preprocess training data
    train_test_rising_target(object_class, database, field)
    rising_class = object_class + '_train_rising'
    print(f'{rising_class} created')
    preprocess_target(rising_class, database, field)
    print(f'{rising_class} preprocessed')

    for fex_function in fex_functions:
        
        # Feature extract the 3 preprocessed database
        feature_extract_target(object_class, fex_function, cores, database, field)
        print(f'{object_class} feature extracted with {fex_function}')
        feature_extract_target(bin_class, fex_function, cores, database, field)
        print(f'{bin_class} feature extracted with {fex_function}')
        feature_extract_target(rising_class, fex_function, cores, database, field)
        print(f'{rising_class} feature extracted with {fex_function}')
    
    print(f'{object_class} COMPLETED FEATURE EXTRACTED')
    

    

    