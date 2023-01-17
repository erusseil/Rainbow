from typing import Dict, List, Union
from pathlib import Path
from argparse import ArgumentParser
import glob
from itertools import chain
from astropy.table import vstack, Table
import astropy.constants as c
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

def generate_plasticc_YSE(object_class, field):
    
    meta = pd.read_csv('data_YSE/YSE_plasticc_meta.csv', index_col='object_id')
    meta = meta[meta['true_target'] == object_class]
    
    file_name = 'data_YSE/YSE_plasticc_data.csv'
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


def preprocess_plasticc(object_class, max_n, field):    
    lcs = []
    for idx, lc in zip(range(max_n), generate_plasticc_lcs(object_class, field)):
        if idx % 100 == 0:
            print(idx)
        lc = normalize(lc)
        lcs.append(lc)
        
    # Save preprocessed data as pkl for later use
    if not os.path.exists("data_plasticc"):
        os.mkdir("data_plasticc")

    if not os.path.exists("data_plasticc/preprocessed"):
        os.mkdir("data_plasticc/preprocessed")
        
    file = f"data_plasticc/preprocessed/{kern.PLASTICC_TARGET_INV.get(object_class)}_{field}.pkl"

    with open(file, "wb") as handle:
        pickle.dump(lcs, handle)

    return lcs


def preprocess_YSE(object_class, max_n, field):    
    lcs = []
    for idx, lc in zip(range(max_n), generate_plasticc_YSE(object_class, field)):
        if idx % 100 == 0:
            print(idx)

        lc = normalize(lc)
        lcs.append(lc)
        
    # Save preprocessed data as pkl for later use
    if not os.path.exists("data_plasticc"):
        os.mkdir("data_plasticc")

    if not os.path.exists("data_plasticc/preprocessed"):
        os.mkdir("data_plasticc/preprocessed")
        
    file = f"data_plasticc/preprocessed/{kern.PLASTICC_TARGET_INV.get(object_class)}_{field}.pkl"

    with open(file, "wb") as handle:
        pickle.dump(lcs, handle)

    return lcs


def normalize(lc):

    gband = lc[lc["BAND"] == "g "]
    maxi = gband["FLUXCAL"].max()

    lc["max_flux"] = maxi
    t_maxi = gband["MJD"][gband["FLUXCAL"].argmax()]
    lc["max_flux_time"] = t_maxi
    lc["FLUXCAL"] = lc["FLUXCAL"] / maxi
    lc["FLUXCALERR"] = lc["FLUXCALERR"] / maxi
    lc["MJD"] = lc["MJD"] - t_maxi

    return lc

def preprocess(object_class, max_n, rising=False):

    """
    Preprocess raw ELASTiCC training sample. Apply cuts on
    passbands and number of points. Apply transformations to light curves.

    Parameters
    ----------
    object_class: str
        Name of the ELASTiCC class to compute
    max_n: int
        Maximum number of object to compute
    rising: bool
        If True keeps only the rising part of the lightcurve
        Default is False
        
    Returns
    -------
    lcs : list
        list of astropy tables. Each table is one light curve.
    """

    heads = sorted(glob.glob(f"{kern.ELASTiCC_path}{object_class}/*_HEAD.FITS.gz"))
    phots = sorted(glob.glob(f"{kern.ELASTiCC_path}{object_class}/*_PHOT.FITS.gz"))
    assert len(heads) != 0, "no *_HEAD_FITS.gz are found"
    assert len(heads) == len(phots), "there are different number of HEAD and PHOT files"

    min_det_per_band = kern.min_det_per_band

    lcs = []
    
    for head, phot in zip(heads, phots):
        if len(lcs)<max_n:
            lcs += list(parse_fits_snana(head, phot, min_det_per_band=min_det_per_band, rising=rising))

    n_objects = int(np.where(max_n<=len(lcs), max_n, len(lcs)))
    lcs = random.sample(lcs, n_objects)
    
    for idx, lc in enumerate(lcs):
        lc = normalize(lc)

    # Save preprocessed data as pkl for later use
    if not os.path.exists("data"):
        os.mkdir("data")

    if not os.path.exists("data/preprocessed"):
        os.mkdir("data/preprocessed")
        
    if rising:
        rising_str = '_rising'
    else:
        rising_str = ''
        
    file = f"data/preprocessed/{object_class}{rising_str}.pkl"

    with open(file, "wb") as handle:
        pickle.dump(lcs, handle)

    return lcs


def parse_fits_snana(
    head: Union[str, Path], phot: Union[str, Path], *, min_det_per_band: Dict[str, int], rising
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
        
        if rising:
            # Remove points after peak
            lc = lc[lc["MJD"] <= lc.meta["SIM_PEAKMJD"]]

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
    Compute flux using Bazin.

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


def plank(nu, T):
    """
    Compute spectral radiance from temperature and frequency.

    Parameters
    ----------
    nu: array
        Frequency for which to compute spectral radiance.
    T: array
        Temperature values at different times.

    Returns:
    --------
        Computed spectral radiance.
    """

    return (2 * c.h.value / c.c.value**2) * nu**3 / np.expm1(c.h.value * nu / (c.k_B.value * T))


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

    return np.pi / c.sigma_sb.value * Fbol * plank(nu, T) / T**4 * amplitude


def perform_fit_mbf(obj):
    """
    Find best fit parameters for MbF method using iminuit.
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


def extract_mbf(lcs, object_class, split, database):
    """
    Apply perform_fit_mbf to each object.
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
        extraction = perform_fit_mbf(obj)
        all_param.append(list(extraction[0].to_dict().values()) + extraction[1])

    features = pd.DataFrame(
        columns=Fnu.__code__.co_varnames[1:8] + ("error", "max_flux", "max_time", "true_peak", "nb_points", "object_id"),
        data=all_param,
    )


    if database == 'elasticc':
        features.to_parquet(f"data/features/mbf/{object_class}/features_{object_class}_{split}.parquet")
    elif (database == 'plasticc') or (database == 'YSE'):
        features.to_parquet(f"data_plasticc/features/mbf/{object_class}/features_{object_class}_{split}.parquet")


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

    if database == 'elasticc':
        features.to_parquet(
            f"data/features/bazin/{object_class}/features_{object_class}_{split}.parquet"
        )
    elif (database == 'plasticc') or (database == 'YSE'):
        features.to_parquet(
            f"data_plasticc/features/bazin/{object_class}/features_{object_class}_{split}.parquet"
        )

        
def train_test_cutting_generator(prepro):
    for obj in prepro:
        invalid_band = False
        all_bands_train, all_bands_test = [], []
        
        for band in ['g ', 'r ', 'i ']:

            if invalid_band:
                continue

            sub_obj = obj[obj['BAND'] == band]

            # We add + windows to ensure that the first window is not selectable.
            # The last window will not exist either because of np.arange combined
            # with the fact that we require mjd to be stricly inferior to upper limitband
            
            bins = np.arange(sub_obj['MJD'].min() + kern.point_cut_window, sub_obj['MJD'].max(), kern.point_cut_window)

            valid = []
            for idx in range(len(bins)-1):
                if ((sub_obj['MJD'] >= bins[idx]) &\
                    (sub_obj['MJD'] < bins[idx + 1]) &\
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

            if (~mask).sum() < kern.min_det_per_band.get(band):
                invalid_band = True
                continue

            all_bands_train.append(sub_obj[~mask])
            all_bands_test.append(sub_obj[mask])

        if invalid_band:
            continue
            
        yield vstack(all_bands_train), vstack(all_bands_test)
        
        
def train_test_rising_cutting_generator(prepro):
    for obj in prepro:
        invalid_band = False
        all_bands_train, all_bands_test = [], []
        
        mini = (obj[obj['detected_bool'] == 1]['MJD']).min()
        maxi = obj.meta['true_peakmjd'] - obj['max_flux_time'][0]
        threshold = random.uniform(mini, maxi)
            
        for band in ['g ', 'r ', 'i ']:

            if invalid_band:
                continue
                
            sub_obj = obj[obj['BAND'] == band]
            mask = sub_obj['MJD'] > threshold

            if ((~mask).sum() < kern.min_det_per_band.get(band)) |\
               ((mask).sum() < 1):
                invalid_band = True
                continue

            all_bands_train.append(sub_obj[~mask])
            all_bands_test.append(sub_obj[mask])

        if invalid_band:
            continue
            
        yield vstack(all_bands_train), vstack(all_bands_test)

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

freq_dic = {"u ": nu_u, "g ": nu_g, "r ": nu_r, "i ": nu_i, "z ": nu_z, "Y ": nu_Y, "PSg": nu_PSg, "PSr": nu_PSr, "PSi": nu_PSi}

# ________________________________________________________________

if __name__ == "__main__":
    """
    Preprocess and feature extract a given class
    of ELASTiCC.
    Take 4 arguments :
        1 : str, Name of the class
        2 : int, Maximum number of objects to consider
        3 : str, Which method to use for feature extraction ('bazin' or 'mbf')
        4 : int, Number of cores to use
        5 : str, has the data been preprocessed already ('True' or 'False')
        6 : str, Do you want to keep only the rising part of the lc ('True' or 'False')
        7 : str, elasticc or plasticc data
    """

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--target', required=True, help='Object type')
    arg_parser.add_argument('--nmax', default=100, type=int, help='Maximum number of objects to feature extract')
    arg_parser.add_argument('--function', required=True, help='mbf or bazin')
    arg_parser.add_argument('--cores', default=1, help='Number of cores to use')
    arg_parser.add_argument('--prepro', default='False', help='Has the data been preprocessed already ?')
    arg_parser.add_argument('--rising', default='False', help='For elasticc data : keep only rising part')
    arg_parser.add_argument('--database', required=True, help='plasticc or elasticc')
    arg_parser.add_argument('--field', default='ddf', help='for plasticc : ddf, wfd or all')
    
    args = arg_parser.parse_args()

    object_class = args.target
    n_max = args.nmax
    fex_function = args.function
    cores = args.cores
    already_prepro = args.prepro
    rising = args.rising
    database = args.database
    field = args.field

    if (fex_function != 'mbf') & (fex_function != 'bazin'):
        raise ValueError('Function must be mbf or bazin')

    if (database != 'elasticc') & (database != 'plasticc') & (database != 'YSE'):
        raise ValueError('Function must be elasticc or plasticc or YSE')

    if (field != 'ddf') & (field != 'wfd') & (field != 'all'):
        raise ValueError('Field must be ddf, wfd or all')

    if rising == 'True':
        rising = True
        rising_str = '_rising'

    elif rising == 'False':
        rising = False
        rising_str = ''

    start_time = time.time()
    
    if database == 'elasticc':
        if not os.path.exists(f"data/features/{fex_function}/"):
            os.mkdir(f"data/features/{fex_function}/")

        if not os.path.exists(f"data/features/{fex_function}/{object_class}{rising_str}"):
            os.mkdir(f"data/features/{fex_function}/{object_class}{rising_str}")

        if already_prepro == "False":
            preprocess(object_class, n_max, rising)
            
        subprocess.call(
            shlex.split(f"sh feature_extraction.sh {cores} {object_class}{rising_str} {fex_function} {database}")
        )

        temp_path = f"data/features/{fex_function}/{object_class}{rising_str}/"

        
    if (database == 'plasticc') or (database == 'YSE'):
        
        if not os.path.exists(f"data_plasticc/features/"):
            os.mkdir(f"data_plasticc/features/")
            
        if not os.path.exists(f"data_plasticc/features/{fex_function}/"):
            os.mkdir(f"data_plasticc/features/{fex_function}/")

        if not os.path.exists(f"data_plasticc/features/{fex_function}/{object_class}_{field}"):
            os.mkdir(f"data_plasticc/features/{fex_function}/{object_class}_{field}")

        if already_prepro == "False":
            
            if (database == 'plasticc'):
                preprocess_plasticc(kern.PLASTICC_TARGET.get(object_class), n_max, field)
            elif (database == 'YSE'):
                preprocess_YSE(kern.PLASTICC_TARGET.get(object_class), n_max, field)
            
        subprocess.call(
            shlex.split(f"sh feature_extraction.sh {cores} {object_class}_{field} {fex_function} {database}")
        )

        temp_path = f"data_plasticc/features/{fex_function}/{object_class}_{field}/"
        
    

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
        
        features.to_parquet(f"data/features/{fex_function}/{object_class}{rising_str}_features.parquet")

        with open(f"data/features/{fex_function}/{object_class}{rising_str}_info.txt", "w") as f:
            f.write(f"Feature extraction took {total_time} sec, over {cores} cores")
            f.write("\n")
            f.write(
                f'The features table take {os.path.getsize(f"data/features/{fex_function}/{object_class}{rising_str}_features.parquet")} bytes of space'
            )
            f.write("\n")
            f.write((features.head()).to_string())

        shutil.rmtree(temp_path)

        print(f"{object_class}{rising_str} features have been computed succesfully")


    elif (database == 'plasticc') or (database == 'YSE'):
        
        features.to_parquet(f"data_plasticc/features/{fex_function}/{object_class}_{field}_features.parquet")

        with open(f"data_plasticc/features/{fex_function}/{object_class}_{field}_info.txt", "w") as f:
            f.write(f"Feature extraction took {total_time} sec, over {cores} cores")
            f.write("\n")
            f.write(
                f'The features table take {os.path.getsize(f"data_plasticc/features/{fex_function}/{object_class}_{field}_features.parquet")} bytes of space'
            )
            f.write("\n")
            f.write((features.head()).to_string())

        shutil.rmtree(temp_path)

        print(f"{object_class}_{field} features have been computed succesfully")