from typing import Dict, List, Union
from pathlib import Path
import glob
from itertools import chain
from astropy.table import Table
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


def preprocess(object_class, max_n):

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

    heads = sorted(glob.glob(f"{kern.ELASTiCC_path}{object_class}/*_HEAD.FITS.gz"))
    phots = sorted(glob.glob(f"{kern.ELASTiCC_path}{object_class}/*_PHOT.FITS.gz"))
    assert len(heads) != 0, "no *_HEAD_FITS.gz are found"
    assert len(heads) == len(phots), "there are different number of HEAD and PHOT files"

    min_det_per_band = {b"g ": 4, b"r ": 4, b"i ": 4}

    lcs = []
    
    for head, phot in zip(heads, phots):
        if len(lcs)<max_n:
            lcs += list(parse_fits_snana(head, phot, min_det_per_band=min_det_per_band))

    n_objects = int(np.where(max_n<=len(lcs), max_n, len(lcs)))
    lcs = random.sample(lcs, n_objects)

    for idx, lc in enumerate(lcs):

        gband = lc[lc["BAND"] == "g "]
        maxi = gband["FLUXCAL"].max()

        lc["max_flux"] = maxi
        t_maxi = gband["MJD"][gband["FLUXCAL"].argmax()]
        lc["max_flux_time"] = t_maxi
        lc["FLUXCAL"] = lc["FLUXCAL"] / maxi
        lc["FLUXCALERR"] = lc["FLUXCALERR"] / maxi
        lc["MJD"] = lc["MJD"] - t_maxi

    # Save preprocessed data as pkl for later use
    if not os.path.exists("data"):
        os.mkdir("data")

    if not os.path.exists("data/preprocessed"):
        os.mkdir("data/preprocessed")

    file = f"data/preprocessed/{object_class}.pkl"

    with open(file, "wb") as handle:
        pickle.dump(lcs, handle)

    return lcs


def parse_fits_snana(
    head: Union[str, Path], phot: Union[str, Path], *, min_det_per_band: Dict[str, int]
) -> List[Table]:

    """
    Reads ELASTiCC training data. Returns it after applying cuts
    on passbands, number of points and saturation flux.

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
        
        # We requiere to have observation before and after peak
        if (lc.meta['SIM_PEAKMJD'] < detections['MJD'][0]
                 or lc.meta['SIM_PEAKMJD'] > detections['MJD'][-1]):
            continue

        # Not enough number of detections in some passband
        for band, min_det in min_det_per_band.items():
            if det_per_band.get(band, 0) < min_det:
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


def Tsig(t, Tmin, dT, ksig, t0, tT):
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
    tT: float
        Difference between t0 and real time of maximum flux

    Returns:
    --------
        Computed temperature at each time t.
    """

    return Tmin + dT / (1 + np.exp((t - (t0 + tT)) / ksig))


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
def Fnu(x, a, t0, tT, tfall, trise, Tmin, dT, ksig):
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
    tT: float
        Difference between t0 and real time of maximum flux
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

    t, nu = x.T
    T = Tsig(t, Tmin, dT, ksig, t0, tT)
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
        "tT": 0,
        "tfall": 30,
        "trise": -5,
        "Tmin": 4000,
        "dT": 7000,
        "ksig": 4,
    }

    least_squares = LeastSquares(
        np.array([global_mjd, global_nu]).T, global_flux, global_fluxerr, Fnu
    )
    fit = Minuit(
        least_squares,
        limit_Tmin=(1000, 100000),
        limit_dT=(0, 100000),
        limit_t0=(-200, 200),
        limit_tT=(-100, 100),
        limit_a=(0.01, 40),
        limit_ksig=(0.01, 50),
        limit_trise=(-30, 0),
        limit_tfall=(0, 500),
        **parameters_dict,
    )
    fit.migrad()

    max_flux = obj["max_flux"][0]
    max_time = obj["max_flux_time"][0]
    fit_error = fit.fval
    peak = obj.meta["PEAKMJD"]
    nb_points = len(global_mjd)

    additionnal = [fit_error, max_flux, max_time, peak, nb_points]

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
            limit_a=(0.01, 40),
            limit_t0=(-200, 200),
            limit_trise=(-30, 0),
            limit_tfall=(0, 500),
            **parameters_dict,
        )

        fit.migrad()

        max_flux = lc["max_flux"][0]
        max_time = lc["max_flux_time"][0]
        fit_error = fit.fval
        peak = lc.meta["PEAKMJD"]
        nb_points = len(global_mjd)

        additionnal = [fit_error, nb_points, max_flux, max_time, peak]

        all_parameters.append([fit.values, additionnal])

    return all_parameters


def extract_mbf(lcs, object_class, split):
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
        all_param.append(extraction[0].values() + extraction[1])

    features = pd.DataFrame(
        columns=Fnu.__code__.co_varnames[1:9] + ("error", "max_flux", "max_time", "true_peak", "nb_points"),
        data=all_param,
    )

    features.to_parquet(f"data/features/mbf/{object_class}/features_{object_class}_{split}.parquet")


def extract_bazin(lcs, object_class, split):
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
                param[idx_band][0].values() + [param[idx_band][1][0]] + [param[idx_band][1][1]]
            )

        flat_obj_param = [x for xs in obj_param for x in xs] + param[0][1][
            2:
        ]
        all_param.append(flat_obj_param)

    flat_name_param = [x for xs in name_param for x in xs] + [
        "max_flux",
        "max_time",
        "true_peak",
    ]

    features = pd.DataFrame(columns=flat_name_param, data=all_param)

    features.to_parquet(
        f"data/features/bazin/{object_class}/features_{object_class}_{split}.parquet"
    )


# __________________________USEFUL VALUES________________________

SATURATION_FLUX = 1e5

# Source of values used for the filters : http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=LSST&asttype=
nu_u = Am_to_Hz(3751)
nu_g = Am_to_Hz(4742)
nu_r = Am_to_Hz(6173)
nu_i = Am_to_Hz(7502)
nu_z = Am_to_Hz(8679)
nu_Y = Am_to_Hz(9711)
freq_dic = {"u ": nu_u, "g ": nu_g, "r ": nu_r, "i ": nu_i, "z ": nu_z, "Y ": nu_Y}

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
    """

    object_class = str(sys.argv[1])
    n_max = int(sys.argv[2])
    fex_function = str(sys.argv[3])
    cores = int(sys.argv[4])
    already_prepro = str(sys.argv[5])

    if not os.path.exists(f"data/features/{fex_function}/"):
        os.mkdir(f"data/features/{fex_function}/")

    if not os.path.exists(f"data/features/{fex_function}/{object_class}"):
        os.mkdir(f"data/features/{fex_function}/{object_class}")

    if already_prepro == "False":
        preprocess(object_class, n_max)

    start_time = time.time()

    subprocess.call(
        shlex.split(f"sh feature_extraction.sh {cores} {object_class} {fex_function}")
    )

    temp_path = f"data/features/{fex_function}/{object_class}/"
    n_computed_files = len(
        [
            entry
            for entry in os.listdir(temp_path)
            if os.path.isfile(os.path.join(temp_path, entry))
        ]
    )

    while (n_computed_files != cores):
        
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

    features.to_parquet(f"data/features/{fex_function}/{object_class}_features.parquet")

    with open(f"data/features/{fex_function}/{object_class}_info.txt", "w") as f:
        f.write(f"Feature extraction took {total_time} sec, over {cores} cores")
        f.write("\n")
        f.write(
            f'The features table take {os.path.getsize(f"data/features/{fex_function}/{object_class}_features.parquet")} bytes of space'
        )
        f.write("\n")
        f.write((features.head()).to_string())

    shutil.rmtree(temp_path)

    print(f"{object_class} features have been computed succesfully")
