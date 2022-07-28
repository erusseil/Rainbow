#!/usr/bin/env python3

from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import sncosmo
from astropy.table import Table
from iminuit import Minuit
from iminuit.cost import LeastSquares
import astropy
import astropy.constants as c
import matplotlib.pyplot as plt


def Am_to_Hz(wl):
    return c.c.value/(wl*1e-10)

SATURATION_FLUX = 1e5

nu_u = Am_to_Hz(3751)
nu_g = Am_to_Hz(4742)
nu_r = Am_to_Hz(6173)
nu_i = Am_to_Hz(7502)
nu_z = Am_to_Hz(8679)
nu_Y = Am_to_Hz(9711)
freq_dic = {'u ':nu_u, 'g ':nu_g, 'r ':nu_r, 'i ':nu_i, 'z ':nu_z, 'Y ':nu_Y}


def parse_fits_snana(head: Union[str, Path], phot: Union[str, Path],
                     *, min_det_per_band: Dict[str, int], min_total_det: int,
                     has_peak: bool) -> List[Table]:
    
    i = head.find('_HEAD.FITS.gz')
    assert head[:i] == phot[:i], f'HEAD and PHOT files name mismatch: {head}, {phot}'

    bands = np.array(list(min_det_per_band), dtype=bytes)

    lcs = []
    for lc in sncosmo.read_snana_fits(head, phot):
        # Keep passbands we need
        lc = lc[np.isin(lc['BAND'], bands)]
        # Remove saturated observations
        lc = lc[lc['FLUXCAL'] <= SATURATION_FLUX]
        # we use this variable for cuts only, while putting the full light curve into dataset
        detections = lc[(lc['PHOTFLAG'] != 0)]
        det_per_band = dict(zip(*np.unique(detections['BAND'], return_counts=True)))
        # Not enough number of detections in all passbands
        if sum(det_per_band.values()) < min_total_det:
            continue
        # We asked to have observations before and after peak
        if (has_peak
            and (lc.meta['SIM_PEAKMJD'] < detections['MJD'][0]
                 or lc.meta['SIM_PEAKMJD'] > detections['MJD'][-1])):
            continue
        # Not enough number of detections in some passband
        for band, min_det in min_det_per_band.items():
            if det_per_band.get(band, 0) < min_det:
                continue
        lcs.append(lc)

    return lcs

# Bolometric flux computed using bazin
def Fbaz(t, a, t0, tfall, trise):
    return a * np.exp(-(t - t0) / tfall) / (1 + np.exp((t - t0) / trise))

# With flux baseline
def FbazB(t, a, b, t0, tfall, trise):
    return Fbaz(t, a, t0, tfall, trise) + b

# Temperature computed using sigmoid
def Tsig(t, Tmin, dT, ksig, t0, tT):
    return Tmin + dT/(1+np.exp((t-(t0+tT))/ksig))

def plank(nu, T):
    return (2*c.h.value/c.c.value**2)*nu**3/np.expm1(c.h.value*nu/(c.k_B.value*T))

# Flux of lightcurves at any time at any frequency
def Fnu(x, a, t0, tT, tfall, trise, Tmin, dT, ksig):
    t, nu = x.T
    T = Tsig(t, Tmin, dT, ksig, t0, tT)
    Fbol = Fbaz(t, a, t0, tfall, trise)
    amplitude = 1e15
    
    return np.pi/c.sigma_sb.value * Fbol * plank(nu, T)/T**4 * amplitude

def Am_to_Hz(wl):
    return c.c.value/(wl*1e-10)


def perform_fit(obj):
    
    obj['NU'] = np.vectorize(freq_dic.get)(list(obj['BAND']))
    
    global_flux = obj['FLUXCAL']
    global_fluxerr = obj['FLUXCALERR']
    global_nu = obj['NU']
    global_mjd = obj['MJD']

    parameters_dict = {"a": global_flux.max(), "t0": global_mjd[np.argmax(global_flux)], "tT": 0,\
                   "tfall": 30, "trise":-5, "Tmin":4000, "dT":7000, "ksig":4}
    
    least_squares = LeastSquares(np.array([global_mjd, global_nu]).T, global_flux, global_fluxerr, Fnu)
    fit = Minuit(least_squares,
             limit_Tmin=(1000, 100000),
             limit_dT=(0, 100000),
             limit_t0=(-200, 200),
             limit_tT=(-100, 100),
             limit_a=(0.01, 40), 
             limit_ksig=(0.01, 50),
             limit_trise=(-30, 0),
             limit_tfall=(0,500),
             **parameters_dict)
    fit.migrad()
    
    
    max_flux = obj['max_flux'][0]
    max_time = obj['max_flux_time'][0]
    fit_error = fit.fval
    peak = obj.meta['PEAKMJD']
    
    additionnal = [fit_error, max_flux, max_time, peak]
    
    return fit.values, additionnal

def perform_bazin_fit(all_obj):
    
    all_parameters = []
    
    for band in ['g ','r ','i ']:
    
        obj = all_obj[all_obj['BAND']==band]
        global_flux = obj['FLUXCAL']
        global_fluxerr = obj['FLUXCALERR']
        global_mjd = obj['MJD']

        parameters_dict = {"a": global_flux.max(), "t0": global_mjd[np.argmax(global_flux)], "tfall": 30, "trise":-5}

        least_squares = LeastSquares(global_mjd, global_flux, global_fluxerr, Fbaz)
        fit = Minuit(least_squares,
                 limit_t0=(-200, 200),
                 limit_a=(0.01, 40), 
                 limit_trise=(-30, 0),
                 limit_tfall=(0,500),
                 **parameters_dict)
        fit.migrad()


        max_flux = obj['max_flux'][0]
        max_time = obj['max_flux_time'][0]
        fit_error = fit.fval
        peak = all_obj.meta['PEAKMJD']

        additionnal = [fit_error, max_flux, max_time, peak]

        all_parameters.append([fit.values, additionnal])
        
        
    return all_parameters  #Of the form  [[param_g, extra_g], [param_r, extra_r], [param_i, extra_i]]


def plot_gri(obj, parameters):
    
    colors = {'g ': 'green', 'r ': 'red', 'i ': 'black'}
    
    plt.figure(figsize=(17,5))
    
    for id_plot, band in enumerate(['g ','r ','i ']):    

        single_band = obj[obj['BAND'] == band]
        flux = single_band['FLUXCAL']
        fluxerr = single_band['FLUXCALERR']
        mjd = single_band['MJD']

        xtime = np.linspace(mjd.min()-100, mjd.max(), 1000)
        x = np.array([xtime,[freq_dic.get(band)]*len(xtime)]).T
        plt.subplot(1, 3, id_plot+1)
        plt.scatter(mjd, flux, label=band, color=colors[band])
        plt.plot(xtime, Fnu(x, **parameters), color=colors[band])
        plt.legend(fontsize=12)
        
    plt.show()
     

def main():
    """Usage example"""
    import glob
    from itertools import chain

    import matplotlib.pyplot as plt
    
    path = 'ELASTICC_TRAIN_SNIa-SALT2'#'/media/ELAsTICC/data/training_sample/ELASTICC_TRAIN_SNIa-SALT2'
    heads = sorted(glob.glob(f'{path}/*_HEAD.FITS.gz'))
    phots = sorted(glob.glob(f'{path}/*_PHOT.FITS.gz'))
    assert len(heads) != 0, 'no *_HEAD_FITS.gz are found'
    assert len(heads) == len(phots), 'there are different number of HEAD and PHOT files'

    min_det_per_band = {b'g ': 2, b'r ': 2, b'i ': 2}
    min_total_det = 10
    has_peak = True

    # merge list of lists into a single list
    lcs = list(chain.from_iterable(
        parse_fits_snana(head, phot, min_det_per_band=min_det_per_band, min_total_det=min_total_det, has_peak=True)
                         for head, phot in zip(heads, phots)
    ))

    colors = {'g ': 'green', 'r ': 'red', 'i ': 'black'}
    rng = np.random.default_rng(0)
    
    for idx in rng.choice(list(range(0, len(lcs))), 10, replace=False):
        plt.scatter(lcs[idx]['MJD'], lcs[idx]['FLUXCAL'], color=[colors[b] for b in lcs[idx]['BAND']])
        plt.show()


if __name__ == '__main__':
    main()
