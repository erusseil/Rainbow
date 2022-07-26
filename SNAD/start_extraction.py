import elasticc_for_etienne as efe
import matplotlib.pyplot as plt
import pickle

import numpy as np
import sncosmo
from astropy.table import Table

from iminuit import Minuit
from iminuit.cost import LeastSquares

import pandas as pd
import sys
import os


def extract(lcs, name, split):
    
    all_param = []
    for obj in lcs:
        extraction = efe.perform_fit(obj)
        all_param.append(extraction[0].values() + extraction[1])

    features = pd.DataFrame(columns=efe.Fnu.__code__.co_varnames[1:9]+('error', 'max_flux', 'max_time'), data = all_param)
    
    if not os.path.exists(f'features/sub_features/{name}'):
        os.mkdir(f'features/sub_features/{name}')
    
    features.to_parquet(f'features/sub_features/{name}/features_{name}_{split}.parquet')
    
def extract_bazin(lcs, name, split):
    
    
    all_param = []
    name_param = []
    
    for idx, obj in enumerate(lcs):
        
        param = efe.perform_bazin_fit(obj)
        
        obj_param = []
        for idx_band, band in enumerate(['g', 'r', 'i']):
            
            if idx == 0:
                baz_name = [x + f'_{band}' for x in list(efe.Fbaz.__code__.co_varnames[1:])+['error']] # For each band add only error as extra param
                name_param.append(baz_name)
            
            obj_param.append(param[idx_band][0].values() + [param[idx_band][1][0]]) # For each band add only error as extra param
            
        flat_obj_param = [x for xs in obj_param for x in xs] + param[0][1][1:] # Add ONCE the max flux and max time paramters
        all_param.append(flat_obj_param)
        
    flat_name_param = [x for xs in name_param for x in xs] + ['max_flux', 'max_time'] # Add ONCE the max flux and max time paramters
    
    
    features = pd.DataFrame(columns=flat_name_param, data = all_param)
    
    if not os.path.exists(f'bazin_features/sub_features/{name}'):
        os.mkdir(f'bazin_features/sub_features/{name}')
    
    features.to_parquet(f'bazin_features/sub_features/{name}/features_{name}_{split}.parquet')
        
if __name__ == '__main__':
    
    
    # HOW TO DIVIDE THE TASKS
    total_div = int(sys.argv[1])
    split = int(sys.argv[2])
    
    ############ USER ############
    name = 'ELASTICC_TRAIN_uLens-Binary_bazin_cut'
    f_extract = extract
    ############ USER ############

    # DATA
    with open(f'preprocess_data/{name}.pkl', 'rb') as handle:
        data = pickle.load(handle)

    
    # DIVIDE DATA INTO SMALL SAMPLES
    
    nb_split = len(data)//total_div

    if split != total_div-1:
        sub_data = data[split*nb_split:(split+1)*nb_split]
    else :
        sub_data = data[split*nb_split:]

        
    f_extract(sub_data, name, split)