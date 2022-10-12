import data_processing as dp
import matplotlib.pyplot as plt
import pickle

import numpy as np
import sncosmo
from astropy.table import Table

import pandas as pd
import sys
import os

        
if __name__ == '__main__':
    
    total_div = int(sys.argv[1])
    split = int(sys.argv[2])
    object_class = str(sys.argv[3])
    str_extract = str(sys.argv[4])
    
    if str_extract == 'bazin':
        f_extract = dp.extract_bazin
        
    elif str_extract == 'mbf':
        f_extract = dp.extract_mbf

    # DATA
    with open(f'data/preprocessed/{object_class}.pkl', 'rb') as handle:
        data = pickle.load(handle)
        
    
    # DIVIDE DATA INTO SMALL SAMPLES
    
    nb_split = len(data)//total_div

    if split != total_div-1:
        sub_data = data[split*nb_split:(split+1)*nb_split]
    else :
        sub_data = data[split*nb_split:]
        
    f_extract(sub_data, object_class, split)