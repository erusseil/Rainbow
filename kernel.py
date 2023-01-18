import numpy as np

ELASTiCC_path = '/media/ELAsTICC/data/training_sample/ELASTICC_TRAIN_'
plasticc_path = '/media/RESSPECT/data/PLAsTiCC/PLAsTiCC_zenodo'
YSE_path = '/media3/etienne/workdir/spark_AGN/ELAsTiCC/Multiple_band_fit/data_YSE'
max_computation_time = 10
min_det_per_band = {"g ": 4, "r ": 4, "i ": 4}
min_det_per_band_YSE = {"g ": 1, "r ": 1, 'PSg':1, 'PSr':1, 'PSi':1}

PLASTICC_TARGET = {'Ia': 90, 'II': 42, 'Ibc': 62, 'SLSN': 95, 'KN': 64, 'TDE': 15,
                   'YSE_SNII':'YSE_SNII', 'YSE_SNIa':'YSE_SNIa', 'YSE_SNIbc':'YSE_SNIbc'}
PLASTICC_TARGET_INV = {90: 'Ia', 42: 'II', 62: 'Ibc', 95: 'SLSN', 64: 'KN', 15:'TDE',
                      'YSE_SNII':'YSE_SNII', 'YSE_SNIa':'YSE_SNIa', 'YSE_SNIbc':'YSE_SNIbc'}
PASSBANDS = np.array(['u ', 'g ', 'r ', 'i ', 'z ', 'Y ', 'PSg', 'PSr', 'PSi'])

point_cut_window = 5
