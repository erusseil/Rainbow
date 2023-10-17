import numpy as np

main_path = '/media3/etienne/workdir/spark_AGN/ELAsTiCC/Multiple_band_fit'

plasticc_path = '/media/RESSPECT/data/PLAsTiCC/PLAsTiCC_zenodo'
YSE_path = main_path + '/data_YSE'

plasticc_train_path = main_path + '/data_plasticc/train-test'
YSE_train_path = YSE_path + '/train-test'

max_computation_time = 10
min_det_per_band = {"g ": 4, "r ": 4, "i ": 4}
min_det_per_band_YSE = {"g ": 4, "r ": 4}

PLASTICC_TARGET = {'Ia': 90, 'II': 42, 'Ibc': 62, 'SLSN': 95, 'KN': 64, 'TDE': 15,
                   'YSE_SNII':'YSE_SNII', 'YSE_SNIa':'YSE_SNIa', 'YSE_SNIbc':'YSE_SNIbc'}
PLASTICC_TARGET_INV = {90: 'Ia', 42: 'II', 62: 'Ibc', 95: 'SLSN', 64: 'KN', 15:'TDE',
                      'YSE_SNII':'YSE_SNII', 'YSE_SNIa':'YSE_SNIa', 'YSE_SNIbc':'YSE_SNIbc'}
PASSBANDS = np.array(['u ', 'g ', 'r ', 'i ', 'z ', 'Y ', 'PSg', 'PSr', 'PSi'])

YSE_fractions = np.linspace(0.3, 1, 8)

point_cut_window = 5

default_seed = 42
seeds = np.arange(10)