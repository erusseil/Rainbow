import data_processing as dp
import kernel as kern
import pickle
import sys

if __name__ == "__main__":

    total_div = int(sys.argv[1])
    split = int(sys.argv[2])
    object_class = str(sys.argv[3])
    str_extract = str(sys.argv[4])
    database = str(sys.argv[5])

    if str_extract == "bazin":
        f_extract = dp.extract_bazin

    elif str_extract == "rainbow":
        f_extract = dp.extract_rainbow

    if database == 'elasticc':
        # DATA
        with open(f"data_elasticc/preprocessed/{object_class}.pkl", "rb") as handle:
            data = pickle.load(handle)
            
    elif (database == 'plasticc') or (database == 'YSE'):
        # DATA
        with open(f"data_plasticc/preprocessed/{object_class}.pkl", "rb") as handle:
            data = pickle.load(handle)

    # DIVIDE DATA INTO SMALL SAMPLES

    nb_split = len(data) // total_div

    if split != total_div - 1:
        sub_data = data[split * nb_split: (split + 1) * nb_split]
    else:
        sub_data = data[split * nb_split:]

    f_extract(sub_data, object_class, split, database)
