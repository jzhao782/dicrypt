from mp_api.matproj import MPRester
import pandas as pd
from tqdm import tqdm
from decouple import config

api_key = config('NG_API_KEY')
data_dir = 'data/'
ids = pd.read_csv(data_dir + 'ids.csv', index_col='Unnamed: 0')
fails = pd.read_csv(data_dir + 'fails.csv', index_col='Unnamed: 0')

with MPRester(api_key=api_key, endpoint="https://api.materialsproject.org/") as mpr:
    pbar = tqdm(total=len(fails))
    
    for id in fails["id"]:
        new_id = mpr.get_materials_id_from_task_id(id)
        if new_id != None:
            ids = ids.replace(id, new_id)
        pbar.update(1)
    pbar.close()

ids.to_csv(data_dir + "ids.csv")