import pandas as pd
import requests
from tqdm import tqdm

api_key = 'OV8HxsEXoIqdA3tr'
data_dir = 'data/'
checkpoint_dir = '.script_checkpoints/e_above_hull.pkl'

properties = ['e_above_hull']
file_name = "e_above_hull"

dict = {'id' : []}
for property in properties:
    dict[property] = []

ids = pd.read_csv(data_dir + 'ids.csv')

for count, i in enumerate(tqdm(ids.id)):
    try:
        res =  requests.get(f"https://www.materialsproject.org/rest/v2/materials/{i}/vasp?API_KEY={api_key}")
        data = res.json()
        dict['id'].append(i)
        
        for property in properties:
            dict[property].append(data["response"][0][property])
            
    except Exception as e:
        print(f"{e} on {ids}")
        dict['id'].append(i)
        for property in properties:
            dict[property].append(None)
    
    if count % 1000 == 0:
        df = pd.DataFrame(dict)
        df.to_pickle(checkpoint_dir + file_name + ".pkl")

df = pd.DataFrame(dict)
df.to_csv(data_dir + file_name + '.csv')