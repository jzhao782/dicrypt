import pandas as pd
import requests
from requests.structures import CaseInsensitiveDict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from decouple import config

api_key = config('NG_API_KEY')
data_dir = 'data/'
checkpoint_dir = '.script_checkpoints/'
file_name = "e_above_hull"

endpoint = 'thermo'
properties = ['energy_above_hull']

dict = {'id' : []}
for property in properties:
    dict[property] = []

ids = pd.read_csv(data_dir + 'ids.csv')

pbar = tqdm(total=len(ids))
def fetch(url, headers, i, count, fails, key):
    try:
        res =  requests.get(url, headers=headers)
        data = res.json()
        
        for property in properties:
            dict[property].append(data[key][0][property])
        dict['id'].append(i)
    except Exception as e:
        fails.append(i)
        dict['id'].append(i)
        for property in properties:
            dict[property].append(None)

    if count % 1000 == 0:
        df = pd.DataFrame(dict)
        df.to_pickle(checkpoint_dir + file_name + ".pkl")
    pbar.update(1)

headers = CaseInsensitiveDict()
headers["accept"] = "application/json"
headers["X-API-KEY"] = api_key
fails = []
with ThreadPoolExecutor(max_workers=10) as executor:
    for count, i in enumerate(ids.id):
        executor.submit(fetch, f"https://api.materialsproject.org/{endpoint}/{i}/?all_fields=true", headers, i, count, fails, "data")
pbar.close()

df = pd.DataFrame(dict)
df.to_csv(data_dir + file_name + '.csv')
print(fails)