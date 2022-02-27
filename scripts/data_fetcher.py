from numpy import add
import pandas as pd
import requests
from requests.structures import CaseInsensitiveDict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from decouple import config

api_key = config('NG_API_KEY')
data_dir = 'data/'
checkpoint_dir = '.script_checkpoints/'
properties = ['formula_pretty'] # example

dict = {'id' : []}
for property in properties:
    dict[property] = []

ids = pd.read_csv(data_dir + 'ids.csv')

pbar = tqdm(total=len(ids))
def fetch(url, headers, i, count):
    try:
        res =  requests.get(url, headers=headers)
        data = res.json()
        
        for property in properties:
            dict[property].append(data["data"][0][property]) # might need to check if properties are nested
        dict['id'].append(i)
    except Exception as e:
        print(e)
        dict['id'].append(i)
        for property in properties:
            dict[property].append(None)

    if count % 1000 == 0:
        df = pd.DataFrame(dict)
        df.to_pickle(checkpoint_dir + "data_fetcher_checkpoint.pkl")
    pbar.update(1)

headers = CaseInsensitiveDict()
headers["accept"] = "application/json"
headers["X-API-KEY"] = api_key
with ThreadPoolExecutor(max_workers=20) as executor:
    for count, i in enumerate(ids.id):
        executor.submit(fetch, f"https://api.materialsproject.org/summary/{i}/?all_fields=true", headers, i, count)
pbar.close()

df = pd.DataFrame(dict)
df = df.set_index("id")
additional_properties = pd.read_csv(data_dir + "additional_properties.csv", index_col="id")
additional_properties = pd.concat([additional_properties, df], axis=1)
additional_properties.to_csv(data_dir + "additional_properties.csv")

'''
Cannot fetch properties that are nested under other labels (e.g. point group), so edit line 27 as necessary
'''