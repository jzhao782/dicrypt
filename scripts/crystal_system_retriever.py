import pandas as pd
import pymatgen
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

data_dir = 'data/'

dict = {'id' : [], 'crystal_system' : [], 'point_group' : []}
ids = pd.read_csv(data_dir + 'ids.csv')

count = 0
for i in ids.id:
    try:
        with MPRester(api_key='scL4gtOTH7U850nI') as m:
            struct = m.get_structure_by_material_id(i)

        analyzer = SpacegroupAnalyzer(struct)
        dict['id'].append(i)
        dict['crystal_system'].append(analyzer.get_crystal_system())
        dict['point_group'].append(analyzer.get_point_group_symbol())
    except:
        dict['id'].append(i)
        dict['crystal_system'].append(None)
        dict['point_group'].append(None)

    count += 1
    if count % 10 == 0:
        print(f'{count} out of {len(ids)} finished')
    
    if count % 1000 == 0:
        df = pd.DataFrame(dict)
        df.to_pickle(data_dir + 'crystal_systems_log.pkl')
        print('Saved')

df = pd.DataFrame(dict)
df.to_csv(data_dir + 'crystal_systems.csv')