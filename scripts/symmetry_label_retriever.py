import pandas as pd
import pymatgen
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

data_dir = 'data/'
checkpoint_dir = '.script_checkpoints'

dict = {'id' : [], 'crystal_system' : [], 'point_group' : []}
ids = pd.read_csv(data_dir + 'ids.csv')

count = 0
for i in ids.id:
    try:
        with MPRester(api_key='') as m: # add your API key
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
        df.to_pickle(checkpoint_dir + 'symmetry_checkpoint.pkl')
        print('Saved')

df = pd.DataFrame(dict)
df.to_csv(data_dir + 'symmetry_labels.csv')