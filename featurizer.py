import os, sys, re
import pandas as pd
import matminer
from os.path import exists
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

data_dir = '/Users/jesse/projects/Python/Element Densities/materials_H-Rn'
os.chdir(data_dir)

def parse_data():
    df = pd.DataFrame()

    for f in os.listdir():
        with open(f'{f}/density.dat') as ff:
            d = float(ff.read()) # extract density

        with open(f'{f}/pretty_formula.dat') as ff:
            s = ff.read() # extract pretty formula

        row = pd.DataFrame([[s, d]], columns=["Input", "Density"])
        df = pd.concat([df, row], ignore_index=True)
        print(str(len(df.index)) + "th row completed!")

    return df

def featurize(df):
    df = StrToComposition(target_col_id="Composition").featurize_dataframe(df, "Input")
    magpie = ElementProperty.from_preset(preset_name='magpie')
    data_comp = magpie.featurize_dataframe(df, col_id="Composition", ignore_errors=True)

    save_name = 'magpiedownloaded.pkl'
    data_comp.to_pickle(save_name)
    print(f"Saved  {save_name}")
    print(data_comp.head())
    

if __name__ == "__main__":
    file_name = "compositions.pkl"
    if not exists(file_name):
        df = parse_data()
        df.to_pickle(file_name)
    else:
        df = pd.read_pickle(file_name)
    featurize(df)

    
    

