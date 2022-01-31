import os, sys, re
import numpy as np
import pandas as pd
import matminer
from os.path import exists
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf

data_dir = '/Users/jesse/projects/Python/Element Densities/materials_H-Rn'
os.chdir(data_dir)

def parse_data():
    dict = {"Id" : [], "Composition" : [], "Density" : []}

    for f in os.listdir():
        with open(f'{f}/pretty_formula.dat') as ff:
            s = ff.read() # extract pretty formula

        with open(f'{f}/density.dat') as ff:
            d = float(ff.read()) # extract density

        dict["Id"].append(f)
        dict["Composition"].append(s)
        dict["Density"].append(d)

        print(str(len(dict["Composition"])) + "th row completed!")
    
    df = pd.DataFrame(dict)
    return df

def featurize(df):
    df = StrToComposition(target_col_id="Composition", overwrite_data=True).featurize_dataframe(df, "Composition")
    featurizers = MultipleFeaturizer([
            cf.ElementProperty.from_preset(preset_name="magpie"),
            cf.Stoichiometry(),
            cf.ValenceOrbital(props=['frac']),
            cf.IonProperty(fast=True),
            # cf.BandCenter(),
            cf.ElementFraction()
        ])

    data_comp = featurizers.featurize_dataframe(df, col_id="Composition", ignore_errors=True)

    data_comp["NComp"] = data_comp["Composition"].apply(len)

    dir = '/Users/jesse/projects/Python/Element Densities/'
    save_name = 'downloaded'
    data_comp.to_pickle(dir + save_name + '.pkl')
    data_comp.to_csv(dir + save_name + '.csv')

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
