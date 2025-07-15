import numpy as np
from pandas import DataFrame

PREPROCESSED_DATASET_PATH = '../dataset/preprocessed'

def save(config, df: DataFrame):
   '''
   Save dataset to CSV file in `PREPROCESSED_DATASET_PATH`.
   '''
   filename = config["dataset"].split('.')[0]
   return df.to_csv(f'{PREPROCESSED_DATASET_PATH}/{filename}.csv', index=False)

def main(config, df):
   '''
   Orchestration function for all process in `load.py`.
   '''
   return save(config, df)