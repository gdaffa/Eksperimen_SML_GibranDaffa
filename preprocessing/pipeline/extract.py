from pandas import DataFrame, read_csv
from scipy.io.arff import loadarff

RAW_DATASET_PATH = '../dataset/raw'

def read_arff(filepath):
   '''
   Read dataset with `arff` type.
   '''
   data, _ = loadarff(filepath)
   df      = DataFrame(data)

   # Convert `byte` data type to a `string` data type.
   category_cols     = df.select_dtypes('object').columns
   df[category_cols] = df[category_cols].apply(lambda series: series.str.decode('utf-8'))

   return df

def read(config) -> DataFrame:
   '''
   Read dataset in `RAW_DATASET_PATH` by file type and return a DataFrame.
   '''
   try:
      filename = config['dataset']
      filetype = filename.split('.')[-1]
      filepath = f'{RAW_DATASET_PATH}/{filename}'

      readers = {
         'read_arff': read_arff,
         'read_csv': read_csv
      }

      return readers[f'read_{filetype}'](filepath)

   except IndexError:
      raise ValueError('File type is invalid.')


def main(config):
   '''
   Orchestration function for all process in `extract.py`.
   '''
   df = read(config)
   return df