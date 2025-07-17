import numpy as np
import joblib
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

from transformer import LogTransformer

def drop_duplicates(config, df: DataFrame):
   '''
   Drop duplicated rows. It should be drop `len(df.duplicated()) / 2` rows.
   '''
   return df.drop_duplicates()

def drop_high_corr(config, df: DataFrame):
   '''
   Drop columns with high correlation (over 0.85)
   to target and another columns.
   '''
   target_col = config['target_col']

   correlation    = df.select_dtypes(['int', 'float']).corr()[target_col]
   correlation    = correlation.drop(target_col)
   high_corr_cols = correlation[correlation > 0.85].index
   highest_corr   = correlation.max()

   # Select all high correlation columns
   # that lower than highest corr to be dropped
   eliminated_cols = [
      col for col in high_corr_cols if correlation[col] < highest_corr
   ]

   return df.drop(columns=eliminated_cols)

def drop_outlier(config, df: DataFrame):
   '''
   Remove outlier using IQR method.
   '''
   df = df.copy()
   numeric_cols = df.select_dtypes(['int', 'float']).columns

   for col in numeric_cols:
      col_val = df[col]

      Q1  = col_val.quantile(0.25)
      Q3  = col_val.quantile(0.75)
      IQR = Q3 - Q1

      # Remove the outlier that lower than `Q1 - IQR_treshold`
      # and higher than `Q3 + IQR_treshold`.
      IQR_treshold = IQR * config['outlier_treshold']
      df = df[(df[col] >= Q1 - IQR_treshold) & (df[col] <= Q3 + IQR_treshold)]

   return df

def preprocess(config, df: DataFrame):
   '''
   Preprocess the data that transform the value
   that does not affect the number of rows and columns.
   '''
   df_feat = df.drop(columns=config['target_col'])
   df_targ = df[[config['target_col']]]

   # Split to prevent data leakage.
   X_train, X_test, y_train, y_test = train_test_split(
      df_feat, df_targ, train_size=0.8, random_state=0
   )

   # Group the columns by type.
   category_cols = df_feat.select_dtypes(['object', 'string']).columns
   numeric_cols  = df_feat.select_dtypes(['int', 'float']).columns

   category_order = list(config['metadata']['category_order'].values())

   pipe_category = Pipeline([
      ('imputer', SimpleImputer(strategy='most_frequent')),
      ('encoder', OrdinalEncoder(categories=category_order))
   ])
   pipe_numeric = Pipeline([
      ('imputer', SimpleImputer(strategy='mean')),
   ])
   x_transformer = ColumnTransformer([
      ('pipe_category', pipe_category, category_cols),
      ('pipe_numeric', pipe_numeric, numeric_cols),
   ])

   # We use `Pipeline` to wrap the `ColumnTransformer` as it's work in pararel.
   x_preprocessor = Pipeline([
      ('transformer', x_transformer),
      ('scaler', StandardScaler())
   ])
   y_preprocessor = Pipeline([
      ('logaritmic', LogTransformer()),
      ('scaler', StandardScaler())
   ])

   # Start preprocess the data.
   x_preprocessor.fit(X_train)
   X_train = x_preprocessor.transform(X_train)
   X_test  = x_preprocessor.transform(X_test)

   y_preprocessor.fit(y_train)
   y_train = y_preprocessor.transform(y_train)
   y_test  = y_preprocessor.transform(y_test)

   # Save the x_preprocessor and y_preprocessor to use it on another time.
   x_preprocessor_path = f'../joblibs/x_preprocessor.joblib'
   y_preprocessor_path = f'../joblibs/y_preprocessor.joblib'
   joblib.dump(x_preprocessor, x_preprocessor_path)
   joblib.dump(y_preprocessor, y_preprocessor_path)

   return {
      'columns'       : df.columns,
      'splitted_data' : [X_train, X_test, y_train, y_test]
   }

def to_dataframe(config, data):
   '''
   Combine splitted data into DataFrame.
   '''
   columns = data['columns']
   X_train, X_test, y_train, y_test = data['splitted_data']

   X = [*X_train, *X_test]
   y = [*y_train, *y_test]

   return DataFrame(np.concatenate((X, y), axis=1), columns=columns)

def main(config, df: DataFrame):
   '''
   Orchestration function for all process in `preprocess.py`.
   '''
   df_prep1 = drop_duplicates(config, df)
   df_prep2 = drop_high_corr(config, df_prep1)
   df_prep3 = drop_outlier(config, df_prep2)
   data     = preprocess(config, df_prep3)
   df_final = to_dataframe(config, data)

   return df_final