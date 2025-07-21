import sys
import os

# Append root folder to access transformer folder.
sys.path.append(os.path.abspath('../'))

import mlflow
import pandas as pd
import config as conf

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

config = conf.get_config()

with mlflow.start_run() as r:
   mlflow.autolog()

   df = pd.read_csv(f'dataset/{config['dataset']}')
   df_feat = df.drop(columns=config['target_col'])
   df_targ = df[[config['target_col']]]

   X_train, X_test, y_train, y_test = train_test_split(
      df_feat, df_targ, train_size=0.8, random_state=0
   )

   rf_model = RandomForestRegressor()
   rf_model.fit(X_train, y_train.to_numpy().reshape(-1))