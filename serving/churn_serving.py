import os
import pickle
import pandas as pd
from config.path_config import PathConfig


def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


modeler = load_model(os.path.join(PathConfig.BASE_DIR, PathConfig.MODEL_PATH, 'churn_model_v1.pickle'))

# Streaming data
df = pd.read_csv(os.path.join(PathConfig.BASE_DIR, PathConfig.DATA, PathConfig.CHURN_DATA_PATH))
df.drop(columns='churn', inplace=True)

predicted = modeler.model.predict(df)
print(predicted)

