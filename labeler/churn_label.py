import os
import pandas as pd
from config.path_config import PathConfig
import numpy as np


class ChurnLabeler:
    """
    An object contains 2 components: Attributes (static) and Methods (dynamic)

    """
    def __init__(self):
        self.data = pd.read_csv(os.path.join(PathConfig.BASE_DIR, PathConfig.DATA, PathConfig.CHURN_DATA_PATH))
        start = 1_000_000_000
        ids = list(range(start, start + self.data.shape[0]))
        self.data['CUSID'] = ids
        self.data.set_index('CUSID', inplace=True)

    def get_features_label(self):

        return self.data.drop(columns='churn'), self.data.churn

    def print_info(self):
        print(f'----> Data shape {self.data.shape}')
        print(f'----> Columns names: {self.data.columns.to_list()}')
        print(f'---->Number of missing values: {self.data.isna().mean()}')

    def feature_selection(self):
        pass


if __name__ == '__main__':
    churn_data = ChurnLabeler()
    churn_data.print_info()
    print("DONE")
    # de-coupling
