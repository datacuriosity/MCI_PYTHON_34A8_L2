from sklearn.datasets import load_boston
import pandas as pd


class BostonLabeler:
    """
    An object contains 2 components: Attributes (static) and Methods (dynamic)


    """
    def __init__(self, is_take_column=True):

        if is_take_column:
            data = load_boston()
            df = pd.DataFrame(data.get('data'), columns=data.get('feature_names'))
            df['LABEL'] = data.get('target')

            self.feature = df.drop(columns='LABEL').values
            self.label = df.LABEL.values
            print(data.get('DESCR'))

        else:
            self.feature, self.label = load_boston(return_X_y=True)

    def feature_selection(self):
        pass


if __name__ == '__main__':
    boston_label = BostonLabeler()
    print("DONE")
    # de-coupling
