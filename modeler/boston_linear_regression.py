from abc import ABC

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from base_modeler import BaseModeler


class BostonLinearRegression(BaseModeler):

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.model = None

    def train(self):
        linear_model = LinearRegression()
        linear_model.fit(self.X_train, self.y_train)

        self.model = linear_model
        return linear_model

    def predict_label(self):
        predicted = self.model.predict(self.X_test)

        return predicted

    def evaluate(self, y_pred):

        return mean_squared_error(self.y_test, y_pred), mean_absolute_error(self.y_test, y_pred), \
               r2_score(self.y_test, y_pred)





