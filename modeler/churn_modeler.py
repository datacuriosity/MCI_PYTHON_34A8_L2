import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_confusion_matrix
from modeler.base_modeler import BaseModeler


class ChurnModel(BaseModeler):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.model = None

    def train(self):
        lr = LogisticRegression()
        lr.fit(self.X_train, self.y_train)

        self.model = lr
        return lr

    def predict_label(self):
        predicted = self.model.predict(self.X_test)

        return predicted

    def evaluate(self, y_pred):
        print(classification_report(self.y_test, y_pred))

        plot_confusion_matrix(self.model, self.X_test, self.y_test)
        plt.show()

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)