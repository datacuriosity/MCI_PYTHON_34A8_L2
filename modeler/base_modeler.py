import abc


class BaseModeler(abc.ABC):

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict_label(self):
        pass

    @abc.abstractmethod
    def evaluate(self, y_pred):
        pass
