import pandas as pd
from labeler.linear_regression_label import BostonLabeler
from data_loader.boston_data_loader import split_train_test_data
from modeler.boston_linear_regression import BostonLinearRegression

boston_label = BostonLabeler()
X_train, X_test, y_train, y_test = split_train_test_data(boston_label.feature, boston_label.label)

boston_model = BostonLinearRegression(X_train, X_test, y_train, y_test)
boston_model.train()

predicted = boston_model.predict_label()

mse, mae, r2_score = boston_model.evaluate(predicted)

print(f"Mean squared err: {mse} - Mean absolute error: {mae} - R2 score: {r2_score}")

result = pd.DataFrame({'actual': y_test, 'predicted': predicted})
print(result)

