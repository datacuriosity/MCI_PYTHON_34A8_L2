import os
import pandas as pd
from labeler.churn_label import ChurnLabeler
from data_loader.data_loader import split_train_test_data
from modeler.churn_modeler import ChurnModel
from config.path_config import PathConfig

# Step 1: Prepare feature and label
labeler = ChurnLabeler()
labeler.print_info()
features, labels = labeler.get_features_label()

# Step 2: Train/ test split
X_train, X_test, y_train, y_test = split_train_test_data(features, labels)

# Step 3: Model trained 2021-09-10
model = ChurnModel(X_train, X_test, y_train, y_test)
model.train()
model.save_model(os.path.join(PathConfig.BASE_DIR, PathConfig.MODEL_PATH, 'churn_model_v1.pickle'))
predicted = model.predict_label()

results = pd.DataFrame({'predicted_label': predicted})
results['CUSID'] = y_test.index
results['true_label'] = y_test.values.tolist()
results.set_index('CUSID', inplace=True)


# Step 4: Evaluation
model.evaluate(predicted)
