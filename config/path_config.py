import os


class PathConfig:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA = 'data'

    # File path
    CHURN_DATA_PATH = 'ChurnData.csv'
    FUEL_DATA_PATH = 'FuelConsumptionCo2.csv'

    # Model
    MODEL_PATH = 'saved_model'
