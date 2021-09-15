from sklearn.model_selection import train_test_split


def split_train_test_data(features, label):
    """
    Split data into train and test sets

    :param features: DataFrame, ndarray - feature set
    :param label: Series, ndarray - label set
    :return: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.15,
                                                        random_state=1234, shuffle=True)

    return X_train, X_test, y_train, y_test
