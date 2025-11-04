"""
K-NEAREST REGRESSOR
Oisin Mc Laughlin
22441106
"""
from sklearn.neighbors import KNeighborsRegressor
from utils import evaluate_model

def knr(x, y, kf):
    """
    :return:
    """
    default_train_mse = []
    default_test_mse = []
    default_train_r2 = []
    default_test_r2 = []

    # Iterate through each fold
    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        print("\nFold:", fold)

        # Split data into train and test for this fold
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        # Train default KNN model
        default_model = KNeighborsRegressor()
        default_model.fit(x_train, y_train)

        # Make predictions
        train_pred = default_model.predict(x_train)
        test_pred = default_model.predict(x_test)

        # Evaluate model
        train_mse, train_r2 = evaluate_model(y_train, train_pred)
        test_mse, test_r2 = evaluate_model(y_test, test_pred)

        # Store results
        default_train_mse.append(train_mse)
        default_test_mse.append(test_mse)
        default_train_r2.append(train_r2)
        default_test_r2.append(test_r2)

        print("Train MSE:", round(train_mse, 4), "Train R2:", round(train_r2, 4))
        print("Test MSE:", round(test_mse, 4), "Test R2:", round(test_r2, 4))
