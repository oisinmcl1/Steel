"""
K-NEAREST REGRESSOR
Oisin Mc Laughlin
22441106
"""
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, cross_validate

scoring = [
    'neg_mean_squared_error',
    'r2'
]

def default_knr(x, y, kf):
    """
    Tests default KNR model with cross validation.
    :param x: Features
    :param y: Target
    :param kf: K-Fold Cross Val obj
    :return: train_mse, test_mse, train_r2, test_r2
    """
    print("\nDefault K-Nearest Regressor Results:")

    # Setup default KNN model and scoring metrics
    default_model = KNeighborsRegressor()

    # Run model with cross validation using default params
    results = cross_validate(
        estimator=default_model,
        X=x,
        y=y,
        scoring=scoring,
        cv=kf,
        verbose=1,
        return_train_score=True
    )

    # No regular mse for cross val so take negative value
    train_mse = -results['train_neg_mean_squared_error']
    test_mse = -results['test_neg_mean_squared_error']
    train_r2 = results['train_r2']
    test_r2 = results['test_r2']

    print("Train MSE:", round(train_mse.mean(), 4))
    print("Test MSE:", round(test_mse.mean(), 4))
    print("Train R2:", round(train_r2.mean(), 4))
    print("Test R2:", round(test_r2.mean(), 4))

    return train_mse, test_mse, train_r2, test_r2


def tuned_knr(x, y, kf):
    """
    Tests tuned KNR model using gridsearch with cross validation.
    :param x: Features
    :param y: Target
    :param kf: K-Fold Cross Val obj
    :return: train_mse, test_mse, train_r2, test_r2
    """
    print("\nTuned K-Nearest Regressor Results:")

    param_grid = {
        'n_neighbors': range(1, 21, 2),
        'weights': ['uniform', 'distance']
    }

    gridsearch_model = KNeighborsRegressor()

    # Using gscv to find best hyperparams with r2 scoring and our kf
    grid_search = GridSearchCV(
        estimator=gridsearch_model,
        param_grid=param_grid,
        scoring='r2',
        cv=kf,
        verbose=1
    )

    # Train model
    grid_search.fit(x, y)

    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best R2 Score:", round(grid_search.best_score_, 4))

    # Evaluate tuned model with cross validation
    tuned_model = grid_search.best_estimator_

    results = cross_validate(
        estimator=tuned_model,
        X=x,
        y=y,
        scoring=scoring,
        cv=kf,
        verbose=1,
        return_train_score=True
    )

    # No regular mse for cross val so take negative value
    train_mse = -results['train_neg_mean_squared_error']
    test_mse = -results['test_neg_mean_squared_error']
    train_r2 = results['train_r2']
    test_r2 = results['test_r2']

    print("Train MSE:", round(train_mse.mean(), 4))
    print("Test MSE:", round(test_mse.mean(), 4))
    print("Train R2:", round(train_r2.mean(), 4))
    print("Test R2:", round(test_r2.mean(), 4))

    return train_mse, test_mse, train_r2, test_r2
