"""
SUPPORT VECTOR REGRESSION
Oisin Mc Laughlin
22441106
"""
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_validate

scoring = [
    'neg_mean_squared_error',
    'r2'
]

def default_svr(x, y, kf):
    """
    Tests default SVR model with cross validation.
    :param x: Features
    :param y: Target
    :param kf: K-Fold Cross Val obj
    :return: train_mse, test_mse, train_r2, test_r2
    """
    print("\nDefault Support Vector Regressor Results:")

    default_model = SVR()

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


def tuned_svr(x, y, kf):
    """
    Tests tuned SVR model using gridsearch with cross validation.
    :param x: Features
    :param y: Target
    :param kf: K-Fold Cross Val obj
    :return: train_mse, test_mse, train_r2, test_r2
    """
    print("\nTuned Support Vector Regressor Results:")

    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100, 1000]
    }

    gridsearch_model = SVR()

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
        return_train_score=True,
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
