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


def tuned_svr(x, y, kf):
    """
    Tests tuned SVR model using gridsearch with cross validation.
    :param x: Features
    :param y: Target
    :param kf: K-Fold Cross Val obj
    :return: train_mse, test_mse, train_r2, test_r2
    """