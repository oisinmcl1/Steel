"""
UTILITY FUNCTIONS
Oisin Mc Laughlin
22441106
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

steel_data = pd.read_csv('steel.csv')


def get_data():
    """
    Splits the steel data into features and target variable.
    Sets up 10 fold cross validation.
    :return: x, y, kf
    """
    print("Data loaded successfully.")
    print("Data Shape:", steel_data.shape)

    # Separating features and target variable
    x = steel_data.drop('tensile_strength', axis=1)
    y = steel_data['tensile_strength']

    print("Features Shape:", x.shape)
    print("Target Shape:", y.shape)

    # Setting up 10 fold cross validation and shuffling the data
    kf = KFold(n_splits=10, shuffle=True, random_state=1)

    return x, y, kf


def visualise_data():
    """
    Visualises the steel data.
    :return:
    """
    steel_data.hist(figsize=(12, 10))
    plt.suptitle('Steel Data')
    plt.tight_layout()
    plt.show()
