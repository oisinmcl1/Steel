"""
UTILITY FUNCTIONS
Oisin Mc Laughlin
22441106
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def get_data():
    """
    Gets the steel data from steel.csv
    Scales the data using StandardScaler.
    Splits the steel data into features and target variable.
    Sets up 10-fold cross validation.
    :return: steel_data, scaled_data, x, y, kf
    """
    steel_data = pd.read_csv('steel.csv')
    print("Data loaded successfully.")
    print("Data Shape:\n", steel_data.shape)
    print("Data Description:\n", steel_data.describe())

    # Scale steel data and convert back to dataframe
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(steel_data), columns=steel_data.columns)

    # Separating features and target variable
    x = scaled_data.drop('tensile_strength', axis=1)
    y = scaled_data['tensile_strength']

    print("Features Shape:", x.shape)
    print("Target Shape:", y.shape)

    # Setting up 10-fold cross validation and shuffling the data
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    return steel_data, scaled_data, x, y, kf


def visualise_data(steel_data, scaled_data):
    """
    Visualises the steel data using histograms.
    Also visualises data before and after scaling using box plots.
    :param steel_data: dataframe read from steel.csv
    :param scaled_data: dataframe of scaled steel data
    :return:
    """
    steel_data.hist(figsize=(12, 10))
    plt.suptitle('Steel Data Before Scaling')
    plt.tight_layout()
    plt.show()

    scaled_data.hist(figsize=(12, 10))
    plt.suptitle('Scaled Steel After Scaling')
    plt.tight_layout()
    plt.show()

    steel_data.boxplot(figsize=(12, 10))
    plt.suptitle('Steel Data Before Scaling')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    scaled_data.boxplot(figsize=(12, 10))
    plt.suptitle('Steel Data After Scaling')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
