"""
UTILITY FUNCTIONS
Oisin Mc Laughlin
22441106
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
    Visualises data before and after scaling using box plots.
    Visualises data with correlation matrix with heatmap.
    :param steel_data: dataframe read from steel.csv
    :param scaled_data: dataframe of scaled steel data
    :return:
    """
    print("\nVisualising Data")

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

    corr = steel_data.corr()
    plt.figure(figsize=(12, 10))
    plt.title('Correlation Matrix of Steel Data')
    sns.heatmap(data=corr, annot=True)
    plt.tight_layout()
    plt.show()


def compare_models(model_x_name, model_y_name,
                   model_x_train_mse, model_x_test_mse, model_x_train_r2, model_x_test_r2,
                   model_y_train_mse, model_y_test_mse, model_y_train_r2, model_y_test_r2):
    """
    Compares two models using a simple bar chart.
    :param model_x_name: Name of first model
    :param model_y_name: Name of second model
    :param model_x_train_mse: Train MSE array for model x
    :param model_x_test_mse: Test MSE array for model x
    :param model_x_train_r2: Train R2 array for model x
    :param model_x_test_r2: Test R2 array for model x
    :param model_y_train_mse: Train MSE array for model y
    :param model_y_test_mse: Test MSE array for model y
    :param model_y_train_r2: Train R2 array for model y
    :param model_y_test_r2: Test R2 array for model y
    :return:
    """
    print("\nVisualising Comparison between" + model_x_name + "and" + model_y_name)

    labels = ['Train MSE', 'Test MSE', 'Train R2', 'Test R2']

    model_x_scores = [
        model_x_train_mse.mean(),
        model_x_test_mse.mean(),
        model_x_train_r2.mean(),
        model_x_test_r2.mean()
    ]

    model_y_scores = [
        model_y_train_mse.mean(),
        model_y_test_mse.mean(),
        model_y_train_r2.mean(),
        model_y_test_r2.mean()
    ]

    x = np.arange(4)

    plt.figure(figsize=(12, 10))
    plt.bar(x=x - 0.2, height=model_x_scores, width=0.4, label=model_x_name)
    plt.bar(x=x + 0.2, height=model_y_scores, width=0.4, label=model_y_name)

    plt.xticks(x, labels)
    plt.ylabel('Score')
    plt.title('Comparison of ' + model_x_name + ' and ' + model_y_name)

    plt.legend()
    plt.tight_layout()
    plt.show()
