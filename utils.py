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
    Eliminates features that have low correlation with tensile_strength.
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

    # Eliminate features with correlation less than 0.2 with tensile_strength
    corr = scaled_data.corr()
    # Abs value to consider negative correlation too
    target_corr = corr['tensile_strength'].abs()
    selected_features = target_corr[target_corr > 0.2].index.tolist()
    print("Number of features after elimination:", len(selected_features) - 1)
    # Convert to dataframe
    selected_features = scaled_data[selected_features]

    # Separating features and target variable
    x = selected_features.drop('tensile_strength', axis=1)
    y = selected_features['tensile_strength']

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


def compare_models(model_x_metrics, model_y_metrics):
    """
    Compares two models using a simple bar chart.
    :param model_x_metrics: Dictionary with model metrics
    :param model_y_metrics: Dictionary with model metrics
    :return:
    """
    model_x_name = model_x_metrics['model_name']
    model_y_name = model_y_metrics['model_name']

    print("\nVisualising Comparison between " + model_x_name + " and " + model_y_name)

    labels = ['Train MSE', 'Test MSE', 'Train R2', 'Test R2']

    model_x_scores = [
        model_x_metrics['train_mse'].mean(),
        model_x_metrics['test_mse'].mean(),
        model_x_metrics['train_r2'].mean(),
        model_x_metrics['test_r2'].mean()
    ]

    model_y_scores = [
        model_y_metrics['train_mse'].mean(),
        model_y_metrics['test_mse'].mean(),
        model_y_metrics['train_r2'].mean(),
        model_y_metrics['test_r2'].mean()
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


def export_results_to_csv(model_metrics):
    """
    Exports model results to a CSV file.
    :param model_metrics: Dictionary with model metrics
    :return:
    """
    model_name = model_metrics['model_name']
    results = {
        'Train MSE': model_metrics['train_mse'],
        'Test MSE': model_metrics['test_mse'],
        'Train R2': model_metrics['train_r2'],
        'Test R2': model_metrics['test_r2']
    }
    results_df = pd.DataFrame(results)

    results_df.to_csv(model_name + '_results.csv', index=False)
    print("\nResults exported to " + model_name + "_results.csv")
