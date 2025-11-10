"""
MAIN PROGRAM
Oisin Mc Laughlin
22441106
"""

from utils import get_data, visualise_data, compare_models, export_results_to_csv
from knr import default_knr, tuned_knr
from svr import default_svr, tuned_svr

def main():
    """
    Main function to execute data retrieval and visualization.
    :return: None
    """
    steel_data, scaled_data, x, y, kf = get_data()
    visualise_data(steel_data, scaled_data)

    (default_knr_train_mse, default_knr_test_mse,
     default_knr_train_r2, default_knr_test_r2) = default_knr(x, y, kf)

    (tuned_knr_train_mse, tuned_knr_test_mse,
     tuned_knr_train_r2, tuned_knr_test_r2) = tuned_knr(x, y, kf)

    (default_svr_train_mse, default_svr_test_mse,
     default_svr_train_r2, default_svr_test_r2) = default_svr(x, y, kf)

    (tuned_svr_train_mse, tuned_svr_test_mse,
     tuned_svr_train_r2, tuned_svr_test_r2) = tuned_svr(x, y, kf)

    # Metric dictionaries
    default_knr_metrics = {
        'model_name': 'Default KNR',
        'train_mse': default_knr_train_mse,
        'test_mse': default_knr_test_mse,
        'train_r2': default_knr_train_r2,
        'test_r2': default_knr_test_r2
    }
    export_results_to_csv(default_knr_metrics)

    tuned_knr_metrics = {
        'model_name': 'Tuned KNR',
        'train_mse': tuned_knr_train_mse,
        'test_mse': tuned_knr_test_mse,
        'train_r2': tuned_knr_train_r2,
        'test_r2': tuned_knr_test_r2
    }
    export_results_to_csv(tuned_knr_metrics)

    default_svr_metrics = {
        'model_name': 'Default SVR',
        'train_mse': default_svr_train_mse,
        'test_mse': default_svr_test_mse,
        'train_r2': default_svr_train_r2,
        'test_r2': default_svr_test_r2
    }
    export_results_to_csv(default_svr_metrics)

    tuned_svr_metrics = {
        'model_name': 'Tuned SVR',
        'train_mse': tuned_svr_train_mse,
        'test_mse': tuned_svr_test_mse,
        'train_r2': tuned_svr_train_r2,
        'test_r2': tuned_svr_test_r2
    }
    export_results_to_csv(tuned_svr_metrics)

    # Compare models
    compare_models(default_knr_metrics, tuned_knr_metrics)
    compare_models(default_svr_metrics, tuned_svr_metrics)
    compare_models(tuned_knr_metrics, tuned_svr_metrics)

if __name__ == "__main__":
    main()
