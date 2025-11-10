"""
MAIN PROGRAM
Oisin Mc Laughlin
22441106
"""

from utils import get_data, visualise_data, compare_models
from knr import default_knr, tuned_knr
from svr import default_svr, tuned_svr

def main():
    """
    Main function to execute data retrieval and visualization.
    :return: None
    """
    steel_data, scaled_data, x, y, kf = get_data()
    visualise_data(steel_data, scaled_data)

    default_knr_train_mse, default_knr_test_mse, default_knr_train_r2, default_knr_test_r2 = default_knr(x, y, kf)
    tuned_knr_train_mse, tuned_knr_test_mse, tuned_knr_train_r2, tuned_knr_test_r2 = tuned_knr(x, y, kf)
    default_svr_train_mse, default_svr_test_mse, default_svr_train_r2, default_svr_test_r2 = default_svr(x, y, kf)
    tuned_svr_train_mse, tuned_svr_test_mse, tuned_svr_train_r2, tuned_svr_test_r2 = tuned_svr(x, y, kf)

    compare_models("Default KNR", "Tuned KNR",
                   default_knr_train_mse, default_knr_test_mse, default_knr_train_r2, default_knr_test_r2,
                   tuned_knr_train_mse, tuned_knr_test_mse, tuned_knr_train_r2, tuned_knr_test_r2)

    compare_models("Default SVR", "Tuned SVR",
                   default_svr_train_mse, default_svr_test_mse, default_svr_train_r2, default_svr_test_r2,
                   tuned_svr_train_mse, tuned_svr_test_mse, tuned_svr_train_r2, tuned_svr_test_r2)

    compare_models("Tuned KNR", "Tuned SVR",
                   tuned_knr_train_mse, tuned_knr_test_mse, tuned_knr_train_r2, tuned_knr_test_r2,
                   tuned_svr_train_mse, tuned_svr_test_mse, tuned_svr_train_r2, tuned_svr_test_r2)

if __name__ == "__main__":
    main()
