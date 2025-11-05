"""
MAIN PROGRAM
Oisin Mc Laughlin
22441106
"""

from utils import get_data, visualise_data
from knr import default_knr, tuned_knr
from svr import default_svr, tuned_svr

def main():
    """
    Main function to execute data retrieval and visualization.
    :return: None
    """
    steel_data, scaled_data, x, y, kf = get_data()
    visualise_data(steel_data, scaled_data)

    default_knr(x, y, kf)
    tuned_knr(x, y, kf)

    default_svr(x, y, kf)
    tuned_svr(x, y, kf)

if __name__ == "__main__":
    main()
