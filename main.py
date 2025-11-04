"""
MAIN PROGRAM
Oisin Mc Laughlin
22441106
"""

from utils import get_data, visualise_data
import knr

def main():
    """
    Main function to execute data retrieval and visualization.
    :return: None
    """
    # Get data
    steel_data, x, y, kf = get_data()
    knr.knr(x, y, kf)

if __name__ == "__main__":
    main()
