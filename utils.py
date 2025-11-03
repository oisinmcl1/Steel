"""
UTILITY FUNCTIONS
Oisin Mc Laughlin
22441106
"""

import pandas as pd

def load_data():
    """
    Loads data from steel.csv
    :return: data
    """
    data = pd.read_csv('steel.csv')

    return data
