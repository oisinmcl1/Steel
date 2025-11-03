"""
MAIN PROGRAM
Oisin Mc Laughlin
22441106
"""

from utils import load_data

def main():
    data = load_data()
    print(data.head())

if __name__ == "__main__":
    main()
