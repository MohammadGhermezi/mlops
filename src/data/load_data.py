from pathlib import Path

import pandas as pd


def load_data():
    data_path = Path("data/raw/german_credit_data.csv")

    df = pd.read_csv(data_path)

    return df