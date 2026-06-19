from pathlib import Path

import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [
        "checking_account",
        "loan_duration",
        "credit_history",
        "loan_purpose",
        "credit_amount",
        "savings_account",
        "employment_duration",
        "installment_rate",
        "personal_status",
        "guarantors",
        "residence_duration",
        "assets",
        "age",
        "other_credits",
        "housing",
        "existing_credits",
        "job",
        "dependents",
        "telephone",
        "foreign_worker",
        "target",
    ]

    return df