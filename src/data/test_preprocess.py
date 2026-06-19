from load_data import load_data
from preprocess import preprocess_data

df = load_data()

df = preprocess_data(df)

print(df.head())
print(df.columns)