import pickle
from pathlib import Path

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data

def predict():
    model_path=Path("artifacts/models/logistic_regresion.pk1")
    
    with open(model_path,"rb")as f:
        model=pickle.load(f)
    
    df=load_data()
    df=preprocess_data(df)
    sample=df.drop("target",axis=1).iloc[[0]]
    
    prediction=model.predict(sample)
    
    print(f"prediction: {prediction}")
    
if __name__=="__main__":
    predict()