import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.data.split_data import split_data
from configs.config import MODEL_PATH,MAX_ITER




def train():
    mlflow.set_experiment("credit-risk-prediction")
    
    with mlflow.start_run():
        df=load_data()
        df=preprocess_data(df)
    
        X_train,X_test,y_train,y_test=split_data(df)   
    
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", MAX_ITER)
    
        model = LogisticRegression(max_iter=MAX_ITER)
    
        model.fit(X_train,y_train)
    
        prediction=model.predict(X_test)
    
        accuracy=accuracy_score(y_test,prediction)
        precision=precision_score(y_test,prediction)
        f1=f1_score(y_test,prediction)
        recall=recall_score(y_test,prediction)
   
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
    
        print(f"accuracy = {accuracy}\n precision= {precision} \n f1= {f1} \n recall= {recall}")
    
        model_path=MODEL_PATH
        with open(model_path,"wb")as f:
            pickle.dump(model,f)
    
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
        )
    
        print(f"model {model} saved to {model_path}")
    
    
if __name__=="__main__":
    train()
    
    