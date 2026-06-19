from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.data.split_data import split_data


def train():
    
    df=load_data()
    df=preprocess_data(df)
    
    X_train,X_test,y_train,y_test=split_data(df)   
    model = LogisticRegression(max_iter=1000)
    
    model.fit(X_train,y_train)
    
    prediction=model.predict(X_test)
    
    accuracy=accuracy_score(y_test,prediction)
    
    print(f"accuracy = {accuracy}")
    
    
    
if __name__=="__main__":
    train()
    
    