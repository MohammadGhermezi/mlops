from sklearn.model_selection import train_test_split
from configs.config import TEST_SIZE,RANDOM_STATE

def split_data(df):
    X=df.drop("target",axis=1)
    y=df["target"]
    
    return train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)