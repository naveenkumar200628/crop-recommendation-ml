import pandas as pd  #importing neccessary packages
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
#Load Data Set
crop_Data=pd.read_csv("data/Crop_recommendation.csv")
#Defining feature to Input And Output(x,y)
X=crop_Data.drop(["label","temperature","ph"],axis=1)#removing Label Because output value and removing temprature after some feature selectin
Y=crop_Data["label"]
#Encoding Output Feature string into numbers using "label encoder" 
le = LabelEncoder()
y_encoded = le.fit(Y)
#Saving The Intelligence of label encoder
joblib.dump(le,"label_Encoded_Intelligence.pkl")
#Splitting Crop Data into training And testing by 7:3 Ratio
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=42,
    stratify=Y
)
y_train_Encoded=le.transform(y_train)
y_test_Encoded=le.transform(y_test)
#Model Training
random_Forest_Model=RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
random_Forest_Model.fit(X_train,y_train_Encoded)
#Evaluate Model By Metrices
def evaluate_model(model,X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print("Train Accuracy:",100* accuracy_score(y_train, y_train_pred))
    print("Test Accuracy:", 100* accuracy_score(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
evaluate_model(random_Forest_Model,X_train,X_test,y_train_Encoded,y_test_Encoded)

#Cross Validation For Knowing the crop model perform is constiency or Knowing(Underfit/Overfit)
scores = cross_val_score(
    random_Forest_Model,
    X_train,
    y_train_Encoded,
    cv=5,
    scoring='accuracy'
)

print("CV Scores:", scores)
print("Mean Accuracy:", scores.mean())
print("Std Dev:", scores.std())

joblib.dump(random_Forest_Model,"Random_forest_model_Intelligence.pkl")
