import joblib
#Loding our Intelligence (Model and Encoder)
Encoder=joblib.load("label_Encoded_Intelligence.pkl")
model=joblib.load("Random_forest_model_Intelligence.pkl")
#Getting User Crop Data to Predict
user_crop_Data=[]
for i in range(5):
    user_Input=float(input("Enter The Crop Data In The Format of 'N,P,K,Humidity,Rainfall':"))
    user_crop_Data.append(user_Input)
user_crop_Data=[user_crop_Data]#Converting 1D to 2D to predict

def predict_crop(data):
    predict=model.predict(data)
    output=Encoder.inverse_transform(predict)#Change Encoded Value into string value
    return output
print(f"The Predicted Crop For Your Soil and Weather Condition is :{predict_crop(user_crop_Data)}")