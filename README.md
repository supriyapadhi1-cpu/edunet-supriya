import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/content/soil_prediction_data.csv")
df.head()
df.info()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Soil_Texture'] = label_encoder.fit_transform(df['Soil_Texture'])
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Soil_Type'] = label_encoder.fit_transform(df['Soil_Type'])
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Soil_Fertility'] = label_encoder.fit_transform(df['Soil_Fertility'])
df.head()
x = df.drop(columns=["Micronutrients"])
x
y = df["Micronutrients"]
y
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.20,random_state = 42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mae = mean_absolute_error(ytest,ypred)
mse = mean_squared_error(ytest,ypred)
r2 = r2_score(ytest,ypred)

#print results
print("Mean absolute Error:",mae)
print("Mean Squared Error:",mse)
print("R-squared Score:",r2)
