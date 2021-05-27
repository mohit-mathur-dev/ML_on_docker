import numpy as np
import pandas as pd
import joblib

df = pd.read_csv("dataset.csv")

X = df['hrs'].values.reshape(-1,1)
y = df['marks']

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fir(x, y)

print("Prediction of 6.5 hrs study", model.predict([[6.5]]))

joblib.dump(model, 'slr_model.pkl')
