import joblib

saved_model = joblib.load('slr_model.pkl')

print("Prediction from saves_model:", saved_model.predict([[6.5]]))
