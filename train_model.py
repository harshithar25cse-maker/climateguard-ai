import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Example training data
data = {
    "temperature": [30, 35, 40, 45, 28, 33, 38, 42],
    "humidity": [40, 45, 50, 55, 35, 48, 52, 60],
    "wind": [5, 6, 7, 8, 4, 5, 7, 9],
    "pollution": [100, 120, 180, 220, 90, 130, 170, 250],
    "risk": [0, 0, 1, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[["temperature", "humidity", "wind", "pollution"]]
y = df["risk"]

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved as model.pkl")