import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load and preprocess data
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
x = df.loc[:, ['time', 'ejection_fraction']].values
y = df.loc[:, "DEATH_EVENT"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=2)

# Train the KNeighborsClassifier model
kn_clf = KNeighborsClassifier(n_neighbors=6)
kn_clf.fit(x_train, y_train)

# Evaluate model (optional, for verification)
kn_pred = kn_clf.predict(x_test)
kn_acc = accuracy_score(y_test, kn_pred)
print("The accuracy of KN Classifier is: " + str(100 * kn_acc) + "%")

# Save the trained model to a file
with open('heart_failure_model.pkl', 'wb') as model_file:
    pickle.dump(kn_clf, model_file)

# Load the saved model
with open('heart_failure_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Define input data model using Pydantic
class HeartFailureInput(BaseModel):
    time: float
    ejection_fraction: float

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: HeartFailureInput):
    try:
        prediction = loaded_model.predict([[input_data.time, input_data.ejection_fraction]])
        return {"prediction": int(prediction[0])}  # Convert prediction to integer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}