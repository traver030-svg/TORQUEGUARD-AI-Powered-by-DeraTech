from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model and features
model = joblib.load("model/model.pkl")
feature_names = joblib.load("model/features.pkl")

@app.get("/")
def home():
    return {"message": "TorqueGuard AI is running 🚀"}

@app.post("/predict")
def predict(
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float
):

    input_data = {
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }

    input_df = pd.DataFrame([input_data])

    # Align with training features
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(input_df)[0]

    result = (
        "Machine Failure Likely ⚠️"
        if prediction == 1
        else "Machine Operating Normally ✅"
    )

    return {
        "prediction": int(prediction),
        "result": result
    }