import pandas as pd
import joblib
import os

# Load saved model and preprocessor
MODEL_PATH = "app/models/trained_model.pkl"
preprocessor_path = "app/models/preprocessor.pkl"

def predict_with_model(input_data: dict) -> dict:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(preprocessor_path):
        return {"error": "Model or preprocessor not found. Please upload and train first."}

    try:
        # Load model
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(preprocessor_path)


        # Convert input dict to DataFrame
        input_df = pd.DataFrame([input_data])  # assuming single row
        transformed_input = preprocessor.transform(input_df)
        # Preprocess manually to match training (this is simplified)
        # If needed, save & reuse preprocessor later
        prediction = model.predict(transformed_input)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
