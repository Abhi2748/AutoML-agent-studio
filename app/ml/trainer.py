import numpy as np
from sklearn.metrics import accuracy_score, root_mean_squared_error
import joblib
import os

def train_and_evaluate_model(
    model, X_train, X_test, y_train, y_test, task_type: str, preprocessor
):
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        if task_type == "classification":
            score = accuracy_score(y_test, y_pred)
            metric = "accuracy"
        else:
            score = root_mean_squared_error(y_test, y_pred)
            metric = "rmse"

        # ✅ Save both model and preprocessor
        save_dir = "app/models"
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "trained_model.pkl")
        preprocessor_path = os.path.join(save_dir, "preprocessor.pkl")

        print(f"Saving model to: {model_path}")
        print(f"Saving preprocessor to: {preprocessor_path}")

        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        # Check if saved
        print("Model saved:", os.path.exists(model_path))
        print("Preprocessor saved:", os.path.exists(preprocessor_path))
        
        return {
            "metric": metric,
            "score": round(score, 4),
            "model_saved": os.path.exists(model_path) and os.path.exists(preprocessor_path)
        }
    except Exception as e:
        return {"error": f"Training failed: {str(e)}"}
