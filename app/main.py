from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.utils.file_handler import save_and_load_csv
from app.agents.task_identifier import identify_ml_task_and_preprocessing
from app.ml.preprocessor import preprocess_and_select_model
from app.ml.trainer import train_and_evaluate_model
from app.ml.predictor import predict_with_model
from pydantic import BaseModel

app = FastAPI()

class InputFeatures(BaseModel):
    data: dict


# Optional CORS middleware (for frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to AutoML Agent Studio"}


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"error": "Only CSV files are supported"}
    
    df, error = save_and_load_csv(file)
    if error:
        return {"error": f"Failed to read CSV: {error}"}
    
    # Agent: classification or regression + preprocessing hints
    agent_response = identify_ml_task_and_preprocessing(df)
    task_type = agent_response["task_type"]

    # Preprocessing and model selection
    try:
        X_train, X_test, y_train, y_test, model, preprocessor = preprocess_and_select_model(df, task_type)
    except Exception as e:
        return {"error": f"Preprocessing failed: {str(e)}"}

    # Training and evaluation
    try:
        training_result = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, task_type, preprocessor)
    except Exception as e:
        return {"error": f"Training failed: {str(e)}"}

    # Final response
    return {
        "filename": file.filename,
        "task_type": task_type,
        "model": type(model).__name__,
        "preprocessing_done": True,
        "columns_used": df.columns.tolist(),
        "X_train_shape": X_train.shape,
        "y_train_shape": y_train.shape,
        "preprocessing_suggestions": agent_response["preprocessing"],
        "evaluation_metric": training_result["metric"],
        "evaluation_score": training_result["score"],
        "model_saved": training_result["model_saved"]
    }


@app.post("/predict")
async def predict(input: InputFeatures):
    result = predict_with_model(input.data)
    return result
