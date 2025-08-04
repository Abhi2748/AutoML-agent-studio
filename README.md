# 🧠 AutoML Agent Studio

A fully automated, LLM-powered machine learning pipeline with real-time model training, evaluation, and prediction.

## 🚀 Features

- 📤 Upload any CSV dataset
- 🧠 LangChain agent determines task type (classification/regression)
- 🧹 Auto preprocessing: imputation, encoding, scaling
- 🤖 Auto model selection (Logistic/Linear Regression)
- 📈 Model evaluation (accuracy or RMSE)
- 🔮 Predict new samples via API
- 🐳 Dockerized + ready for deployment

## 🛠 Tech Stack

- Python, FastAPI
- scikit-learn, pandas, joblib
- LangChain + OpenAI
- Docker

## 🐳 Run with Docker

```bash
docker build -t automl-agent-studio .
docker run -p 8000:8000 automl-agent-studio
