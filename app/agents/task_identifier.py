from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.schema import AgentFinish

load_dotenv()

def identify_ml_task_and_preprocessing(df: pd.DataFrame) -> dict:
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    def df_summary():
        return f"""
        Columns: {df.columns.tolist()}
        Nulls: {df.isnull().sum().to_dict()}
        Data Types: {df.dtypes.to_dict()}
        Target Sample: {df.iloc[:, -1].dropna().unique()[:5].tolist()}
        """

    tools = [
        Tool(
            name="DatasetSummary",
            func=lambda _: df_summary(),
            description="Summarizes dataset columns, types, and missing values"
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True
    )

    # First prompt: task type
    try:
        task = agent.run(
            "Use DatasetSummary tool and answer ONLY with one word: 'classification' or 'regression'."
        )
    except AgentFinish as e:
        task = e.return_values["output"]

    # Second prompt: preprocessing suggestions
    try:
        preprocessing = agent.run(
            "Based on the DatasetSummary tool, suggest preprocessing steps in bullet points."
        )
    except AgentFinish as e:
        preprocessing = e.return_values["output"]

    return {
        "task_type": task.strip().lower(),
        "preprocessing": preprocessing.strip()
    }
