import pandas as pd
from langchain_community.llms import Ollama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from src.config import OLLAMA_MODEL

def load_data(file_path: str) -> pd.DataFrame:
    """Read the csv as a Pandas DataFrame"""
    try:
        df = pd.read_csv(file_path)
        print(f"[+] Data loaded succefully in  {file_path}")
        return df
    except Exception as e:
        print(f"[-] Error readind the file csv: {e}")
        raise

def build_pandas_agent(df: pd.DataFrame):
    """Creates and return the agent with LangChain linked to Ollama and the DataFrame."""
    # 1. Inizializza il modello locale tramite Ollama
    llm = Ollama(model=OLLAMA_MODEL)

    # 2. Istanzia l'agente per l'analisi del DataFrame
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,                # Mostra la catena di ragionamento (CoT) nel terminale
        allow_dangerous_code=True    # Permette all'agente di eseguire codice Python generato
    )
    return agent