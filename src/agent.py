import os
import numpy as np
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from src.config import OLLAMA_MODEL

def load_data(file_path: str) -> pd.DataFrame:
    """Read the data as a Pandas DataFrame"""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Estensione '{ext}' non supportata dall'Agente Pandas.")
        
        print(f"[+] Dati caricati con successo da: {os.path.basename(file_path)} ({len(df)} righe)")

        # --- METRICHE IN CODICE (Pre-processing) ---
        # Cerchiamo dinamicamente i prefissi delle metriche (es. 'ua', 'pv', 'ts')
        # basandoci sulle colonne che contengono '_dcr'
        dcr_columns = [col for col in df.columns if '_dcr' in col.lower()]
        
        for dcr_col in dcr_columns:
            # Estraiamo il prefisso (es. da 'ua_dcr' otteniamo 'ua')
            prefix = dcr_col.split('_dcr')[0]
            
            # Cerchiamo la colonna srld corrispondente per questo prefisso
            # (es. 'ua_srld_ANS')
            srld_col_matches = [col for col in df.columns if col.startswith(prefix) and 'srld' in col.lower()]
            
            if srld_col_matches:
                srld_col = srld_col_matches[0]
                
                # Nome per la nuova colonna APE
                ape_col_name = f"ape_{prefix}"
                
                # Calcolo APE: |(DCR - SRLD) / DCR| * 100
                # Usiamo np.where per evitare divisioni per zero (se DCR è 0, mettiamo NaN o 0)
                df[ape_col_name] = np.where(
                    df[dcr_col] != 0, 
                    np.abs((df[dcr_col] - df[srld_col]) / df[dcr_col]) * 100, 
                    np.nan # Mettiamo NaN (Not a Number) così Pandas non lo conta nella media (MAPE)
                )
                print(f"[+] Calcolata colonna '{ape_col_name}' confrontando {dcr_col} e {srld_col}")

        return df
    
    except Exception as e:
        print(f"[-] Errore nel caricamento del file: {e}")
        raise

def build_pandas_agent(df: pd.DataFrame):
    """Creates and return the agent with LangChain linked to Ollama and the DataFrame."""
    # 1. Inizializza il modello locale tramite Ollama
    llm = OllamaLLM(model=OLLAMA_MODEL)

    # --- REGOLE NEL PROMPT ---
    custom_prefix = """
                    Sei un Data Analyst esperto. Il tuo compito è analizzare il DataFrame 'df'.
                    
                    ATTENZIONE - REGOLE DI FORMATTAZIONE RIGIDE (DEVI RISPETTARLE O IL SISTEMA CRASHERA'):
                    Devi rispondere ESATTAMENTE in questo formato passo-passo. 
                    VIETATO usare parentesi quadre per il nome dell'azione.
                    VIETATO usare blocchi di codice markdown (```python) per l'Action Input.

                    Question: la domanda dell'utente
                    Thought: il tuo ragionamento su cosa fare
                    Action: python_repl_ast
                    Action Input: df['ape_ua'].mean()
                    Observation: il risultato del comando
                    Thought: Ora conosco la risposta
                    Final Answer: La risposta finale in italiano.
                    
                    REGOLE SUI DATI:
                    1. L'Errore Percentuale Assoluto (APE) è GIA' CALCOLATO nelle colonne 'ape_ua', 'ape_pv', 'ape_ts'.
                    2. Il MAPE è semplicemente la media dell'APE. Esempio per 'ua': df['ape_ua'].mean()
                    3. Ignora le operazioni non richieste e NON inventare nuove metriche.
                    """

    # 2. Istanzia l'agente per l'analisi del DataFrame
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        handle_parsing_errors=True,
        prefix=custom_prefix,
        verbose=True,                # Mostra la catena di ragionamento (CoT) nel terminale
        allow_dangerous_code=True    # Permette all'agente di eseguire codice Python generato
    )
    return agent