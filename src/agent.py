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
        else:
            raise ValueError(f"Estensione '{ext}' non supportata.")
        
        print(f"[+] Dati caricati con successo da: {os.path.basename(file_path)} ({len(df)} righe)")

        df["Sub_Brand"].fillna(-1, inplace=True)
        df["SUB_BRAND_NAME"].fillna("all", inplace=True)

        # 1. EREDITARIETÀ SICURA: Riempiamo SOLO i buchi (NaN), senza sovrascrivere i dati esistenti!
        if 'BRAND' in df.columns and 'TOPBrand' in df.columns:
            # Creiamo un dizionario "BRAND -> Primo TOPBrand valido"
            mappa_brand = df.dropna(subset=['TOPBrand']).groupby('BRAND')['TOPBrand'].first()
            # Usiamo fillna per tappare solo i buchi
            df['TOPBrand'] = df['TOPBrand'].fillna(df['BRAND'].map(mappa_brand))
            # Se rimane qualche orfano, lo chiamiamo (blank) per non far scartare la riga al groupby
            df['TOPBrand'] = df['TOPBrand'].fillna('(blank)')
            print("[+] Propagato 'TOPBrand' in modo sicuro sui valori mancanti.")

        # 2. CALCOLO APE: Logica Excel (Tutti gli errori e i vuoti diventano 0)
        dcr_columns = [col for col in df.columns if '_dcr' in col.lower()]
        for dcr_col in dcr_columns:
            prefix = dcr_col.split('_dcr')[0]
            srld_col_matches = [col for col in df.columns if col.startswith(prefix) and 'srld' in col.lower()]
            
            if srld_col_matches:
                srld_col = srld_col_matches[0]
                ape_col_name = f"ape_{prefix}"
                
                # Forziamo in numerico
                dcr_num = pd.to_numeric(df[dcr_col], errors='coerce')
                srld_num = pd.to_numeric(df[srld_col], errors='coerce')
                
                # Calcolo nudo e crudo
                ape_calc = np.abs((dcr_num - srld_num) / dcr_num) * 100
                
                # REPLICA EXCEL: Se c'è divisione per zero o DCR è vuoto, forziamo a 0
                df[ape_col_name] = ape_calc.replace([np.inf, -np.inf, np.nan], 0)
                
                print(f"[+] Calcolata colonna '{ape_col_name}' (modalità Excel con Zeri)")

        return df
    
    except Exception as e:
        print(f"[-] Errore nel caricamento del file: {e}")
        raise

def build_pandas_agent(df: pd.DataFrame):
    """Creates and return the agent with LangChain linked to Ollama and the DataFrame."""

    # Disabilita il troncamento delle righe e colonne in Pandas
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # 1. Inizializza il modello locale tramite Ollama
    llm = OllamaLLM(model=OLLAMA_MODEL)

    # --- REGOLE NEL PROMPT ---
    custom_prefix = """
                                    Sei un Data Analyst esperto. Rispondi alle domande dell'utente analizzando il DataFrame 'df'.
                                    
                                    REGOLE SUI DATI:
                                    1. L'Errore Percentuale Assoluto (APE) è già stato calcolato nelle colonne 'ape_ua', 'ape_pv', 'ape_ts'.
                                    2. Il MAPE (Mean Absolute Percentage Error) è la media semplice dell'APE. Ad esempio, per la 'unique audience' (ua) devi eseguire: df['ape_ua'].mean()
                                    3. Rispondi sempre in italiano, in modo chiaro e preciso.
                                    4. Non appena hai ottenuto un risultato numerico, rispondi.
                                    4.1. Esempio: 'Il MAPE della unique audience è del 12.3%'
                                    5. Quando hai ottenuto il risultato del calcolo numerico, formula subito la risposta finale, con una frase che lo includa in modo naturale.
                                    6. Non speculare troppo sui dati, se non sei sicuro di qualcosa chiedi spiegazioni.
                                    7. Quando fai raggruppamenti complessi (es. groupby con più colonne), assicurati di analizzare TUTTE le righe restituite dal comando Python prima di scrivere la risposta finale.
                                    
                                    REGOLE DI SINTASSI (IMPORTANTE):
                                    Se devi eseguire più righe di codice Python (es. filtrare, poi raggruppare, poi salvare in CSV), scrivile tutte sotto un UNICO 'Action Input' andando semplicemente a capo. 
                                    NON ripetere MAI la dicitura 'Action Input:' per ogni riga. 
                                    Esempio corretto:
                                        Action Input: 
                                        filtered = df[df['col'] > 0]
                                        result = filtered.groupby('X')['Y'].mean()
                                        result.to_csv('data/report.csv')

                                    REGOLE PER I REPORT E SALVATAGGI:
                                    - Se l'utente ti chiede di salvare, esportare o creare un report (CSV o Excel), usa i comandi Pandas come .to_csv() o .to_excel().
                                    - Salva SEMPRE i file generati dentro la cartella 'data/' (es. 'data/report_mape.csv').
                                    - Avvisa l'utente nella risposta finale quando il file è stato creato con successo.

                                    REGOLE DI FORMATTAZIONE OBBLIGATORIE:
                                    Quando hai finito i calcoli o hai salvato un file, la tua risposta all'utente DEVE iniziare SEMPRE e SOLO con la dicitura esatta "Final Answer: ". 
                                    NON iniziare mai a parlare senza aver prima scritto "Final Answer: ".
                                    """

    # 2. Istanzia l'agente per l'analisi del DataFrame
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        handle_parsing_errors=True,
        prefix=custom_prefix,
        max_iterations=10,            # <--- BLOCCO LOOP: si ferma se non risponde in 10 passaggi
        verbose=True,                # Mostra la catena di ragionamento (CoT) nel terminale
        allow_dangerous_code=True    # Permette all'agente di eseguire codice Python generato
    )
    return agent