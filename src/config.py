import os
import glob

# Modello Ollama da utilizzare (assicurati che sia tra quelli di 'ollama list')
# list of models
## qwen2.5:14b                  7cdf5a0187d5    9.0 GB
## llama3.1:latest              46e0c10c039e    4.9 GB
## llama3.2:3b-instruct-q4_0    9b9453afbdd6    1.9 GB
## gemma:2b                     b50d6c999e59    1.7 GB
## llama3.2:3b                  a80c4f17acd5    2.0 GB
## llama3.2:1b                  baf6a787fdff    1.3 GB
## tinyllama:latest             2644915ede35    637 MB
## phi3:latest                  4f2222927938    2.2 GB
OLLAMA_MODEL = "qwen2.5:14b"

# Percorso relativo del file dati
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def get_dynamic_data_path() -> str:
    """
    Cerca automaticamente il primo file valido (CSV, Excel, JSON, Parquet)
    presente nella cartella 'data/'.
    """
    estensioni_valide = ["*.csv", "*.xlsx", "*.xls", "*.json", "*.parquet"]
    file_trovati = []

    for ext in estensioni_valide:
        file_trovati.extend(glob.glob(os.path.join(DATA_DIR, ext)))

    # Filtra eventuali file temporanei o nascosti
    file_validi = [f for f in file_trovati if not os.path.basename(f).startswith(".")]

    if not file_validi:
        raise FileNotFoundError(f"[-] Nessun file di dati trovato nella cartella '{DATA_DIR}'.")

    # Se ci sono più file, prende il primo (o quello modificato più di recente)
    file_selezionato = file_validi[0]
    print(f"[+] File rilevato ed elaborato: {os.path.basename(file_selezionato)}")
    return file_selezionato