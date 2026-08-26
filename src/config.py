import os
import glob

# Modello Ollama da utilizzare (assicurati che sia tra quelli di 'ollama list')
OLLAMA_MODEL = "phi3:latest" # Oppure "llama3.2:3b-instruct-q4_0"

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