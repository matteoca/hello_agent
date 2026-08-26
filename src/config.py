import os

# Modello Ollama da utilizzare (assicurati che sia tra quelli di 'ollama list')
OLLAMA_MODEL = "phi3:latest" # Oppure "llama3.2:3b-instruct-q4_0"

# Percorso relativo del file dati
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "input.csv")