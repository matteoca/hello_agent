from src.config import DATA_PATH
from src.agent import load_data, build_pandas_agent

def main():
    print("=== AVVIO LOCAL DATA AGENT ===")
    
    # 1. Carica il file dati
    df = load_data(DATA_PATH)

    # 2. Costruisci l'agente
    agent = build_pandas_agent(df)

    # 3. Definisci la query o chiedi input all'utente
    domanda = "Qual è il totale delle vendite per la categoria Elettronica a Milano nel 2024?"
    print(f"\n[?] Domanda: {domanda}\n")

    # 4. Esegui l'agente
    try:
        risposta = agent.invoke(domanda)
        print("\n--- RISPOSTA FINALE ---")
        print(risposta['output'])
    except Exception as e:
        print(f"\n[-] Errore durante l'esecuzione dell'agente: {e}")

if __name__ == "__main__":
    main()