import os
import sys
from src.config import get_dynamic_data_path
from src.agent import load_data, build_pandas_agent

def main():
    print("=== LOCAL DATA AGENT CHAT ===")
    
    # 1. Trova il file automaticamente senza specificare il nome
    try:
        data_path = get_dynamic_data_path()
        # 2. Carica i dati e costruisci l'agente
        df = load_data(data_path)
        agent = build_pandas_agent(df)
    except Exception as e:
        print(e)
        return

    print("\n[!] Agente pronto! Scrivi 'esci' o 'exit' per terminare la sessione.\n")

    # 3. Loop interattivo di domande e risposte
    while True:
        try:
            domanda = input("\n[Tu] > ").strip()
            
            if domanda.lower() in ['esci', 'exit', 'q']:
                print("\nArrivederci!")
                break
                
            if not domanda:
                continue

            print("\n--- L'AGENTE STA RAGIONANDO ---")
            risposta = agent.invoke(domanda)
            
            print("\n--- RISPOSTA AGENTE ---")
            print(risposta['output'])
            
        except KeyboardInterrupt:
            print("\nSessione interrotta.")
            break
        except Exception as e:
            print(f"\n[-] Si è verificato un errore: {e}")

if __name__ == "__main__":
    main()