"""
Script pour obtenir votre Chat ID Telegram
Envoyez /start a votre bot TKTradingV4_Bot, puis executez ce script.
"""
import os
import sys

# Charger le token depuis .env
from dotenv import load_dotenv
load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

if not BOT_TOKEN or BOT_TOKEN == 'your_telegram_bot_token_here':
    print("[ERREUR] Token Telegram non configure dans .env")
    sys.exit(1)

print("=" * 50)
print("RECUPERATION CHAT ID TELEGRAM")
print("=" * 50)
print(f"\n1. Ouvrez Telegram")
print(f"2. Cherchez votre bot: @TKTradingV4_Bot")
print(f"3. Envoyez-lui /start")
print(f"4. Attendez quelques secondes...")
print("\nRecuperation en cours...")

try:
    import requests
except ImportError:
    print("[ERREUR] requests non installe. Executez: pip install requests")
    sys.exit(1)

url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

try:
    response = requests.get(url, timeout=10)
    data = response.json()

    if not data.get('ok'):
        print(f"[ERREUR] API Telegram: {data.get('description', 'Erreur inconnue')}")
        sys.exit(1)

    updates = data.get('result', [])

    if not updates:
        print("\n[INFO] Aucun message recu.")
        print("       Assurez-vous d'avoir envoye /start au bot!")
        print("       Puis relancez ce script.")
        sys.exit(0)

    # Prendre le dernier message
    for update in updates:
        message = update.get('message', {})
        chat = message.get('chat', {})
        chat_id = chat.get('id')
        first_name = chat.get('first_name', 'Inconnu')
        username = chat.get('username', '')

        if chat_id:
            print(f"\n[OK] Chat ID trouve!")
            print(f"     Nom: {first_name}")
            if username:
                print(f"     Username: @{username}")
            print(f"\n" + "=" * 50)
            print(f"     VOTRE CHAT ID: {chat_id}")
            print("=" * 50)
            print(f"\nCopiez ce Chat ID et donnez-le moi pour")
            print(f"finaliser la configuration!")
            break

except requests.exceptions.RequestException as e:
    print(f"[ERREUR] Connexion: {e}")
    sys.exit(1)
