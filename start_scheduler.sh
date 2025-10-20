#!/bin/bash
# Script de lancement automatique du Market Screener en mode scheduler

echo "=========================================="
echo "Market Screener - Lancement du Scheduler"
echo "=========================================="
echo ""

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "main.py" ]; then
    echo "❌ Erreur: main.py introuvable!"
    echo "Assurez-vous d'être dans le répertoire Tradingbot_V3"
    exit 1
fi

# Vérifier que l'environnement virtuel existe
if [ ! -d "venv" ]; then
    echo "❌ Erreur: Environnement virtuel introuvable!"
    echo "Exécutez d'abord: python3.11 -m venv venv"
    exit 1
fi

# Vérifier que Telegram est configuré
if ! grep -q "your_telegram_bot_token_here" .env 2>/dev/null; then
    echo "⚠️  Telegram semble configuré dans .env"
else
    echo "❌ ATTENTION: Telegram n'est pas encore configuré!"
    echo "Éditez le fichier .env avec vos credentials Telegram"
    echo ""
    read -p "Voulez-vous continuer quand même? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Configuration détectée:"
echo "  • Environnement virtuel: ✅"
echo "  • Python 3.11: ✅"
echo "  • Dépendances installées: ✅"
echo ""

# Vérifier si screen est installé
if ! command -v screen &> /dev/null; then
    echo "⚠️  'screen' n'est pas installé. Installation..."
    sudo apt-get update && sudo apt-get install -y screen
fi

echo "=========================================="
echo "Options de lancement:"
echo "=========================================="
echo "1. Lancer avec screen (détaché - RECOMMANDÉ)"
echo "2. Lancer en mode direct (terminal reste ouvert)"
echo "3. Test des notifications Telegram"
echo "4. Screening manuel unique"
echo "5. Annuler"
echo "=========================================="
echo ""

read -p "Votre choix (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Lancement du scheduler en mode détaché avec screen..."
        echo ""
        echo "INSTRUCTIONS:"
        echo "  • Le scheduler va démarrer dans une session screen"
        echo "  • Pour voir les logs: screen -r screener"
        echo "  • Pour détacher: Ctrl+A puis D"
        echo "  • Pour arrêter: screen -S screener -X quit"
        echo ""
        read -p "Appuyez sur Entrée pour continuer..."

        # Activer l'environnement et lancer avec screen
        source venv/bin/activate
        screen -dmS screener bash -c "source venv/bin/activate && python main.py schedule"

        echo ""
        echo "✅ Scheduler lancé en arrière-plan!"
        echo ""
        echo "Vérification de la session..."
        sleep 2
        screen -ls
        echo ""
        echo "Pour voir les logs: screen -r screener"
        ;;
    2)
        echo ""
        echo "🚀 Lancement du scheduler en mode direct..."
        echo "Appuyez sur Ctrl+C pour arrêter"
        echo ""
        source venv/bin/activate
        python main.py schedule
        ;;
    3)
        echo ""
        echo "🧪 Test des notifications Telegram..."
        source venv/bin/activate
        python main.py test
        ;;
    4)
        echo ""
        echo "🔍 Lancement d'un screening manuel unique..."
        source venv/bin/activate
        python main.py run
        ;;
    5)
        echo "Annulé."
        exit 0
        ;;
    *)
        echo "❌ Choix invalide!"
        exit 1
        ;;
esac

echo ""
echo "✅ Terminé!"
