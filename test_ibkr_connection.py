"""
Test de connexion IBKR - TradingBot V4.1
"""
import asyncio
import sys

async def test_connection():
    print("=" * 50)
    print("TEST CONNEXION IBKR")
    print("=" * 50)

    try:
        from ib_insync import IB
        print("[OK] ib_insync importe")
    except ImportError:
        print("[ERREUR] ib_insync non installe")
        print("  -> pip install ib_insync")
        return False

    # Connexion
    ib = IB()

    try:
        print("\nConnexion a TWS (port 7497)...")
        await ib.connectAsync('127.0.0.1', 7497, clientId=1)
        print("[OK] Connecte a IBKR!")

        # Infos compte
        accounts = ib.managedAccounts()
        print(f"\nCompte(s): {accounts}")

        # Verifier si paper trading
        for acc in accounts:
            if acc.startswith('DU'):
                print(f"[OK] Compte Paper Trading detecte: {acc}")
            else:
                print(f"[WARN] Compte Live detecte: {acc}")

        # Solde
        summary = ib.accountSummary()
        for item in summary:
            if item.tag in ['TotalCashValue', 'NetLiquidation', 'BuyingPower']:
                print(f"  {item.tag}: {item.value} {item.currency}")

        # Test: recuperer le prix d'une action
        print("\nTest recuperation prix AAPL...")
        from ib_insync import Stock
        contract = Stock('AAPL', 'SMART', 'USD')
        ib.qualifyContracts(contract)

        ticker = ib.reqMktData(contract)
        await asyncio.sleep(2)  # Attendre les donnees

        if ticker.last:
            print(f"[OK] Prix AAPL: ${ticker.last}")
        elif ticker.close:
            print(f"[OK] Prix cloture AAPL: ${ticker.close}")
        else:
            print("[INFO] Donnees de marche non disponibles (marche ferme?)")

        ib.cancelMktData(contract)

        print("\n" + "=" * 50)
        print("[OK] CONNEXION REUSSIE!")
        print("=" * 50)

        ib.disconnect()
        return True

    except ConnectionRefusedError:
        print("\n[ERREUR] Connexion refusee")
        print("  Verifiez que TWS est lance et l'API activee")
        return False
    except Exception as e:
        print(f"\n[ERREUR] {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
