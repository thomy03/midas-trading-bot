# Injected by Jarvis - signals endpoint
# To be included in main.py

def register_signals_endpoint(app):
    import json
    from pathlib import Path
    from fastapi import Depends

    @app.get("/api/v1/signals", tags=["Signals"])
    async def get_signals(limit: int = 100):
        """Get recent signals from live loop"""
        signals_path = Path("data/signals_log.json")
        if not signals_path.exists():
            return {"signals": [], "total": 0}
        try:
            with open(signals_path, "r") as f:
                signals = json.load(f)
            # Return most recent first
            signals = list(reversed(signals[-limit:]))
            return {"signals": signals, "total": len(signals)}
        except Exception:
            return {"signals": [], "total": 0}
