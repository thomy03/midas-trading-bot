# Injected by Jarvis - signals endpoint with agent support

def register_signals_endpoint(app):
    import json
    from pathlib import Path
    from fastapi import Depends

    @app.get("/api/v1/signals", tags=["Signals"])
    async def get_signals(limit: int = 100, agent: str = "llm"):
        """Get recent signals from live loop"""
        data_dir = "data" if agent == "llm" else "data-nollm"
        signals_path = Path(f"{data_dir}/signals_log.json")
        if not signals_path.exists():
            return {"signals": [], "total": 0}
        try:
            with open(signals_path, "r") as f:
                signals = json.load(f)
            signals = list(reversed(signals[-limit:]))
            return {"signals": signals, "total": len(signals)}
        except Exception:
            return {"signals": [], "total": 0}
