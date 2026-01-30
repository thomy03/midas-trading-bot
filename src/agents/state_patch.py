# Patch pour ajouter les méthodes manquantes à AgentState
# À appliquer dans state.py après la définition de AgentState

def get_daily_pnl(self) -> float:
    """Retourne le PnL du jour"""
    today = datetime.now().strftime('%Y-%m-%d')
    for day in self.daily_history:
        if day.get('date') == today:
            return day.get('pnl', 0.0)
    return 0.0

def get_current_drawdown(self) -> float:
    """Calcule le drawdown actuel"""
    if self.peak_capital <= 0:
        return 0.0
    return (self.peak_capital - self.current_capital) / self.peak_capital * 100

def get_open_positions_count(self) -> int:
    """Retourne le nombre de positions ouvertes"""
    return len(self.positions)
