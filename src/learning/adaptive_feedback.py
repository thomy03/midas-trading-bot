"""
Adaptive Feedback Loop - Sprint 1C
Runs daily (called by nightly_auditor or cron).

1. Collects all signals of the day (taken + rejected)
2. Gets actual returns for signals from J-1, J-5, J-20
3. Computes per-indicator attribution
4. Updates learned weights
5. Tracks missed opportunities
6. Saves daily feedback report
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get('MIDAS_DATA_DIR', '/app/data')
SIGNALS_DIR = os.path.join(DATA_DIR, 'signals', 'tracked')
FEEDBACK_DIR = os.path.join(DATA_DIR, 'feedback')
WEIGHTS_DIR = os.path.join(DATA_DIR, 'learned_weights')
RELIABILITY_FILE = os.path.join(DATA_DIR, 'indicator_reliability.json')

LEARNING_RATE = 0.05
MIN_WEIGHT = 0.02
MAX_WEIGHT = 0.25
MISSED_OPP_THRESHOLD = 0.05  # 5%

HORIZONS = [1, 5, 20]

REGIMES = ['bull', 'bear', 'sideways', 'volatile', 'unknown']
SECTOR_GROUPS = ['tech', 'defensive', 'cyclical', 'financial', 'energy', 'other']

SECTOR_TO_GROUP = {
    'Technology': 'tech', 'Communication Services': 'tech',
    'Consumer Defensive': 'defensive', 'Healthcare': 'defensive', 'Utilities': 'defensive',
    'Consumer Cyclical': 'cyclical', 'Industrials': 'cyclical', 'Basic Materials': 'cyclical',
    'Financial Services': 'financial',
    'Energy': 'energy',
    'Real Estate': 'other',
}


def _ensure_dirs():
    for d in [FEEDBACK_DIR, WEIGHTS_DIR, SIGNALS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def _get_sector_group(sector: str) -> str:
    return SECTOR_TO_GROUP.get(sector, 'other')


def _get_return(symbol: str, signal_date: str, horizon: int) -> Optional[float]:
    """Get actual return for symbol, horizon days after signal_date."""
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        
        dt = datetime.strptime(signal_date[:10], '%Y-%m-%d')
        start = dt
        end = dt + timedelta(days=horizon + 5)  # buffer for weekends
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
        
        if len(hist) < 2:
            return None
        
        entry_price = hist['Close'].iloc[0]
        
        # Find the closest trading day to horizon
        target_idx = min(horizon, len(hist) - 1)
        exit_price = hist['Close'].iloc[target_idx]
        
        return (exit_price - entry_price) / entry_price
    except Exception as e:
        logger.debug(f"Could not get return for {symbol} +{horizon}d: {e}")
        return None


def _load_json(path: str, default=None):
    if default is None:
        default = {}
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return default


def _save_json(path: str, data):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def _load_signals_for_date(date_str: str) -> List[Dict]:
    path = os.path.join(SIGNALS_DIR, f'{date_str}.json')
    return _load_json(path, [])


def _load_reliability() -> Dict:
    return _load_json(RELIABILITY_FILE, {})


def _save_reliability(data: Dict):
    _save_json(RELIABILITY_FILE, data)


def _update_indicator_reliability(reliability: Dict, indicator: str, regime: str, 
                                   sector_group: str, was_right: bool):
    """Update reliability score for an indicator in a given context."""
    key = f"{indicator}|{regime}|{sector_group}"
    if key not in reliability:
        reliability[key] = {'correct': 0, 'total': 0, 'score': 0.5}
    
    reliability[key]['total'] += 1
    if was_right:
        reliability[key]['correct'] += 1
    
    total = reliability[key]['total']
    correct = reliability[key]['correct']
    reliability[key]['score'] = correct / total if total > 0 else 0.5


def _get_reliability_score(reliability: Dict, indicator: str, regime: str, sector_group: str) -> float:
    key = f"{indicator}|{regime}|{sector_group}"
    if key in reliability:
        return reliability[key]['score']
    # Fallback: indicator across all contexts
    scores = [v['score'] for k, v in reliability.items() if k.startswith(f"{indicator}|")]
    return sum(scores) / len(scores) if scores else 0.5


def _update_weights(reliability: Dict):
    """Update learned weights based on reliability scores."""
    tech_weights_path = os.path.join(WEIGHTS_DIR, 'technical_weights.json')
    fund_weights_path = os.path.join(WEIGHTS_DIR, 'fundamental_weights.json')
    
    tech_weights = _load_json(tech_weights_path, {})
    fund_weights = _load_json(fund_weights_path, {})
    
    # Get all known indicators from reliability data
    tech_indicators = set()
    fund_indicators = set()
    for key in reliability:
        indicator = key.split('|')[0]
        # Simple heuristic: fundamental indicators
        if indicator in ('pe_ratio', 'pb_ratio', 'fcf_yield', 'debt_equity', 'roe', 
                         'revenue_growth', 'earnings_growth', 'dividend_yield', 'current_ratio'):
            fund_indicators.add(indicator)
        else:
            tech_indicators.add(indicator)
    
    for regime in REGIMES:
        for sg in SECTOR_GROUPS:
            # Technical weights
            if tech_indicators:
                reliabilities = {ind: _get_reliability_score(reliability, ind, regime, sg) 
                                for ind in tech_indicators}
                total_rel = sum(reliabilities.values())
                if total_rel > 0:
                    if regime not in tech_weights:
                        tech_weights[regime] = {}
                    if sg not in tech_weights[regime]:
                        tech_weights[regime][sg] = {}
                    
                    for ind, rel in reliabilities.items():
                        target = rel / total_rel
                        current = tech_weights.get(regime, {}).get(sg, {}).get(ind, target)
                        new_w = current + LEARNING_RATE * (target - current)
                        new_w = max(MIN_WEIGHT, min(MAX_WEIGHT, new_w))
                        tech_weights[regime][sg][ind] = round(new_w, 4)
            
            # Fundamental weights
            if fund_indicators:
                reliabilities = {ind: _get_reliability_score(reliability, ind, regime, sg) 
                                for ind in fund_indicators}
                total_rel = sum(reliabilities.values())
                if total_rel > 0:
                    if regime not in fund_weights:
                        fund_weights[regime] = {}
                    if sg not in fund_weights[regime]:
                        fund_weights[regime][sg] = {}
                    
                    for ind, rel in reliabilities.items():
                        target = rel / total_rel
                        current = fund_weights.get(regime, {}).get(sg, {}).get(ind, target)
                        new_w = current + LEARNING_RATE * (target - current)
                        new_w = max(MIN_WEIGHT, min(MAX_WEIGHT, new_w))
                        fund_weights[regime][sg][ind] = round(new_w, 4)
    
    _save_json(tech_weights_path, tech_weights)
    _save_json(fund_weights_path, fund_weights)
    
    return tech_weights, fund_weights


async def run_feedback(today: Optional[str] = None):
    """
    Main feedback loop entry point. Call once per day.
    
    Args:
        today: Override date (YYYY-MM-DD), defaults to today
    """
    _ensure_dirs()
    
    if today is None:
        today = datetime.utcnow().strftime('%Y-%m-%d')
    
    logger.info(f"[FEEDBACK] Starting adaptive feedback for {today}")
    
    # 1. Collect today's signals
    today_signals = _load_signals_for_date(today)
    taken = [s for s in today_signals if s.get('decision') in ('BUY', 'STRONG_BUY', 'SELL')]
    rejected = [s for s in today_signals if s.get('decision') not in ('BUY', 'STRONG_BUY', 'SELL')]
    
    logger.info(f"[FEEDBACK] Today: {len(taken)} taken, {len(rejected)} rejected signals")
    
    # 2. Evaluate past signals at various horizons
    reliability = _load_reliability()
    
    feedback_results = {}
    for horizon in HORIZONS:
        past_date = (datetime.strptime(today, '%Y-%m-%d') - timedelta(days=horizon)).strftime('%Y-%m-%d')
        past_signals = _load_signals_for_date(past_date)
        
        if not past_signals:
            continue
        
        correct = 0
        incorrect = 0
        evaluated = 0
        
        for signal in past_signals:
            symbol = signal.get('symbol')
            if not symbol:
                continue
            
            actual_return = _get_return(symbol, past_date, horizon)
            if actual_return is None:
                continue
            
            evaluated += 1
            decision = signal.get('decision', '')
            was_correct = (actual_return > 0 and decision in ('BUY', 'STRONG_BUY')) or \
                          (actual_return < 0 and decision == 'SELL')
            
            if decision in ('BUY', 'STRONG_BUY', 'SELL'):
                if was_correct:
                    correct += 1
                else:
                    incorrect += 1
            
            # 3. Per-indicator attribution
            normalized = signal.get('normalized_values', {})
            regime = signal.get('regime', 'unknown')
            sector_group = _get_sector_group(signal.get('sector', ''))
            
            for indicator, norm_val in normalized.items():
                try:
                    norm_val = float(norm_val)
                except (ValueError, TypeError):
                    continue
                indicator_said_buy = norm_val > 60
                indicator_was_right = (indicator_said_buy and actual_return > 0) or \
                                      (not indicator_said_buy and actual_return < 0)
                _update_indicator_reliability(reliability, indicator, regime, sector_group, indicator_was_right)
        
        if evaluated > 0:
            feedback_results[f'horizon_{horizon}d'] = {
                'date_evaluated': past_date,
                'signals_evaluated': evaluated,
                'correct_predictions': correct,
                'incorrect_predictions': incorrect,
                'accuracy': round(correct / (correct + incorrect), 3) if (correct + incorrect) > 0 else None
            }
    
    _save_reliability(reliability)
    
    # 4. Update weights
    tech_weights, fund_weights = _update_weights(reliability)
    
    # 5. Missed opportunities
    missed_opportunities = []
    for horizon in [5]:
        past_date = (datetime.strptime(today, '%Y-%m-%d') - timedelta(days=horizon)).strftime('%Y-%m-%d')
        past_signals = _load_signals_for_date(past_date)
        
        for signal in past_signals:
            if signal.get('decision') in ('BUY', 'STRONG_BUY', 'SELL'):
                continue  # only check rejected
            
            symbol = signal.get('symbol')
            if not symbol:
                continue
            
            actual_return = _get_return(symbol, past_date, 5)
            if actual_return is not None and actual_return > MISSED_OPP_THRESHOLD:
                missed_opportunities.append({
                    'symbol': symbol,
                    'rejected_score': signal.get('score', 0),
                    'actual_return_5d': f"+{actual_return*100:.1f}%",
                    'rejection_reason': signal.get('rejection_reason', 'unknown'),
                    'date': past_date
                })
    
    if missed_opportunities:
        logger.warning(f"[FEEDBACK] {len(missed_opportunities)} missed opportunities detected!")
    
    # 6. Compute top/worst indicators
    indicator_scores = {}
    for key, val in reliability.items():
        ind = key.split('|')[0]
        if ind not in indicator_scores:
            indicator_scores[ind] = []
        indicator_scores[ind].append(val['score'])
    
    avg_scores = {ind: sum(scores)/len(scores) for ind, scores in indicator_scores.items() if scores}
    sorted_inds = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    top_indicators = [ind for ind, _ in sorted_inds[:5]]
    worst_indicators = [ind for ind, _ in sorted_inds[-5:]] if len(sorted_inds) > 5 else []
    
    # 7. Save report
    report = {
        'date': today,
        'signals_evaluated': len(today_signals),
        'signals_taken': len(taken),
        'signals_rejected': len(rejected),
        'feedback_results': feedback_results,
        'top_indicators': top_indicators,
        'worst_indicators': worst_indicators,
        'missed_opportunities': missed_opportunities[:20],
        'indicator_reliability_summary': {ind: round(avg, 3) for ind, avg in avg_scores.items()},
        'weight_files_updated': True,
    }
    
    report_path = os.path.join(FEEDBACK_DIR, f'{today}.json')
    _save_json(report_path, report)
    
    logger.info(f"[FEEDBACK] Report saved to {report_path}")
    logger.info(f"[FEEDBACK] Top indicators: {top_indicators}")
    if missed_opportunities:
        logger.info(f"[FEEDBACK] Missed opps: {len(missed_opportunities)}")
    
    return report
