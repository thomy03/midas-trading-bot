#!/usr/bin/env python3
"""Patch nightly_auditor.py to add ParameterOptimizer integration"""

import sys

file_path = sys.argv[1] if len(sys.argv) > 1 else '/root/tradingbot-github/src/agents/nightly_auditor.py'

# Read the file
with open(file_path, 'r') as f:
    content = f.read()

# Check if already patched
if 'get_parameter_optimizer' in content:
    print("Already patched - ParameterOptimizer integration exists")
    sys.exit(0)

# The integration code to insert
integration_code = '''
        # =========================================================================
        # 6.5 NEW: Parameter Optimization via ParameterOptimizer
        # =========================================================================
        try:
            from src.agents.parameter_optimizer import get_parameter_optimizer
            
            optimizer = get_parameter_optimizer()
            
            # Record trades with full indicator data
            for trade in trades:
                indicators = trade.get('indicators', {})
                if not indicators:
                    indicators = {
                        'rsi': trade.get('rsi_at_entry', 50),
                        'volume_ratio': trade.get('volume_ratio', 1.0),
                        'market_regime': trade.get('market_regime', 'neutral'),
                        'vix_level': trade.get('vix_level', 20)
                    }
                
                pillar_scores = {
                    'technical': trade.get('technical_score', 50),
                    'fundamental': trade.get('fundamental_score', 50),
                    'sentiment': trade.get('sentiment_score', 50),
                    'news': trade.get('news_score', 50),
                    'combined': trade.get('combined_score', 50),
                    'confidence': trade.get('confidence_score', trade.get('confidence', 50))
                }
                weights = trade.get('weights', {
                    'technical': 0.25, 'fundamental': 0.25,
                    'sentiment': 0.25, 'news': 0.25
                })
                optimizer.record_trade(trade, pillar_scores, indicators, weights)
            
            # Run optimization analysis
            opt_report = await optimizer.optimize_from_trades()
            
            if opt_report:
                for pattern in opt_report.winning_patterns[:3]:
                    if pattern.confidence.value in ['high', 'medium']:
                        recommendations.append(
                            f"📈 Pattern gagnant: {pattern.description} "
                            f"(win rate: {pattern.win_rate:.0%}, {pattern.total_trades} trades)"
                        )
                
                for pattern in opt_report.losing_patterns[:3]:
                    if pattern.confidence.value in ['high', 'medium']:
                        recommendations.append(
                            f"⚠️ Pattern à éviter: {pattern.description} "
                            f"(win rate: {pattern.win_rate:.0%})"
                        )
                
                for change in opt_report.recommended_changes[:3]:
                    recommendations.append(
                        f"🔧 Suggestion: {change.parameter_name} "
                        f"{change.current_value} → {change.proposed_value} "
                        f"[{change.confidence.value}]"
                    )
                
                high_conf = [c for c in opt_report.recommended_changes if c.confidence.value == 'high']
                if high_conf:
                    optimizer.apply_optimizations(high_conf)
                    logger.info(f"Applied {len(high_conf)} high-confidence parameter changes")
            
            logger.info(f"Parameter optimization: {len(opt_report.winning_patterns)} patterns, {len(opt_report.recommended_changes)} suggestions")
            
        except ImportError:
            logger.debug("ParameterOptimizer not available")
        except Exception as e:
            logger.warning(f"Parameter optimization error: {e}")

'''

# Find the insertion point: after "# 6. Ajustements de risque" block and before "# 7. Créer le rapport"
marker = '        # 7. Créer le rapport'
if marker in content:
    content = content.replace(marker, integration_code + marker)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Successfully patched nightly_auditor.py")
else:
    # Try alternative marker
    marker2 = '        # 7. Créer le rapport\n        report = AuditReport('
    marker3 = 'risk_adjustments = self._calculate_risk_adjustments(analyses)\n\n        # 7.'
    
    if 'risk_adjustments = self._calculate_risk_adjustments(analyses)' in content:
        old = 'risk_adjustments = self._calculate_risk_adjustments(analyses)\n\n        # 7.'
        new = 'risk_adjustments = self._calculate_risk_adjustments(analyses)' + integration_code + '        # 7.'
        content = content.replace(old, new)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ Successfully patched nightly_auditor.py (alt)")
    else:
        print("❌ Could not find insertion point")
        sys.exit(1)
