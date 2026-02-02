"""
Integration patch for NightlyAuditor to use ParameterOptimizer.

This adds the optimization step to the nightly audit cycle.
Apply by adding the import and modifying run_audit().
"""

# ADD TO IMPORTS at top of nightly_auditor.py:
# ---------------------------------------------
# from src.agents.parameter_optimizer import get_parameter_optimizer, OptimizationReport


# REPLACE run_audit() method with this version:
# ---------------------------------------------

async def run_audit(
    self,
    trades: List[Dict],
    market_data: Optional[Dict] = None
) -> AuditReport:
    """
    Exécute l'audit nocturne avec optimisation des paramètres.

    Args:
        trades: Liste des trades de la journée
        market_data: Données de marché optionnelles

    Returns:
        Rapport d'audit
    """
    logger.info(f"Starting nightly audit for {len(trades)} trades...")

    # 1. Analyser chaque trade
    analyses = []
    for trade in trades:
        analysis = await self._analyze_trade(trade, market_data)
        analyses.append(analysis)

    # 2. Calculer les stats
    wins = [a for a in analyses if a.outcome == TradeOutcome.WIN]
    losses = [a for a in analyses if a.outcome == TradeOutcome.LOSS]

    total_pnl = sum(a.pnl for a in analyses)
    win_rate = len(wins) / len(analyses) if analyses else 0
    avg_win = sum(a.pnl for a in wins) / len(wins) if wins else 0
    avg_loss = sum(a.pnl for a in losses) / len(losses) if losses else 0

    # 3. Identifier les patterns
    new_guidelines = []
    confirmed_guidelines = []
    invalidated_guidelines = []

    # Analyser les trades gagnants
    if wins:
        win_patterns = await self._identify_patterns(wins, is_winning=True)
        for pattern, confidence in win_patterns:
            guideline = self._create_or_update_guideline(
                pattern=pattern,
                is_positive=True,
                trades=[a.symbol for a in wins],
                confidence=confidence
            )
            if guideline:
                if guideline.confirmation_count == 1:
                    new_guidelines.append(guideline)
                else:
                    confirmed_guidelines.append(guideline.id)

    # Analyser les trades perdants
    if losses:
        loss_patterns = await self._identify_patterns(losses, is_winning=False)
        for pattern, confidence in loss_patterns:
            guideline = self._create_or_update_guideline(
                pattern=pattern,
                is_positive=False,
                trades=[a.symbol for a in losses],
                confidence=confidence
            )
            if guideline:
                if guideline.confirmation_count == 1:
                    new_guidelines.append(guideline)
                else:
                    confirmed_guidelines.append(guideline.id)

    # 4. Vérifier les guidelines existantes
    for gid, guideline in list(self._guidelines.items()):
        was_correct = await self._verify_guideline(guideline, analyses)

        if was_correct is True:
            if gid not in confirmed_guidelines:
                confirmed_guidelines.append(gid)
        elif was_correct is False:
            guideline.invalidation_count += 1
            guideline.confidence = max(
                guideline.confidence - self.CONFIDENCE_DECREMENT,
                0
            )
            if guideline.confidence < self.MIN_CONFIDENCE_ACTIVE:
                invalidated_guidelines.append(gid)

    # 5. Générer les recommandations
    recommendations = await self._generate_recommendations(
        analyses, new_guidelines, market_data
    )

    # 6. Ajustements de risque
    risk_adjustments = self._calculate_risk_adjustments(analyses)

    # =========================================================================
    # 7. NEW: Parameter Optimization via ParameterOptimizer
    # =========================================================================
    optimization_report = None
    try:
        from src.agents.parameter_optimizer import get_parameter_optimizer
        
        optimizer = get_parameter_optimizer()
        
        # Record trades with full indicator data
        for trade in trades:
            if trade.get('indicators'):  # Only if indicators available
                pillar_scores = {
                    'technical': trade.get('technical_score', 0),
                    'fundamental': trade.get('fundamental_score', 0),
                    'sentiment': trade.get('sentiment_score', 0),
                    'news': trade.get('news_score', 0),
                    'combined': trade.get('combined_score', 0),
                    'confidence': trade.get('confidence_score', 0)
                }
                weights = trade.get('weights', {
                    'technical': 0.25,
                    'fundamental': 0.25,
                    'sentiment': 0.25,
                    'news': 0.25
                })
                optimizer.record_trade(trade, pillar_scores, trade['indicators'], weights)
        
        # Run optimization
        optimization_report = await optimizer.optimize_from_trades()
        
        # Add optimization insights to recommendations
        if optimization_report:
            # Add winning patterns
            for pattern in optimization_report.winning_patterns[:3]:
                if pattern.confidence.value in ['high', 'medium']:
                    recommendations.append(
                        f"📈 Pattern gagnant: {pattern.description} "
                        f"(win rate: {pattern.win_rate:.0%}, {pattern.total_trades} trades)"
                    )
            
            # Add losing patterns to avoid
            for pattern in optimization_report.losing_patterns[:3]:
                if pattern.confidence.value in ['high', 'medium']:
                    recommendations.append(
                        f"⚠️ Pattern à éviter: {pattern.description} "
                        f"(win rate: {pattern.win_rate:.0%}, {pattern.total_trades} trades)"
                    )
            
            # Add parameter changes
            for change in optimization_report.recommended_changes[:5]:
                if change.confidence.value in ['high', 'medium']:
                    recommendations.append(
                        f"🔧 Suggestion: {change.parameter_name} "
                        f"{change.current_value} → {change.proposed_value} "
                        f"({change.reason})"
                    )
            
            # Apply high-confidence changes automatically
            high_conf_changes = [
                c for c in optimization_report.recommended_changes
                if c.confidence.value == 'high'
            ]
            if high_conf_changes:
                optimizer.apply_optimizations(high_conf_changes)
                recommendations.append(
                    f"✅ Appliqué automatiquement {len(high_conf_changes)} optimisations haute confiance"
                )
        
        logger.info(
            f"Parameter optimization complete: "
            f"{len(optimization_report.winning_patterns)} winning patterns, "
            f"{len(optimization_report.recommended_changes)} suggestions"
        )
        
    except ImportError:
        logger.warning("ParameterOptimizer not available - skipping optimization")
    except Exception as e:
        logger.error(f"Error in parameter optimization: {e}")

    # 8. Créer le rapport
    report = AuditReport(
        date=datetime.now(),
        trades_analyzed=len(analyses),
        wins=len(wins),
        losses=len(losses),
        total_pnl=total_pnl,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        new_guidelines=new_guidelines,
        confirmed_guidelines=confirmed_guidelines,
        invalidated_guidelines=invalidated_guidelines,
        recommendations=recommendations,
        risk_adjustments=risk_adjustments
    )

    # 9. Sauvegarder
    self._save_guidelines()
    self._save_report(report)

    logger.info(
        f"Audit complete: {len(wins)}W/{len(losses)}L, "
        f"PnL: ${total_pnl:.2f}, "
        f"{len(new_guidelines)} new rules, "
        f"{len(confirmed_guidelines)} confirmed"
    )

    return report
