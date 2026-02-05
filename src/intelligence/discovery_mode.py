"""
Discovery Mode - Recherche de pépites en régime BULL
Inspiré de Citron Research : trouver les futures stars "under the radar"

S'active UNIQUEMENT en régime BULL pour chercher des small/mid caps avec:
- Early buzz social (X/Twitter via Grok)
- Volume accumulation (smart money)
- Breakout patterns
- Catalyseurs imminents

En régime BEAR/RANGE → Focus sur grosses caps avec 5 piliers classiques
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class DiscoveryRegime(Enum):
    """Régime de découverte basé sur le marché"""
    DISCOVERY = "discovery"  # BULL → cherche pépites
    CLASSIC = "classic"      # BEAR/RANGE → grosses caps, 5 piliers


@dataclass
class DiscoveryCandidate:
    """Un candidat découvert par le Discovery Mode"""
    symbol: str
    company_name: str
    market_cap: float  # en millions
    sector: str
    
    # Scores Discovery (0-100)
    buzz_score: float = 0.0         # Early social buzz
    accumulation_score: float = 0.0  # Volume accumulation pattern
    breakout_score: float = 0.0      # Technical breakout potential
    catalyst_score: float = 0.0      # Upcoming catalysts
    fundamental_score: float = 0.0   # Growth metrics
    
    # Total Discovery Score
    discovery_score: float = 0.0
    
    # Détails
    catalyst_details: str = ""
    buzz_details: str = ""
    confidence: float = 0.0
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass 
class DiscoveryConfig:
    """Configuration du Discovery Mode"""
    # Filtres market cap (en millions USD)
    min_market_cap: float = 500      # Pas trop petit (penny stocks)
    max_market_cap: float = 15000    # Pas les mega caps
    
    # Filtres volume
    min_avg_volume: int = 100000     # Assez liquide
    max_avg_volume: int = 5000000    # Pas trop mainstream
    
    # Filtres coverage
    max_analyst_coverage: int = 15   # Pas trop couvert
    
    # Scoring weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        'buzz': 0.25,           # Early social buzz
        'accumulation': 0.25,   # Volume accumulation
        'breakout': 0.20,       # Technical breakout
        'catalyst': 0.15,       # Upcoming catalysts
        'fundamental': 0.15     # Growth metrics
    })
    
    # Seuils
    min_discovery_score: float = 60.0  # Score minimum pour être candidat
    max_volatility: float = 80.0       # Volatilité max (leçon du backtest)


class DiscoveryMode:
    """
    Mode Discovery pour trouver des pépites en régime BULL.
    
    Logique:
    1. Vérifie le régime de marché (BULL → Discovery, sinon → Classic)
    2. Si Discovery: cherche small/mid caps avec critères spéciaux
    3. Si Classic: utilise les 5 piliers standards sur grosses caps
    """
    
    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()
        self.current_regime = DiscoveryRegime.CLASSIC
        self.candidates: List[DiscoveryCandidate] = []
        self.last_scan: Optional[datetime] = None
        
    def detect_regime(self, market_data: Dict[str, Any]) -> DiscoveryRegime:
        """
        Détecte le régime de marché pour décider du mode.
        
        Args:
            market_data: Données de marché (SPY, VIX, breadth, etc.)
            
        Returns:
            DiscoveryRegime: DISCOVERY si BULL, CLASSIC sinon
        """
        # Récupérer les indicateurs
        spy_above_ema50 = market_data.get('spy_above_ema50', False)
        spy_above_ema200 = market_data.get('spy_above_ema200', False)
        vix = market_data.get('vix', 20)
        market_breadth = market_data.get('breadth_percent', 50)  # % stocks above 50 EMA
        spy_momentum = market_data.get('spy_momentum_20d', 0)
        
        # Conditions BULL (toutes doivent être vraies)
        bull_conditions = [
            spy_above_ema50,                    # SPY au-dessus de EMA50
            spy_above_ema200,                   # SPY au-dessus de EMA200
            vix < 25,                           # VIX pas trop élevé
            market_breadth > 55,                # Majorité des stocks en hausse
            spy_momentum > 0                    # Momentum positif
        ]
        
        is_bull = sum(bull_conditions) >= 4  # Au moins 4/5 conditions
        
        self.current_regime = DiscoveryRegime.DISCOVERY if is_bull else DiscoveryRegime.CLASSIC
        
        logger.info(f"[DISCOVERY] Régime détecté: {self.current_regime.value} "
                   f"(SPY>EMA50={spy_above_ema50}, VIX={vix:.1f}, Breadth={market_breadth:.0f}%)")
        
        return self.current_regime
    
    async def calculate_buzz_score(self, symbol: str, grok_client: Any) -> Tuple[float, str]:
        """
        Calcule le score de buzz social via Grok/X.
        Cherche les mentions "early" avant que ça devienne mainstream.
        
        Returns:
            Tuple[score, details]
        """
        if not grok_client:
            return 0.0, "Grok not available"
            
        try:
            # Query Grok pour les mentions récentes
            query = f"${symbol} stock mentions sentiment last 7 days growth"
            response = await grok_client.search(query)
            
            # Analyser la réponse
            mention_count = response.get('mention_count', 0)
            mention_growth = response.get('mention_growth_7d', 0)  # % croissance
            sentiment = response.get('sentiment', 0.5)  # 0-1
            
            # Score basé sur croissance des mentions (pas le volume absolu)
            # On cherche les stocks qui COMMENCENT à buzzer
            if mention_growth > 100:  # +100% de mentions
                growth_score = 90
            elif mention_growth > 50:
                growth_score = 70
            elif mention_growth > 20:
                growth_score = 50
            else:
                growth_score = 20
                
            # Bonus si sentiment positif
            sentiment_bonus = (sentiment - 0.5) * 40  # -20 à +20
            
            # Malus si trop de mentions (déjà mainstream)
            if mention_count > 10000:
                mainstream_malus = -30
            elif mention_count > 5000:
                mainstream_malus = -15
            else:
                mainstream_malus = 0
                
            score = max(0, min(100, growth_score + sentiment_bonus + mainstream_malus))
            details = f"Mentions: {mention_count}, Growth: {mention_growth:+.0f}%, Sentiment: {sentiment:.2f}"
            
            return score, details
            
        except Exception as e:
            logger.warning(f"[DISCOVERY] Buzz score error for {symbol}: {e}")
            return 0.0, str(e)
    
    def calculate_accumulation_score(self, price_data: Dict[str, Any]) -> float:
        """
        Détecte l'accumulation par les institutionnels (smart money).
        Signes: Volume en hausse sans mouvement de prix significatif.
        
        Args:
            price_data: OHLCV data
            
        Returns:
            Score 0-100
        """
        try:
            volume = price_data.get('volume', [])
            close = price_data.get('close', [])
            
            if len(volume) < 20 or len(close) < 20:
                return 0.0
                
            # Volume moyen récent vs historique
            vol_recent = sum(volume[-5:]) / 5
            vol_history = sum(volume[-20:-5]) / 15
            vol_ratio = vol_recent / vol_history if vol_history > 0 else 1
            
            # Variation de prix
            price_change = (close[-1] / close[-20] - 1) * 100
            
            # Accumulation = volume en hausse MAIS prix stable ou légèrement en hausse
            # (Les institutionnels achètent sans faire monter le prix)
            if vol_ratio > 1.5 and -5 < price_change < 15:
                # Fort volume, prix stable = accumulation
                score = 80 + (vol_ratio - 1.5) * 20
            elif vol_ratio > 1.2 and -3 < price_change < 10:
                score = 60
            elif vol_ratio > 1.0 and price_change > 0:
                score = 40
            else:
                score = 20
                
            return min(100, max(0, score))
            
        except Exception as e:
            logger.warning(f"[DISCOVERY] Accumulation score error: {e}")
            return 0.0
    
    def calculate_breakout_score(self, price_data: Dict[str, Any]) -> float:
        """
        Détecte les patterns de breakout imminents.
        Signes: Consolidation prolongée, compression de volatilité.
        
        Returns:
            Score 0-100
        """
        try:
            high = price_data.get('high', [])
            low = price_data.get('low', [])
            close = price_data.get('close', [])
            
            if len(close) < 60:
                return 0.0
                
            # Range des 20 derniers jours vs 60 jours
            range_20d = (max(high[-20:]) - min(low[-20:])) / close[-20] * 100
            range_60d = (max(high[-60:]) - min(low[-60:])) / close[-60] * 100
            
            # Compression = range récent < range historique
            compression_ratio = range_20d / range_60d if range_60d > 0 else 1
            
            # Prix proche des plus hauts récents
            high_20d = max(high[-20:])
            distance_to_high = (high_20d - close[-1]) / close[-1] * 100
            
            # Score de breakout
            if compression_ratio < 0.5 and distance_to_high < 3:
                # Forte compression + proche des hauts = breakout imminent
                score = 90
            elif compression_ratio < 0.7 and distance_to_high < 5:
                score = 70
            elif compression_ratio < 0.8:
                score = 50
            else:
                score = 30
                
            return min(100, max(0, score))
            
        except Exception as e:
            logger.warning(f"[DISCOVERY] Breakout score error: {e}")
            return 0.0
    
    async def calculate_catalyst_score(self, symbol: str, news_client: Any) -> Tuple[float, str]:
        """
        Détecte les catalyseurs imminents.
        FDA approvals, earnings, M&A rumors, product launches, etc.
        
        Returns:
            Tuple[score, details]
        """
        catalyst_keywords = [
            'FDA approval', 'earnings beat', 'guidance raised',
            'acquisition', 'merger', 'partnership', 'contract',
            'product launch', 'breakthrough', 'patent'
        ]
        
        try:
            if not news_client:
                return 0.0, "News client not available"
                
            # Chercher les news récentes
            news = await news_client.get_news(symbol, days=14)
            
            score = 0
            catalysts_found = []
            
            for article in news:
                title = article.get('title', '').lower()
                for keyword in catalyst_keywords:
                    if keyword.lower() in title:
                        score += 15
                        catalysts_found.append(keyword)
                        break
                        
            score = min(100, score)
            details = f"Catalysts: {', '.join(set(catalysts_found))}" if catalysts_found else "No catalyst found"
            
            return score, details
            
        except Exception as e:
            logger.warning(f"[DISCOVERY] Catalyst score error for {symbol}: {e}")
            return 0.0, str(e)
    
    def calculate_fundamental_score(self, fundamentals: Dict[str, Any]) -> float:
        """
        Score fondamental axé sur la croissance.
        Pour Discovery, on cherche des métriques de croissance élevées.
        
        Returns:
            Score 0-100
        """
        try:
            revenue_growth = fundamentals.get('revenue_growth', 0) * 100  # %
            earnings_growth = fundamentals.get('earnings_growth', 0) * 100
            pe_ratio = fundamentals.get('pe_ratio', 50)
            profit_margin = fundamentals.get('profit_margin', 0) * 100
            
            score = 0
            
            # Revenue growth (le plus important pour discovery)
            if revenue_growth > 50:
                score += 40
            elif revenue_growth > 30:
                score += 30
            elif revenue_growth > 15:
                score += 20
            else:
                score += 10
                
            # P/E raisonnable (pas trop cher)
            if 0 < pe_ratio < 30:
                score += 25
            elif 30 <= pe_ratio < 50:
                score += 15
            elif pe_ratio < 0:  # Non profitable mais en croissance
                score += 10 if revenue_growth > 30 else 5
                
            # Earnings growth
            if earnings_growth > 30:
                score += 20
            elif earnings_growth > 15:
                score += 15
            elif earnings_growth > 0:
                score += 10
                
            # Profit margin
            if profit_margin > 15:
                score += 15
            elif profit_margin > 5:
                score += 10
                
            return min(100, max(0, score))
            
        except Exception as e:
            logger.warning(f"[DISCOVERY] Fundamental score error: {e}")
            return 0.0
    
    def calculate_discovery_score(self, candidate: DiscoveryCandidate) -> float:
        """
        Calcule le score total de découverte.
        
        Returns:
            Score pondéré 0-100
        """
        weights = self.config.weights
        
        score = (
            candidate.buzz_score * weights['buzz'] +
            candidate.accumulation_score * weights['accumulation'] +
            candidate.breakout_score * weights['breakout'] +
            candidate.catalyst_score * weights['catalyst'] +
            candidate.fundamental_score * weights['fundamental']
        )
        
        return round(score, 1)
    
    async def scan_for_candidates(
        self,
        universe: List[str],
        market_data_fetcher: Any,
        grok_client: Any = None,
        news_client: Any = None
    ) -> List[DiscoveryCandidate]:
        """
        Scanne l'univers pour trouver des candidats Discovery.
        
        Args:
            universe: Liste de symboles à scanner
            market_data_fetcher: Client pour les données de marché
            grok_client: Client Grok pour le buzz social
            news_client: Client pour les news
            
        Returns:
            Liste de candidats triés par score
        """
        logger.info(f"[DISCOVERY] Scanning {len(universe)} symbols...")
        
        candidates = []
        
        for symbol in universe:
            try:
                # Récupérer les données
                info = await market_data_fetcher.get_stock_info(symbol)
                price_data = await market_data_fetcher.get_price_history(symbol, period='3mo')
                
                # Filtres de base
                market_cap = info.get('market_cap', 0) / 1e6  # En millions
                avg_volume = info.get('avg_volume', 0)
                volatility = price_data.get('volatility_60d', 50)
                
                # Appliquer les filtres
                if not (self.config.min_market_cap <= market_cap <= self.config.max_market_cap):
                    continue
                if not (self.config.min_avg_volume <= avg_volume <= self.config.max_avg_volume):
                    continue
                if volatility > self.config.max_volatility:
                    continue
                    
                # Créer le candidat
                candidate = DiscoveryCandidate(
                    symbol=symbol,
                    company_name=info.get('name', symbol),
                    market_cap=market_cap,
                    sector=info.get('sector', 'Unknown')
                )
                
                # Calculer les scores
                candidate.buzz_score, candidate.buzz_details = await self.calculate_buzz_score(
                    symbol, grok_client
                )
                candidate.accumulation_score = self.calculate_accumulation_score(price_data)
                candidate.breakout_score = self.calculate_breakout_score(price_data)
                candidate.catalyst_score, candidate.catalyst_details = await self.calculate_catalyst_score(
                    symbol, news_client
                )
                candidate.fundamental_score = self.calculate_fundamental_score(info)
                
                # Score total
                candidate.discovery_score = self.calculate_discovery_score(candidate)
                candidate.confidence = candidate.discovery_score / 100
                
                # Filtrer par score minimum
                if candidate.discovery_score >= self.config.min_discovery_score:
                    candidates.append(candidate)
                    logger.info(f"[DISCOVERY] Found: {symbol} - Score: {candidate.discovery_score:.1f}")
                    
            except Exception as e:
                logger.debug(f"[DISCOVERY] Error scanning {symbol}: {e}")
                continue
                
        # Trier par score décroissant
        candidates.sort(key=lambda c: c.discovery_score, reverse=True)
        
        self.candidates = candidates[:20]  # Top 20
        self.last_scan = datetime.now()
        
        logger.info(f"[DISCOVERY] Found {len(self.candidates)} candidates")
        
        return self.candidates
    
    def get_trading_mode(self, market_regime: str) -> Dict[str, Any]:
        """
        Retourne la configuration de trading selon le régime.
        
        Args:
            market_regime: 'BULL', 'BEAR', 'RANGE', 'VOLATILE'
            
        Returns:
            Configuration pour le trading
        """
        if market_regime == 'BULL':
            return {
                'mode': 'DISCOVERY',
                'description': 'Recherche de pépites small/mid caps',
                'focus': 'small_mid_caps',
                'market_cap_range': (self.config.min_market_cap, self.config.max_market_cap),
                'use_discovery_pillars': True,
                'pillar_weights': {
                    'buzz': 0.25,
                    'accumulation': 0.25,
                    'breakout': 0.20,
                    'catalyst': 0.15,
                    'fundamental': 0.15
                },
                'risk_tolerance': 'medium',
                'position_size_multiplier': 0.8  # Plus prudent sur les small caps
            }
        else:
            return {
                'mode': 'CLASSIC',
                'description': 'Focus grosses capitalisations avec 5 piliers',
                'focus': 'large_caps',
                'market_cap_range': (10000, None),  # > $10B
                'use_discovery_pillars': False,
                'pillar_weights': {
                    'technical': 0.25,
                    'fundamental': 0.20,
                    'sentiment': 0.20,
                    'news': 0.10,
                    'ml': 0.25
                },
                'risk_tolerance': 'low',
                'position_size_multiplier': 1.0
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise l'état du Discovery Mode"""
        return {
            'current_regime': self.current_regime.value,
            'last_scan': self.last_scan.isoformat() if self.last_scan else None,
            'candidates_count': len(self.candidates),
            'top_candidates': [
                {
                    'symbol': c.symbol,
                    'name': c.company_name,
                    'score': c.discovery_score,
                    'market_cap_m': c.market_cap,
                    'sector': c.sector
                }
                for c in self.candidates[:5]
            ],
            'config': {
                'min_market_cap': self.config.min_market_cap,
                'max_market_cap': self.config.max_market_cap,
                'min_score': self.config.min_discovery_score
            }
        }


# =========================================================================
# FACTORY
# =========================================================================

_discovery_mode: Optional[DiscoveryMode] = None

def get_discovery_mode(config: Optional[DiscoveryConfig] = None) -> DiscoveryMode:
    """Factory pour obtenir l'instance unique du Discovery Mode"""
    global _discovery_mode
    if _discovery_mode is None:
        _discovery_mode = DiscoveryMode(config)
    return _discovery_mode


if __name__ == "__main__":
    # Test basique
    dm = get_discovery_mode()
    
    # Simuler des données de marché BULL
    market_data = {
        'spy_above_ema50': True,
        'spy_above_ema200': True,
        'vix': 15,
        'breadth_percent': 65,
        'spy_momentum_20d': 5.0
    }
    
    regime = dm.detect_regime(market_data)
    print(f"Régime détecté: {regime.value}")
    
    config = dm.get_trading_mode('BULL')
    print(f"Mode de trading: {config['mode']}")
    print(f"Description: {config['description']}")
