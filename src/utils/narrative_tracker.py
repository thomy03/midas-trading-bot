"""
Market Narrative Tracker
Capture les narratifs dominants du marché par période temporelle.

Narratifs historiques:
- 2024-25: IA / Intelligence Artificielle (NVDA, AMD, MSFT, GOOGL, META)
- 2023-24: Crypto / Bitcoin (crypto-related)
- 2022-23: Pharma / Médicaments anti-obésité (LLY, NVO, AMGN)
- 2020-21: Crypto boom + Work from home
- 2018-19: Réseaux sociaux (META, SNAP, TWTR, PINS)
- 2016-17: FAANG growth

Usage:
    from src.utils.narrative_tracker import narrative_tracker

    # Vérifier si un symbole est aligné avec le narratif actuel
    is_aligned = narrative_tracker.is_symbol_aligned('NVDA', '2024-06-15')

    # Obtenir le boost de score pour un symbole
    boost = narrative_tracker.get_narrative_boost('NVDA', '2024-06-15')

    # Filtrer une liste de signaux par narratif
    filtered = narrative_tracker.filter_signals_by_narrative(signals, as_of_date)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, date
from enum import Enum


class MarketNarrative(Enum):
    """Narratifs de marché identifiés."""
    AI_REVOLUTION = "ai_revolution"           # 2024-2025: IA, LLM, semiconducteurs
    CRYPTO_BULL = "crypto_bull"               # 2023-2024 et 2020-2021: Bitcoin, crypto
    PHARMA_OBESITY = "pharma_obesity"         # 2022-2023: GLP-1, Ozempic, Wegovy
    WORK_FROM_HOME = "work_from_home"         # 2020-2021: Remote work, e-commerce
    SOCIAL_MEDIA = "social_media"             # 2018-2019: Réseaux sociaux
    FAANG_GROWTH = "faang_growth"             # 2016-2017: FAANG domination
    EV_CLEANTECH = "ev_cleantech"             # 2020-2021: Véhicules électriques
    METAVERSE = "metaverse"                   # 2021-2022: Metaverse, VR/AR
    FINTECH = "fintech"                       # 2019-2021: Paiements digitaux
    NONE = "none"                             # Pas de narratif dominant


@dataclass
class NarrativePeriod:
    """Définition d'une période de narratif."""
    narrative: MarketNarrative
    start_date: str  # Format YYYY-MM-DD
    end_date: str    # Format YYYY-MM-DD
    description: str
    strength: float  # 0.0 à 1.0 - force du narratif

    def contains_date(self, check_date) -> bool:
        """Vérifie si une date est dans cette période."""
        import pandas as pd

        # Convert check_date to date object if needed
        if isinstance(check_date, pd.Timestamp):
            check_date = check_date.date()
        elif isinstance(check_date, datetime):
            check_date = check_date.date()
        elif isinstance(check_date, str):
            check_date = datetime.strptime(check_date, "%Y-%m-%d").date()

        start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        return start <= check_date <= end


@dataclass
class NarrativeBoost:
    """Résultat d'un boost de narratif."""
    is_aligned: bool
    narrative: MarketNarrative
    boost_pct: float  # Boost en pourcentage (0-30%)
    themes: List[str]
    description: str


class NarrativeTracker:
    """
    Tracker de narratifs de marché.

    Identifie les thèmes dominants par période et permet de:
    - Booster les scores des actions alignées avec le narratif
    - Filtrer les signaux par thème
    """

    # Définition des périodes de narratifs
    NARRATIVE_PERIODS: List[NarrativePeriod] = [
        # 2024-2025: Révolution IA
        NarrativePeriod(
            narrative=MarketNarrative.AI_REVOLUTION,
            start_date="2024-01-01",
            end_date="2025-12-31",
            description="Révolution IA: LLMs, Semiconducteurs, Cloud AI",
            strength=1.0
        ),
        # 2023-2024: Crypto recovery
        NarrativePeriod(
            narrative=MarketNarrative.CRYPTO_BULL,
            start_date="2023-10-01",
            end_date="2024-03-31",
            description="Rally Bitcoin / Crypto après approbation ETF",
            strength=0.8
        ),
        # 2022-2023: Pharma anti-obésité
        NarrativePeriod(
            narrative=MarketNarrative.PHARMA_OBESITY,
            start_date="2022-06-01",
            end_date="2024-06-30",
            description="GLP-1: Ozempic, Wegovy, Mounjaro",
            strength=0.9
        ),
        # 2021-2022: Metaverse hype
        NarrativePeriod(
            narrative=MarketNarrative.METAVERSE,
            start_date="2021-10-01",
            end_date="2022-06-30",
            description="Metaverse, VR/AR, NFTs",
            strength=0.7
        ),
        # 2020-2021: Crypto boom
        NarrativePeriod(
            narrative=MarketNarrative.CRYPTO_BULL,
            start_date="2020-10-01",
            end_date="2021-11-30",
            description="Bull run crypto / Bitcoin ATH",
            strength=1.0
        ),
        # 2020-2021: Work from home
        NarrativePeriod(
            narrative=MarketNarrative.WORK_FROM_HOME,
            start_date="2020-03-01",
            end_date="2021-12-31",
            description="Remote work, E-commerce, Streaming",
            strength=0.9
        ),
        # 2020-2021: EV / Clean tech
        NarrativePeriod(
            narrative=MarketNarrative.EV_CLEANTECH,
            start_date="2020-06-01",
            end_date="2022-01-31",
            description="Véhicules électriques, Énergie propre",
            strength=0.8
        ),
        # 2019-2021: Fintech
        NarrativePeriod(
            narrative=MarketNarrative.FINTECH,
            start_date="2019-01-01",
            end_date="2021-12-31",
            description="Paiements digitaux, Néobanques",
            strength=0.7
        ),
        # 2018-2019: Social media
        NarrativePeriod(
            narrative=MarketNarrative.SOCIAL_MEDIA,
            start_date="2018-01-01",
            end_date="2019-12-31",
            description="Réseaux sociaux, Publicité digitale",
            strength=0.8
        ),
        # 2016-2017: FAANG growth
        NarrativePeriod(
            narrative=MarketNarrative.FAANG_GROWTH,
            start_date="2016-01-01",
            end_date="2018-12-31",
            description="FAANG domination, Tech growth",
            strength=0.9
        ),
    ]

    # Mapping Narratif -> Symboles concernés
    NARRATIVE_SYMBOLS: Dict[MarketNarrative, Set[str]] = {
        MarketNarrative.AI_REVOLUTION: {
            # Semiconducteurs IA
            'NVDA', 'AMD', 'AVGO', 'MRVL', 'TSM', 'INTC', 'QCOM', 'ARM',
            # Cloud AI
            'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'ORCL', 'CRM',
            # Software IA
            'PLTR', 'AI', 'SNOW', 'DDOG', 'MDB', 'PATH',
            # Autres bénéficiaires
            'SMCI', 'DELL', 'HPE',
        },
        MarketNarrative.CRYPTO_BULL: {
            # Crypto-related stocks
            'COIN', 'MARA', 'RIOT', 'HUT', 'BITF', 'CLSK',
            # Bénéficiaires indirects (hardware)
            'NVDA', 'AMD',
            # Paypal/Square (crypto trading)
            'PYPL', 'SQ',
        },
        MarketNarrative.PHARMA_OBESITY: {
            # Leaders GLP-1
            'LLY', 'NVO',
            # Autres pharma
            'AMGN', 'PFE', 'MRK', 'ABBV', 'BMY',
            # Santé Europe
            'SAN.PA', 'ROG.SW', 'NOVN.SW',
        },
        MarketNarrative.WORK_FROM_HOME: {
            # Video conferencing
            'ZM', 'MSFT', 'GOOGL',
            # E-commerce
            'AMZN', 'SHOP', 'ETSY', 'W', 'CHWY',
            # Streaming
            'NFLX', 'DIS', 'ROKU',
            # Cloud
            'CRM', 'DDOG', 'ZS', 'OKTA', 'CRWD',
            # Hardware
            'AAPL', 'DELL', 'HPQ', 'LOGI',
        },
        MarketNarrative.SOCIAL_MEDIA: {
            'META', 'SNAP', 'PINS', 'TWTR', 'GOOGL',
            # Publicité digitale
            'TTD', 'DV', 'MGNI',
        },
        MarketNarrative.FAANG_GROWTH: {
            'META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL', 'GOOG',
            'MSFT', 'TSLA', 'NVDA',
        },
        MarketNarrative.EV_CLEANTECH: {
            # Véhicules électriques
            'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'FSR',
            # Batteries
            'ALB', 'LTHM', 'SQM',
            # Énergie solaire
            'ENPH', 'SEDG', 'FSLR', 'RUN', 'NOVA',
            # Hydrogène
            'PLUG', 'BLDP', 'BE',
            # Recharge
            'CHPT', 'EVGO', 'BLNK',
        },
        MarketNarrative.METAVERSE: {
            'META', 'RBLX', 'U', 'SNAP',
            # Hardware VR
            'AAPL', 'SONY',
            # Gaming
            'NVDA', 'AMD',
        },
        MarketNarrative.FINTECH: {
            # Paiements
            'V', 'MA', 'PYPL', 'SQ', 'ADYEN.AS',
            # Néobanques
            'SOFI', 'NU', 'UPST', 'AFRM',
            # Crypto trading
            'COIN',
            # Brokers
            'HOOD', 'IBKR', 'SCHW',
        },
        MarketNarrative.NONE: set(),
    }

    # Boost par force de narratif
    MAX_NARRATIVE_BOOST = 25  # Points de boost max

    def __init__(self):
        """Initialise le tracker."""
        self._cache: Dict[str, List[NarrativePeriod]] = {}

    def _parse_date(self, date_input) -> date:
        """Parse une date en objet date."""
        if isinstance(date_input, date):
            return date_input
        if isinstance(date_input, datetime):
            return date_input.date()
        if isinstance(date_input, str):
            return datetime.strptime(date_input[:10], "%Y-%m-%d").date()
        raise ValueError(f"Format de date non reconnu: {date_input}")

    def get_active_narratives(self, as_of_date=None) -> List[NarrativePeriod]:
        """
        Retourne les narratifs actifs pour une date donnée.

        Args:
            as_of_date: Date de référence (défaut: aujourd'hui)

        Returns:
            Liste des NarrativePeriod actifs
        """
        if as_of_date is None:
            check_date = date.today()
        else:
            check_date = self._parse_date(as_of_date)

        active = []
        for period in self.NARRATIVE_PERIODS:
            if period.contains_date(check_date):
                active.append(period)

        # Trier par force décroissante
        active.sort(key=lambda x: x.strength, reverse=True)
        return active

    def get_dominant_narrative(self, as_of_date=None) -> Optional[NarrativePeriod]:
        """
        Retourne le narratif dominant pour une date donnée.

        Args:
            as_of_date: Date de référence

        Returns:
            NarrativePeriod le plus fort ou None
        """
        active = self.get_active_narratives(as_of_date)
        return active[0] if active else None

    def get_narrative_symbols(self, narrative: MarketNarrative) -> Set[str]:
        """
        Retourne les symboles associés à un narratif.

        Args:
            narrative: Type de narratif

        Returns:
            Set de symboles
        """
        return self.NARRATIVE_SYMBOLS.get(narrative, set())

    def is_symbol_aligned(self, symbol: str, as_of_date=None) -> bool:
        """
        Vérifie si un symbole est aligné avec le narratif dominant.

        Args:
            symbol: Symbole boursier
            as_of_date: Date de référence

        Returns:
            True si aligné
        """
        active = self.get_active_narratives(as_of_date)

        for period in active:
            if symbol.upper() in self.NARRATIVE_SYMBOLS.get(period.narrative, set()):
                return True

        return False

    def get_narrative_boost(self, symbol: str, as_of_date=None) -> NarrativeBoost:
        """
        Calcule le boost de narratif pour un symbole.

        Args:
            symbol: Symbole boursier
            as_of_date: Date de référence

        Returns:
            NarrativeBoost avec le boost calculé
        """
        symbol_upper = symbol.upper()
        active = self.get_active_narratives(as_of_date)

        if not active:
            return NarrativeBoost(
                is_aligned=False,
                narrative=MarketNarrative.NONE,
                boost_pct=0.0,
                themes=[],
                description="Pas de narratif dominant"
            )

        # Trouver le meilleur match
        best_boost = 0.0
        best_narrative = None
        matched_themes = []

        for period in active:
            symbols_set = self.NARRATIVE_SYMBOLS.get(period.narrative, set())
            if symbol_upper in symbols_set:
                boost = period.strength * self.MAX_NARRATIVE_BOOST
                if boost > best_boost:
                    best_boost = boost
                    best_narrative = period
                matched_themes.append(period.narrative.value)

        if best_narrative:
            return NarrativeBoost(
                is_aligned=True,
                narrative=best_narrative.narrative,
                boost_pct=best_boost,
                themes=matched_themes,
                description=best_narrative.description
            )
        else:
            # Symbole non aligné - retourne le narratif dominant sans boost
            dominant = active[0]
            return NarrativeBoost(
                is_aligned=False,
                narrative=dominant.narrative,
                boost_pct=0.0,
                themes=[],
                description=f"Non aligné avec: {dominant.description}"
            )

    def filter_signals_by_narrative(
        self,
        signals: List[dict],
        as_of_date=None,
        require_alignment: bool = False,
        boost_scores: bool = True
    ) -> List[dict]:
        """
        Filtre et/ou booste une liste de signaux selon le narratif.

        Args:
            signals: Liste de signaux (dicts avec 'symbol')
            as_of_date: Date de référence
            require_alignment: Si True, ne garde que les signaux alignés
            boost_scores: Si True, ajoute le boost au confidence_score

        Returns:
            Liste de signaux (filtrée et/ou boostée)
        """
        if not signals:
            return []

        result = []

        for signal in signals:
            symbol = signal.get('symbol', '')
            boost_info = self.get_narrative_boost(symbol, as_of_date)

            # Filtrer si requis
            if require_alignment and not boost_info.is_aligned:
                continue

            # Copier le signal
            new_signal = signal.copy()

            # Ajouter les infos de narratif
            new_signal['narrative_aligned'] = boost_info.is_aligned
            new_signal['narrative'] = boost_info.narrative.value
            new_signal['narrative_boost'] = boost_info.boost_pct
            new_signal['narrative_themes'] = boost_info.themes

            # Booster le score si demandé
            if boost_scores and 'confidence_score' in new_signal:
                original_score = new_signal['confidence_score']
                boosted_score = min(100, original_score + boost_info.boost_pct)
                new_signal['confidence_score'] = boosted_score
                new_signal['confidence_score_raw'] = original_score

            result.append(new_signal)

        return result

    def get_period_summary(self, start_date: str, end_date: str) -> Dict:
        """
        Retourne un résumé des narratifs pour une période.

        Args:
            start_date: Date de début (YYYY-MM-DD)
            end_date: Date de fin (YYYY-MM-DD)

        Returns:
            Dict avec statistiques
        """
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)

        # Collecter tous les narratifs actifs sur la période
        narratives_active: Dict[MarketNarrative, int] = {}

        current = start
        while current <= end:
            for period in self.NARRATIVE_PERIODS:
                if period.contains_date(current):
                    if period.narrative not in narratives_active:
                        narratives_active[period.narrative] = 0
                    narratives_active[period.narrative] += 1

            # Avancer d'une semaine pour accélérer
            from datetime import timedelta
            current = current + timedelta(days=7)

        # Calculer le dominant
        if narratives_active:
            dominant = max(narratives_active.items(), key=lambda x: x[1])
            dominant_narrative = dominant[0]
        else:
            dominant_narrative = MarketNarrative.NONE

        return {
            'start_date': start_date,
            'end_date': end_date,
            'dominant_narrative': dominant_narrative.value,
            'narratives_count': len(narratives_active),
            'narratives': {k.value: v for k, v in narratives_active.items()},
            'recommended_symbols': list(self.get_narrative_symbols(dominant_narrative))[:10]
        }

    def print_narrative_summary(self, as_of_date=None):
        """Affiche un résumé des narratifs actifs."""
        active = self.get_active_narratives(as_of_date)

        if as_of_date:
            date_str = self._parse_date(as_of_date).strftime("%Y-%m-%d")
        else:
            date_str = date.today().strftime("%Y-%m-%d")

        print(f"\n{'='*70}")
        print(f"NARRATIFS DE MARCHÉ au {date_str}")
        print(f"{'='*70}")

        if not active:
            print("  Aucun narratif dominant identifié")
            return

        for i, period in enumerate(active, 1):
            symbols = list(self.get_narrative_symbols(period.narrative))[:8]
            symbols_str = ", ".join(symbols) + ("..." if len(self.get_narrative_symbols(period.narrative)) > 8 else "")

            print(f"\n  [{i}] {period.narrative.value.upper()} (Force: {period.strength*100:.0f}%)")
            print(f"      {period.description}")
            print(f"      Période: {period.start_date} → {period.end_date}")
            print(f"      Symboles: {symbols_str}")

        print(f"\n{'='*70}")


# Singleton instance
narrative_tracker = NarrativeTracker()
