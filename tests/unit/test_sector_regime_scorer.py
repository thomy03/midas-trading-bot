"""Tests for SectorRegimeScorer."""
import sys
import importlib.util
import pytest

# Direct file import to avoid src.agents.__init__ which pulls heavy deps
_spec = importlib.util.spec_from_file_location(
    "sector_regime_scorer",
    "src/agents/sector_regime_scorer.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sector_regime_scorer"] = _mod
_spec.loader.exec_module(_mod)

SectorRegimeScorer = _mod.SectorRegimeScorer
SectorCategory = _mod.SectorCategory
SECTOR_CATEGORY_MAP = _mod.SECTOR_CATEGORY_MAP
REGIME_SECTOR_ADJUSTMENTS = _mod.REGIME_SECTOR_ADJUSTMENTS
get_sector = _mod.get_sector
get_symbol_sector_map = _mod.get_symbol_sector_map
_market_cap_adjustment = _mod._market_cap_adjustment


class TestSectorCategories:
    """Test sector classification."""

    def test_tech_is_growth(self):
        assert SECTOR_CATEGORY_MAP['Technology'] == SectorCategory.GROWTH

    def test_utilities_is_defensive(self):
        assert SECTOR_CATEGORY_MAP['Utilities'] == SectorCategory.DEFENSIVE

    def test_healthcare_is_defensive(self):
        assert SECTOR_CATEGORY_MAP['Healthcare'] == SectorCategory.DEFENSIVE

    def test_consumer_cyclical_is_cyclical(self):
        assert SECTOR_CATEGORY_MAP['Consumer Cyclical'] == SectorCategory.CYCLICAL

    def test_financials_is_neutral(self):
        assert SECTOR_CATEGORY_MAP['Financial Services'] == SectorCategory.NEUTRAL

    def test_all_yfinance_sectors_mapped(self):
        """All common yfinance sectors should be in the map."""
        expected = [
            'Technology', 'Healthcare', 'Financial Services',
            'Consumer Cyclical', 'Consumer Defensive', 'Energy',
            'Industrials', 'Basic Materials', 'Communication Services',
            'Utilities', 'Real Estate',
        ]
        for sector in expected:
            assert sector in SECTOR_CATEGORY_MAP, f"{sector} not mapped"


class TestRegimeAdjustments:
    """Test the bonus/malus values per regime."""

    def test_bull_favors_growth(self):
        adj = REGIME_SECTOR_ADJUSTMENTS['BULL']
        assert adj[SectorCategory.GROWTH] > 0
        assert adj[SectorCategory.DEFENSIVE] < 0

    def test_bear_favors_defensive(self):
        adj = REGIME_SECTOR_ADJUSTMENTS['BEAR']
        assert adj[SectorCategory.DEFENSIVE] > 0
        assert adj[SectorCategory.GROWTH] < 0

    def test_volatile_strongly_favors_defensive(self):
        adj = REGIME_SECTOR_ADJUSTMENTS['VOLATILE']
        assert adj[SectorCategory.DEFENSIVE] > adj[SectorCategory.GROWTH]
        assert adj[SectorCategory.DEFENSIVE] >= 7

    def test_range_is_neutral(self):
        adj = REGIME_SECTOR_ADJUSTMENTS['RANGE']
        for cat in SectorCategory:
            assert adj[cat] == 0

    def test_adjustments_are_moderate(self):
        """No adjustment should exceed +/- 10 to avoid overfitting."""
        for regime, adjustments in REGIME_SECTOR_ADJUSTMENTS.items():
            for cat, val in adjustments.items():
                assert -10 <= val <= 10, f"{regime}/{cat.value} = {val} is too extreme"


class TestMarketCapAdjustment:
    """Test market cap bonus/malus."""

    def test_bull_small_cap_bonus(self):
        adj = _market_cap_adjustment(1.5, 'BULL')  # $1.5B
        assert adj > 0

    def test_bull_mega_cap_no_bonus(self):
        adj = _market_cap_adjustment(200, 'BULL')  # $200B
        assert adj == 0

    def test_bear_small_cap_penalty(self):
        adj = _market_cap_adjustment(1.5, 'BEAR')
        assert adj < 0

    def test_bear_mega_cap_bonus(self):
        adj = _market_cap_adjustment(200, 'BEAR')
        assert adj > 0

    def test_volatile_mega_cap_safe(self):
        adj = _market_cap_adjustment(200, 'VOLATILE')
        assert adj > 0

    def test_volatile_small_cap_dangerous(self):
        adj = _market_cap_adjustment(1.5, 'VOLATILE')
        assert adj < 0

    def test_range_mostly_neutral(self):
        adj = _market_cap_adjustment(50, 'RANGE')
        assert adj == 0

    def test_range_small_penalty(self):
        adj = _market_cap_adjustment(0.5, 'RANGE')
        assert adj < 0

    def test_zero_cap_no_adjustment(self):
        assert _market_cap_adjustment(0, 'BULL') == 0


class TestSymbolSectorMap:
    """Test the static symbol -> sector mapping."""

    def test_aapl_is_tech(self):
        m = get_symbol_sector_map()
        assert m.get('AAPL') == 'Technology'

    def test_jnj_is_healthcare(self):
        m = get_symbol_sector_map()
        assert m.get('JNJ') == 'Healthcare'

    def test_jpm_is_financials(self):
        m = get_symbol_sector_map()
        assert m.get('JPM') == 'Financial Services'

    def test_ko_is_consumer_defensive(self):
        m = get_symbol_sector_map()
        assert m.get('KO') == 'Consumer Defensive'

    def test_xom_is_energy(self):
        m = get_symbol_sector_map()
        assert m.get('XOM') == 'Energy'

    def test_nee_is_utilities(self):
        m = get_symbol_sector_map()
        assert m.get('NEE') == 'Utilities'

    def test_map_has_hundreds_of_entries(self):
        m = get_symbol_sector_map()
        assert len(m) > 500

    def test_get_sector_from_map(self):
        assert get_sector('NVDA') == 'Technology'

    def test_get_sector_unknown_symbol(self):
        assert get_sector('ZZZZZ') == 'Unknown'

    def test_get_sector_from_info_fallback(self):
        assert get_sector('ZZZZZ', info={'sector': 'Energy'}) == 'Energy'


class TestSectorRegimeScorer:
    """Test the main scorer class."""

    def test_tech_in_bull_positive(self):
        scorer = SectorRegimeScorer(enable_market_cap_adj=False)
        adj = scorer.get_adjustment('AAPL', regime='BULL')
        assert adj > 0

    def test_tech_in_bear_negative(self):
        scorer = SectorRegimeScorer(enable_market_cap_adj=False)
        adj = scorer.get_adjustment('AAPL', regime='BEAR')
        assert adj < 0

    def test_utility_in_bear_positive(self):
        scorer = SectorRegimeScorer(enable_market_cap_adj=False)
        adj = scorer.get_adjustment('NEE', regime='BEAR')
        assert adj > 0

    def test_utility_in_bull_negative(self):
        scorer = SectorRegimeScorer(enable_market_cap_adj=False)
        adj = scorer.get_adjustment('NEE', regime='BULL')
        assert adj < 0

    def test_range_is_zero(self):
        scorer = SectorRegimeScorer(enable_market_cap_adj=False)
        adj = scorer.get_adjustment('AAPL', regime='RANGE')
        assert adj == 0

    def test_unknown_symbol_returns_zero(self):
        scorer = SectorRegimeScorer()
        adj = scorer.get_adjustment('ZZZZZ', regime='BULL')
        assert adj == 0

    def test_with_market_cap(self):
        scorer = SectorRegimeScorer(enable_market_cap_adj=True)
        # Tech mega cap in BULL = sector bonus + no cap bonus
        adj_mega = scorer.get_adjustment('AAPL', regime='BULL', market_cap=2.8e12)
        # Tech small cap in BULL = sector bonus + cap bonus
        adj_small = scorer.get_adjustment('AAPL', regime='BULL', market_cap=1.5e9)
        assert adj_small > adj_mega

    def test_sector_override(self):
        scorer = SectorRegimeScorer(enable_market_cap_adj=False)
        adj = scorer.get_adjustment('AAPL', regime='BEAR', sector='Utilities')
        assert adj > 0  # Treated as utility despite being AAPL

    def test_batch_adjustment(self):
        scorer = SectorRegimeScorer(enable_market_cap_adj=False)
        results = scorer.get_adjustment_batch(
            ['AAPL', 'NEE', 'JPM'], regime='BULL'
        )
        assert results['AAPL'] > 0     # Growth bonus
        assert results['NEE'] < 0      # Defensive penalty
        assert results['JPM'] == 0     # Neutral

    def test_total_adjustment_bounded(self):
        """Combined sector + market cap should be reasonable."""
        scorer = SectorRegimeScorer(enable_market_cap_adj=True)
        for regime in ('BULL', 'BEAR', 'RANGE', 'VOLATILE'):
            adj = scorer.get_adjustment('AAPL', regime=regime, market_cap=1e9)
            assert -15 <= adj <= 15, f"Adjustment {adj} in {regime} too extreme"


class TestIntegrationWithScore:
    """Test that adjustments work correctly when applied to scores."""

    def test_bull_tech_improves_score(self):
        scorer = SectorRegimeScorer(enable_market_cap_adj=False)
        base_score = 65.0
        adj = scorer.get_adjustment('NVDA', regime='BULL')
        final = max(0, min(100, base_score + adj))
        assert final > base_score

    def test_bear_tech_reduces_score(self):
        scorer = SectorRegimeScorer(enable_market_cap_adj=False)
        base_score = 65.0
        adj = scorer.get_adjustment('NVDA', regime='BEAR')
        final = max(0, min(100, base_score + adj))
        assert final < base_score

    def test_score_stays_in_bounds(self):
        scorer = SectorRegimeScorer(enable_market_cap_adj=True)
        for base in (0, 5, 50, 95, 100):
            for regime in ('BULL', 'BEAR', 'VOLATILE'):
                adj = scorer.get_adjustment('NVDA', regime=regime, market_cap=1e9)
                final = max(0, min(100, base + adj))
                assert 0 <= final <= 100
