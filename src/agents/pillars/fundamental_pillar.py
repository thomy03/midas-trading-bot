"""
Fundamental Pillar - Redirects to Adaptive Fundamental Pillar V8.1.

This file maintains backward compatibility. The actual implementation
is in adaptive_fundamental_pillar.py.
"""
from .adaptive_fundamental_pillar import (
    AdaptiveFundamentalPillar as FundamentalPillar,
    get_adaptive_fundamental_pillar as get_fundamental_pillar,
)

__all__ = ["FundamentalPillar", "get_fundamental_pillar"]
