"""
Watchlist Manager for TradingBot V3

Allows users to create, manage, and scan custom watchlists.
Features:
- Create/delete watchlists
- Add/remove symbols
- Quick scan watchlist for signals
- Persistence to JSON file
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WatchlistManager:
    """Manages user watchlists with persistence"""

    # Default watchlists to create on first run
    DEFAULT_WATCHLISTS = {
        "Tech Leaders": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "Favorites": []
    }

    def __init__(self, data_file: str = "data/watchlists.json"):
        """
        Initialize WatchlistManager

        Args:
            data_file: Path to JSON file for persistence
        """
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.watchlists: Dict[str, List[str]] = {}
        self._load()

    def _load(self):
        """Load watchlists from JSON file"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.watchlists = data.get('watchlists', {})
                    logger.info(f"Loaded {len(self.watchlists)} watchlists")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading watchlists: {e}")
                self.watchlists = self.DEFAULT_WATCHLISTS.copy()
                self._save()
        else:
            # Create default watchlists
            self.watchlists = self.DEFAULT_WATCHLISTS.copy()
            self._save()
            logger.info("Created default watchlists")

    def _save(self):
        """Save watchlists to JSON file"""
        try:
            data = {
                'watchlists': self.watchlists,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Error saving watchlists: {e}")

    def get_all_watchlists(self) -> Dict[str, List[str]]:
        """Get all watchlists"""
        return self.watchlists.copy()

    def get_watchlist(self, name: str) -> Optional[List[str]]:
        """Get a specific watchlist by name"""
        return self.watchlists.get(name, None)

    def get_watchlist_names(self) -> List[str]:
        """Get list of all watchlist names"""
        return list(self.watchlists.keys())

    def create_watchlist(self, name: str, symbols: List[str] = None) -> bool:
        """
        Create a new watchlist

        Args:
            name: Name of the watchlist
            symbols: Optional list of initial symbols

        Returns:
            True if created, False if already exists
        """
        if name in self.watchlists:
            return False

        self.watchlists[name] = symbols or []
        self._save()
        logger.info(f"Created watchlist '{name}' with {len(self.watchlists[name])} symbols")
        return True

    def delete_watchlist(self, name: str) -> bool:
        """
        Delete a watchlist

        Args:
            name: Name of the watchlist to delete

        Returns:
            True if deleted, False if not found
        """
        if name not in self.watchlists:
            return False

        del self.watchlists[name]
        self._save()
        logger.info(f"Deleted watchlist '{name}'")
        return True

    def rename_watchlist(self, old_name: str, new_name: str) -> bool:
        """
        Rename a watchlist

        Args:
            old_name: Current name
            new_name: New name

        Returns:
            True if renamed, False if old not found or new exists
        """
        if old_name not in self.watchlists:
            return False
        if new_name in self.watchlists:
            return False

        self.watchlists[new_name] = self.watchlists.pop(old_name)
        self._save()
        logger.info(f"Renamed watchlist '{old_name}' to '{new_name}'")
        return True

    def add_symbol(self, watchlist_name: str, symbol: str) -> bool:
        """
        Add a symbol to a watchlist

        Args:
            watchlist_name: Name of the watchlist
            symbol: Symbol to add (will be uppercased)

        Returns:
            True if added, False if watchlist not found or symbol exists
        """
        if watchlist_name not in self.watchlists:
            return False

        symbol = symbol.upper().strip()
        if symbol in self.watchlists[watchlist_name]:
            return False

        self.watchlists[watchlist_name].append(symbol)
        self._save()
        logger.info(f"Added '{symbol}' to watchlist '{watchlist_name}'")
        return True

    def remove_symbol(self, watchlist_name: str, symbol: str) -> bool:
        """
        Remove a symbol from a watchlist

        Args:
            watchlist_name: Name of the watchlist
            symbol: Symbol to remove

        Returns:
            True if removed, False if watchlist or symbol not found
        """
        if watchlist_name not in self.watchlists:
            return False

        symbol = symbol.upper().strip()
        if symbol not in self.watchlists[watchlist_name]:
            return False

        self.watchlists[watchlist_name].remove(symbol)
        self._save()
        logger.info(f"Removed '{symbol}' from watchlist '{watchlist_name}'")
        return True

    def add_symbols_bulk(self, watchlist_name: str, symbols: List[str]) -> int:
        """
        Add multiple symbols to a watchlist

        Args:
            watchlist_name: Name of the watchlist
            symbols: List of symbols to add

        Returns:
            Number of symbols actually added
        """
        if watchlist_name not in self.watchlists:
            return 0

        added = 0
        for symbol in symbols:
            symbol = symbol.upper().strip()
            if symbol and symbol not in self.watchlists[watchlist_name]:
                self.watchlists[watchlist_name].append(symbol)
                added += 1

        if added > 0:
            self._save()
            logger.info(f"Added {added} symbols to watchlist '{watchlist_name}'")

        return added

    def get_symbol_count(self, watchlist_name: str) -> int:
        """Get number of symbols in a watchlist"""
        if watchlist_name not in self.watchlists:
            return 0
        return len(self.watchlists[watchlist_name])

    def get_all_symbols(self) -> List[str]:
        """Get all unique symbols across all watchlists"""
        all_symbols = set()
        for symbols in self.watchlists.values():
            all_symbols.update(symbols)
        return sorted(list(all_symbols))

    def find_symbol(self, symbol: str) -> List[str]:
        """
        Find which watchlists contain a symbol

        Args:
            symbol: Symbol to search for

        Returns:
            List of watchlist names containing the symbol
        """
        symbol = symbol.upper().strip()
        return [name for name, symbols in self.watchlists.items() if symbol in symbols]

    def export_watchlist(self, watchlist_name: str) -> Optional[str]:
        """
        Export watchlist as comma-separated string

        Args:
            watchlist_name: Name of the watchlist

        Returns:
            Comma-separated string of symbols, or None if not found
        """
        if watchlist_name not in self.watchlists:
            return None
        return ", ".join(self.watchlists[watchlist_name])

    def import_from_string(self, watchlist_name: str, symbols_str: str) -> int:
        """
        Import symbols from a comma/space/newline separated string

        Args:
            watchlist_name: Name of the watchlist (creates if doesn't exist)
            symbols_str: String of symbols separated by comma, space, or newline

        Returns:
            Number of symbols added
        """
        # Parse symbols from various formats
        import re
        symbols = re.split(r'[,\s\n]+', symbols_str)
        symbols = [s.strip().upper() for s in symbols if s.strip()]

        if watchlist_name not in self.watchlists:
            self.create_watchlist(watchlist_name)

        return self.add_symbols_bulk(watchlist_name, symbols)

    def clear_watchlist(self, watchlist_name: str) -> bool:
        """
        Remove all symbols from a watchlist

        Args:
            watchlist_name: Name of the watchlist

        Returns:
            True if cleared, False if not found
        """
        if watchlist_name not in self.watchlists:
            return False

        self.watchlists[watchlist_name] = []
        self._save()
        logger.info(f"Cleared watchlist '{watchlist_name}'")
        return True


# Singleton instance
watchlist_manager = WatchlistManager()
