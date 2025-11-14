"""
Trendline Analysis Configuration

Settings for RSI trendline detection and breakout analysis
"""

# RSI Parameters
RSI_PERIOD = 14
RSI_EMA_PERIODS = [9, 14]

# Peak Detection Parameters
PEAK_PROMINENCE = 2.0  # Minimum prominence for a peak to be considered (RÉDUIT de 3.0 → 2.0 pour détecter plus d'obliques)
PEAK_DISTANCE = 5      # Minimum distance between peaks (in periods)
MIN_PEAKS_FOR_TRENDLINE = 3  # Minimum number of peaks to draw a trendline

# Trendline Validation
MIN_R_SQUARED = 0.25   # Minimum R² for trendline quality (TRÈS ASSOUPLI de 0.40 → 0.25 pour accepter plus d'obliques)
MIN_SLOPE = -1.0       # Minimum negative slope (descending resistance) - allows steeper slopes
MAX_SLOPE = 0.0        # Maximum negative slope (must be descending)
MAX_RESIDUAL_DISTANCE = 8.0  # Maximum distance of peaks from trendline (RSI points) - ASSOUPLI de 6.0 → 8.0 pour accepter plus d'obliques

# Breakout Detection
BREAKOUT_THRESHOLD = 0.0  # RSI must cross above trendline (0 = detect even slight crossings)
CONFIRMATION_PERIODS = 1   # Number of periods to confirm breakout
MAX_BREAKOUT_AGE = 3      # Maximum age of breakout (periods) to consider "recent"

# Timeframe Priorities
TIMEFRAME_PRIORITY = ['weekly', 'daily']  # Order of analysis
WEEKLY_LOOKBACK_PERIODS = 104  # ~2 years of weekly data
DAILY_LOOKBACK_PERIODS = 252   # ~1 year of daily data

# Scoring
TRENDLINE_QUALITY_WEIGHTS = {
    'r_squared': 0.4,        # How well points fit the line
    'num_points': 0.3,       # More points = better
    'slope_consistency': 0.2, # Slope within optimal range
    'recency': 0.1          # Recent trendline preferred
}

# Visualization
TRENDLINE_COLOR = '#FF9800'  # Orange for resistance lines
BREAKOUT_COLOR = '#4CAF50'   # Green for breakout points
TRENDLINE_WIDTH = 2
TRENDLINE_DASH = 'dash'
