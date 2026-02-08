"""
MIDAS Symbol Universe - ~800 liquid US equities

Covers S&P 500 + Russell 1000 mid-caps across all sectors.
All symbols are traded on NYSE/NASDAQ with sufficient liquidity.
"""

# S&P 500 Large Caps (sorted by sector)
SP500_TECH = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'META', 'AMZN', 'AVGO', 'TSLA',
    'CRM', 'AMD', 'ADBE', 'ORCL', 'CSCO', 'ACN', 'INTC', 'TXN', 'QCOM', 'IBM',
    'AMAT', 'NOW', 'INTU', 'PANW', 'ADI', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS',
    'CRWD', 'MRVL', 'FTNT', 'ADSK', 'NXPI', 'MCHP', 'ON', 'MPWR', 'KEYS', 'TEL',
    'CDW', 'HPQ', 'HPE', 'WDC', 'STX', 'SWKS', 'ZBRA', 'TRMB', 'TER',
    'GEN', 'NTAP', 'CTSH', 'IT', 'EPAM', 'AKAM', 'FFIV',
]

SP500_HEALTHCARE = [
    'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'PFE', 'AMGN', 'DHR',
    'BMY', 'ISRG', 'GILD', 'MDT', 'ELV', 'CI', 'SYK', 'VRTX', 'REGN', 'BSX',
    'ZTS', 'BDX', 'HCA', 'MCK', 'EW', 'IDXX', 'A', 'IQV', 'DXCM', 'MTD',
    'RMD', 'BAX', 'ALGN', 'HOLX', 'LH', 'ILMN', 'CAH', 'COR', 'PODD', 'TECH',
    'BIO', 'INCY', 'TFX', 'XRAY', 'DGX', 'VTRS', 'CRL', 'MOH',
    'HSIC', 'OGN',
]

SP500_FINANCIALS = [
    'JPM', 'V', 'MA', 'BAC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C',
    'SPGI', 'ICE', 'CB', 'PGR', 'MMC', 'AON', 'CME', 'USB', 'PNC', 'TFC',
    'AIG', 'MET', 'PRU', 'AFL', 'ALL', 'TRV', 'BK', 'STT', 'FITB', 'CFG',
    'HBAN', 'MTB', 'KEY', 'NTRS', 'RF', 'WRB', 'CINF', 'RJF', 'SYF',
    'GL', 'BRO', 'L', 'FDS', 'MKTX', 'NDAQ', 'MSCI', 'COF',
    'TROW', 'IVZ', 'BEN',
]

SP500_CONSUMER_DISC = [
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'CMG',
    'ORLY', 'AZO', 'ROST', 'DHI', 'LEN', 'GM', 'F', 'GPC', 'POOL', 'BBY',
    'ULTA', 'DRI', 'YUM', 'MAR', 'HLT', 'LVS', 'WYNN', 'MGM', 'CCL', 'RCL',
    'NCLH', 'EXPE', 'EBAY', 'ETSY', 'TPR', 'RL', 'HAS', 'APTV', 'BWA',
    'CZR', 'PHM', 'NVR', 'GRMN', 'DECK', 'LKQ',
]

SP500_CONSUMER_STAPLES = [
    'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
    'STZ', 'GIS', 'SYY', 'ADM', 'HSY', 'MKC', 'K', 'SJM', 'TSN', 'HRL',
    'CPB', 'KHC', 'CAG', 'CLX', 'CHD', 'BG', 'TAP', 'MNST', 'KDP',
    'EL', 'TGT', 'DG', 'DLTR',
]

SP500_ENERGY = [
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY',
    'WMB', 'HAL', 'DVN', 'BKR', 'FANG', 'CTRA', 'OKE', 'TRGP', 'KMI',
    'APA',
]

SP500_INDUSTRIALS = [
    'CAT', 'HON', 'UPS', 'GE', 'BA', 'RTX', 'DE', 'LMT', 'UNP', 'ADP',
    'ETN', 'ITW', 'EMR', 'FDX', 'GD', 'NOC', 'WM', 'RSG', 'CSX', 'NSC',
    'JCI', 'CMI', 'PH', 'ROK', 'FAST', 'PCAR', 'OTIS', 'CARR', 'DAL', 'LUV',
    'UAL', 'AAL', 'IR', 'DOV', 'AME', 'SWK', 'GWW', 'TT', 'XYL', 'IEX',
    'HUBB', 'WAB', 'PAYC', 'ROP', 'VRSK', 'CTAS', 'PAYX', 'LDOS', 'J',
    'PWR', 'GNRC', 'AXON', 'TDG', 'HWM', 'NDSN', 'SNA',
]

SP500_MATERIALS = [
    'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'ECL', 'PPG',
    'VMC', 'MLM', 'ALB', 'CF', 'MOS', 'IFF', 'CE', 'IP', 'PKG', 'SEE',
    'AVY', 'EMN', 'FMC', 'AMCR',
]

SP500_COMMUNICATION = [
    'GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR',
    'EA', 'TTWO', 'WBD', 'OMC', 'LYV', 'MTCH', 'FOXA', 'FOX',
    'NWSA', 'NWS',
]

SP500_UTILITIES = [
    'NEE', 'SO', 'DUK', 'D', 'SRE', 'AEP', 'EXC', 'XEL', 'ED', 'WEC',
    'ES', 'EIX', 'AWK', 'DTE', 'PPL', 'FE', 'AES', 'CMS', 'ATO', 'NI',
    'EVRG', 'LNT', 'PNW', 'NRG', 'CEG',
]

SP500_REALESTATE = [
    'PLD', 'AMT', 'EQIX', 'SPG', 'CCI', 'O', 'PSA', 'DLR', 'WELL', 'AVB',
    'EQR', 'VTR', 'ARE', 'ESS', 'MAA', 'UDR', 'HST', 'KIM', 'REG',
    'FRT', 'BXP', 'CPT', 'IRM', 'SBAC', 'INVH',
]

# Russell 1000 Mid-Caps and Additional Liquid Stocks
MIDCAP_TECH = [
    'SHOP', 'NET', 'WDAY', 'SNOW', 'ZS', 'DDOG', 'TEAM', 'HUBS', 'VEEV',
    'BILL', 'CFLT', 'MDB', 'OKTA', 'ESTC', 'FIVN', 'DOCN', 'PTC', 'MANH',
    'BSY', 'TYL', 'SMCI', 'ARM', 'PLTR', 'COIN', 'U', 'RBLX',
    'SNAP', 'PINS', 'PATH', 'APP', 'TTD', 'ROKU', 'PCOR',
    'QLYS', 'RPD', 'VRNS', 'TENB', 'CYBR', 'ALRM', 'DT', 'GLOB',
    'WK', 'JAMF', 'SPT', 'CWAN', 'FOUR', 'FICO',
    'GWRE', 'NICE', 'RNG', 'TWLO', 'DOCU', 'ZM', 'ABNB',
    'DASH', 'LYFT', 'UBER', 'PYPL', 'AFRM', 'SOFI', 'HOOD',
    'IONQ', 'RGTI', 'QUBT',
]

MIDCAP_HEALTHCARE = [
    'MRNA', 'BNTX', 'JAZZ', 'RARE', 'NBIX', 'EXAS', 'NVCR',
    'HALO', 'PCVX', 'RVMD', 'SRPT', 'BMRN', 'ALNY', 'IONS', 'EXEL',
    'IOVA', 'ARWR', 'VCEL', 'PRCT', 'GMED', 'TNDM', 'IRTC',
    'RGEN', 'AZTA', 'NTRA', 'GH', 'TWST', 'CRSP', 'BEAM', 'EDIT',
    'NTLA', 'FATE', 'RCKT', 'SRRK', 'LEGN', 'BHVN',
    'ACAD', 'AXSM', 'CRNX', 'INSM', 'ARVN', 'DNLI',
    'RYTM', 'VERA',
]

MIDCAP_FINANCIALS = [
    'HOOD', 'LPLA', 'IBKR', 'MARA', 'RIOT', 'HIG', 'ACGL', 'FNF',
    'EWBC', 'ZION', 'CMA', 'FHN', 'SNV', 'WAL', 'PNFP',
    'SBNY', 'OZK', 'UMBF', 'BOH', 'FFIN', 'CBSH', 'ABCB', 'NBTB',
    'WTFC', 'GBCI', 'FNB', 'HWC', 'SSB', 'UBSI', 'TMP', 'CADE',
    'PIPR', 'VIRT', 'EVR', 'PJT', 'HLI', 'LAZ', 'MC',
]

MIDCAP_CONSUMER = [
    'LULU', 'RH', 'FIVE', 'WING', 'CAVA', 'SHAK', 'PLAY', 'EAT', 'DIN',
    'TXRH', 'CAKE', 'DENN', 'JACK', 'LOCO', 'BJRI',
    'ANF', 'AEO', 'URBN', 'VSCO', 'DKS',
    'WSM', 'W', 'CVNA', 'CARG', 'KMX',
    'BROS', 'SG', 'PENN', 'DKNG', 'GENI', 'CHDN',
    'BURL', 'MNSO', 'CELH', 'OLPX', 'ELF', 'COTY',
]

MIDCAP_INDUSTRIAL = [
    'BLDR', 'SITE', 'UFPI', 'FBIN', 'MAS', 'OC', 'TREX',
    'STRL', 'MTZ', 'FIX', 'PRIM', 'GVA', 'DY', 'AAON',
    'WFRD', 'TPC', 'AGX', 'KNF', 'RBC', 'GATX', 'AIT',
    'SPXC', 'ITT', 'RRX', 'ESE', 'CXT', 'THR', 'MIDD',
    'TTC', 'AL', 'VLTO', 'AGCO', 'ATKR',
]

MIDCAP_ENERGY_MATERIALS = [
    'AR', 'RRC', 'EQT', 'CNX', 'MTDR', 'CHRD', 'SM', 'NOV',
    'PTEN', 'HP', 'LBRT', 'RES', 'WHD', 'PUMP',
    'MP', 'LAC', 'SQM', 'VALE', 'RIO', 'BHP', 'SCCO',
    'AA', 'CENX', 'HCC', 'BTU', 'STLD', 'CLF', 'RS',
    'ATI', 'CRS', 'KALU', 'IOSP', 'CBT', 'KWR',
]

MIDCAP_COMM_REIT = [
    'SPOT', 'RDDT', 'GRAB', 'SE', 'BIDU', 'JD', 'PDD', 'BABA', 'TME',
    'BILI', 'IQ', 'ZTO', 'VNET', 'WB',
    'GLPI', 'VICI', 'SUI', 'ELS', 'REXR', 'STAG', 'COLD', 'IIPR',
    'NNN', 'ADC', 'EPRT', 'GTY', 'WPC', 'PINE', 'SAFE',
    'MPW', 'OHI', 'SBRA', 'HR', 'DOC', 'LTC', 'UHT',
]

# Additional popular / high-liquidity tickers
POPULAR_ADDITIONS = [
    # EV / Clean Energy
    'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'QS', 'CHPT', 'BLNK',
    'PLUG', 'FCEL', 'BE', 'ENPH', 'SEDG', 'RUN', 'ARRY', 'MAXN',
    'FSLR', 'CSIQ', 'JKS', 'DQ',
    # AI / Semiconductor
    'TSM', 'ASML', 'OLED', 'LSCC', 'RMBS', 'POWI',
    'SLAB', 'DIOD', 'SITM', 'AMBA', 'CEVA', 'ALGM',
    # Biotech
    'BIIB', 'RPRX', 'UTHR', 'MEDP', 'ENSG', 'OMCL',
    'ANGO', 'ATEC', 'SIBN', 'LIVN', 'NVST',
    # Fintech
    'FIS', 'FISV', 'GPN', 'WEX', 'GDOT', 'CPAY',
    # Defense
    'HII', 'LHX', 'TDG', 'HEI', 'KTOS', 'MRCY', 'CACI', 'BWXT',
    # REITs extra
    'CUBE', 'NSA', 'UNIT', 'CTO', 'AKR',
    # Cannabis/Misc
    'TLRY', 'CGC', 'ACB',
    # SPACs turned real companies
    'DNA', 'ASTS', 'RDW', 'LUNR',
    # China ADRs
    'NTES', 'LU', 'MNSO', 'YMM', 'FUTU', 'TIGR',
]


def get_universe(size: str = 'full') -> list:
    """Get symbol universe by size.

    Args:
        size: 'quick' (30), 'medium' (200), 'full' (500+), 'mega' (800+)
    """
    quick = [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'CRM', 'AVGO',
        'JNJ', 'UNH', 'PFE', 'ABBV',
        'JPM', 'BAC', 'GS', 'V',
        'WMT', 'HD', 'MCD', 'NKE',
        'XOM', 'CVX', 'LIN',
        'CAT', 'HON', 'UPS',
        'DIS', 'NFLX',
        'NEE', 'PLD',
    ]

    if size == 'quick':
        return quick

    # Combine all S&P 500 sectors
    sp500 = (
        SP500_TECH + SP500_HEALTHCARE + SP500_FINANCIALS +
        SP500_CONSUMER_DISC + SP500_CONSUMER_STAPLES + SP500_ENERGY +
        SP500_INDUSTRIALS + SP500_MATERIALS + SP500_COMMUNICATION +
        SP500_UTILITIES + SP500_REALESTATE
    )

    if size == 'medium':
        return list(dict.fromkeys(sp500[:200]))

    if size == 'full':
        return list(dict.fromkeys(sp500))

    # 'mega' = everything
    all_symbols = (
        sp500 +
        MIDCAP_TECH + MIDCAP_HEALTHCARE + MIDCAP_FINANCIALS +
        MIDCAP_CONSUMER + MIDCAP_INDUSTRIAL + MIDCAP_ENERGY_MATERIALS +
        MIDCAP_COMM_REIT + POPULAR_ADDITIONS
    )

    # Deduplicate while preserving order
    return list(dict.fromkeys(all_symbols))


if __name__ == '__main__':
    for size in ['quick', 'medium', 'full', 'mega']:
        symbols = get_universe(size)
        print(f"{size:>8}: {len(symbols)} symbols")
