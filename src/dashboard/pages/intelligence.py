"""
🧠 Intelligence page for Market Screener Dashboard
"""
import sys
import os

# Ensure src is in path
_src_path = os.path.join(os.path.dirname(__file__), '../../..')
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Import all shared dependencies
from src.dashboard.shared_imports import *


def render():
    """Render the 🧠 Intelligence page"""
    st.title("🧠 Market Intelligence")
    st.markdown("### Veille sectorielle et analyse de momentum")

    # Initialize sector analyzer
    if 'sector_analyzer' not in st.session_state:
        st.session_state.sector_analyzer = SectorAnalyzer(min_momentum_vs_spy=0.0)

    # Tabs for different features
    intel_tab1, intel_tab2, intel_tab3 = st.tabs(["📊 Sector Momentum", "📰 News Search", "📚 Research Papers"])

    with intel_tab1:
        st.subheader("Sector Momentum vs SPY")
        st.markdown("Analyse le momentum de chaque secteur par rapport au S&P 500")

        col1, col2 = st.columns([1, 1])

        with col1:
            min_vs_spy = st.slider(
                "Min. performance vs SPY (%)",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                help="Seuil minimum de surperformance par rapport au SPY"
            )

        with col2:
            if st.button("🔄 Refresh Sector Data", type="primary"):
                st.session_state.sector_data_loaded = False

        # Load sector data if needed
        if 'sector_data_loaded' not in st.session_state or not st.session_state.sector_data_loaded:
            with st.spinner("Loading sector ETF data..."):
                try:
                    st.session_state.sector_analyzer = SectorAnalyzer(min_momentum_vs_spy=min_vs_spy)
                    success = st.session_state.sector_analyzer.load_sector_data(market_data_fetcher, period='1y')
                    st.session_state.sector_data_loaded = success
                    if success:
                        st.success("Sector data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading sector data: {e}")
                    st.session_state.sector_data_loaded = False

        # Display sector momentum if data is loaded
        if st.session_state.get('sector_data_loaded', False):
            try:
                momentum_data = st.session_state.sector_analyzer.calculate_sector_momentum()

                if momentum_data:
                    # Create dataframe for display
                    sector_df_data = []
                    for sector, sm in sorted(momentum_data.items(), key=lambda x: x[1].rank):
                        status = "BULLISH" if sm.is_bullish else "BEARISH"
                        status_color = "green" if sm.is_bullish else "red"
                        sector_df_data.append({
                            'Rank': sm.rank,
                            'Sector': sector,
                            'ETF': sm.etf_symbol,
                            'Perf 20d': f"{sm.perf_20d:+.1f}%",
                            'vs SPY': f"{sm.vs_spy:+.1f}%",
                            'Status': status
                        })

                    sector_df = pd.DataFrame(sector_df_data)

                    # Color code status
                    def color_status(val):
                        if val == 'BULLISH':
                            return 'background-color: #00C853; color: white'
                        elif val == 'BEARISH':
                            return 'background-color: #FF5252; color: white'
                        return ''

                    styled_sector_df = sector_df.style.applymap(color_status, subset=['Status'])
                    st.dataframe(styled_sector_df, use_container_width=True, height=500)

                    # Summary metrics
                    st.markdown("---")
                    bullish_count = sum(1 for sm in momentum_data.values() if sm.is_bullish)
                    bearish_count = len(momentum_data) - bullish_count

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Bullish Sectors", bullish_count)
                    with col2:
                        st.metric("Bearish Sectors", bearish_count)
                    with col3:
                        market_breadth = (bullish_count / len(momentum_data)) * 100 if momentum_data else 0
                        st.metric("Market Breadth", f"{market_breadth:.0f}%")

                    # Top sectors recommendation
                    st.markdown("---")
                    st.subheader("Recommended Sectors")
                    bullish_sectors = st.session_state.sector_analyzer.get_bullish_sectors()
                    if bullish_sectors:
                        for sm in bullish_sectors[:5]:
                            st.success(f"**{sm.sector}** ({sm.etf_symbol}): +{sm.perf_20d:.1f}% | vs SPY: +{sm.vs_spy:.1f}%")
                    else:
                        st.warning("No bullish sectors detected with current threshold")

                else:
                    st.info("No sector data available. Click 'Refresh Sector Data' to load.")

            except Exception as e:
                st.error(f"Error calculating momentum: {e}")

    with intel_tab2:
        st.subheader("Financial News Search")
        st.markdown("Search for news from various sources (Reddit, NewsAPI, Alpha Vantage)")

        col1, col2 = st.columns([3, 1])

        with col1:
            news_query = st.text_input(
                "Search keywords",
                placeholder="e.g., AI semiconductor, Tesla earnings, Fed rate",
                key="news_query"
            )

        with col2:
            news_source = st.selectbox(
                "Source",
                ["Reddit", "All Sources"],
                key="news_source"
            )

        if st.button("🔍 Search News", type="primary"):
            if news_query:
                with st.spinner(f"Searching for '{news_query}'..."):
                    import asyncio

                    async def fetch_news():
                        fetcher = get_news_fetcher()
                        try:
                            if news_source == "Reddit":
                                posts = await fetcher.fetch_reddit_posts(
                                    query=news_query,
                                    limit=20,
                                    time_filter='week'
                                )
                                return posts
                            else:
                                # Fetch from multiple sources
                                posts = await fetcher.fetch_reddit_posts(query=news_query, limit=10)
                                return posts
                        finally:
                            await fetcher.close()

                    try:
                        # Run async function
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        articles = loop.run_until_complete(fetch_news())
                        loop.close()

                        if articles:
                            st.success(f"Found {len(articles)} results")
                            for article in articles[:15]:
                                with st.expander(f"📰 {article.title[:80]}..."):
                                    st.markdown(f"**Source:** {article.source}")
                                    st.markdown(f"**Date:** {article.published_at.strftime('%Y-%m-%d %H:%M')}")
                                    if article.content:
                                        st.markdown(article.content[:500] + "..." if len(article.content) > 500 else article.content)
                                    st.markdown(f"[Read more]({article.url})")
                                    st.markdown(f"**Relevance:** {article.relevance_score:.2f}")
                        else:
                            st.info("No results found. Try different keywords.")

                    except Exception as e:
                        st.error(f"Error fetching news: {e}")
            else:
                st.warning("Please enter search keywords")

    with intel_tab3:
        st.subheader("Research Papers (Arxiv)")
        st.markdown("Search for academic publications on AI, Finance, and Technology")

        col1, col2 = st.columns([3, 1])

        with col1:
            arxiv_query = st.text_input(
                "Search query",
                placeholder="e.g., machine learning stock prediction, LLM finance",
                key="arxiv_query"
            )

        with col2:
            arxiv_categories = st.multiselect(
                "Categories",
                ["cs.AI", "cs.LG", "q-fin.ST", "q-fin.TR", "cs.CL"],
                default=["cs.AI", "cs.LG"],
                key="arxiv_categories"
            )

        if st.button("🔍 Search Papers", type="primary"):
            if arxiv_query:
                with st.spinner(f"Searching Arxiv for '{arxiv_query}'..."):
                    import asyncio

                    async def fetch_papers():
                        fetcher = get_news_fetcher()
                        try:
                            papers = await fetcher.fetch_arxiv_papers(
                                query=arxiv_query,
                                categories=arxiv_categories if arxiv_categories else None,
                                max_results=20
                            )
                            return papers
                        finally:
                            await fetcher.close()

                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        papers = loop.run_until_complete(fetch_papers())
                        loop.close()

                        if papers:
                            st.success(f"Found {len(papers)} papers")
                            for paper in papers:
                                with st.expander(f"📄 {paper.title[:80]}..."):
                                    st.markdown(f"**Authors:** {', '.join(paper.authors[:5])}")
                                    st.markdown(f"**Date:** {paper.published_at.strftime('%Y-%m-%d')}")
                                    st.markdown(f"**Categories:** {', '.join(paper.categories)}")
                                    st.markdown("**Abstract:**")
                                    st.markdown(paper.abstract[:800] + "..." if len(paper.abstract) > 800 else paper.abstract)
                                    if paper.url:
                                        st.markdown(f"[Read PDF]({paper.url})")
                        else:
                            st.info("No papers found. Try different keywords.")

                    except Exception as e:
                        st.error(f"Error fetching papers: {e}")
            else:
                st.warning("Please enter a search query")

    # API Configuration hint
    st.markdown("---")
    with st.expander("🔧 API Configuration"):
        st.markdown("""
        **Optional API keys for enhanced functionality:**

        - `NEWSAPI_KEY`: For NewsAPI access (newsapi.org)
        - `ALPHAVANTAGE_KEY`: For Alpha Vantage news sentiment
        - `OPENROUTER_API_KEY`: For LLM-powered analysis

        Set these in your `.env` file or environment variables.
        """)

