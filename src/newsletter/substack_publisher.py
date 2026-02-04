"""
Substack Publisher - Automated newsletter publication (FIXED v2)

Publishes:
- Pre-market analysis (8h EU, 15h30 US)
- Trade signals (at each BUY/SELL)
- Daily recap (22h)

Fixed: Handles "Add subscribe buttons" popup
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

SESSION_FILE = Path('/root/substack_session.json')
SUBSTACK_URL = 'https://aitradingradar.substack.com'


class SubstackPublisher:
    """Publishes content to Substack newsletter"""
    
    def __init__(self):
        self.session_file = SESSION_FILE
        self.base_url = SUBSTACK_URL
        
    async def publish_post(
        self,
        title: str,
        content: str,
        subtitle: str = '',
        is_premium: bool = False,
        publish_now: bool = True
    ) -> Optional[str]:
        """
        Publish a post to Substack.
        
        Args:
            title: Post title
            content: Post content (markdown)
            subtitle: Optional subtitle
            is_premium: If True, only for paid subscribers
            publish_now: If True, publish immediately; else save as draft
            
        Returns:
            Post URL if successful, None otherwise
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error('Playwright not installed')
            return None
            
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    storage_state=str(self.session_file)
                )
                page = await context.new_page()
                
                # Go to new post
                logger.info('Creating new Substack post...')
                await page.goto(f'{self.base_url}/publish/post', wait_until='networkidle')
                await page.wait_for_timeout(3000)
                
                # Type title
                await page.keyboard.type(title)
                await page.keyboard.press('Tab')
                await page.wait_for_timeout(500)
                
                # Type subtitle if provided
                if subtitle:
                    await page.keyboard.type(subtitle)
                    await page.keyboard.press('Tab')
                    await page.wait_for_timeout(500)
                
                # Type content
                await page.keyboard.type(content)
                await page.wait_for_timeout(1000)
                
                post_url = page.url
                logger.info(f'Draft created: {post_url}')
                
                if publish_now:
                    # Step 1: Click Continue
                    logger.info('Clicking Continue...')
                    continue_btn = page.locator('button:has-text("Continue")')
                    if await continue_btn.count() > 0:
                        await continue_btn.first.click()
                        await page.wait_for_timeout(2000)
                    
                    # Step 2: Set visibility (free or premium)
                    if is_premium:
                        paid_radio = page.locator('input[value="only_paid"]')
                        if await paid_radio.count() > 0:
                            await paid_radio.click()
                    
                    # Step 3: Click Send to everyone now
                    logger.info('Clicking Send to everyone now...')
                    send_btn = page.locator('button:has-text("Send to everyone now")')
                    if await send_btn.count() > 0:
                        await send_btn.first.click()
                        await page.wait_for_timeout(2000)
                    
                    # Step 4: Handle "Add subscribe buttons" popup (FIXED)
                    logger.info('Handling subscribe buttons popup...')
                    pub_without = page.locator('button:has-text("Publish without buttons")')
                    if await pub_without.count() > 0:
                        await pub_without.click()
                        await page.wait_for_timeout(2000)
                    else:
                        # Try Add subscribe buttons
                        add_sub = page.locator('button:has-text("Add subscribe buttons")')
                        if await add_sub.count() > 0:
                            await add_sub.click()
                            await page.wait_for_timeout(2000)
                    
                    post_url = page.url
                    logger.info(f'Published: {post_url}')
                
                await browser.close()
                return post_url
                
        except Exception as e:
            logger.error(f'Failed to publish: {e}')
            return None
    
    async def publish_first_draft(self) -> Optional[str]:
        """
        Publish the first draft in the queue.
        Useful for publishing existing drafts.
        
        Returns:
            Post URL if successful, None otherwise
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error('Playwright not installed')
            return None
            
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    storage_state=str(self.session_file)
                )
                page = await context.new_page()
                
                # Go to posts page
                logger.info('Loading posts page...')
                await page.goto(f'{self.base_url}/publish/posts', wait_until='networkidle')
                await page.wait_for_timeout(2000)
                
                # Click Drafts tab
                logger.info('Clicking Drafts tab...')
                drafts_tab = page.locator('button:has-text("Drafts")')
                await drafts_tab.click()
                await page.wait_for_timeout(2000)
                
                # Find drafts
                draft_links = await page.locator('a[href*="/publish/post/"]').all()
                if len(draft_links) == 0:
                    logger.info('No drafts to publish')
                    await browser.close()
                    return None
                
                # Open first draft
                title = await draft_links[0].text_content()
                logger.info(f'Opening draft: {title[:60] if title else "Unknown"}')
                await draft_links[0].click()
                await page.wait_for_timeout(3000)
                
                # Click Continue
                logger.info('Clicking Continue...')
                continue_btn = page.locator('button:has-text("Continue")')
                await continue_btn.first.click()
                await page.wait_for_timeout(2000)
                
                # Click Send to everyone now
                logger.info('Clicking Send to everyone now...')
                send_btn = page.locator('button:has-text("Send to everyone now")')
                await send_btn.first.click()
                await page.wait_for_timeout(2000)
                
                # Handle "Add subscribe buttons" popup
                logger.info('Handling subscribe buttons popup...')
                pub_without = page.locator('button:has-text("Publish without buttons")')
                if await pub_without.count() > 0:
                    await pub_without.click()
                    await page.wait_for_timeout(2000)
                
                post_url = page.url
                logger.info(f'Published: {post_url}')
                
                await browser.close()
                return post_url
                
        except Exception as e:
            logger.error(f'Failed to publish draft: {e}')
            return None
    
    async def publish_premarket_eu(self, analysis: Dict[str, Any]) -> Optional[str]:
        """Publish EU pre-market analysis (8h Paris)"""
        date_str = datetime.now().strftime('%b %d, %Y')
        title = f"EU Pre-Market | {date_str}"
        
        content = f"""
## Market Overview

{analysis.get('overview', 'Markets opening...')}

## Key Levels to Watch

{analysis.get('key_levels', '')}

## Top Opportunities

{analysis.get('opportunities', '')}

---
*AI Trading Radar - Your daily edge*
"""
        
        return await self.publish_post(title, content, is_premium=False)
    
    async def publish_trade_signal(self, trade: Dict[str, Any]) -> Optional[str]:
        """Publish trade signal (PREMIUM)"""
        symbol = trade.get('symbol', 'UNKNOWN')
        action = trade.get('action', 'BUY')
        price = trade.get('price', 0)
        score = trade.get('score', 0)
        sl = trade.get('stop_loss', 0)
        tp = trade.get('take_profit', 0)
        
        title = f"TRADE: {action} {symbol} @ ${price:.2f}"
        
        content = f"""
## Trade Alert: {symbol}

**Action:** {action}
**Entry Price:** ${price:.2f}
**Confidence Score:** {score}/100

### Risk Management

- **Stop Loss:** ${sl:.2f}
- **Take Profit:** ${tp:.2f}

### Analysis

{trade.get('analysis', 'Signal generated by AI Trading Radar.')}

---
*Not financial advice. Trade at your own risk.*
"""
        
        return await self.publish_post(title, content, is_premium=True)
    
    async def publish_daily_recap(self, recap: Dict[str, Any]) -> Optional[str]:
        """Publish end-of-day recap (22h Paris)"""
        date_str = datetime.now().strftime('%b %d, %Y')
        title = f"Daily Recap | {date_str}"
        
        content = f"""
## Market Summary

{recap.get('summary', '')}

## Today's Trades

{recap.get('trades', 'No trades today.')}

## Performance

{recap.get('performance', '')}

---
*See you tomorrow! - Midas*
"""
        
        return await self.publish_post(title, content, is_premium=False)


# Singleton
_publisher: Optional[SubstackPublisher] = None

def get_substack_publisher() -> SubstackPublisher:
    global _publisher
    if _publisher is None:
        _publisher = SubstackPublisher()
    return _publisher
