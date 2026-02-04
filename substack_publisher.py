#!/usr/bin/env python3
"""Substack Publisher V5 - Fix publish button"""

import asyncio
import json
import os
import sys
from datetime import datetime
from playwright.async_api import async_playwright

SESSION_PATH = '/root/substack_session.json'
SUBSTACK_URL = 'https://aitradingradar.substack.com'

async def publish_to_substack(title: str, subtitle: str, content: str, publish: bool = False):
    if not os.path.exists(SESSION_PATH):
        return {'error': 'Session file not found', 'status': 'error'}
    
    with open(SESSION_PATH, 'r') as f:
        session = json.load(f)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1280, 'height': 900})
        await context.add_cookies(session['cookies'])
        
        page = await context.new_page()
        await page.goto(f'{SUBSTACK_URL}/publish/post')
        await page.wait_for_load_state('networkidle')
        await asyncio.sleep(3)
        
        try:
            # Click on title area
            await page.mouse.click(640, 220)
            await asyncio.sleep(0.3)
            await page.keyboard.press('Control+a')
            await page.keyboard.type(title, delay=10)
            
            # Tab to subtitle
            await page.keyboard.press('Tab')
            await asyncio.sleep(0.3)
            await page.keyboard.type(subtitle, delay=10)
            
            # Press Escape and click on content area
            await page.keyboard.press('Escape')
            await asyncio.sleep(0.3)
            await page.mouse.click(640, 420)
            await asyncio.sleep(0.5)
            
            # Type content
            await page.keyboard.type(content, delay=5)
            
            await asyncio.sleep(2)
            draft_url = page.url
            status = 'draft'
            
            if publish:
                # Click Continue button
                continue_btn = page.locator('button:has-text("Continue")').first
                if await continue_btn.count():
                    await continue_btn.click()
                    await asyncio.sleep(3)
                    
                    # Click "Send to everyone now" or "Publish now"
                    send_btn = page.locator('button:has-text("Send to everyone")').first
                    if await send_btn.count():
                        await send_btn.click()
                        await asyncio.sleep(5)
                        status = 'published'
                    else:
                        publish_btn = page.locator('button:has-text("Publish")').first
                        if await publish_btn.count():
                            await publish_btn.click()
                            await asyncio.sleep(5)
                            status = 'published'
            
            await page.screenshot(path='/tmp/substack_final.png')
            
            result = {
                'url': page.url,
                'draft_url': draft_url,
                'status': status
            }
            
        except Exception as e:
            await page.screenshot(path='/tmp/substack_error.png')
            result = {'error': str(e), 'status': 'error'}
        
        await browser.close()
        return result

async def main():
    title = sys.argv[1] if len(sys.argv) > 1 else 'Test'
    subtitle = sys.argv[2] if len(sys.argv) > 2 else ''
    content = sys.argv[3] if len(sys.argv) > 3 else 'Test content'
    publish = '--publish' in sys.argv
    
    result = await publish_to_substack(title, subtitle, content, publish)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    asyncio.run(main())
