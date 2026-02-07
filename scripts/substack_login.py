import asyncio
import sys
from playwright.async_api import async_playwright

async def login_and_save(email, password):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        print("Navigating to Substack login...")
        await page.goto("https://substack.com/sign-in")
        await asyncio.sleep(3)
        
        # Click "Sign in with password"
        try:
            pwd_btn = page.locator("text=Sign in with password")
            if await pwd_btn.count() > 0:
                await pwd_btn.click()
                await asyncio.sleep(2)
        except:
            pass
        
        # Enter email
        print(f"Entering email: {email}")
        email_input = page.locator("input[type=email], input[name=email]")
        await email_input.fill(email)
        await asyncio.sleep(1)
        
        # Click continue/next
        try:
            continue_btn = page.locator("button:has-text(\"Continue\"), button:has-text(\"Next\")")
            if await continue_btn.count() > 0:
                await continue_btn.first.click()
                await asyncio.sleep(2)
        except:
            pass
        
        # Enter password
        print("Entering password...")
        pwd_input = page.locator("input[type=password]")
        await pwd_input.fill(password)
        await asyncio.sleep(1)
        
        # Click sign in
        signin_btn = page.locator("button:has-text(\"Sign in\"), button[type=submit]")
        await signin_btn.first.click()
        await asyncio.sleep(5)
        
        print(f"Current URL: {page.url}")
        
        # Save session
        await context.storage_state(path="/root/substack_session.json")
        print("Session saved to /root/substack_session.json")
        
        await browser.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python substack_login.py <email> <password>")
        sys.exit(1)
    asyncio.run(login_and_save(sys.argv[1], sys.argv[2]))
