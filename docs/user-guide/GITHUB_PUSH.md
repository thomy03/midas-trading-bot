# 🚀 How to Push to GitHub

Your repository is ready to push, but needs authentication.

## Current Status

✅ Git repository initialized
✅ All files committed (38 files)
✅ Remote configured: https://github.com/thomy03/tradingbot-ema-screener.git
✅ Branch renamed to `main`
❌ Push blocked by authentication

## Solution: Use Personal Access Token

### Step 1: Create GitHub Token

1. Go to https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Settings:
   - **Name**: `WSL Tradingbot`
   - **Expiration**: 90 days (or custom)
   - **Scope**: Check `repo` (full control of repositories)
4. Click **"Generate token"**
5. **COPY THE TOKEN** (you won't see it again!)

### Step 2: Push to GitHub

Run this command in your terminal:

```bash
git push -u origin main
```

When prompted:
- **Username**: Enter your GitHub username (e.g., `thomy03`)
- **Password**: **Paste your token** (NOT your GitHub password!)

### Step 3: Verify

After successful push, verify at:
https://github.com/thomy03/tradingbot-ema-screener

## Alternative: SSH Authentication (More Secure)

If you prefer SSH:

```bash
# 1. Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. Start SSH agent
eval "$(ssh-agent -s)"

# 3. Add key
ssh-add ~/.ssh/id_ed25519

# 4. Display public key
cat ~/.ssh/id_ed25519.pub
```

Then:
- Copy the output from step 4
- Go to https://github.com/settings/keys
- Click "New SSH key"
- Paste and save

Finally:
```bash
# Change remote to SSH
git remote set-url origin git@github.com:thomy03/tradingbot-ema-screener.git

# Push
git push -u origin main
```

## Troubleshooting

### "fatal: Authentication failed"
- Make sure you're using the **token**, not your password
- Check token has `repo` scope enabled

### "Permission denied (publickey)" (SSH)
- Make sure SSH key is added to GitHub
- Verify SSH agent: `ssh-add -l`

## After Successful Push

Your repository will be available at:
**https://github.com/thomy03/tradingbot-ema-screener**

It will include:
- ✅ Complete EMA-based screener code
- ✅ Streamlit dashboard with RSI subplot
- ✅ Historical trade visualization
- ✅ Documentation (WSL setup, testing guide, signal visualization)
- ✅ Telegram notification system
- ✅ Automated scheduler scripts

---

**Ready to push?** Follow Step 1 & 2 above! 🚀
