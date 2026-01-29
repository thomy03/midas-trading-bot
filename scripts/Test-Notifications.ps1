<#
.SYNOPSIS
    Script de test des canaux de notification TradingBot

.DESCRIPTION
    Teste les différents canaux de notification:
    - Telegram
    - Email (si configuré)
    Affiche un rapport détaillé du statut de chaque canal.

.PARAMETER Channel
    Canal à tester: 'all', 'telegram', 'email'

.PARAMETER Message
    Message de test personnalisé

.EXAMPLE
    .\Test-Notifications.ps1

.EXAMPLE
    .\Test-Notifications.ps1 -Channel telegram -Message "Test personnalisé"

.NOTES
    Auteur: TradingBot Team
    Version: 1.0.0
#>

[CmdletBinding()]
param(
    [ValidateSet('all', 'telegram', 'email')]
    [string]$Channel = 'all',

    [string]$Message = "Test de notification TradingBot - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
)

# Configuration stricte
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Obtenir les chemins
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptRoot "..")

# Charger les modules communs
. "$ScriptRoot\common\Write-Log.ps1"
. "$ScriptRoot\common\Initialize-Venv.ps1"

function Get-EnvConfiguration {
    <#
    .SYNOPSIS
        Lit la configuration du fichier .env
    #>
    param()

    $envPath = Join-Path $ProjectRoot ".env"
    $config = @{
        TelegramToken  = $null
        TelegramChatId = $null
        EmailEnabled   = $false
        EmailFrom      = $null
        EmailTo        = $null
        SmtpServer     = $null
        SmtpPort       = $null
    }

    if (-not (Test-Path $envPath)) {
        return $config
    }

    $envContent = Get-Content $envPath

    foreach ($line in $envContent) {
        if ($line -match '^\s*#' -or $line -match '^\s*$') {
            continue
        }

        if ($line -match '^([^=]+)=(.*)$') {
            $key = $Matches[1].Trim()
            $value = $Matches[2].Trim()

            switch ($key) {
                'TELEGRAM_BOT_TOKEN' { $config.TelegramToken = $value }
                'TELEGRAM_CHAT_ID' { $config.TelegramChatId = $value }
                'EMAIL_ENABLED' { $config.EmailEnabled = ($value -eq 'True') }
                'EMAIL_FROM' { $config.EmailFrom = $value }
                'EMAIL_TO' { $config.EmailTo = $value }
                'SMTP_SERVER' { $config.SmtpServer = $value }
                'SMTP_PORT' { $config.SmtpPort = $value }
            }
        }
    }

    return $config
}

function Test-TelegramConfiguration {
    <#
    .SYNOPSIS
        Vérifie si Telegram est correctement configuré
    #>
    param(
        [hashtable]$Config
    )

    $result = [PSCustomObject]@{
        Configured = $false
        Token      = $false
        ChatId     = $false
        Issues     = @()
    }

    if ($Config.TelegramToken -and $Config.TelegramToken -notmatch '^your_') {
        $result.Token = $true
    }
    else {
        $result.Issues += "TELEGRAM_BOT_TOKEN non configuré"
    }

    if ($Config.TelegramChatId -and $Config.TelegramChatId -notmatch '^your_') {
        $result.ChatId = $true
    }
    else {
        $result.Issues += "TELEGRAM_CHAT_ID non configuré"
    }

    $result.Configured = $result.Token -and $result.ChatId

    return $result
}

function Test-EmailConfiguration {
    <#
    .SYNOPSIS
        Vérifie si l'email est correctement configuré
    #>
    param(
        [hashtable]$Config
    )

    $result = [PSCustomObject]@{
        Configured = $false
        Enabled    = $Config.EmailEnabled
        Issues     = @()
    }

    if (-not $Config.EmailEnabled) {
        $result.Issues += "Email désactivé (EMAIL_ENABLED=False)"
        return $result
    }

    if (-not $Config.EmailFrom -or $Config.EmailFrom -match '^your_') {
        $result.Issues += "EMAIL_FROM non configuré"
    }

    if (-not $Config.EmailTo -or $Config.EmailTo -match '^recipient') {
        $result.Issues += "EMAIL_TO non configuré"
    }

    if (-not $Config.SmtpServer) {
        $result.Issues += "SMTP_SERVER non configuré"
    }

    $result.Configured = ($Config.EmailEnabled -and $result.Issues.Count -eq 0)

    return $result
}

function Send-TestTelegramMessage {
    <#
    .SYNOPSIS
        Envoie un message de test via Telegram
    #>
    param(
        [Parameter(Mandatory)]
        [string]$TestMessage
    )

    Write-LogInfo "Envoi du message de test Telegram..."

    $pythonPath = Get-VenvPythonPath
    $testScript = @"
import asyncio
import sys
sys.path.insert(0, r'$ProjectRoot')

from src.notifications.notifier import notifier

async def test():
    try:
        result = await notifier.send_telegram_message('''$TestMessage''')
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        return False

result = asyncio.run(test())
print("SUCCESS" if result else "FAILED")
"@

    try {
        $result = $testScript | & $pythonPath - 2>&1

        if ($result -match "SUCCESS") {
            Write-LogSuccess "Message Telegram envoyé avec succès"
            return $true
        }
        else {
            Write-LogError "Échec de l'envoi Telegram: $result"
            return $false
        }
    }
    catch {
        Write-LogError "Erreur lors de l'envoi Telegram: $_"
        return $false
    }
}

function Send-TestEmailMessage {
    <#
    .SYNOPSIS
        Envoie un email de test
    #>
    param(
        [Parameter(Mandatory)]
        [string]$TestMessage
    )

    Write-LogInfo "Envoi de l'email de test..."

    $pythonPath = Get-VenvPythonPath
    $testScript = @"
import sys
sys.path.insert(0, r'$ProjectRoot')

from src.notifications.notifier import notifier

try:
    result = notifier.send_email("Test TradingBot", '''$TestMessage''')
    print("SUCCESS" if result else "FAILED")
except Exception as e:
    print(f"ERROR: {e}")
"@

    try {
        $result = $testScript | & $pythonPath - 2>&1

        if ($result -match "SUCCESS") {
            Write-LogSuccess "Email envoyé avec succès"
            return $true
        }
        else {
            Write-LogError "Échec de l'envoi email: $result"
            return $false
        }
    }
    catch {
        Write-LogError "Erreur lors de l'envoi email: $_"
        return $false
    }
}

function Show-ConfigurationSummary {
    <#
    .SYNOPSIS
        Affiche un résumé de la configuration
    #>
    param(
        [hashtable]$Config
    )

    Write-Section "Configuration Actuelle"

    # Telegram
    $telegramStatus = Test-TelegramConfiguration -Config $Config
    Write-Host "  TELEGRAM:" -ForegroundColor Cyan
    Write-Result -Item "  Token" -Success $telegramStatus.Token -Message $(if ($telegramStatus.Token) { "Configuré" } else { "Non configuré" })
    Write-Result -Item "  Chat ID" -Success $telegramStatus.ChatId -Message $(if ($telegramStatus.ChatId) { "Configuré" } else { "Non configuré" })

    Write-Host ""

    # Email
    $emailStatus = Test-EmailConfiguration -Config $Config
    Write-Host "  EMAIL:" -ForegroundColor Cyan
    Write-Result -Item "  Activé" -Success $emailStatus.Enabled -Message $(if ($emailStatus.Enabled) { "Oui" } else { "Non" })

    if ($emailStatus.Enabled) {
        Write-Host "    From: $($Config.EmailFrom)" -ForegroundColor Gray
        Write-Host "    To: $($Config.EmailTo)" -ForegroundColor Gray
        Write-Host "    SMTP: $($Config.SmtpServer):$($Config.SmtpPort)" -ForegroundColor Gray
    }

    Write-Host ""
}

# Point d'entrée principal
function Start-NotificationTest {
    param()

    Write-Banner -Title "TradingBot" -Subtitle "Test des Notifications"

    try {
        # Vérifier le venv
        if (-not (Enable-PythonVenv)) {
            Write-LogError "Impossible d'activer l'environnement virtuel"
            return $false
        }

        # Lire la configuration
        $config = Get-EnvConfiguration

        # Afficher le résumé
        Show-ConfigurationSummary -Config $config

        $results = @{
            Telegram = $null
            Email    = $null
        }

        # Test Telegram
        if ($Channel -eq 'all' -or $Channel -eq 'telegram') {
            Write-Section "Test Telegram"

            $telegramConfig = Test-TelegramConfiguration -Config $config

            if (-not $telegramConfig.Configured) {
                Write-LogWarning "Telegram n'est pas configuré"
                foreach ($issue in $telegramConfig.Issues) {
                    Write-Host "  - $issue" -ForegroundColor Yellow
                }
                $results.Telegram = $false
            }
            else {
                $results.Telegram = Send-TestTelegramMessage -TestMessage $Message
            }
        }

        # Test Email
        if ($Channel -eq 'all' -or $Channel -eq 'email') {
            Write-Section "Test Email"

            $emailConfig = Test-EmailConfiguration -Config $config

            if (-not $emailConfig.Enabled) {
                Write-LogInfo "Email désactivé dans la configuration"
                $results.Email = $null  # Non testé car désactivé
            }
            elseif (-not $emailConfig.Configured) {
                Write-LogWarning "Email activé mais mal configuré"
                foreach ($issue in $emailConfig.Issues) {
                    Write-Host "  - $issue" -ForegroundColor Yellow
                }
                $results.Email = $false
            }
            else {
                $results.Email = Send-TestEmailMessage -TestMessage $Message
            }
        }

        # Résumé final
        Write-Section "Résumé des Tests"

        if ($results.Telegram -eq $true) {
            Write-Result -Item "Telegram" -Success $true -Message "Fonctionnel"
        }
        elseif ($results.Telegram -eq $false) {
            Write-Result -Item "Telegram" -Success $false -Message "Échec"
        }

        if ($results.Email -eq $true) {
            Write-Result -Item "Email" -Success $true -Message "Fonctionnel"
        }
        elseif ($results.Email -eq $false) {
            Write-Result -Item "Email" -Success $false -Message "Échec"
        }
        elseif ($results.Email -eq $null -and ($Channel -eq 'all' -or $Channel -eq 'email')) {
            Write-Host "  [SKIP] Email - Désactivé" -ForegroundColor Gray
        }

        Write-Host ""

        # Suggestions
        if ($results.Telegram -eq $false -or $results.Email -eq $false) {
            Write-Section "Suggestions"

            if ($results.Telegram -eq $false) {
                Write-Host "  Telegram:" -ForegroundColor Yellow
                Write-Host "    1. Vérifiez votre TELEGRAM_BOT_TOKEN" -ForegroundColor Gray
                Write-Host "    2. Vérifiez votre TELEGRAM_CHAT_ID" -ForegroundColor Gray
                Write-Host "    3. Assurez-vous d'avoir démarré une conversation avec le bot" -ForegroundColor Gray
            }

            if ($results.Email -eq $false) {
                Write-Host "  Email:" -ForegroundColor Yellow
                Write-Host "    1. Vérifiez vos paramètres SMTP" -ForegroundColor Gray
                Write-Host "    2. Pour Gmail, utilisez un 'App Password'" -ForegroundColor Gray
                Write-Host "    3. Vérifiez que le port n'est pas bloqué" -ForegroundColor Gray
            }
        }

        # Retourner succès si au moins un canal fonctionne
        return ($results.Telegram -eq $true -or $results.Email -eq $true)
    }
    catch {
        Write-LogError "Erreur: $_"
        Write-LogError $_.ScriptStackTrace
        return $false
    }
}

# Exécution
if ($MyInvocation.InvocationName -ne '.') {
    Push-Location $ProjectRoot

    try {
        $result = Start-NotificationTest

        if (-not $result) {
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}
