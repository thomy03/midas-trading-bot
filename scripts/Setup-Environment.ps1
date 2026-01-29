<#
.SYNOPSIS
    Script d'installation et de configuration de l'environnement TradingBot

.DESCRIPTION
    Ce script effectue l'installation complète de l'environnement:
    - Vérification de Python (>= 3.8)
    - Création de l'environnement virtuel
    - Installation des dépendances
    - Création des dossiers nécessaires
    - Vérification/création du fichier .env
    - Rapport de santé du système

.PARAMETER Force
    Recréer l'environnement virtuel même s'il existe

.PARAMETER SkipDeps
    Ne pas installer les dépendances Python

.PARAMETER SkipEnvCheck
    Ne pas vérifier le fichier .env

.EXAMPLE
    .\Setup-Environment.ps1

.EXAMPLE
    .\Setup-Environment.ps1 -Force -Verbose

.NOTES
    Auteur: TradingBot Team
    Version: 1.0.0
#>

[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$SkipDeps,
    [switch]$SkipEnvCheck
)

# Configuration stricte
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Obtenir le chemin du projet
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptRoot "..")

# Charger les modules communs
. "$ScriptRoot\common\Write-Log.ps1"
. "$ScriptRoot\common\Initialize-Venv.ps1"

# Configuration des dossiers à créer
$RequiredDirectories = @(
    "logs",
    "data",
    "data\tickers",
    "data\backups",
    "data\cache"
)

# Template du fichier .env
$EnvTemplate = @"
# TradingBot Configuration
# ========================
# Renommez ce fichier en .env et remplissez les valeurs

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Email Configuration (optionnel)
EMAIL_ENABLED=False
EMAIL_FROM=your_email@example.com
EMAIL_TO=recipient@example.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@example.com
SMTP_PASSWORD=your_email_password_or_app_password

# API Keys (optionnel)
# ALPHA_VANTAGE_API_KEY=your_api_key
# FMP_API_KEY=your_financial_modeling_prep_key
"@

function Test-EnvFile {
    <#
    .SYNOPSIS
        Vérifie la présence et le contenu du fichier .env
    #>
    param()

    $envPath = Join-Path $ProjectRoot ".env"
    $envExamplePath = Join-Path $ProjectRoot ".env.example"

    $result = [PSCustomObject]@{
        Exists = $false
        IsConfigured = $false
        TelegramConfigured = $false
        EmailConfigured = $false
        MissingVariables = @()
    }

    if (Test-Path $envPath) {
        $result.Exists = $true

        # Lire le contenu
        $envContent = Get-Content $envPath -Raw

        # Vérifier Telegram
        if ($envContent -match "TELEGRAM_BOT_TOKEN=(?!your_).+" -and
            $envContent -match "TELEGRAM_CHAT_ID=(?!your_).+") {
            $result.TelegramConfigured = $true
        } else {
            $result.MissingVariables += "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"
        }

        # Vérifier Email
        if ($envContent -match "EMAIL_ENABLED=True" -and
            $envContent -match "EMAIL_FROM=(?!your_).+" -and
            $envContent -match "EMAIL_TO=(?!recipient).+") {
            $result.EmailConfigured = $true
        }

        $result.IsConfigured = $result.TelegramConfigured
    }

    return $result
}

function New-EnvFile {
    <#
    .SYNOPSIS
        Crée un fichier .env template
    #>
    param()

    $envPath = Join-Path $ProjectRoot ".env"
    $envExamplePath = Join-Path $ProjectRoot ".env.example"

    # Créer .env.example si n'existe pas
    if (-not (Test-Path $envExamplePath)) {
        $EnvTemplate | Out-File -FilePath $envExamplePath -Encoding UTF8
        Write-LogInfo "Fichier .env.example créé"
    }

    # Créer .env si n'existe pas
    if (-not (Test-Path $envPath)) {
        $EnvTemplate | Out-File -FilePath $envPath -Encoding UTF8
        Write-LogWarning "Fichier .env créé - VEUILLEZ LE CONFIGURER"
        return $false
    }

    return $true
}

function New-RequiredDirectories {
    <#
    .SYNOPSIS
        Crée les dossiers nécessaires pour le projet
    #>
    param()

    $allCreated = $true

    foreach ($dir in $RequiredDirectories) {
        $fullPath = Join-Path $ProjectRoot $dir

        if (-not (Test-Path $fullPath)) {
            try {
                New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
                Write-LogDebug "Dossier créé: $dir"
            }
            catch {
                Write-LogError "Impossible de créer le dossier: $dir"
                $allCreated = $false
            }
        }
    }

    return $allCreated
}

function Test-DatabaseExists {
    <#
    .SYNOPSIS
        Vérifie si la base de données SQLite existe
    #>
    param()

    $dbPath = Join-Path $ProjectRoot "data\screener.db"
    return Test-Path $dbPath
}

function Get-SystemHealthReport {
    <#
    .SYNOPSIS
        Génère un rapport de santé du système
    #>
    param()

    Write-Section "Rapport de Santé du Système"

    # Python
    $pythonInfo = Get-PythonInfo
    Write-Result -Item "Python installé" -Success $pythonInfo.PythonInstalled -Message $(if ($pythonInfo.PythonInstalled) { "Version $($pythonInfo.PythonVersion)" } else { "Non trouvé" })

    # Venv
    Write-Result -Item "Environnement virtuel" -Success $pythonInfo.VenvExists -Message $(if ($pythonInfo.VenvExists) { $pythonInfo.VenvPath } else { "Non créé" })

    # Requirements
    Write-Result -Item "requirements.txt" -Success $pythonInfo.RequirementsExists

    # Dossiers
    $dirsOk = $true
    foreach ($dir in $RequiredDirectories) {
        $fullPath = Join-Path $ProjectRoot $dir
        if (-not (Test-Path $fullPath)) {
            $dirsOk = $false
            break
        }
    }
    Write-Result -Item "Dossiers requis" -Success $dirsOk

    # .env
    $envStatus = Test-EnvFile
    Write-Result -Item "Fichier .env" -Success $envStatus.Exists -Message $(if (-not $envStatus.Exists) { "Non trouvé" } elseif ($envStatus.IsConfigured) { "Configuré" } else { "Non configuré" })

    # Database
    $dbExists = Test-DatabaseExists
    Write-Result -Item "Base de données" -Success $dbExists -Message $(if ($dbExists) { "Initialisée" } else { "Sera créée au premier run" })

    # Packages critiques
    if ($pythonInfo.VenvActivated) {
        $criticalPackages = @("yfinance", "pandas", "streamlit", "sqlalchemy")
        foreach ($pkg in $criticalPackages) {
            $pkgInstalled = Test-PythonPackage -PackageName $pkg
            Write-Result -Item "Package: $pkg" -Success $pkgInstalled
        }
    }

    Write-Host ""
}

function Start-Setup {
    <#
    .SYNOPSIS
        Fonction principale d'installation
    #>
    param()

    Write-Banner -Title "TradingBot Setup" -Subtitle "Installation et Configuration"

    $startTime = Get-Date
    $success = $true

    try {
        # Étape 1: Vérification Python
        Write-Section "Vérification de Python"

        $pythonInfo = Test-PythonInstalled
        if (-not $pythonInfo.Installed) {
            Write-LogError "Python n'est pas installé sur ce système."
            Write-LogError "Veuillez installer Python 3.8 ou supérieur depuis https://python.org"
            return $false
        }

        Write-Result -Item "Python" -Success $true -Message "Version $($pythonInfo.PythonVersion) - $($pythonInfo.Path)"

        if (-not (Test-PythonVersion)) {
            Write-LogError "La version de Python ($($pythonInfo.PythonVersion)) est trop ancienne."
            Write-LogError "Version 3.8 ou supérieure requise."
            return $false
        }

        # Étape 2: Création des dossiers
        Write-Section "Création des Dossiers"

        if (New-RequiredDirectories) {
            Write-LogSuccess "Tous les dossiers ont été créés/vérifiés"
        } else {
            Write-LogWarning "Certains dossiers n'ont pas pu être créés"
        }

        # Étape 3: Environnement virtuel
        Write-Section "Environnement Virtuel Python"

        if (-not (Initialize-PythonVenv -Force:$Force -SkipDependencies:$SkipDeps)) {
            Write-LogError "Échec de l'initialisation de l'environnement virtuel"
            $success = $false
        }

        # Étape 4: Fichier .env
        if (-not $SkipEnvCheck) {
            Write-Section "Configuration (.env)"

            $envStatus = Test-EnvFile

            if (-not $envStatus.Exists) {
                New-EnvFile | Out-Null
                Write-LogWarning "Un fichier .env a été créé."
                Write-LogWarning "IMPORTANT: Veuillez le configurer avec vos credentials Telegram avant d'utiliser le bot."
            }
            elseif (-not $envStatus.IsConfigured) {
                Write-LogWarning "Le fichier .env existe mais n'est pas configuré."
                Write-LogWarning "Variables manquantes: $($envStatus.MissingVariables -join ', ')"
            }
            else {
                Write-LogSuccess "Fichier .env configuré"
                if ($envStatus.TelegramConfigured) {
                    Write-LogInfo "  - Telegram: Configuré"
                }
                if ($envStatus.EmailConfigured) {
                    Write-LogInfo "  - Email: Configuré"
                }
            }
        }

        # Étape 5: Rapport final
        Get-SystemHealthReport

        # Résumé
        $duration = (Get-Date) - $startTime
        Write-Host ""

        if ($success) {
            Write-LogSuccess "Installation terminée avec succès en $([int]$duration.TotalSeconds) secondes"
            Write-Host ""
            Write-Host "Prochaines étapes:" -ForegroundColor Cyan
            Write-Host "  1. Configurez le fichier .env avec vos credentials Telegram" -ForegroundColor White
            Write-Host "  2. Exécutez: .\Start-Screening.ps1 -Mode run" -ForegroundColor White
            Write-Host "  3. Ou lancez le dashboard: .\Start-Dashboard.ps1" -ForegroundColor White
        }
        else {
            Write-LogError "Installation terminée avec des erreurs"
        }

        return $success
    }
    catch {
        Write-LogError "Erreur fatale lors de l'installation: $_"
        Write-LogError $_.ScriptStackTrace
        return $false
    }
}

# Point d'entrée
if ($MyInvocation.InvocationName -ne '.') {
    # Script exécuté directement (pas sourcé)
    Push-Location $ProjectRoot

    try {
        $result = Start-Setup

        if (-not $result) {
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}
