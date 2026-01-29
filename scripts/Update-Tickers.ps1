<#
.SYNOPSIS
    Script de mise à jour des listes de tickers pour TradingBot

.DESCRIPTION
    Met à jour les fichiers JSON locaux contenant les listes de tickers:
    - Téléchargement depuis Wikipedia ou autres sources
    - Validation et nettoyage des données
    - Backup des anciennes versions
    - Rapport des changements (ajouts/suppressions)

.PARAMETER Market
    Marché à mettre à jour: 'all', 'nasdaq', 'sp500', 'europe', 'asia'

.PARAMETER Force
    Forcer la mise à jour même si les fichiers sont récents

.PARAMETER BackupDays
    Nombre de jours à conserver pour les backups (défaut: 7)

.EXAMPLE
    .\Update-Tickers.ps1 -Market all

.EXAMPLE
    .\Update-Tickers.ps1 -Market nasdaq -Force

.NOTES
    Auteur: TradingBot Team
    Version: 1.0.0
#>

[CmdletBinding()]
param(
    [ValidateSet('all', 'nasdaq', 'sp500', 'europe', 'asia')]
    [string]$Market = 'all',

    [switch]$Force,

    [int]$BackupDays = 7
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

# Configuration
$TickersDir = Join-Path $ProjectRoot "data\tickers"
$BackupDir = Join-Path $ProjectRoot "data\backups\tickers"
$MaxTickerAge = 7  # Jours avant rafraîchissement recommandé

# URLs Wikipedia pour les tickers
$WikipediaUrls = @{
    nasdaq  = "https://en.wikipedia.org/wiki/Nasdaq-100"
    sp500   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    europe  = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
}

# Tickers Asia ADR (liste statique)
$AsiaADRTickers = @(
    @{symbol = "BABA"; name = "Alibaba Group"; country = "China"},
    @{symbol = "JD"; name = "JD.com"; country = "China"},
    @{symbol = "PDD"; name = "PDD Holdings"; country = "China"},
    @{symbol = "NIO"; name = "NIO Inc"; country = "China"},
    @{symbol = "BIDU"; name = "Baidu Inc"; country = "China"},
    @{symbol = "NTES"; name = "NetEase Inc"; country = "China"},
    @{symbol = "TME"; name = "Tencent Music"; country = "China"},
    @{symbol = "BILI"; name = "Bilibili Inc"; country = "China"},
    @{symbol = "LI"; name = "Li Auto Inc"; country = "China"},
    @{symbol = "XPEV"; name = "XPeng Inc"; country = "China"},
    @{symbol = "SONY"; name = "Sony Group"; country = "Japan"},
    @{symbol = "TM"; name = "Toyota Motor"; country = "Japan"},
    @{symbol = "HMC"; name = "Honda Motor"; country = "Japan"},
    @{symbol = "MUFG"; name = "Mitsubishi UFJ"; country = "Japan"},
    @{symbol = "TSM"; name = "TSMC"; country = "Taiwan"},
    @{symbol = "UMC"; name = "United Microelectronics"; country = "Taiwan"},
    @{symbol = "ASML"; name = "ASML Holding"; country = "Netherlands"},
    @{symbol = "SE"; name = "Sea Limited"; country = "Singapore"},
    @{symbol = "GRAB"; name = "Grab Holdings"; country = "Singapore"},
    @{symbol = "INFY"; name = "Infosys Ltd"; country = "India"},
    @{symbol = "WIT"; name = "Wipro Ltd"; country = "India"},
    @{symbol = "HDB"; name = "HDFC Bank"; country = "India"},
    @{symbol = "IBN"; name = "ICICI Bank"; country = "India"},
    @{symbol = "KB"; name = "KB Financial"; country = "South Korea"},
    @{symbol = "SHG"; name = "Shinhan Financial"; country = "South Korea"}
)

function Test-TickerFileAge {
    <#
    .SYNOPSIS
        Vérifie si un fichier de tickers doit être mis à jour
    #>
    param(
        [Parameter(Mandatory)]
        [string]$FilePath
    )

    if (-not (Test-Path $FilePath)) {
        return $true  # Fichier n'existe pas, mise à jour nécessaire
    }

    $fileAge = (Get-Date) - (Get-Item $FilePath).LastWriteTime

    return $fileAge.TotalDays -gt $MaxTickerAge
}

function Backup-TickerFile {
    <#
    .SYNOPSIS
        Crée une backup du fichier de tickers
    #>
    param(
        [Parameter(Mandatory)]
        [string]$FilePath
    )

    if (-not (Test-Path $FilePath)) {
        return
    }

    # Créer le dossier de backup si nécessaire
    if (-not (Test-Path $BackupDir)) {
        New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    }

    $fileName = Split-Path -Leaf $FilePath
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($fileName)
    $extension = [System.IO.Path]::GetExtension($fileName)
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupName = "${baseName}_${timestamp}${extension}"
    $backupPath = Join-Path $BackupDir $backupName

    Copy-Item -Path $FilePath -Destination $backupPath -Force
    Write-LogDebug "Backup créé: $backupName"

    # Nettoyer les anciennes backups
    Get-ChildItem -Path $BackupDir -Filter "${baseName}_*${extension}" |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$BackupDays) } |
        Remove-Item -Force
}

function Get-TickersFromWikipedia {
    <#
    .SYNOPSIS
        Récupère les tickers depuis Wikipedia via Python
    #>
    param(
        [Parameter(Mandatory)]
        [string]$MarketName
    )

    $pythonPath = Get-VenvPythonPath

    if (-not (Test-Path $pythonPath)) {
        Write-LogError "Python non trouvé. Exécutez Setup-Environment.ps1"
        return $null
    }

    $pythonScript = @"
import pandas as pd
import json
from datetime import datetime

def get_nasdaq100():
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        df = tables[4]  # Table des composants
        tickers = []
        for _, row in df.iterrows():
            tickers.append({
                'symbol': row.get('Ticker', row.get('Symbol', '')),
                'name': row.get('Company', ''),
                'sector': row.get('GICS Sector', row.get('Sector', ''))
            })
        return tickers
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_sp500():
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = tables[0]
        tickers = []
        for _, row in df.iterrows():
            tickers.append({
                'symbol': row.get('Symbol', ''),
                'name': row.get('Security', ''),
                'sector': row.get('GICS Sector', '')
            })
        return tickers
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_eurostoxx50():
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/EURO_STOXX_50')
        df = tables[2]  # Table des composants
        tickers = []
        for _, row in df.iterrows():
            tickers.append({
                'symbol': row.get('Ticker', row.get('Symbol', '')),
                'name': row.get('Name', row.get('Company', '')),
                'sector': row.get('Sector', '')
            })
        return tickers
    except Exception as e:
        print(f"Error: {e}")
        return []

market = '$MarketName'
if market == 'nasdaq':
    result = get_nasdaq100()
elif market == 'sp500':
    result = get_sp500()
elif market == 'europe':
    result = get_eurostoxx50()
else:
    result = []

print(json.dumps(result))
"@

    try {
        $result = $pythonScript | & $pythonPath - 2>&1

        if ($LASTEXITCODE -ne 0) {
            Write-LogError "Erreur lors de la récupération des tickers: $result"
            return $null
        }

        # Parser le JSON
        $tickers = $result | ConvertFrom-Json

        # Filtrer et nettoyer
        $cleanTickers = $tickers | Where-Object {
            $_.symbol -and $_.symbol -match '^[A-Z0-9.]+$'
        } | ForEach-Object {
            @{
                symbol = $_.symbol.Trim()
                name   = if ($_.name) { $_.name.Trim() } else { $_.symbol }
                sector = if ($_.sector) { $_.sector.Trim() } else { "Unknown" }
            }
        }

        return $cleanTickers
    }
    catch {
        Write-LogError "Exception lors de la récupération des tickers: $_"
        return $null
    }
}

function Save-TickersToJson {
    <#
    .SYNOPSIS
        Sauvegarde les tickers dans un fichier JSON
    #>
    param(
        [Parameter(Mandatory)]
        [string]$MarketName,

        [Parameter(Mandatory)]
        [array]$Tickers
    )

    $filePath = Join-Path $TickersDir "$MarketName.json"

    # Créer le dossier si nécessaire
    if (-not (Test-Path $TickersDir)) {
        New-Item -ItemType Directory -Path $TickersDir -Force | Out-Null
    }

    # Backup de l'ancien fichier
    Backup-TickerFile -FilePath $filePath

    # Préparer les données
    $data = @{
        updated = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
        source  = "wikipedia"
        market  = $MarketName
        count   = $Tickers.Count
        tickers = $Tickers
    }

    # Sauvegarder
    $data | ConvertTo-Json -Depth 10 | Out-File -FilePath $filePath -Encoding UTF8 -Force

    Write-LogSuccess "Fichier sauvegardé: $MarketName.json ($($Tickers.Count) tickers)"

    return $filePath
}

function Update-SingleMarket {
    <#
    .SYNOPSIS
        Met à jour les tickers d'un marché spécifique
    #>
    param(
        [Parameter(Mandatory)]
        [string]$MarketName
    )

    $filePath = Join-Path $TickersDir "$MarketName.json"

    Write-Section "Mise à jour: $($MarketName.ToUpper())"

    # Vérifier si mise à jour nécessaire
    if (-not $Force -and -not (Test-TickerFileAge -FilePath $filePath)) {
        $fileAge = [int]((Get-Date) - (Get-Item $filePath).LastWriteTime).TotalDays
        Write-LogInfo "Fichier récent ($fileAge jours). Utilisez -Force pour forcer."
        return $true
    }

    # Charger les anciens tickers pour comparaison
    $oldTickers = @()
    if (Test-Path $filePath) {
        try {
            $oldData = Get-Content $filePath -Raw | ConvertFrom-Json
            $oldTickers = $oldData.tickers | ForEach-Object { $_.symbol }
        }
        catch {
            Write-LogWarning "Impossible de lire l'ancien fichier"
        }
    }

    # Récupérer les nouveaux tickers
    if ($MarketName -eq 'asia') {
        $newTickers = $AsiaADRTickers
    }
    else {
        Write-LogInfo "Téléchargement depuis Wikipedia..."
        $newTickers = Get-TickersFromWikipedia -MarketName $MarketName

        if (-not $newTickers -or $newTickers.Count -eq 0) {
            Write-LogError "Aucun ticker récupéré pour $MarketName"
            return $false
        }
    }

    # Validation minimale
    if ($newTickers.Count -lt 10) {
        Write-LogError "Trop peu de tickers récupérés ($($newTickers.Count)). Abandon."
        return $false
    }

    # Sauvegarder
    Save-TickersToJson -MarketName $MarketName -Tickers $newTickers

    # Rapport des changements
    $newSymbols = $newTickers | ForEach-Object { $_.symbol }

    $added = $newSymbols | Where-Object { $_ -notin $oldTickers }
    $removed = $oldTickers | Where-Object { $_ -notin $newSymbols }

    if ($added.Count -gt 0) {
        Write-LogInfo "Ajoutés: $($added.Count) - $($added -join ', ')"
    }
    if ($removed.Count -gt 0) {
        Write-LogWarning "Supprimés: $($removed.Count) - $($removed -join ', ')"
    }
    if ($added.Count -eq 0 -and $removed.Count -eq 0) {
        Write-LogInfo "Aucun changement détecté"
    }

    return $true
}

function Update-AllMarkets {
    <#
    .SYNOPSIS
        Met à jour tous les marchés
    #>
    param()

    $markets = @('nasdaq', 'sp500', 'europe', 'asia')
    $results = @{
        Success = @()
        Failed  = @()
    }

    foreach ($market in $markets) {
        if (Update-SingleMarket -MarketName $market) {
            $results.Success += $market
        }
        else {
            $results.Failed += $market
        }
    }

    return $results
}

function Show-TickersSummary {
    <#
    .SYNOPSIS
        Affiche un résumé des fichiers de tickers
    #>
    param()

    Write-Section "Résumé des Fichiers de Tickers"

    $markets = @('nasdaq', 'sp500', 'europe', 'asia')

    foreach ($market in $markets) {
        $filePath = Join-Path $TickersDir "$market.json"

        if (Test-Path $filePath) {
            $data = Get-Content $filePath -Raw | ConvertFrom-Json
            $fileAge = [int]((Get-Date) - (Get-Item $filePath).LastWriteTime).TotalDays
            $status = if ($fileAge -gt $MaxTickerAge) { "Ancien" } else { "OK" }

            Write-Host "  $($market.ToUpper().PadRight(10)) " -NoNewline
            Write-Host "$($data.count.ToString().PadLeft(4)) tickers" -NoNewline -ForegroundColor Cyan
            Write-Host " | Màj: $($data.updated.Substring(0,10))" -NoNewline -ForegroundColor Gray
            Write-Host " | $status" -ForegroundColor $(if ($status -eq "OK") { "Green" } else { "Yellow" })
        }
        else {
            Write-Host "  $($market.ToUpper().PadRight(10)) " -NoNewline
            Write-Host "Non trouvé" -ForegroundColor Red
        }
    }

    Write-Host ""
}

# Point d'entrée principal
function Start-TickerUpdate {
    param()

    Write-Banner -Title "TradingBot" -Subtitle "Mise à jour des Tickers"

    try {
        # Vérifier le venv
        if (-not (Enable-PythonVenv)) {
            Write-LogError "Impossible d'activer l'environnement virtuel"
            return $false
        }

        # Afficher le résumé actuel
        Show-TickersSummary

        # Mettre à jour
        if ($Market -eq 'all') {
            $results = Update-AllMarkets

            Write-Section "Résultat Final"
            Write-LogSuccess "Succès: $($results.Success.Count) - $($results.Success -join ', ')"

            if ($results.Failed.Count -gt 0) {
                Write-LogError "Échecs: $($results.Failed.Count) - $($results.Failed -join ', ')"
            }

            return ($results.Failed.Count -eq 0)
        }
        else {
            return Update-SingleMarket -MarketName $Market
        }
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
        $result = Start-TickerUpdate

        if (-not $result) {
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}
