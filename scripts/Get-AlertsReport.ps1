<#
.SYNOPSIS
    Script d'extraction et de rapport des alertes TradingBot

.DESCRIPTION
    Extrait les alertes de la base de données et génère des rapports:
    - Affichage en tableau
    - Export CSV ou JSON
    - Filtrage par période et recommandation
    - Statistiques résumées

.PARAMETER Days
    Nombre de jours à inclure (défaut: 7)

.PARAMETER Format
    Format de sortie: 'table', 'csv', 'json'

.PARAMETER Output
    Chemin du fichier de sortie (optionnel)

.PARAMETER Recommendation
    Filtrer par type de recommandation: 'all', 'STRONG_BUY', 'BUY', 'WATCH', 'OBSERVE'

.PARAMETER Top
    Limiter aux N premières alertes

.EXAMPLE
    .\Get-AlertsReport.ps1 -Days 30

.EXAMPLE
    .\Get-AlertsReport.ps1 -Days 7 -Format csv -Output "alerts.csv"

.EXAMPLE
    .\Get-AlertsReport.ps1 -Recommendation STRONG_BUY -Top 10

.NOTES
    Auteur: TradingBot Team
    Version: 1.0.0
#>

[CmdletBinding()]
param(
    [int]$Days = 7,

    [ValidateSet('table', 'csv', 'json')]
    [string]$Format = 'table',

    [string]$Output,

    [ValidateSet('all', 'STRONG_BUY', 'BUY', 'WATCH', 'OBSERVE')]
    [string]$Recommendation = 'all',

    [int]$Top = 0
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
$DatabasePath = Join-Path $ProjectRoot "data\screener.db"

function Get-AlertsFromDatabase {
    <#
    .SYNOPSIS
        Récupère les alertes depuis la base de données SQLite
    #>
    param(
        [int]$DaysBack,
        [string]$FilterRecommendation
    )

    if (-not (Test-Path $DatabasePath)) {
        Write-LogError "Base de données non trouvée: $DatabasePath"
        return @()
    }

    $pythonPath = Get-VenvPythonPath

    $whereClause = "WHERE alert_date >= datetime('now', '-$DaysBack days')"
    if ($FilterRecommendation -ne 'all') {
        $whereClause += " AND recommendation = '$FilterRecommendation'"
    }

    $queryScript = @"
import sqlite3
import json
from datetime import datetime

db_path = r'$DatabasePath'
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

try:
    cursor.execute('''
        SELECT
            symbol,
            company_name,
            alert_date,
            timeframe,
            current_price,
            support_level,
            distance_to_support_pct,
            ema_24,
            ema_38,
            ema_62,
            ema_alignment,
            recommendation,
            is_notified
        FROM stock_alerts
        $whereClause
        ORDER BY alert_date DESC
    ''')

    rows = cursor.fetchall()
    alerts = []

    for row in rows:
        alerts.append({
            'symbol': row['symbol'],
            'company_name': row['company_name'] or row['symbol'],
            'alert_date': row['alert_date'],
            'timeframe': row['timeframe'],
            'current_price': round(row['current_price'], 2) if row['current_price'] else 0,
            'support_level': round(row['support_level'], 2) if row['support_level'] else 0,
            'distance_pct': round(row['distance_to_support_pct'], 2) if row['distance_to_support_pct'] else 0,
            'ema_alignment': row['ema_alignment'],
            'recommendation': row['recommendation'],
            'is_notified': bool(row['is_notified'])
        })

    print(json.dumps(alerts))

except Exception as e:
    print(json.dumps({'error': str(e)}))

finally:
    conn.close()
"@

    try {
        $result = $queryScript | & $pythonPath - 2>&1

        if ($result -match '"error"') {
            $error = $result | ConvertFrom-Json
            Write-LogError "Erreur SQL: $($error.error)"
            return @()
        }

        $alerts = $result | ConvertFrom-Json
        return $alerts
    }
    catch {
        Write-LogError "Erreur lors de la requête: $_"
        return @()
    }
}

function Get-AlertStatistics {
    <#
    .SYNOPSIS
        Calcule les statistiques des alertes
    #>
    param(
        [array]$Alerts
    )

    $stats = [PSCustomObject]@{
        TotalAlerts   = $Alerts.Count
        ByRecommendation = @{
            STRONG_BUY = ($Alerts | Where-Object { $_.recommendation -eq 'STRONG_BUY' }).Count
            BUY        = ($Alerts | Where-Object { $_.recommendation -eq 'BUY' }).Count
            WATCH      = ($Alerts | Where-Object { $_.recommendation -eq 'WATCH' }).Count
            OBSERVE    = ($Alerts | Where-Object { $_.recommendation -eq 'OBSERVE' }).Count
        }
        ByTimeframe = @{
            weekly = ($Alerts | Where-Object { $_.timeframe -eq 'weekly' }).Count
            daily  = ($Alerts | Where-Object { $_.timeframe -eq 'daily' }).Count
        }
        AverageDistance = 0
        NotifiedCount   = ($Alerts | Where-Object { $_.is_notified }).Count
    }

    if ($Alerts.Count -gt 0) {
        $stats.AverageDistance = [math]::Round(($Alerts | Measure-Object -Property distance_pct -Average).Average, 2)
    }

    return $stats
}

function Show-AlertsTable {
    <#
    .SYNOPSIS
        Affiche les alertes en format tableau
    #>
    param(
        [array]$Alerts
    )

    if ($Alerts.Count -eq 0) {
        Write-LogInfo "Aucune alerte trouvée pour la période spécifiée"
        return
    }

    Write-Host ""
    Write-Host ("{0,-8} {1,-25} {2,-10} {3,-10} {4,-10} {5,-8} {6,-12}" -f `
        "Symbol", "Company", "Date", "Price", "Support", "Dist%", "Recomm.") -ForegroundColor Cyan
    Write-Host ("-" * 90) -ForegroundColor Gray

    foreach ($alert in $Alerts) {
        $dateStr = if ($alert.alert_date) { $alert.alert_date.Substring(0, 10) } else { "N/A" }
        $recommColor = switch ($alert.recommendation) {
            'STRONG_BUY' { 'Green' }
            'BUY' { 'Yellow' }
            'WATCH' { 'White' }
            'OBSERVE' { 'Gray' }
            default { 'White' }
        }

        $company = if ($alert.company_name.Length -gt 24) {
            $alert.company_name.Substring(0, 21) + "..."
        } else {
            $alert.company_name
        }

        Write-Host ("{0,-8} {1,-25} {2,-10} {3,9:N2} {4,9:N2} {5,7:N2} " -f `
            $alert.symbol, $company, $dateStr, $alert.current_price, $alert.support_level, $alert.distance_pct) -NoNewline

        Write-Host ("{0,-12}" -f $alert.recommendation) -ForegroundColor $recommColor
    }

    Write-Host ""
}

function Export-AlertsCsv {
    <#
    .SYNOPSIS
        Exporte les alertes en CSV
    #>
    param(
        [array]$Alerts,
        [string]$FilePath
    )

    if ($Alerts.Count -eq 0) {
        Write-LogWarning "Aucune alerte à exporter"
        return $false
    }

    try {
        $Alerts | ForEach-Object {
            [PSCustomObject]@{
                Symbol = $_.symbol
                Company = $_.company_name
                Date = $_.alert_date
                Timeframe = $_.timeframe
                Price = $_.current_price
                Support = $_.support_level
                Distance = $_.distance_pct
                EMA_Alignment = $_.ema_alignment
                Recommendation = $_.recommendation
                Notified = $_.is_notified
            }
        } | Export-Csv -Path $FilePath -NoTypeInformation -Encoding UTF8

        Write-LogSuccess "Exporté vers: $FilePath"
        return $true
    }
    catch {
        Write-LogError "Erreur d'export CSV: $_"
        return $false
    }
}

function Export-AlertsJson {
    <#
    .SYNOPSIS
        Exporte les alertes en JSON
    #>
    param(
        [array]$Alerts,
        [string]$FilePath
    )

    if ($Alerts.Count -eq 0) {
        Write-LogWarning "Aucune alerte à exporter"
        return $false
    }

    try {
        $exportData = @{
            exported_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
            count = $Alerts.Count
            alerts = $Alerts
        }

        $exportData | ConvertTo-Json -Depth 10 | Out-File -FilePath $FilePath -Encoding UTF8

        Write-LogSuccess "Exporté vers: $FilePath"
        return $true
    }
    catch {
        Write-LogError "Erreur d'export JSON: $_"
        return $false
    }
}

function Show-Statistics {
    <#
    .SYNOPSIS
        Affiche les statistiques
    #>
    param(
        [PSCustomObject]$Stats
    )

    Write-Section "Statistiques"

    Write-Host "  Total alertes: $($Stats.TotalAlerts)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Par recommandation:" -ForegroundColor White
    Write-Host "    STRONG_BUY: $($Stats.ByRecommendation.STRONG_BUY)" -ForegroundColor Green
    Write-Host "    BUY:        $($Stats.ByRecommendation.BUY)" -ForegroundColor Yellow
    Write-Host "    WATCH:      $($Stats.ByRecommendation.WATCH)" -ForegroundColor White
    Write-Host "    OBSERVE:    $($Stats.ByRecommendation.OBSERVE)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Par timeframe:" -ForegroundColor White
    Write-Host "    Weekly: $($Stats.ByTimeframe.weekly)" -ForegroundColor Cyan
    Write-Host "    Daily:  $($Stats.ByTimeframe.daily)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Distance moyenne: $($Stats.AverageDistance)%" -ForegroundColor White
    Write-Host "  Notifiées: $($Stats.NotifiedCount)" -ForegroundColor White
    Write-Host ""
}

# Point d'entrée principal
function Start-AlertsReport {
    param()

    Write-Banner -Title "TradingBot" -Subtitle "Rapport des Alertes"

    try {
        # Vérifier le venv
        if (-not (Enable-PythonVenv)) {
            Write-LogError "Impossible d'activer l'environnement virtuel"
            return $false
        }

        # Vérifier la base de données
        if (-not (Test-Path $DatabasePath)) {
            Write-LogError "Base de données non trouvée"
            Write-LogInfo "Exécutez d'abord un screening pour créer la base de données"
            return $false
        }

        Write-LogInfo "Période: $Days derniers jours"
        if ($Recommendation -ne 'all') {
            Write-LogInfo "Filtre: $Recommendation"
        }

        # Récupérer les alertes
        $alerts = Get-AlertsFromDatabase -DaysBack $Days -FilterRecommendation $Recommendation

        # Limiter si -Top spécifié
        if ($Top -gt 0 -and $alerts.Count -gt $Top) {
            $alerts = $alerts | Select-Object -First $Top
            Write-LogInfo "Limité aux $Top premières alertes"
        }

        # Afficher selon le format
        switch ($Format) {
            'table' {
                Show-AlertsTable -Alerts $alerts

                # Afficher les statistiques
                if ($alerts.Count -gt 0) {
                    $stats = Get-AlertStatistics -Alerts $alerts
                    Show-Statistics -Stats $stats
                }
            }
            'csv' {
                $outputPath = if ($Output) { $Output } else { Join-Path $ProjectRoot "alerts_export.csv" }
                Export-AlertsCsv -Alerts $alerts -FilePath $outputPath
            }
            'json' {
                $outputPath = if ($Output) { $Output } else { Join-Path $ProjectRoot "alerts_export.json" }
                Export-AlertsJson -Alerts $alerts -FilePath $outputPath
            }
        }

        return $true
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
        $result = Start-AlertsReport

        if (-not $result) {
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}
