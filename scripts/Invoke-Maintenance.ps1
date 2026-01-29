<#
.SYNOPSIS
    Script de maintenance automatique pour TradingBot

.DESCRIPTION
    Effectue les tâches de maintenance suivantes:
    - Rotation des fichiers de logs
    - Nettoyage du cache
    - Backup de la base de données
    - Vérification de l'intégrité de la DB
    - Rapport de santé du système

.PARAMETER Task
    Tâche à exécuter: 'all', 'logs', 'cache', 'backup', 'health', 'cleanup'

.PARAMETER DaysToKeep
    Nombre de jours de rétention pour les logs (défaut: 30)

.PARAMETER Verbose
    Afficher plus de détails

.EXAMPLE
    .\Invoke-Maintenance.ps1 -Task all

.EXAMPLE
    .\Invoke-Maintenance.ps1 -Task logs -DaysToKeep 14

.NOTES
    Auteur: TradingBot Team
    Version: 1.0.0
#>

[CmdletBinding()]
param(
    [ValidateSet('all', 'logs', 'cache', 'backup', 'health', 'cleanup')]
    [string]$Task = 'all',

    [int]$DaysToKeep = 30
)

# Configuration stricte
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Obtenir les chemins
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptRoot "..")

# Charger les modules communs
. "$ScriptRoot\common\Write-Log.ps1"

# Configuration
$LogsDir = Join-Path $ProjectRoot "logs"
$DataDir = Join-Path $ProjectRoot "data"
$BackupDir = Join-Path $DataDir "backups"
$CacheDir = Join-Path $DataDir "cache"
$DatabasePath = Join-Path $DataDir "screener.db"

function Invoke-LogRotation {
    <#
    .SYNOPSIS
        Effectue la rotation des fichiers de logs
    #>
    param()

    Write-Section "Rotation des Logs"

    if (-not (Test-Path $LogsDir)) {
        Write-LogInfo "Dossier de logs non trouvé"
        return $true
    }

    $cutoffDate = (Get-Date).AddDays(-$DaysToKeep)
    $oldLogs = Get-ChildItem -Path $LogsDir -File -Recurse |
        Where-Object { $_.LastWriteTime -lt $cutoffDate }

    if ($oldLogs.Count -eq 0) {
        Write-LogInfo "Aucun fichier de log à supprimer"
        return $true
    }

    $totalSize = ($oldLogs | Measure-Object -Property Length -Sum).Sum / 1MB

    Write-LogInfo "Fichiers à supprimer: $($oldLogs.Count)"
    Write-LogInfo "Espace à libérer: $([math]::Round($totalSize, 2)) MB"

    foreach ($log in $oldLogs) {
        try {
            Remove-Item -Path $log.FullName -Force
            Write-LogDebug "Supprimé: $($log.Name)"
        }
        catch {
            Write-LogWarning "Impossible de supprimer: $($log.Name)"
        }
    }

    Write-LogSuccess "Rotation des logs terminée"
    return $true
}

function Clear-Cache {
    <#
    .SYNOPSIS
        Nettoie le cache
    #>
    param()

    Write-Section "Nettoyage du Cache"

    $cacheLocations = @(
        $CacheDir,
        (Join-Path $env:USERPROFILE ".cache\yfinance"),
        (Join-Path $env:LOCALAPPDATA "yfinance")
    )

    $totalCleared = 0

    foreach ($cachePath in $cacheLocations) {
        if (Test-Path $cachePath) {
            try {
                $size = (Get-ChildItem -Path $cachePath -Recurse -File -ErrorAction SilentlyContinue |
                    Measure-Object -Property Length -Sum).Sum / 1MB

                Remove-Item -Path $cachePath -Recurse -Force -ErrorAction SilentlyContinue
                New-Item -ItemType Directory -Path $cachePath -Force | Out-Null

                $totalCleared += $size
                Write-LogDebug "Nettoyé: $cachePath ($([math]::Round($size, 2)) MB)"
            }
            catch {
                Write-LogWarning "Impossible de nettoyer: $cachePath"
            }
        }
    }

    Write-LogSuccess "Cache nettoyé: $([math]::Round($totalCleared, 2)) MB libérés"
    return $true
}

function Backup-Database {
    <#
    .SYNOPSIS
        Crée une backup de la base de données
    #>
    param()

    Write-Section "Backup de la Base de Données"

    if (-not (Test-Path $DatabasePath)) {
        Write-LogWarning "Base de données non trouvée: $DatabasePath"
        return $true
    }

    # Créer le dossier de backup
    $dbBackupDir = Join-Path $BackupDir "database"
    if (-not (Test-Path $dbBackupDir)) {
        New-Item -ItemType Directory -Path $dbBackupDir -Force | Out-Null
    }

    # Nom du fichier de backup
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupName = "screener_$timestamp.db"
    $backupPath = Join-Path $dbBackupDir $backupName

    try {
        # Copier la base de données
        Copy-Item -Path $DatabasePath -Destination $backupPath -Force

        $backupSize = (Get-Item $backupPath).Length / 1MB
        Write-LogSuccess "Backup créé: $backupName ($([math]::Round($backupSize, 2)) MB)"

        # Nettoyer les anciennes backups (garder les 7 dernières)
        $oldBackups = Get-ChildItem -Path $dbBackupDir -Filter "screener_*.db" |
            Sort-Object LastWriteTime -Descending |
            Select-Object -Skip 7

        foreach ($old in $oldBackups) {
            Remove-Item -Path $old.FullName -Force
            Write-LogDebug "Ancienne backup supprimée: $($old.Name)"
        }

        return $true
    }
    catch {
        Write-LogError "Erreur lors de la backup: $_"
        return $false
    }
}

function Test-DatabaseIntegrity {
    <#
    .SYNOPSIS
        Vérifie l'intégrité de la base de données SQLite
    #>
    param()

    Write-Section "Vérification de l'Intégrité DB"

    if (-not (Test-Path $DatabasePath)) {
        Write-LogWarning "Base de données non trouvée"
        return $true
    }

    # Charger le module venv pour utiliser Python
    . "$ScriptRoot\common\Initialize-Venv.ps1"

    if (-not (Enable-PythonVenv)) {
        Write-LogWarning "Impossible de vérifier l'intégrité sans Python"
        return $true
    }

    $pythonPath = Get-VenvPythonPath
    $checkScript = @"
import sqlite3
import sys

db_path = r'$DatabasePath'
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Vérifier l'intégrité
    cursor.execute('PRAGMA integrity_check')
    result = cursor.fetchone()[0]

    if result == 'ok':
        print('OK')
        sys.exit(0)
    else:
        print(f'ERREUR: {result}')
        sys.exit(1)
except Exception as e:
    print(f'ERREUR: {e}')
    sys.exit(1)
finally:
    if 'conn' in dir():
        conn.close()
"@

    try {
        $result = $checkScript | & $pythonPath - 2>&1

        if ($result -eq 'OK') {
            Write-LogSuccess "Intégrité de la base de données: OK"
            return $true
        }
        else {
            Write-LogError "Problème d'intégrité: $result"
            return $false
        }
    }
    catch {
        Write-LogError "Erreur lors de la vérification: $_"
        return $false
    }
}

function Get-SystemHealthReport {
    <#
    .SYNOPSIS
        Génère un rapport de santé du système
    #>
    param()

    Write-Section "Rapport de Santé"

    $report = @{
        Timestamp = Get-Date
        Issues    = @()
    }

    # Espace disque
    $drive = (Get-Item $ProjectRoot).PSDrive
    $freeSpace = [math]::Round((Get-PSDrive $drive.Name).Free / 1GB, 2)

    if ($freeSpace -lt 5) {
        $report.Issues += "Espace disque faible: $freeSpace GB"
        Write-Result -Item "Espace disque" -Success $false -Message "$freeSpace GB (critique)"
    }
    elseif ($freeSpace -lt 20) {
        Write-Result -Item "Espace disque" -Success $true -Message "$freeSpace GB (attention)"
    }
    else {
        Write-Result -Item "Espace disque" -Success $true -Message "$freeSpace GB"
    }

    # Taille des logs
    if (Test-Path $LogsDir) {
        $logsSize = [math]::Round((Get-ChildItem -Path $LogsDir -Recurse -File |
            Measure-Object -Property Length -Sum).Sum / 1MB, 2)

        if ($logsSize -gt 500) {
            $report.Issues += "Logs volumineux: $logsSize MB"
            Write-Result -Item "Taille des logs" -Success $false -Message "$logsSize MB"
        }
        else {
            Write-Result -Item "Taille des logs" -Success $true -Message "$logsSize MB"
        }
    }

    # Taille de la base de données
    if (Test-Path $DatabasePath) {
        $dbSize = [math]::Round((Get-Item $DatabasePath).Length / 1MB, 2)

        if ($dbSize -gt 500) {
            $report.Issues += "Base de données volumineuse: $dbSize MB"
            Write-Result -Item "Taille DB" -Success $false -Message "$dbSize MB"
        }
        else {
            Write-Result -Item "Taille DB" -Success $true -Message "$dbSize MB"
        }
    }
    else {
        Write-Result -Item "Base de données" -Success $false -Message "Non trouvée"
    }

    # Fichier .env
    $envPath = Join-Path $ProjectRoot ".env"
    if (Test-Path $envPath) {
        Write-Result -Item "Configuration (.env)" -Success $true
    }
    else {
        $report.Issues += "Fichier .env manquant"
        Write-Result -Item "Configuration (.env)" -Success $false
    }

    # Environnement virtuel
    $venvPath = Join-Path $ProjectRoot "venv"
    if (Test-Path $venvPath) {
        Write-Result -Item "Environnement virtuel" -Success $true
    }
    else {
        $report.Issues += "Environnement virtuel manquant"
        Write-Result -Item "Environnement virtuel" -Success $false
    }

    # Fichiers de tickers
    $tickersDir = Join-Path $DataDir "tickers"
    if (Test-Path $tickersDir) {
        $tickerFiles = Get-ChildItem -Path $tickersDir -Filter "*.json"
        Write-Result -Item "Fichiers tickers" -Success ($tickerFiles.Count -gt 0) -Message "$($tickerFiles.Count) fichiers"
    }
    else {
        Write-Result -Item "Fichiers tickers" -Success $false -Message "Dossier manquant"
    }

    Write-Host ""

    # Résumé
    if ($report.Issues.Count -eq 0) {
        Write-LogSuccess "Système en bonne santé - Aucun problème détecté"
    }
    else {
        Write-LogWarning "Problèmes détectés: $($report.Issues.Count)"
        foreach ($issue in $report.Issues) {
            Write-Host "  - $issue" -ForegroundColor Yellow
        }
    }

    return $report.Issues.Count -eq 0
}

function Invoke-Cleanup {
    <#
    .SYNOPSIS
        Nettoie les fichiers temporaires
    #>
    param()

    Write-Section "Nettoyage des Fichiers Temporaires"

    $tempPatterns = @(
        "*.pyc",
        "*.pyo",
        "__pycache__",
        "*.tmp",
        "*.bak",
        ".pytest_cache",
        "*.log.1", "*.log.2", "*.log.3", "*.log.4", "*.log.5"
    )

    $totalRemoved = 0

    foreach ($pattern in $tempPatterns) {
        $items = Get-ChildItem -Path $ProjectRoot -Filter $pattern -Recurse -Force -ErrorAction SilentlyContinue

        foreach ($item in $items) {
            try {
                if ($item.PSIsContainer) {
                    Remove-Item -Path $item.FullName -Recurse -Force
                }
                else {
                    Remove-Item -Path $item.FullName -Force
                }
                $totalRemoved++
            }
            catch {
                Write-LogDebug "Impossible de supprimer: $($item.FullName)"
            }
        }
    }

    Write-LogSuccess "Nettoyage terminé: $totalRemoved éléments supprimés"
    return $true
}

# Point d'entrée principal
function Start-Maintenance {
    param()

    Write-Banner -Title "TradingBot Maintenance" -Subtitle "Tâches de Maintenance Automatiques"

    $startTime = Get-Date
    $results = @{
        Success = @()
        Failed  = @()
    }

    try {
        $tasks = @()

        switch ($Task) {
            'all' {
                $tasks = @('logs', 'cache', 'backup', 'cleanup', 'health')
            }
            default {
                $tasks = @($Task)
            }
        }

        foreach ($t in $tasks) {
            $success = $false

            switch ($t) {
                'logs' {
                    $success = Invoke-LogRotation
                }
                'cache' {
                    $success = Clear-Cache
                }
                'backup' {
                    $success = Backup-Database
                }
                'health' {
                    $success = Get-SystemHealthReport
                }
                'cleanup' {
                    $success = Invoke-Cleanup
                }
            }

            if ($success) {
                $results.Success += $t
            }
            else {
                $results.Failed += $t
            }
        }

        # Résumé final
        $duration = (Get-Date) - $startTime

        Write-Section "Résumé"
        Write-LogInfo "Durée: $([int]$duration.TotalSeconds) secondes"
        Write-LogSuccess "Tâches réussies: $($results.Success.Count) - $($results.Success -join ', ')"

        if ($results.Failed.Count -gt 0) {
            Write-LogError "Tâches échouées: $($results.Failed.Count) - $($results.Failed -join ', ')"
        }

        return ($results.Failed.Count -eq 0)
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
        $result = Start-Maintenance

        if (-not $result) {
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}
