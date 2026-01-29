<#
.SYNOPSIS
    Script de planification automatique du screening TradingBot

.DESCRIPTION
    Gère le scheduler pour l'exécution automatique du screening:
    - Exécution quotidienne à heure configurable
    - Mode arrière-plan avec Jobs PowerShell
    - Gestion robuste des erreurs avec retry
    - Logging continu et notifications

.PARAMETER Time
    Heure d'exécution quotidienne au format HH:mm (défaut: 08:00)

.PARAMETER Background
    Exécuter le scheduler en arrière-plan

.PARAMETER RunNow
    Exécuter immédiatement un screening au démarrage

.PARAMETER Stop
    Arrêter le scheduler en cours

.PARAMETER Status
    Afficher le statut du scheduler

.EXAMPLE
    .\Start-Scheduler.ps1

.EXAMPLE
    .\Start-Scheduler.ps1 -Time "09:30" -Background

.EXAMPLE
    .\Start-Scheduler.ps1 -Stop

.NOTES
    Auteur: TradingBot Team
    Version: 1.0.0
#>

[CmdletBinding()]
param(
    [ValidatePattern('^\d{2}:\d{2}$')]
    [string]$Time = "08:00",

    [switch]$Background,

    [switch]$RunNow,

    [switch]$Stop,

    [switch]$Status
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
$JobName = "TradingBot-Scheduler"
$LockFile = Join-Path $ProjectRoot "data\.scheduler.lock"
$StatusFile = Join-Path $ProjectRoot "data\.scheduler.status"

function Get-SchedulerStatus {
    <#
    .SYNOPSIS
        Retourne le statut détaillé du scheduler
    #>
    param()

    $status = [PSCustomObject]@{
        Running         = $false
        JobState        = $null
        ProcessId       = $null
        ScheduledTime   = $null
        LastRun         = $null
        LastRunStatus   = $null
        NextRun         = $null
        StartedAt       = $null
    }

    # Vérifier le job PowerShell
    $job = Get-Job -Name $JobName -ErrorAction SilentlyContinue
    if ($job) {
        $status.JobState = $job.State
        $status.Running = ($job.State -eq 'Running')
    }

    # Vérifier le fichier de verrouillage
    if (Test-Path $LockFile) {
        $lockContent = Get-Content $LockFile -ErrorAction SilentlyContinue
        if ($lockContent) {
            $status.ProcessId = $lockContent
            $status.Running = $true
        }
    }

    # Lire le fichier de statut
    if (Test-Path $StatusFile) {
        try {
            $statusData = Get-Content $StatusFile -Raw | ConvertFrom-Json
            $status.ScheduledTime = $statusData.ScheduledTime
            $status.LastRun = $statusData.LastRun
            $status.LastRunStatus = $statusData.LastRunStatus
            $status.StartedAt = $statusData.StartedAt
        }
        catch {
            Write-LogDebug "Impossible de lire le fichier de statut"
        }
    }

    # Calculer la prochaine exécution
    if ($status.ScheduledTime) {
        $now = Get-Date
        $scheduledToday = Get-Date $status.ScheduledTime

        if ($now -gt $scheduledToday) {
            $status.NextRun = $scheduledToday.AddDays(1)
        }
        else {
            $status.NextRun = $scheduledToday
        }
    }

    return $status
}

function Save-SchedulerStatus {
    <#
    .SYNOPSIS
        Sauvegarde le statut du scheduler
    #>
    param(
        [string]$ScheduledTime,
        [string]$LastRun,
        [string]$LastRunStatus,
        [string]$StartedAt
    )

    $statusData = @{
        ScheduledTime = $ScheduledTime
        LastRun       = $LastRun
        LastRunStatus = $LastRunStatus
        StartedAt     = $StartedAt
    }

    $statusData | ConvertTo-Json | Out-File -FilePath $StatusFile -Encoding UTF8 -Force
}

function New-SchedulerLock {
    <#
    .SYNOPSIS
        Crée un fichier de verrouillage
    #>
    param()

    $PID | Out-File -FilePath $LockFile -Force
}

function Remove-SchedulerLock {
    <#
    .SYNOPSIS
        Supprime le fichier de verrouillage
    #>
    param()

    if (Test-Path $LockFile) {
        Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
    }
}

function Stop-Scheduler {
    <#
    .SYNOPSIS
        Arrête le scheduler en cours
    #>
    param()

    Write-LogInfo "Arrêt du scheduler..."

    # Arrêter le job
    $job = Get-Job -Name $JobName -ErrorAction SilentlyContinue
    if ($job) {
        $job | Stop-Job -PassThru | Remove-Job -Force
        Write-LogInfo "Job arrêté"
    }

    # Supprimer le verrou
    Remove-SchedulerLock

    # Mettre à jour le statut
    if (Test-Path $StatusFile) {
        $statusData = Get-Content $StatusFile -Raw | ConvertFrom-Json
        $statusData.LastRunStatus = "Stopped"
        $statusData | ConvertTo-Json | Out-File -FilePath $StatusFile -Encoding UTF8 -Force
    }

    Write-LogSuccess "Scheduler arrêté"
}

function Show-SchedulerStatus {
    <#
    .SYNOPSIS
        Affiche le statut du scheduler
    #>
    param()

    Write-Banner -Title "TradingBot Scheduler" -Subtitle "Statut"

    $status = Get-SchedulerStatus

    Write-Section "État du Scheduler"

    if ($status.Running) {
        Write-Result -Item "Scheduler" -Success $true -Message "En cours d'exécution"

        if ($status.JobState) {
            Write-Host "  État du Job: $($status.JobState)" -ForegroundColor Gray
        }
        if ($status.ProcessId) {
            Write-Host "  PID: $($status.ProcessId)" -ForegroundColor Gray
        }
        if ($status.StartedAt) {
            Write-Host "  Démarré le: $($status.StartedAt)" -ForegroundColor Gray
        }
    }
    else {
        Write-Result -Item "Scheduler" -Success $false -Message "Arrêté"
    }

    Write-Host ""

    if ($status.ScheduledTime) {
        Write-Host "  Heure planifiée: $($status.ScheduledTime)" -ForegroundColor Cyan
    }
    if ($status.NextRun) {
        Write-Host "  Prochaine exécution: $($status.NextRun)" -ForegroundColor Cyan
    }
    if ($status.LastRun) {
        Write-Host "  Dernière exécution: $($status.LastRun)" -ForegroundColor Gray
        Write-Host "  Statut: $($status.LastRunStatus)" -ForegroundColor $(if ($status.LastRunStatus -eq 'Success') { 'Green' } else { 'Red' })
    }

    Write-Host ""
}

function Invoke-ScheduledScreening {
    <#
    .SYNOPSIS
        Exécute le screening planifié
    #>
    param()

    $startTime = Get-Date

    Write-LogInfo "Démarrage du screening planifié..."

    try {
        $pythonPath = Get-VenvPythonPath
        $mainPath = Join-Path $ProjectRoot "main.py"

        Push-Location $ProjectRoot
        try {
            & $pythonPath $mainPath "run"
            $exitCode = $LASTEXITCODE
        }
        finally {
            Pop-Location
        }

        $duration = (Get-Date) - $startTime

        if ($exitCode -eq 0) {
            Write-LogSuccess "Screening terminé avec succès (durée: $([int]$duration.TotalMinutes) min)"
            Save-SchedulerStatus -ScheduledTime $Time -LastRun (Get-Date).ToString() -LastRunStatus "Success" -StartedAt $script:SchedulerStartTime
            return $true
        }
        else {
            Write-LogError "Screening échoué (code: $exitCode)"
            Save-SchedulerStatus -ScheduledTime $Time -LastRun (Get-Date).ToString() -LastRunStatus "Failed" -StartedAt $script:SchedulerStartTime
            return $false
        }
    }
    catch {
        Write-LogError "Erreur lors du screening: $_"
        Save-SchedulerStatus -ScheduledTime $Time -LastRun (Get-Date).ToString() -LastRunStatus "Error: $_" -StartedAt $script:SchedulerStartTime
        return $false
    }
}

function Start-SchedulerLoop {
    <#
    .SYNOPSIS
        Boucle principale du scheduler
    #>
    param()

    $script:SchedulerStartTime = (Get-Date).ToString()

    Write-LogInfo "Scheduler démarré"
    Write-LogInfo "Heure d'exécution planifiée: $Time"

    # Créer le verrou
    New-SchedulerLock

    # Sauvegarder le statut initial
    Save-SchedulerStatus -ScheduledTime $Time -LastRun $null -LastRunStatus "Started" -StartedAt $script:SchedulerStartTime

    # Exécuter immédiatement si demandé
    if ($RunNow) {
        Write-LogInfo "Exécution immédiate demandée..."
        Invoke-ScheduledScreening
    }

    # Boucle principale
    try {
        while ($true) {
            $now = Get-Date
            $scheduledTime = Get-Date $Time

            # Si l'heure planifiée est passée pour aujourd'hui
            if ($now.TimeOfDay -gt $scheduledTime.TimeOfDay) {
                $nextRun = $scheduledTime.AddDays(1)
            }
            else {
                $nextRun = $scheduledTime
            }

            $timeUntilRun = $nextRun - $now

            # Vérifier si c'est l'heure
            if ($timeUntilRun.TotalMinutes -le 1 -and $timeUntilRun.TotalMinutes -ge 0) {
                Write-LogInfo "Heure d'exécution atteinte!"
                Invoke-ScheduledScreening

                # Attendre 2 minutes pour éviter double exécution
                Start-Sleep -Seconds 120
            }
            else {
                # Afficher le temps restant toutes les heures
                if ($now.Minute -eq 0 -and $now.Second -lt 60) {
                    $hours = [int]$timeUntilRun.TotalHours
                    $minutes = $timeUntilRun.Minutes
                    Write-LogInfo "Prochaine exécution dans ${hours}h ${minutes}min"
                }

                # Attendre 1 minute
                Start-Sleep -Seconds 60
            }
        }
    }
    catch {
        Write-LogError "Erreur dans la boucle du scheduler: $_"
    }
    finally {
        Remove-SchedulerLock
    }
}

function Start-BackgroundScheduler {
    <#
    .SYNOPSIS
        Démarre le scheduler en arrière-plan
    #>
    param()

    # Vérifier si déjà en cours
    $existingJob = Get-Job -Name $JobName -ErrorAction SilentlyContinue
    if ($existingJob -and $existingJob.State -eq 'Running') {
        Write-LogWarning "Le scheduler est déjà en cours d'exécution"
        Write-LogInfo "Utilisez -Stop pour l'arrêter"
        return $true
    }

    # Supprimer les anciens jobs
    Get-Job -Name $JobName -ErrorAction SilentlyContinue | Remove-Job -Force

    Write-LogInfo "Démarrage du scheduler en arrière-plan..."

    # Créer le script à exécuter
    $schedulerScript = {
        param($scriptRoot, $projectRoot, $time, $runNow)

        Set-Location $projectRoot

        # Charger les modules
        . "$scriptRoot\common\Write-Log.ps1"
        . "$scriptRoot\common\Initialize-Venv.ps1"

        # Activer le venv
        Enable-PythonVenv | Out-Null

        # Configuration
        $LockFile = Join-Path $projectRoot "data\.scheduler.lock"
        $StatusFile = Join-Path $projectRoot "data\.scheduler.status"
        $startTime = (Get-Date).ToString()

        # Créer le verrou
        $PID | Out-File -FilePath $LockFile -Force

        # Sauvegarder le statut
        @{
            ScheduledTime = $time
            LastRun       = $null
            LastRunStatus = "Started"
            StartedAt     = $startTime
        } | ConvertTo-Json | Out-File -FilePath $StatusFile -Encoding UTF8 -Force

        # Exécuter immédiatement si demandé
        if ($runNow) {
            $pythonPath = Join-Path $projectRoot "venv\Scripts\python.exe"
            $mainPath = Join-Path $projectRoot "main.py"
            & $pythonPath $mainPath "run"
        }

        # Boucle principale
        while ($true) {
            $now = Get-Date
            $scheduledDateTime = Get-Date $time

            if ($now.TimeOfDay -gt $scheduledDateTime.TimeOfDay) {
                $nextRun = $scheduledDateTime.AddDays(1)
            }
            else {
                $nextRun = $scheduledDateTime
            }

            $timeUntilRun = $nextRun - $now

            if ($timeUntilRun.TotalMinutes -le 1 -and $timeUntilRun.TotalMinutes -ge 0) {
                # Exécuter le screening
                $pythonPath = Join-Path $projectRoot "venv\Scripts\python.exe"
                $mainPath = Join-Path $projectRoot "main.py"
                & $pythonPath $mainPath "run"
                $exitCode = $LASTEXITCODE

                # Mettre à jour le statut
                @{
                    ScheduledTime = $time
                    LastRun       = (Get-Date).ToString()
                    LastRunStatus = if ($exitCode -eq 0) { "Success" } else { "Failed" }
                    StartedAt     = $startTime
                } | ConvertTo-Json | Out-File -FilePath $StatusFile -Encoding UTF8 -Force

                Start-Sleep -Seconds 120
            }
            else {
                Start-Sleep -Seconds 60
            }
        }
    }

    # Démarrer le job
    $job = Start-Job -Name $JobName -ScriptBlock $schedulerScript `
        -ArgumentList $ScriptRoot, $ProjectRoot, $Time, $RunNow

    Start-Sleep -Seconds 2

    if ($job.State -eq 'Running') {
        Write-LogSuccess "Scheduler démarré en arrière-plan"
        Write-LogInfo "Job ID: $($job.Id)"
        Write-LogInfo "Heure planifiée: $Time"
        Write-LogInfo "Pour arrêter: .\Start-Scheduler.ps1 -Stop"
        Write-LogInfo "Pour le statut: .\Start-Scheduler.ps1 -Status"
        return $true
    }
    else {
        $output = Receive-Job -Job $job
        Write-LogError "Échec du démarrage du scheduler"
        Write-LogError "Sortie: $output"
        Remove-Job -Job $job -Force
        return $false
    }
}

# Point d'entrée principal
function Start-Scheduler {
    param()

    try {
        # Mode statut
        if ($Status) {
            Show-SchedulerStatus
            return $true
        }

        # Mode arrêt
        if ($Stop) {
            Stop-Scheduler
            return $true
        }

        Write-Banner -Title "TradingBot Scheduler" -Subtitle "Planificateur Automatique"

        # Vérifier le venv
        if (-not (Enable-PythonVenv)) {
            Write-LogError "Impossible d'activer l'environnement virtuel"
            Write-LogError "Exécutez d'abord: .\Setup-Environment.ps1"
            return $false
        }

        # Valider l'heure
        try {
            $testTime = Get-Date $Time
            Write-LogInfo "Heure d'exécution: $Time"
        }
        catch {
            Write-LogError "Format d'heure invalide: $Time"
            Write-LogError "Utilisez le format HH:mm (ex: 08:00)"
            return $false
        }

        if ($Background) {
            return Start-BackgroundScheduler
        }
        else {
            Write-LogInfo "Démarrage du scheduler en mode interactif..."
            Write-LogInfo "Appuyez sur Ctrl+C pour arrêter"
            Write-Host ""

            Start-SchedulerLoop
        }
    }
    catch {
        Write-LogError "Erreur: $_"
        Write-LogError $_.ScriptStackTrace
        return $false
    }
    finally {
        Remove-SchedulerLock
    }
}

# Exécution
if ($MyInvocation.InvocationName -ne '.') {
    Push-Location $ProjectRoot

    try {
        $result = Start-Scheduler

        if (-not $result) {
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}
