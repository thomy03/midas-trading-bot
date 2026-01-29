<#
.SYNOPSIS
    Script de lancement du dashboard Streamlit TradingBot

.DESCRIPTION
    Lance le dashboard web Streamlit avec les options suivantes:
    - Configuration du port
    - Mode arrière-plan (Job PowerShell)
    - Ouverture automatique du navigateur
    - Gestion des processus existants

.PARAMETER Port
    Port d'écoute pour Streamlit (défaut: 8501)

.PARAMETER Host
    Adresse d'écoute (défaut: localhost, 0.0.0.0 pour accès réseau)

.PARAMETER Background
    Lancer en arrière-plan comme Job PowerShell

.PARAMETER NoBrowser
    Ne pas ouvrir automatiquement le navigateur

.PARAMETER Force
    Forcer l'arrêt des processus Streamlit existants

.PARAMETER Stop
    Arrêter le dashboard en cours d'exécution

.EXAMPLE
    .\Start-Dashboard.ps1

.EXAMPLE
    .\Start-Dashboard.ps1 -Port 8080 -Background

.EXAMPLE
    .\Start-Dashboard.ps1 -Stop

.NOTES
    Auteur: TradingBot Team
    Version: 1.0.0
#>

[CmdletBinding()]
param(
    [int]$Port = 8501,

    [string]$Host = "localhost",

    [switch]$Background,

    [switch]$NoBrowser,

    [switch]$Force,

    [switch]$Stop
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

# Nom du job pour le mode background
$JobName = "TradingBot-Dashboard"

function Test-PortAvailable {
    <#
    .SYNOPSIS
        Vérifie si un port est disponible
    #>
    param(
        [Parameter(Mandatory)]
        [int]$PortNumber
    )

    try {
        $tcpListener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $PortNumber)
        $tcpListener.Start()
        $tcpListener.Stop()
        return $true
    }
    catch {
        return $false
    }
}

function Get-StreamlitProcesses {
    <#
    .SYNOPSIS
        Retourne les processus Streamlit en cours
    #>
    param()

    return Get-Process -Name "streamlit", "python" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -like "*streamlit*" -or
            $_.CommandLine -like "*dashboard.py*"
        }
}

function Stop-StreamlitProcesses {
    <#
    .SYNOPSIS
        Arrête tous les processus Streamlit
    #>
    param()

    $processes = Get-StreamlitProcesses

    if ($processes) {
        Write-LogInfo "Arrêt de $($processes.Count) processus Streamlit..."

        foreach ($proc in $processes) {
            try {
                $proc | Stop-Process -Force
                Write-LogDebug "Processus $($proc.Id) arrêté"
            }
            catch {
                Write-LogWarning "Impossible d'arrêter le processus $($proc.Id)"
            }
        }

        Start-Sleep -Seconds 1
    }

    # Arrêter le job si existant
    $job = Get-Job -Name $JobName -ErrorAction SilentlyContinue
    if ($job) {
        Write-LogInfo "Arrêt du job $JobName..."
        $job | Stop-Job -PassThru | Remove-Job
    }

    Write-LogSuccess "Dashboard arrêté"
}

function Get-DashboardStatus {
    <#
    .SYNOPSIS
        Retourne le statut du dashboard
    #>
    param()

    $status = [PSCustomObject]@{
        Running     = $false
        ProcessId   = $null
        Port        = $null
        Url         = $null
        JobRunning  = $false
    }

    # Vérifier le job
    $job = Get-Job -Name $JobName -ErrorAction SilentlyContinue
    if ($job -and $job.State -eq 'Running') {
        $status.JobRunning = $true
        $status.Running = $true
    }

    # Vérifier les processus
    $processes = Get-StreamlitProcesses
    if ($processes) {
        $status.Running = $true
        $status.ProcessId = $processes[0].Id
    }

    # Vérifier si le port par défaut est utilisé
    if (-not (Test-PortAvailable -PortNumber 8501)) {
        $status.Port = 8501
        $status.Url = "http://localhost:8501"
    }

    return $status
}

function Start-StreamlitDashboard {
    <#
    .SYNOPSIS
        Démarre le dashboard Streamlit
    #>
    param()

    $dashboardPath = Join-Path $ProjectRoot "dashboard.py"

    if (-not (Test-Path $dashboardPath)) {
        Write-LogError "dashboard.py non trouvé: $dashboardPath"
        return $false
    }

    # Vérifier/activer le venv
    if (-not (Enable-PythonVenv)) {
        Write-LogError "Impossible d'activer l'environnement virtuel"
        return $false
    }

    # Vérifier si Streamlit est installé
    if (-not (Test-PythonPackage -PackageName "streamlit")) {
        Write-LogError "Streamlit n'est pas installé. Exécutez Setup-Environment.ps1"
        return $false
    }

    $pythonPath = Get-VenvPythonPath
    $streamlitArgs = @(
        "-m", "streamlit", "run",
        $dashboardPath,
        "--server.port", $Port,
        "--server.address", $Host
    )

    if ($NoBrowser) {
        $streamlitArgs += "--server.headless", "true"
    }

    $url = "http://${Host}:${Port}"

    if ($Background) {
        # Mode arrière-plan avec Job
        Write-LogInfo "Démarrage du dashboard en arrière-plan..."

        $scriptBlock = {
            param($pythonPath, $args, $projectRoot)
            Set-Location $projectRoot
            & $pythonPath @args
        }

        $job = Start-Job -Name $JobName -ScriptBlock $scriptBlock `
            -ArgumentList $pythonPath, $streamlitArgs, $ProjectRoot

        # Attendre un peu pour vérifier le démarrage
        Start-Sleep -Seconds 3

        if ($job.State -eq 'Running') {
            Write-LogSuccess "Dashboard démarré en arrière-plan"
            Write-LogInfo "URL: $url"
            Write-LogInfo "Job ID: $($job.Id)"
            Write-LogInfo "Pour arrêter: .\Start-Dashboard.ps1 -Stop"

            # Ouvrir le navigateur
            if (-not $NoBrowser) {
                Start-Sleep -Seconds 2
                Start-Process $url
            }

            return $true
        }
        else {
            $output = Receive-Job -Job $job
            Write-LogError "Le dashboard n'a pas démarré correctement"
            Write-LogError "Sortie: $output"
            Remove-Job -Job $job -Force
            return $false
        }
    }
    else {
        # Mode interactif
        Write-LogInfo "Démarrage du dashboard..."
        Write-LogInfo "URL: $url"
        Write-LogInfo "Appuyez sur Ctrl+C pour arrêter"
        Write-Host ""

        # Ouvrir le navigateur après un délai
        if (-not $NoBrowser) {
            $browserJob = Start-Job -ScriptBlock {
                param($url)
                Start-Sleep -Seconds 3
                Start-Process $url
            } -ArgumentList $url
        }

        # Exécuter Streamlit (bloquant)
        Push-Location $ProjectRoot
        try {
            & $pythonPath @streamlitArgs

            return ($LASTEXITCODE -eq 0)
        }
        finally {
            Pop-Location
            if ($browserJob) {
                Remove-Job -Job $browserJob -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

function Show-DashboardMenu {
    <#
    .SYNOPSIS
        Affiche un menu interactif pour le dashboard
    #>
    param()

    Write-Banner -Title "TradingBot Dashboard" -Subtitle "Interface Web Streamlit"

    $status = Get-DashboardStatus

    Write-Section "Statut Actuel"

    if ($status.Running) {
        Write-Result -Item "Dashboard" -Success $true -Message "En cours d'exécution"
        if ($status.Url) {
            Write-Host "  URL: $($status.Url)" -ForegroundColor Cyan
        }
        if ($status.ProcessId) {
            Write-Host "  PID: $($status.ProcessId)" -ForegroundColor Gray
        }
        if ($status.JobRunning) {
            Write-Host "  Mode: Arrière-plan (Job)" -ForegroundColor Gray
        }
    }
    else {
        Write-Result -Item "Dashboard" -Success $false -Message "Arrêté"
    }

    Write-Host ""
}

# Point d'entrée principal
function Start-Dashboard {
    param()

    try {
        # Mode arrêt
        if ($Stop) {
            Write-Banner -Title "TradingBot Dashboard" -Subtitle "Arrêt"
            Stop-StreamlitProcesses
            return $true
        }

        Write-Banner -Title "TradingBot Dashboard" -Subtitle "Démarrage"

        # Vérifier le statut actuel
        $status = Get-DashboardStatus

        if ($status.Running) {
            if ($Force) {
                Write-LogWarning "Dashboard déjà en cours. Arrêt forcé..."
                Stop-StreamlitProcesses
                Start-Sleep -Seconds 2
            }
            else {
                Write-LogWarning "Le dashboard est déjà en cours d'exécution"
                Write-LogInfo "URL: $($status.Url)"
                Write-LogInfo "Utilisez -Force pour redémarrer ou -Stop pour arrêter"
                return $true
            }
        }

        # Vérifier le port
        if (-not (Test-PortAvailable -PortNumber $Port)) {
            if ($Force) {
                Write-LogWarning "Port $Port occupé. Tentative de libération..."
                Stop-StreamlitProcesses
                Start-Sleep -Seconds 2
            }
            else {
                Write-LogError "Le port $Port est déjà utilisé"
                Write-LogError "Utilisez -Force pour forcer ou -Port pour changer"
                return $false
            }
        }

        Write-Result -Item "Port $Port" -Success $true -Message "Disponible"

        # Démarrer le dashboard
        return Start-StreamlitDashboard
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
        $result = Start-Dashboard

        if (-not $result) {
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}
