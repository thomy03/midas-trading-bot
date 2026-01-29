<#
.SYNOPSIS
    Enregistre les taches planifiees Windows pour l'agent TradingBot V4

.DESCRIPTION
    Configure le Planificateur de taches Windows pour:
    - Discovery: 06:00 (scan social + decouverte)
    - Trading: 09:30-16:00 (screening + execution)
    - Audit: 20:00 (analyse des trades)

.PARAMETER Register
    Enregistrer les taches

.PARAMETER Unregister
    Supprimer les taches

.PARAMETER List
    Lister les taches existantes

.PARAMETER Docker
    Utiliser Docker au lieu de Python

.EXAMPLE
    .\Register-AgentTasks.ps1 -Register

.EXAMPLE
    .\Register-AgentTasks.ps1 -Unregister

.NOTES
    Necessite des droits administrateur pour enregistrer les taches.
    Auteur: TradingBot V4
#>

[CmdletBinding()]
param(
    [switch]$Register,
    [switch]$Unregister,
    [switch]$List,
    [switch]$Docker
)

# Configuration stricte
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Obtenir les chemins
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptRoot "..")

# Charger les modules communs
. "$ScriptRoot\common\Write-Log.ps1"

# Configuration des taches
$TaskFolder = "TradingBot"
$Tasks = @(
    @{
        Name = "TradingBot-Discovery"
        Description = "Scan social + decouverte de stocks (06:00)"
        Time = "06:00"
        Mode = "discovery"
    },
    @{
        Name = "TradingBot-Trading"
        Description = "Screening + execution (15:30 heure Paris = 09:30 ET)"
        Time = "15:30"
        Mode = "trading"
    },
    @{
        Name = "TradingBot-Audit"
        Description = "Analyse des trades du jour (02:00 = 20:00 ET)"
        Time = "02:00"
        Mode = "full"  # Le mode full inclut l'audit
    }
)

function Test-Administrator {
    <#
    .SYNOPSIS
        Verifie si le script s'execute en tant qu'administrateur
    #>
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-TaskCommand {
    <#
    .SYNOPSIS
        Retourne la commande pour une tache
    #>
    param([string]$Mode)

    if ($Docker) {
        return @{
            Execute = "docker-compose"
            Arguments = "-f `"$ProjectRoot\docker-compose.yml`" run --rm agent python run_agent.py --mode $Mode --mock"
            WorkingDir = $ProjectRoot
        }
    }
    else {
        $pythonPath = Join-Path $ProjectRoot "venv\Scripts\python.exe"
        $agentPath = Join-Path $ProjectRoot "run_agent.py"

        return @{
            Execute = $pythonPath
            Arguments = "`"$agentPath`" --mode $Mode --mock"
            WorkingDir = $ProjectRoot
        }
    }
}

function Register-AgentTask {
    <#
    .SYNOPSIS
        Enregistre une tache planifiee
    #>
    param(
        [hashtable]$TaskConfig
    )

    $taskName = $TaskConfig.Name
    $description = $TaskConfig.Description
    $time = $TaskConfig.Time
    $mode = $TaskConfig.Mode

    # Supprimer si existe deja
    $existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        Write-LogInfo "Suppression de la tache existante: $taskName"
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    }

    # Obtenir la commande
    $cmd = Get-TaskCommand -Mode $mode

    # Creer l'action
    $action = New-ScheduledTaskAction `
        -Execute $cmd.Execute `
        -Argument $cmd.Arguments `
        -WorkingDirectory $cmd.WorkingDir

    # Creer le declencheur (quotidien a l'heure specifiee)
    $trigger = New-ScheduledTaskTrigger -Daily -At $time

    # Creer les parametres
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable `
        -MultipleInstances IgnoreNew

    # Utiliser l'utilisateur actuel
    $principal = New-ScheduledTaskPrincipal `
        -UserId $env:USERNAME `
        -RunLevel Limited

    # Enregistrer la tache
    Register-ScheduledTask `
        -TaskName $taskName `
        -Description $description `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Force

    Write-LogSuccess "Tache enregistree: $taskName ($time)"
}

function Unregister-AgentTasks {
    <#
    .SYNOPSIS
        Supprime toutes les taches agent
    #>

    foreach ($task in $Tasks) {
        $existingTask = Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue
        if ($existingTask) {
            Unregister-ScheduledTask -TaskName $task.Name -Confirm:$false
            Write-LogSuccess "Tache supprimee: $($task.Name)"
        }
        else {
            Write-LogInfo "Tache non trouvee: $($task.Name)"
        }
    }
}

function Show-AgentTasks {
    <#
    .SYNOPSIS
        Affiche les taches agent existantes
    #>

    Write-Section "Taches TradingBot"

    foreach ($task in $Tasks) {
        $existingTask = Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue
        if ($existingTask) {
            $taskInfo = Get-ScheduledTaskInfo -TaskName $task.Name
            Write-Result -Item $task.Name -Success $true -Message "Prochaine: $($taskInfo.NextRunTime)"
        }
        else {
            Write-Result -Item $task.Name -Success $false -Message "Non enregistree"
        }
    }
}

# Point d'entree principal
function Main {
    Write-Banner -Title "TradingBot Agent V4" -Subtitle "Taches Planifiees"

    if ($List) {
        Show-AgentTasks
        return
    }

    if ($Unregister) {
        Write-Section "Suppression des taches"
        Unregister-AgentTasks
        return
    }

    if ($Register) {
        # Verifier les droits admin
        if (-not (Test-Administrator)) {
            Write-LogWarning "Ce script necessite des droits administrateur pour enregistrer les taches."
            Write-LogInfo "Relancez PowerShell en tant qu'administrateur."
            return
        }

        Write-Section "Enregistrement des taches"

        # Verifier que le venv existe (si pas Docker)
        if (-not $Docker) {
            $venvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
            if (-not (Test-Path $venvPython)) {
                Write-LogError "Environnement virtuel non trouve. Executez Setup-Environment.ps1 d'abord."
                return
            }
        }

        foreach ($task in $Tasks) {
            Register-AgentTask -TaskConfig $task
        }

        Write-Host ""
        Write-LogSuccess "Toutes les taches ont ete enregistrees!"
        Write-LogInfo ""
        Write-LogInfo "Horaires (heure Paris):"
        Write-LogInfo "  - Discovery: 06:00 (scan social)"
        Write-LogInfo "  - Trading:   15:30 (= 09:30 ET, ouverture US)"
        Write-LogInfo "  - Audit:     02:00 (= 20:00 ET, apres cloture)"
        Write-LogInfo ""
        Write-LogInfo "Pour voir les taches: .\Register-AgentTasks.ps1 -List"
        Write-LogInfo "Pour supprimer:       .\Register-AgentTasks.ps1 -Unregister"
        return
    }

    # Par defaut, afficher l'aide
    Write-Section "Usage"
    Write-Host "  -Register     Enregistrer les taches planifiees"
    Write-Host "  -Unregister   Supprimer les taches"
    Write-Host "  -List         Lister les taches existantes"
    Write-Host "  -Docker       Utiliser Docker au lieu de Python"
    Write-Host ""
    Write-Host "Exemples:"
    Write-Host "  .\Register-AgentTasks.ps1 -Register"
    Write-Host "  .\Register-AgentTasks.ps1 -Register -Docker"
    Write-Host "  .\Register-AgentTasks.ps1 -List"
}

# Execution
Main
