<#
.SYNOPSIS
    Script d'enregistrement des tâches planifiées Windows pour TradingBot

.DESCRIPTION
    Configure les tâches planifiées dans le Planificateur de Tâches Windows:
    - TradingBot-DailyScreening: Screening quotidien à 8h00 (lun-ven)
    - TradingBot-TickerUpdate: Mise à jour des tickers (dimanche 22h00)
    - TradingBot-Maintenance: Maintenance hebdomadaire (samedi 03h00)

.PARAMETER Action
    Action à effectuer: 'register', 'unregister', 'status', 'enable', 'disable'

.PARAMETER TaskName
    Tâche spécifique: 'all', 'screening', 'tickers', 'maintenance'

.PARAMETER ScreeningTime
    Heure du screening quotidien (défaut: 08:00)

.EXAMPLE
    .\Register-ScheduledTasks.ps1 -Action register

.EXAMPLE
    .\Register-ScheduledTasks.ps1 -Action status

.EXAMPLE
    .\Register-ScheduledTasks.ps1 -Action unregister -TaskName screening

.NOTES
    Auteur: TradingBot Team
    Version: 1.0.0
    IMPORTANT: Nécessite des privilèges Administrateur pour enregistrer les tâches
#>

[CmdletBinding()]
param(
    [ValidateSet('register', 'unregister', 'status', 'enable', 'disable')]
    [string]$Action = 'status',

    [ValidateSet('all', 'screening', 'tickers', 'maintenance')]
    [string]$TaskName = 'all',

    [string]$ScreeningTime = "08:00"
)

# Configuration stricte
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Obtenir les chemins
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptRoot "..")

# Charger les modules communs
. "$ScriptRoot\common\Write-Log.ps1"

# Configuration des tâches
$TaskPrefix = "TradingBot"
$TaskFolder = "\TradingBot\"

$Tasks = @{
    screening = @{
        Name        = "$TaskPrefix-DailyScreening"
        Description = "Screening quotidien du marché boursier"
        Script      = "Start-Screening.ps1"
        Arguments   = "-Mode run"
        TriggerType = "Daily"
        TriggerTime = $ScreeningTime
        TriggerDays = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
    }
    tickers = @{
        Name        = "$TaskPrefix-TickerUpdate"
        Description = "Mise à jour hebdomadaire des listes de tickers"
        Script      = "Update-Tickers.ps1"
        Arguments   = "-Market all"
        TriggerType = "Weekly"
        TriggerTime = "22:00"
        TriggerDays = @("Sunday")
    }
    maintenance = @{
        Name        = "$TaskPrefix-Maintenance"
        Description = "Maintenance hebdomadaire (logs, backup, nettoyage)"
        Script      = "Invoke-Maintenance.ps1"
        Arguments   = "-Task all"
        TriggerType = "Weekly"
        TriggerTime = "03:00"
        TriggerDays = @("Saturday")
    }
}

function Test-Administrator {
    <#
    .SYNOPSIS
        Vérifie si le script s'exécute en tant qu'administrateur
    #>
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-TaskStatus {
    <#
    .SYNOPSIS
        Récupère le statut d'une tâche planifiée
    #>
    param(
        [string]$Name
    )

    try {
        $task = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue

        if ($task) {
            $info = Get-ScheduledTaskInfo -TaskName $Name -ErrorAction SilentlyContinue

            return [PSCustomObject]@{
                Exists      = $true
                State       = $task.State
                LastRun     = $info.LastRunTime
                LastResult  = $info.LastTaskResult
                NextRun     = $info.NextRunTime
            }
        }
        else {
            return [PSCustomObject]@{
                Exists      = $false
                State       = "NotFound"
                LastRun     = $null
                LastResult  = $null
                NextRun     = $null
            }
        }
    }
    catch {
        return [PSCustomObject]@{
            Exists      = $false
            State       = "Error"
            LastRun     = $null
            LastResult  = $null
            NextRun     = $null
        }
    }
}

function Register-TradingBotTask {
    <#
    .SYNOPSIS
        Enregistre une tâche planifiée
    #>
    param(
        [hashtable]$TaskConfig
    )

    $taskName = $TaskConfig.Name
    $scriptPath = Join-Path $ScriptRoot $TaskConfig.Script

    Write-LogInfo "Enregistrement de la tâche: $taskName"

    # Vérifier que le script existe
    if (-not (Test-Path $scriptPath)) {
        Write-LogError "Script non trouvé: $scriptPath"
        return $false
    }

    try {
        # Supprimer si existe déjà
        $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($existing) {
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
            Write-LogDebug "Ancienne tâche supprimée"
        }

        # Créer l'action
        $action = New-ScheduledTaskAction -Execute "powershell.exe" `
            -Argument "-ExecutionPolicy Bypass -NoProfile -File `"$scriptPath`" $($TaskConfig.Arguments)" `
            -WorkingDirectory $ProjectRoot

        # Créer le trigger selon le type
        if ($TaskConfig.TriggerType -eq "Daily") {
            # Trigger quotidien pour jours spécifiques
            $triggers = @()
            foreach ($day in $TaskConfig.TriggerDays) {
                $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $day -At $TaskConfig.TriggerTime
                $triggers += $trigger
            }
        }
        else {
            # Trigger hebdomadaire
            $triggers = @()
            foreach ($day in $TaskConfig.TriggerDays) {
                $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $day -At $TaskConfig.TriggerTime
                $triggers += $trigger
            }
        }

        # Paramètres de la tâche
        $settings = New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -StartWhenAvailable `
            -RunOnlyIfNetworkAvailable `
            -ExecutionTimeLimit (New-TimeSpan -Hours 2)

        # Principal (utilisateur actuel)
        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

        # Enregistrer la tâche
        Register-ScheduledTask `
            -TaskName $taskName `
            -Description $TaskConfig.Description `
            -Action $action `
            -Trigger $triggers `
            -Settings $settings `
            -Principal $principal | Out-Null

        Write-LogSuccess "Tâche enregistrée: $taskName"
        return $true
    }
    catch {
        Write-LogError "Erreur lors de l'enregistrement: $_"
        return $false
    }
}

function Unregister-TradingBotTask {
    <#
    .SYNOPSIS
        Supprime une tâche planifiée
    #>
    param(
        [string]$Name
    )

    Write-LogInfo "Suppression de la tâche: $Name"

    try {
        $task = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue

        if ($task) {
            Unregister-ScheduledTask -TaskName $Name -Confirm:$false
            Write-LogSuccess "Tâche supprimée: $Name"
            return $true
        }
        else {
            Write-LogWarning "Tâche non trouvée: $Name"
            return $true
        }
    }
    catch {
        Write-LogError "Erreur lors de la suppression: $_"
        return $false
    }
}

function Set-TradingBotTaskState {
    <#
    .SYNOPSIS
        Active ou désactive une tâche
    #>
    param(
        [string]$Name,
        [bool]$Enable
    )

    $action = if ($Enable) { "Activation" } else { "Désactivation" }
    Write-LogInfo "$action de la tâche: $Name"

    try {
        $task = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue

        if (-not $task) {
            Write-LogWarning "Tâche non trouvée: $Name"
            return $false
        }

        if ($Enable) {
            Enable-ScheduledTask -TaskName $Name | Out-Null
        }
        else {
            Disable-ScheduledTask -TaskName $Name | Out-Null
        }

        Write-LogSuccess "Tâche ${action}: $Name"
        return $true
    }
    catch {
        Write-LogError "Erreur: $_"
        return $false
    }
}

function Show-TasksStatus {
    <#
    .SYNOPSIS
        Affiche le statut de toutes les tâches
    #>
    param()

    Write-Section "Statut des Tâches Planifiées"

    foreach ($key in $Tasks.Keys) {
        $taskConfig = $Tasks[$key]
        $status = Get-TaskStatus -Name $taskConfig.Name

        Write-Host "  $($taskConfig.Name)" -ForegroundColor Cyan

        if ($status.Exists) {
            $stateColor = switch ($status.State) {
                'Ready' { 'Green' }
                'Running' { 'Yellow' }
                'Disabled' { 'Gray' }
                default { 'Red' }
            }

            Write-Host "    État: " -NoNewline
            Write-Host $status.State -ForegroundColor $stateColor

            if ($status.LastRun) {
                Write-Host "    Dernière exécution: $($status.LastRun)" -ForegroundColor Gray
                $resultText = if ($status.LastResult -eq 0) { "Succès" } else { "Code: $($status.LastResult)" }
                Write-Host "    Résultat: $resultText" -ForegroundColor $(if ($status.LastResult -eq 0) { 'Green' } else { 'Yellow' })
            }

            if ($status.NextRun) {
                Write-Host "    Prochaine exécution: $($status.NextRun)" -ForegroundColor Cyan
            }
        }
        else {
            Write-Host "    État: Non enregistrée" -ForegroundColor Red
        }

        Write-Host ""
    }
}

function Get-TasksToProcess {
    <#
    .SYNOPSIS
        Retourne les tâches à traiter selon le paramètre TaskName
    #>
    param()

    if ($TaskName -eq 'all') {
        return $Tasks.Keys
    }
    else {
        return @($TaskName)
    }
}

# Point d'entrée principal
function Start-TaskRegistration {
    param()

    Write-Banner -Title "TradingBot" -Subtitle "Tâches Planifiées Windows"

    try {
        # Vérifier les privilèges admin pour register/unregister
        if ($Action -in @('register', 'unregister', 'enable', 'disable')) {
            if (-not (Test-Administrator)) {
                Write-LogWarning "Ce script nécessite des privilèges Administrateur pour modifier les tâches."
                Write-LogInfo "Relancez PowerShell en tant qu'Administrateur."

                # Proposer de relancer en admin
                $response = Read-Host "Voulez-vous relancer en tant qu'Administrateur? (O/N)"
                if ($response -eq 'O' -or $response -eq 'o') {
                    Start-Process powershell -Verb RunAs -ArgumentList "-File `"$PSCommandPath`" -Action $Action -TaskName $TaskName"
                    return $true
                }

                return $false
            }
        }

        switch ($Action) {
            'status' {
                Show-TasksStatus
            }

            'register' {
                Write-Section "Enregistrement des Tâches"

                $tasksToProcess = Get-TasksToProcess
                $results = @{ Success = @(); Failed = @() }

                foreach ($key in $tasksToProcess) {
                    if ($Tasks.ContainsKey($key)) {
                        if (Register-TradingBotTask -TaskConfig $Tasks[$key]) {
                            $results.Success += $key
                        }
                        else {
                            $results.Failed += $key
                        }
                    }
                }

                Write-Host ""
                Write-LogSuccess "Enregistrées: $($results.Success -join ', ')"
                if ($results.Failed.Count -gt 0) {
                    Write-LogError "Échecs: $($results.Failed -join ', ')"
                }

                # Afficher le statut final
                Show-TasksStatus
            }

            'unregister' {
                Write-Section "Suppression des Tâches"

                $tasksToProcess = Get-TasksToProcess

                foreach ($key in $tasksToProcess) {
                    if ($Tasks.ContainsKey($key)) {
                        Unregister-TradingBotTask -Name $Tasks[$key].Name
                    }
                }
            }

            'enable' {
                Write-Section "Activation des Tâches"

                $tasksToProcess = Get-TasksToProcess

                foreach ($key in $tasksToProcess) {
                    if ($Tasks.ContainsKey($key)) {
                        Set-TradingBotTaskState -Name $Tasks[$key].Name -Enable $true
                    }
                }
            }

            'disable' {
                Write-Section "Désactivation des Tâches"

                $tasksToProcess = Get-TasksToProcess

                foreach ($key in $tasksToProcess) {
                    if ($Tasks.ContainsKey($key)) {
                        Set-TradingBotTaskState -Name $Tasks[$key].Name -Enable $false
                    }
                }
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
        $result = Start-TaskRegistration

        if (-not $result) {
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}
