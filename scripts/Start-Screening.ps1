<#
.SYNOPSIS
    Script de lancement du screening de marché TradingBot

.DESCRIPTION
    Ce script permet d'exécuter le screening de marché avec différentes options:
    - Mode unique (run): Exécute le screening une fois et quitte
    - Mode planifié (schedule): Lance le scheduler pour exécution quotidienne
    - Mode symbole: Analyse un ou plusieurs symboles spécifiques

.PARAMETER Mode
    Mode d'exécution: 'run' (unique) ou 'schedule' (planifié)

.PARAMETER Symbol
    Symbole spécifique à analyser (ex: AAPL)

.PARAMETER Symbols
    Liste de symboles séparés par des virgules (ex: AAPL,MSFT,GOOGL)

.PARAMETER Days
    Nombre de jours de lookback pour l'analyse (défaut: 365)

.PARAMETER Quiet
    Mode silencieux - moins de sortie console

.PARAMETER NoNotification
    Désactiver l'envoi de notifications

.EXAMPLE
    .\Start-Screening.ps1 -Mode run

.EXAMPLE
    .\Start-Screening.ps1 -Symbol AAPL

.EXAMPLE
    .\Start-Screening.ps1 -Symbols "AAPL,MSFT,GOOGL"

.EXAMPLE
    .\Start-Screening.ps1 -Mode schedule

.NOTES
    Auteur: TradingBot Team
    Version: 1.0.0
#>

[CmdletBinding(DefaultParameterSetName = 'Mode')]
param(
    [Parameter(ParameterSetName = 'Mode')]
    [ValidateSet('run', 'schedule')]
    [string]$Mode = 'run',

    [Parameter(ParameterSetName = 'Symbol')]
    [string]$Symbol,

    [Parameter(ParameterSetName = 'Symbols')]
    [string]$Symbols,

    [int]$Days = 365,

    [switch]$Quiet,

    [switch]$NoNotification
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

# Configuration du logging
if ($Quiet) {
    Set-LogConfig -MinLogLevel WARNING
}

function Test-Configuration {
    <#
    .SYNOPSIS
        Vérifie que la configuration est correcte avant le screening
    #>
    param()

    $configOk = $true

    # Vérifier le fichier .env
    $envPath = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path $envPath)) {
        Write-LogWarning "Fichier .env non trouvé - les notifications ne fonctionneront pas"
    }

    # Vérifier main.py
    $mainPath = Join-Path $ProjectRoot "main.py"
    if (-not (Test-Path $mainPath)) {
        Write-LogError "main.py non trouvé dans $ProjectRoot"
        $configOk = $false
    }

    # Vérifier requirements.txt
    $reqPath = Join-Path $ProjectRoot "requirements.txt"
    if (-not (Test-Path $reqPath)) {
        Write-LogError "requirements.txt non trouvé"
        $configOk = $false
    }

    return $configOk
}

function Invoke-PythonScreening {
    <#
    .SYNOPSIS
        Exécute le script Python de screening

    .PARAMETER Command
        Commande à exécuter (run, schedule, screen, test, alerts)

    .PARAMETER Arguments
        Arguments additionnels
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Command,

        [hashtable]$Arguments = @{}
    )

    $pythonPath = Get-VenvPythonPath
    $mainPath = Join-Path $ProjectRoot "main.py"

    if (-not (Test-Path $pythonPath)) {
        Write-LogError "Python non trouvé dans le venv. Exécutez d'abord Setup-Environment.ps1"
        return $false
    }

    # Construire la commande
    $cmdArgs = @($mainPath, $Command)

    foreach ($key in $Arguments.Keys) {
        $value = $Arguments[$key]
        if ($value -is [switch] -or $value -eq $true) {
            $cmdArgs += "--$key"
        }
        elseif ($value) {
            $cmdArgs += "--$key"
            $cmdArgs += $value
        }
    }

    Write-LogDebug "Commande: $pythonPath $($cmdArgs -join ' ')"

    try {
        # Exécuter Python
        $process = Start-Process -FilePath $pythonPath `
            -ArgumentList $cmdArgs `
            -WorkingDirectory $ProjectRoot `
            -NoNewWindow `
            -PassThru `
            -Wait

        if ($process.ExitCode -ne 0) {
            Write-LogError "Le screening a échoué avec le code: $($process.ExitCode)"
            return $false
        }

        return $true
    }
    catch {
        Write-LogError "Erreur lors de l'exécution: $_"
        return $false
    }
}

function Invoke-InteractiveScreening {
    <#
    .SYNOPSIS
        Exécute le screening en mode interactif (sortie visible)
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Command,

        [hashtable]$Arguments = @{}
    )

    $pythonPath = Get-VenvPythonPath
    $mainPath = Join-Path $ProjectRoot "main.py"

    # Construire les arguments
    $argString = $Command
    foreach ($key in $Arguments.Keys) {
        $value = $Arguments[$key]
        if ($value -is [switch] -or $value -eq $true) {
            $argString += " --$key"
        }
        elseif ($value) {
            $argString += " --$key `"$value`""
        }
    }

    Write-LogInfo "Exécution: python main.py $argString"
    Write-Host ""

    # Exécuter avec sortie visible
    Push-Location $ProjectRoot
    try {
        & $pythonPath $mainPath $Command @(
            $Arguments.GetEnumerator() | ForEach-Object {
                if ($_.Value -is [switch] -or $_.Value -eq $true) {
                    "--$($_.Key)"
                }
                elseif ($_.Value) {
                    "--$($_.Key)", $_.Value
                }
            }
        )

        $exitCode = $LASTEXITCODE
        Write-Host ""

        if ($exitCode -eq 0) {
            Write-LogSuccess "Screening terminé avec succès"
            return $true
        }
        else {
            Write-LogError "Screening terminé avec erreur (code: $exitCode)"
            return $false
        }
    }
    finally {
        Pop-Location
    }
}

function Start-SingleScreening {
    <#
    .SYNOPSIS
        Exécute un screening unique
    #>
    param()

    Write-Banner -Title "TradingBot Screening" -Subtitle "Mode: Exécution Unique"

    $startTime = Get-Date

    Write-LogInfo "Démarrage du screening..."

    $success = Invoke-InteractiveScreening -Command "run"

    $duration = (Get-Date) - $startTime
    Write-LogInfo "Durée totale: $([int]$duration.TotalMinutes) min $([int]($duration.TotalSeconds % 60)) sec"

    return $success
}

function Start-ScheduledScreening {
    <#
    .SYNOPSIS
        Démarre le scheduler pour le screening quotidien
    #>
    param()

    Write-Banner -Title "TradingBot Scheduler" -Subtitle "Mode: Planifié (8h00 quotidien)"

    Write-LogInfo "Démarrage du scheduler..."
    Write-LogInfo "Le screening sera exécuté quotidiennement à 8h00 (Europe/Paris)"
    Write-LogInfo "Appuyez sur Ctrl+C pour arrêter"
    Write-Host ""

    # Exécuter le scheduler (bloquant)
    $success = Invoke-InteractiveScreening -Command "schedule"

    return $success
}

function Start-SymbolScreening {
    <#
    .SYNOPSIS
        Analyse un symbole spécifique
    #>
    param(
        [Parameter(Mandatory)]
        [string]$SymbolToScreen
    )

    $symbolUpper = $SymbolToScreen.ToUpper().Trim()

    Write-Banner -Title "TradingBot" -Subtitle "Analyse: $symbolUpper"

    Write-LogInfo "Analyse du symbole $symbolUpper..."

    $success = Invoke-InteractiveScreening -Command "screen" -Arguments @{
        symbol = $symbolUpper
    }

    return $success
}

function Start-MultiSymbolScreening {
    <#
    .SYNOPSIS
        Analyse plusieurs symboles
    #>
    param(
        [Parameter(Mandatory)]
        [string]$SymbolList
    )

    $symbolArray = $SymbolList.Split(',') | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ }

    Write-Banner -Title "TradingBot" -Subtitle "Analyse Multi-Symboles"

    Write-LogInfo "Analyse de $($symbolArray.Count) symboles: $($symbolArray -join ', ')"
    Write-Host ""

    $results = @{
        Success = @()
        Failed  = @()
    }

    foreach ($sym in $symbolArray) {
        Write-Section "Analyse de $sym"

        $success = Invoke-InteractiveScreening -Command "screen" -Arguments @{
            symbol = $sym
        }

        if ($success) {
            $results.Success += $sym
        }
        else {
            $results.Failed += $sym
        }

        Write-Host ""
    }

    # Résumé
    Write-Section "Résumé"
    Write-LogInfo "Symboles analysés: $($symbolArray.Count)"
    Write-LogSuccess "Succès: $($results.Success.Count) - $($results.Success -join ', ')"

    if ($results.Failed.Count -gt 0) {
        Write-LogError "Échecs: $($results.Failed.Count) - $($results.Failed -join ', ')"
    }

    return ($results.Failed.Count -eq 0)
}

function Start-Screening {
    <#
    .SYNOPSIS
        Point d'entrée principal du script
    #>
    param()

    $startTime = Get-Date

    try {
        # Vérifier la configuration
        if (-not (Test-Configuration)) {
            Write-LogError "Configuration invalide. Exécutez Setup-Environment.ps1"
            return $false
        }

        # Initialiser l'environnement Python
        Write-LogInfo "Initialisation de l'environnement..."

        if (-not (Enable-PythonVenv)) {
            Write-LogError "Impossible d'activer l'environnement virtuel"
            Write-LogError "Exécutez d'abord: .\Setup-Environment.ps1"
            return $false
        }

        # Déterminer le mode d'exécution
        if ($Symbol) {
            return Start-SymbolScreening -SymbolToScreen $Symbol
        }
        elseif ($Symbols) {
            return Start-MultiSymbolScreening -SymbolList $Symbols
        }
        elseif ($Mode -eq 'schedule') {
            return Start-ScheduledScreening
        }
        else {
            return Start-SingleScreening
        }
    }
    catch {
        Write-LogError "Erreur fatale: $_"
        Write-LogError $_.ScriptStackTrace

        # Tenter d'envoyer une notification d'erreur
        if (-not $NoNotification) {
            try {
                $pythonPath = Get-VenvPythonPath
                if (Test-Path $pythonPath) {
                    $errorScript = @"
import asyncio
from src.notifications.notifier import notifier
asyncio.run(notifier.send_telegram_message('TradingBot Error: $($_.Exception.Message -replace "'", "")'))
"@
                    $errorScript | & $pythonPath - 2>$null
                }
            }
            catch {
                # Ignorer les erreurs de notification
            }
        }

        return $false
    }
    finally {
        $duration = (Get-Date) - $startTime
        Write-LogDebug "Durée totale d'exécution: $([int]$duration.TotalSeconds) secondes"
    }
}

# Point d'entrée
if ($MyInvocation.InvocationName -ne '.') {
    Push-Location $ProjectRoot

    try {
        $result = Start-Screening

        if (-not $result) {
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}
