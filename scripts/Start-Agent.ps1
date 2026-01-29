<#
.SYNOPSIS
    Script de lancement de l'agent TradingBot V4

.DESCRIPTION
    Lance l'agent de trading autonome avec les modes suivants:
    - test: Test avec mock IBKR
    - discovery: Scan social + decouverte de stocks
    - analysis: Analyse LLM des tendances
    - trading: Screening + execution
    - full: Cycle quotidien complet

.PARAMETER Mode
    Mode d'execution (test, discovery, analysis, trading, full)

.PARAMETER Mock
    Utiliser le mock IBKR (pas de trading reel)

.PARAMETER Background
    Lancer en arriere-plan comme Job PowerShell

.PARAMETER Stop
    Arreter l'agent en cours d'execution

.PARAMETER Docker
    Lancer via Docker au lieu de Python directement

.EXAMPLE
    .\Start-Agent.ps1 -Mode test

.EXAMPLE
    .\Start-Agent.ps1 -Mode discovery -Mock -Background

.EXAMPLE
    .\Start-Agent.ps1 -Mode trading

.EXAMPLE
    .\Start-Agent.ps1 -Docker -Mode full

.NOTES
    Auteur: TradingBot V4
    Version: 1.0.0
#>

[CmdletBinding()]
param(
    [ValidateSet("test", "discovery", "analysis", "trading", "full")]
    [string]$Mode = "test",

    [switch]$Mock,

    [switch]$Background,

    [switch]$Stop,

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
. "$ScriptRoot\common\Initialize-Venv.ps1"

# Nom du job pour le mode background
$JobName = "TradingBot-Agent"

function Get-AgentProcesses {
    <#
    .SYNOPSIS
        Retourne les processus agent en cours
    #>
    param()

    return Get-Process -Name "python" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -like "*run_agent.py*"
        }
}

function Stop-AgentProcesses {
    <#
    .SYNOPSIS
        Arrete tous les processus agent
    #>
    param()

    $processes = Get-AgentProcesses

    if ($processes) {
        Write-LogInfo "Arret de $($processes.Count) processus agent..."

        foreach ($proc in $processes) {
            try {
                $proc | Stop-Process -Force
                Write-LogDebug "Processus $($proc.Id) arrete"
            }
            catch {
                Write-LogWarning "Impossible d'arreter le processus $($proc.Id)"
            }
        }

        Start-Sleep -Seconds 1
    }

    # Arreter le job si existant
    $job = Get-Job -Name $JobName -ErrorAction SilentlyContinue
    if ($job) {
        Write-LogInfo "Arret du job $JobName..."
        $job | Stop-Job -PassThru | Remove-Job
    }

    # Arreter les containers Docker
    $dockerContainer = docker ps -q --filter "name=tradingbot-agent" 2>$null
    if ($dockerContainer) {
        Write-LogInfo "Arret du container Docker..."
        docker stop tradingbot-agent 2>$null
    }

    Write-LogSuccess "Agent arrete"
}

function Get-AgentStatus {
    <#
    .SYNOPSIS
        Retourne le statut de l'agent
    #>
    param()

    $status = [PSCustomObject]@{
        Running     = $false
        ProcessId   = $null
        Mode        = $null
        JobRunning  = $false
        DockerRunning = $false
    }

    # Verifier le job
    $job = Get-Job -Name $JobName -ErrorAction SilentlyContinue
    if ($job -and $job.State -eq 'Running') {
        $status.JobRunning = $true
        $status.Running = $true
    }

    # Verifier les processus
    $processes = Get-AgentProcesses
    if ($processes) {
        $status.Running = $true
        $status.ProcessId = $processes[0].Id
    }

    # Verifier Docker
    $dockerContainer = docker ps -q --filter "name=tradingbot-agent" 2>$null
    if ($dockerContainer) {
        $status.DockerRunning = $true
        $status.Running = $true
    }

    return $status
}

function Start-AgentPython {
    <#
    .SYNOPSIS
        Demarre l'agent via Python directement
    #>
    param()

    $agentPath = Join-Path $ProjectRoot "run_agent.py"

    if (-not (Test-Path $agentPath)) {
        Write-LogError "run_agent.py non trouve: $agentPath"
        return $false
    }

    # Verifier/activer le venv
    if (-not (Enable-PythonVenv)) {
        Write-LogError "Impossible d'activer l'environnement virtuel"
        return $false
    }

    $pythonPath = Get-VenvPythonPath
    $pythonArgs = @($agentPath, "--mode", $Mode)

    if ($Mock -or $Mode -eq "test") {
        $pythonArgs += "--mock"
    }

    if ($Background) {
        # Mode arriere-plan avec Job
        Write-LogInfo "Demarrage de l'agent en arriere-plan..."

        $scriptBlock = {
            param($pythonPath, $args, $projectRoot)
            Set-Location $projectRoot
            & $pythonPath @args
        }

        $job = Start-Job -Name $JobName -ScriptBlock $scriptBlock `
            -ArgumentList $pythonPath, $pythonArgs, $ProjectRoot

        # Attendre un peu pour verifier le demarrage
        Start-Sleep -Seconds 5

        if ($job.State -eq 'Running') {
            Write-LogSuccess "Agent demarre en arriere-plan"
            Write-LogInfo "Mode: $Mode"
            Write-LogInfo "Job ID: $($job.Id)"
            Write-LogInfo "Pour arreter: .\Start-Agent.ps1 -Stop"
            Write-LogInfo "Pour voir les logs: Receive-Job -Name $JobName"
            return $true
        }
        else {
            $output = Receive-Job -Job $job
            Write-LogError "L'agent n'a pas demarre correctement"
            Write-LogError "Sortie: $output"
            Remove-Job -Job $job -Force
            return $false
        }
    }
    else {
        # Mode interactif
        Write-LogInfo "Demarrage de l'agent..."
        Write-LogInfo "Mode: $Mode"
        Write-LogInfo "Mock: $($Mock -or $Mode -eq 'test')"
        Write-LogInfo "Appuyez sur Ctrl+C pour arreter"
        Write-Host ""

        # Executer l'agent (bloquant)
        Push-Location $ProjectRoot
        try {
            & $pythonPath @pythonArgs
            return ($LASTEXITCODE -eq 0)
        }
        finally {
            Pop-Location
        }
    }
}

function Start-AgentDocker {
    <#
    .SYNOPSIS
        Demarre l'agent via Docker
    #>
    param()

    # Verifier que Docker est installe
    $dockerVersion = docker --version 2>$null
    if (-not $dockerVersion) {
        Write-LogError "Docker n'est pas installe ou n'est pas dans le PATH"
        return $false
    }

    Write-LogInfo "Docker detecte: $dockerVersion"

    # Verifier que le fichier .env existe
    $envFile = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path $envFile)) {
        Write-LogWarning ".env non trouve. Copie de .env.example..."
        $envExample = Join-Path $ProjectRoot ".env.example"
        if (Test-Path $envExample) {
            Copy-Item $envExample $envFile
        }
        else {
            Write-LogError ".env.example non trouve"
            return $false
        }
    }

    # Definir le mode d'agent
    $env:AGENT_MODE = $Mode

    Push-Location $ProjectRoot
    try {
        if ($Background) {
            Write-LogInfo "Demarrage de l'agent Docker en arriere-plan..."
            docker-compose up -d agent
        }
        else {
            Write-LogInfo "Demarrage de l'agent Docker..."
            Write-LogInfo "Mode: $Mode"
            Write-LogInfo "Appuyez sur Ctrl+C pour arreter"
            docker-compose up agent
        }

        return ($LASTEXITCODE -eq 0)
    }
    finally {
        Pop-Location
    }
}

# Point d'entree principal
function Start-Agent {
    param()

    try {
        # Mode arret
        if ($Stop) {
            Write-Banner -Title "TradingBot Agent V4" -Subtitle "Arret"
            Stop-AgentProcesses
            return $true
        }

        Write-Banner -Title "TradingBot Agent V4" -Subtitle "Mode: $Mode"

        # Verifier le statut actuel
        $status = Get-AgentStatus

        if ($status.Running) {
            Write-LogWarning "L'agent est deja en cours d'execution"
            if ($status.DockerRunning) {
                Write-LogInfo "Mode: Docker"
            }
            elseif ($status.JobRunning) {
                Write-LogInfo "Mode: Background Job"
            }
            else {
                Write-LogInfo "PID: $($status.ProcessId)"
            }
            Write-LogInfo "Utilisez -Stop pour arreter"
            return $true
        }

        Write-Section "Configuration"
        Write-Result -Item "Mode" -Success $true -Message $Mode
        Write-Result -Item "Mock IBKR" -Success $true -Message $(if ($Mock -or $Mode -eq "test") { "Oui" } else { "Non" })
        Write-Result -Item "Backend" -Success $true -Message $(if ($Docker) { "Docker" } else { "Python" })
        Write-Host ""

        # Demarrer l'agent
        if ($Docker) {
            return Start-AgentDocker
        }
        else {
            return Start-AgentPython
        }
    }
    catch {
        Write-LogError "Erreur: $_"
        Write-LogError $_.ScriptStackTrace
        return $false
    }
}

# Execution
if ($MyInvocation.InvocationName -ne '.') {
    Push-Location $ProjectRoot

    try {
        $result = Start-Agent

        if (-not $result) {
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}
