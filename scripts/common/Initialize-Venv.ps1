<#
.SYNOPSIS
    Module de gestion de l'environnement virtuel Python pour TradingBot

.DESCRIPTION
    Fournit des fonctions pour:
    - Créer un environnement virtuel Python
    - Activer l'environnement virtuel
    - Vérifier l'installation de Python
    - Installer les dépendances

.EXAMPLE
    . .\common\Initialize-Venv.ps1
    Initialize-PythonVenv
#>

# Importer le module de logging
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$scriptRoot\Write-Log.ps1"

# Configuration
$script:VenvConfig = @{
    ProjectRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
    VenvName    = "venv"
    MinPythonVersion = [Version]"3.8.0"
}

function Get-VenvPath {
    <#
    .SYNOPSIS
        Retourne le chemin de l'environnement virtuel
    #>
    return Join-Path $script:VenvConfig.ProjectRoot $script:VenvConfig.VenvName
}

function Get-VenvActivatePath {
    <#
    .SYNOPSIS
        Retourne le chemin du script d'activation
    #>
    $venvPath = Get-VenvPath
    return Join-Path $venvPath "Scripts\Activate.ps1"
}

function Get-VenvPythonPath {
    <#
    .SYNOPSIS
        Retourne le chemin de l'exécutable Python dans le venv
    #>
    $venvPath = Get-VenvPath
    return Join-Path $venvPath "Scripts\python.exe"
}

function Get-VenvPipPath {
    <#
    .SYNOPSIS
        Retourne le chemin de l'exécutable pip dans le venv
    #>
    $venvPath = Get-VenvPath
    return Join-Path $venvPath "Scripts\pip.exe"
}

function Test-PythonInstalled {
    <#
    .SYNOPSIS
        Vérifie si Python est installé et retourne la version

    .OUTPUTS
        PSObject avec propriétés: Installed, Version, Path
    #>
    [CmdletBinding()]
    param()

    $result = [PSCustomObject]@{
        Installed = $false
        Version   = $null
        Path      = $null
    }

    try {
        # Essayer python puis python3
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCmd) {
            $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
        }

        if ($pythonCmd) {
            $result.Path = $pythonCmd.Source

            # Obtenir la version
            $versionOutput = & $pythonCmd.Source --version 2>&1
            if ($versionOutput -match "Python (\d+\.\d+\.\d+)") {
                $result.Version = [Version]$Matches[1]
                $result.Installed = $true
            }
        }
    }
    catch {
        Write-LogDebug "Erreur lors de la vérification de Python: $_"
    }

    return $result
}

function Test-PythonVersion {
    <#
    .SYNOPSIS
        Vérifie si la version de Python est compatible

    .OUTPUTS
        $true si compatible, $false sinon
    #>
    [CmdletBinding()]
    param()

    $pythonInfo = Test-PythonInstalled

    if (-not $pythonInfo.Installed) {
        return $false
    }

    return $pythonInfo.Version -ge $script:VenvConfig.MinPythonVersion
}

function Test-VenvExists {
    <#
    .SYNOPSIS
        Vérifie si l'environnement virtuel existe

    .OUTPUTS
        $true si le venv existe et est valide
    #>
    [CmdletBinding()]
    param()

    $activatePath = Get-VenvActivatePath
    $pythonPath = Get-VenvPythonPath

    return (Test-Path $activatePath) -and (Test-Path $pythonPath)
}

function New-PythonVenv {
    <#
    .SYNOPSIS
        Crée un nouvel environnement virtuel Python

    .PARAMETER Force
        Recréer le venv même s'il existe déjà

    .OUTPUTS
        $true si succès
    #>
    [CmdletBinding()]
    param(
        [switch]$Force
    )

    $venvPath = Get-VenvPath

    # Vérifier si le venv existe déjà
    if ((Test-VenvExists) -and -not $Force) {
        Write-LogInfo "L'environnement virtuel existe déjà: $venvPath"
        return $true
    }

    # Supprimer l'ancien venv si -Force
    if ($Force -and (Test-Path $venvPath)) {
        Write-LogWarning "Suppression de l'ancien environnement virtuel..."
        Remove-Item -Path $venvPath -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Vérifier Python
    if (-not (Test-PythonVersion)) {
        $pythonInfo = Test-PythonInstalled
        if (-not $pythonInfo.Installed) {
            Write-LogError "Python n'est pas installé. Veuillez installer Python $($script:VenvConfig.MinPythonVersion) ou supérieur."
        } else {
            Write-LogError "Version Python $($pythonInfo.Version) détectée, mais version $($script:VenvConfig.MinPythonVersion) ou supérieure requise."
        }
        return $false
    }

    # Créer le venv
    Write-LogInfo "Création de l'environnement virtuel..."

    try {
        $pythonInfo = Test-PythonInstalled
        $createResult = & $pythonInfo.Path -m venv $venvPath 2>&1

        if ($LASTEXITCODE -ne 0) {
            Write-LogError "Échec de la création du venv: $createResult"
            return $false
        }

        Write-LogSuccess "Environnement virtuel créé: $venvPath"
        return $true
    }
    catch {
        Write-LogError "Erreur lors de la création du venv: $_"
        return $false
    }
}

function Enable-PythonVenv {
    <#
    .SYNOPSIS
        Active l'environnement virtuel Python

    .DESCRIPTION
        Active le venv en modifiant les variables d'environnement PATH et VIRTUAL_ENV.
        Note: L'activation dans PowerShell est persistante pour la session courante.

    .OUTPUTS
        $true si succès
    #>
    [CmdletBinding()]
    param()

    if (-not (Test-VenvExists)) {
        Write-LogError "L'environnement virtuel n'existe pas. Exécutez d'abord New-PythonVenv."
        return $false
    }

    $venvPath = Get-VenvPath
    $activatePath = Get-VenvActivatePath

    try {
        # Vérifier si déjà activé
        if ($env:VIRTUAL_ENV -eq $venvPath) {
            Write-LogDebug "L'environnement virtuel est déjà activé"
            return $true
        }

        # Activer le venv en utilisant le script d'activation
        Write-LogInfo "Activation de l'environnement virtuel..."

        # Exécuter le script d'activation
        . $activatePath

        # Vérifier l'activation
        if ($env:VIRTUAL_ENV -eq $venvPath) {
            Write-LogSuccess "Environnement virtuel activé"
            return $true
        } else {
            # Activation manuelle si le script ne fonctionne pas
            $env:VIRTUAL_ENV = $venvPath
            $env:PATH = "$(Join-Path $venvPath 'Scripts');$env:PATH"
            Write-LogSuccess "Environnement virtuel activé (méthode alternative)"
            return $true
        }
    }
    catch {
        Write-LogError "Erreur lors de l'activation du venv: $_"
        return $false
    }
}

function Disable-PythonVenv {
    <#
    .SYNOPSIS
        Désactive l'environnement virtuel Python
    #>
    [CmdletBinding()]
    param()

    if ($env:VIRTUAL_ENV) {
        $venvScripts = Join-Path $env:VIRTUAL_ENV "Scripts"
        $env:PATH = $env:PATH -replace [regex]::Escape("$venvScripts;"), ""
        Remove-Item Env:\VIRTUAL_ENV -ErrorAction SilentlyContinue
        Write-LogInfo "Environnement virtuel désactivé"
    }
}

function Install-PythonDependencies {
    <#
    .SYNOPSIS
        Installe les dépendances Python depuis requirements.txt

    .PARAMETER Upgrade
        Mettre à jour les packages existants

    .PARAMETER Quiet
        Mode silencieux (moins de sortie)

    .OUTPUTS
        $true si succès
    #>
    [CmdletBinding()]
    param(
        [switch]$Upgrade,
        [switch]$Quiet
    )

    $requirementsPath = Join-Path $script:VenvConfig.ProjectRoot "requirements.txt"

    if (-not (Test-Path $requirementsPath)) {
        Write-LogError "Fichier requirements.txt non trouvé: $requirementsPath"
        return $false
    }

    # Vérifier que le venv est activé
    if (-not $env:VIRTUAL_ENV) {
        Write-LogWarning "L'environnement virtuel n'est pas activé. Activation..."
        if (-not (Enable-PythonVenv)) {
            return $false
        }
    }

    $pipPath = Get-VenvPipPath

    if (-not (Test-Path $pipPath)) {
        Write-LogError "pip non trouvé dans le venv: $pipPath"
        return $false
    }

    Write-LogInfo "Installation des dépendances..."

    try {
        $pipArgs = @("install", "-r", $requirementsPath)

        if ($Upgrade) {
            $pipArgs += "--upgrade"
        }

        if ($Quiet) {
            $pipArgs += "-q"
        }

        $installResult = & $pipPath @pipArgs 2>&1

        if ($LASTEXITCODE -ne 0) {
            Write-LogError "Échec de l'installation des dépendances"
            Write-LogDebug "Détails: $installResult"
            return $false
        }

        Write-LogSuccess "Dépendances installées avec succès"
        return $true
    }
    catch {
        Write-LogError "Erreur lors de l'installation des dépendances: $_"
        return $false
    }
}

function Test-PythonPackage {
    <#
    .SYNOPSIS
        Vérifie si un package Python est installé

    .PARAMETER PackageName
        Nom du package à vérifier

    .OUTPUTS
        $true si le package est installé
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$PackageName
    )

    $pipPath = Get-VenvPipPath

    if (-not (Test-Path $pipPath)) {
        return $false
    }

    try {
        $result = & $pipPath show $PackageName 2>&1

        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Initialize-PythonVenv {
    <#
    .SYNOPSIS
        Initialise complètement l'environnement Python (création + activation + dépendances)

    .PARAMETER Force
        Recréer le venv même s'il existe

    .PARAMETER SkipDependencies
        Ne pas installer les dépendances

    .OUTPUTS
        $true si tout est OK
    #>
    [CmdletBinding()]
    param(
        [switch]$Force,
        [switch]$SkipDependencies
    )

    Write-LogInfo "Initialisation de l'environnement Python..."

    # Vérifier Python
    $pythonInfo = Test-PythonInstalled
    if (-not $pythonInfo.Installed) {
        Write-LogError "Python n'est pas installé"
        return $false
    }

    Write-Result -Item "Python" -Success $true -Message "Version $($pythonInfo.Version)"

    # Vérifier version
    if (-not (Test-PythonVersion)) {
        Write-Result -Item "Version Python" -Success $false -Message "Minimum requis: $($script:VenvConfig.MinPythonVersion)"
        return $false
    }

    Write-Result -Item "Version Python" -Success $true -Message ">= $($script:VenvConfig.MinPythonVersion)"

    # Créer le venv si nécessaire
    if (-not (Test-VenvExists) -or $Force) {
        if (-not (New-PythonVenv -Force:$Force)) {
            Write-Result -Item "Création venv" -Success $false
            return $false
        }
    }

    Write-Result -Item "Environnement virtuel" -Success $true

    # Activer le venv
    if (-not (Enable-PythonVenv)) {
        Write-Result -Item "Activation venv" -Success $false
        return $false
    }

    Write-Result -Item "Activation venv" -Success $true

    # Installer les dépendances
    if (-not $SkipDependencies) {
        if (-not (Install-PythonDependencies -Quiet)) {
            Write-Result -Item "Dépendances" -Success $false
            return $false
        }

        Write-Result -Item "Dépendances" -Success $true
    }

    Write-LogSuccess "Environnement Python initialisé avec succès"
    return $true
}

function Get-PythonInfo {
    <#
    .SYNOPSIS
        Retourne les informations sur l'environnement Python actuel

    .OUTPUTS
        PSObject avec les informations
    #>
    [CmdletBinding()]
    param()

    $info = [PSCustomObject]@{
        PythonInstalled    = $false
        PythonVersion      = $null
        PythonPath         = $null
        VenvExists         = $false
        VenvPath           = Get-VenvPath
        VenvActivated      = $false
        RequirementsExists = $false
    }

    # Python système
    $pythonInfo = Test-PythonInstalled
    $info.PythonInstalled = $pythonInfo.Installed
    $info.PythonVersion = $pythonInfo.Version
    $info.PythonPath = $pythonInfo.Path

    # Venv
    $info.VenvExists = Test-VenvExists
    $info.VenvActivated = ($env:VIRTUAL_ENV -eq $info.VenvPath)

    # Requirements
    $requirementsPath = Join-Path $script:VenvConfig.ProjectRoot "requirements.txt"
    $info.RequirementsExists = Test-Path $requirementsPath

    return $info
}

# Exporter les fonctions
Export-ModuleMember -Function Initialize-PythonVenv, New-PythonVenv, Enable-PythonVenv, `
    Disable-PythonVenv, Install-PythonDependencies, Test-VenvExists, Test-PythonInstalled, `
    Test-PythonVersion, Test-PythonPackage, Get-VenvPath, Get-VenvPythonPath, Get-PythonInfo `
    -ErrorAction SilentlyContinue
