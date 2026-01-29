<#
.SYNOPSIS
    Module de logging pour TradingBot - Gestion des logs avec couleurs et fichiers

.DESCRIPTION
    Fournit des fonctions de logging avec:
    - Niveaux: DEBUG, INFO, WARNING, ERROR, SUCCESS
    - Sortie console colorée
    - Écriture dans fichier avec rotation
    - Horodatage précis

.EXAMPLE
    . .\common\Write-Log.ps1
    Write-Log -Message "Démarrage du screening" -Level INFO
    Write-Log "Erreur critique" -Level ERROR
#>

# Configuration par défaut
$script:LogConfig = @{
    LogDirectory = Join-Path $PSScriptRoot "..\..\logs"
    LogFileName  = "tradingbot_ps.log"
    MaxLogSizeMB = 10
    MaxLogFiles  = 5
    MinLogLevel  = "DEBUG"
    ConsoleOutput = $true
    FileOutput   = $true
}

# Niveaux de log avec priorité
$script:LogLevels = @{
    "DEBUG"   = @{ Priority = 0; Color = "Gray";    Symbol = "[DEBUG]  " }
    "INFO"    = @{ Priority = 1; Color = "White";   Symbol = "[INFO]   " }
    "SUCCESS" = @{ Priority = 2; Color = "Green";   Symbol = "[SUCCESS]" }
    "WARNING" = @{ Priority = 3; Color = "Yellow";  Symbol = "[WARNING]" }
    "ERROR"   = @{ Priority = 4; Color = "Red";     Symbol = "[ERROR]  " }
}

function Initialize-LogDirectory {
    <#
    .SYNOPSIS
        Initialise le répertoire de logs
    #>
    param()

    $logDir = $script:LogConfig.LogDirectory

    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
}

function Get-LogFilePath {
    <#
    .SYNOPSIS
        Retourne le chemin complet du fichier de log
    #>
    param()

    return Join-Path $script:LogConfig.LogDirectory $script:LogConfig.LogFileName
}

function Invoke-LogRotation {
    <#
    .SYNOPSIS
        Effectue la rotation des fichiers de log si nécessaire
    #>
    param()

    $logPath = Get-LogFilePath

    if (Test-Path $logPath) {
        $logFile = Get-Item $logPath
        $maxSizeBytes = $script:LogConfig.MaxLogSizeMB * 1MB

        if ($logFile.Length -gt $maxSizeBytes) {
            # Rotation des anciens fichiers
            for ($i = $script:LogConfig.MaxLogFiles - 1; $i -ge 1; $i--) {
                $oldFile = "$logPath.$i"
                $newFile = "$logPath.$($i + 1)"

                if (Test-Path $oldFile) {
                    if ($i -eq ($script:LogConfig.MaxLogFiles - 1)) {
                        Remove-Item $oldFile -Force
                    } else {
                        Move-Item $oldFile $newFile -Force
                    }
                }
            }

            # Renommer le fichier actuel
            Move-Item $logPath "$logPath.1" -Force
        }
    }
}

function Write-Log {
    <#
    .SYNOPSIS
        Écrit un message de log avec niveau et horodatage

    .DESCRIPTION
        Écrit un message formaté dans la console (avec couleur) et/ou dans un fichier.
        Supporte différents niveaux de log: DEBUG, INFO, SUCCESS, WARNING, ERROR

    .PARAMETER Message
        Le message à logger

    .PARAMETER Level
        Niveau de log (DEBUG, INFO, SUCCESS, WARNING, ERROR). Par défaut: INFO

    .PARAMETER NoConsole
        Désactive l'affichage console

    .PARAMETER NoFile
        Désactive l'écriture fichier

    .EXAMPLE
        Write-Log -Message "Démarrage du bot" -Level INFO

    .EXAMPLE
        Write-Log "Erreur de connexion" ERROR

    .EXAMPLE
        Write-Log "Debug info" -Level DEBUG -NoFile
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string]$Message,

        [Parameter(Position = 1)]
        [ValidateSet("DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR")]
        [string]$Level = "INFO",

        [switch]$NoConsole,
        [switch]$NoFile
    )

    # Vérifier le niveau minimum
    $currentLevelPriority = $script:LogLevels[$Level].Priority
    $minLevelPriority = $script:LogLevels[$script:LogConfig.MinLogLevel].Priority

    if ($currentLevelPriority -lt $minLevelPriority) {
        return
    }

    # Formater le timestamp
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"

    # Obtenir les informations de niveau
    $levelInfo = $script:LogLevels[$Level]

    # Construire le message formaté
    $formattedMessage = "$timestamp $($levelInfo.Symbol) $Message"

    # Sortie console
    if ($script:LogConfig.ConsoleOutput -and -not $NoConsole) {
        Write-Host $formattedMessage -ForegroundColor $levelInfo.Color
    }

    # Sortie fichier
    if ($script:LogConfig.FileOutput -and -not $NoFile) {
        try {
            Initialize-LogDirectory
            Invoke-LogRotation

            $logPath = Get-LogFilePath
            Add-Content -Path $logPath -Value $formattedMessage -Encoding UTF8
        }
        catch {
            # En cas d'erreur d'écriture, afficher uniquement en console
            Write-Host "ERREUR: Impossible d'écrire dans le fichier de log: $_" -ForegroundColor Red
        }
    }
}

function Write-LogDebug {
    <#
    .SYNOPSIS
        Raccourci pour Write-Log avec niveau DEBUG
    #>
    param([Parameter(Mandatory)][string]$Message)
    Write-Log -Message $Message -Level DEBUG
}

function Write-LogInfo {
    <#
    .SYNOPSIS
        Raccourci pour Write-Log avec niveau INFO
    #>
    param([Parameter(Mandatory)][string]$Message)
    Write-Log -Message $Message -Level INFO
}

function Write-LogSuccess {
    <#
    .SYNOPSIS
        Raccourci pour Write-Log avec niveau SUCCESS
    #>
    param([Parameter(Mandatory)][string]$Message)
    Write-Log -Message $Message -Level SUCCESS
}

function Write-LogWarning {
    <#
    .SYNOPSIS
        Raccourci pour Write-Log avec niveau WARNING
    #>
    param([Parameter(Mandatory)][string]$Message)
    Write-Log -Message $Message -Level WARNING
}

function Write-LogError {
    <#
    .SYNOPSIS
        Raccourci pour Write-Log avec niveau ERROR
    #>
    param([Parameter(Mandatory)][string]$Message)
    Write-Log -Message $Message -Level ERROR
}

function Set-LogConfig {
    <#
    .SYNOPSIS
        Configure les paramètres de logging

    .PARAMETER LogDirectory
        Répertoire pour les fichiers de log

    .PARAMETER MinLogLevel
        Niveau minimum de log à enregistrer

    .PARAMETER ConsoleOutput
        Activer/désactiver la sortie console

    .PARAMETER FileOutput
        Activer/désactiver l'écriture fichier

    .EXAMPLE
        Set-LogConfig -MinLogLevel WARNING -ConsoleOutput $true
    #>
    [CmdletBinding()]
    param(
        [string]$LogDirectory,
        [ValidateSet("DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR")]
        [string]$MinLogLevel,
        [bool]$ConsoleOutput,
        [bool]$FileOutput
    )

    if ($LogDirectory) {
        $script:LogConfig.LogDirectory = $LogDirectory
    }
    if ($MinLogLevel) {
        $script:LogConfig.MinLogLevel = $MinLogLevel
    }
    if ($PSBoundParameters.ContainsKey('ConsoleOutput')) {
        $script:LogConfig.ConsoleOutput = $ConsoleOutput
    }
    if ($PSBoundParameters.ContainsKey('FileOutput')) {
        $script:LogConfig.FileOutput = $FileOutput
    }
}

function Write-Banner {
    <#
    .SYNOPSIS
        Affiche une bannière décorative pour les scripts

    .PARAMETER Title
        Titre à afficher dans la bannière

    .PARAMETER Subtitle
        Sous-titre optionnel

    .EXAMPLE
        Write-Banner -Title "TradingBot" -Subtitle "Market Screener v3"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Title,

        [string]$Subtitle
    )

    $width = 60
    $border = "=" * $width

    Write-Host ""
    Write-Host $border -ForegroundColor Cyan

    # Centrer le titre
    $padding = [Math]::Max(0, ($width - $Title.Length) / 2)
    $centeredTitle = (" " * [int]$padding) + $Title
    Write-Host $centeredTitle -ForegroundColor Cyan

    if ($Subtitle) {
        $padding = [Math]::Max(0, ($width - $Subtitle.Length) / 2)
        $centeredSubtitle = (" " * [int]$padding) + $Subtitle
        Write-Host $centeredSubtitle -ForegroundColor Gray
    }

    Write-Host $border -ForegroundColor Cyan
    Write-Host ""
}

function Write-Section {
    <#
    .SYNOPSIS
        Affiche un titre de section formaté

    .PARAMETER Title
        Titre de la section

    .EXAMPLE
        Write-Section "Configuration"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Title
    )

    Write-Host ""
    Write-Host "--- $Title ---" -ForegroundColor Yellow
    Write-Host ""
}

function Write-Result {
    <#
    .SYNOPSIS
        Affiche un résultat avec statut (succès/échec)

    .PARAMETER Item
        Élément vérifié

    .PARAMETER Success
        $true si succès, $false si échec

    .PARAMETER Message
        Message optionnel

    .EXAMPLE
        Write-Result -Item "Python" -Success $true -Message "Version 3.11 détectée"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Item,

        [Parameter(Mandatory)]
        [bool]$Success,

        [string]$Message
    )

    if ($Success) {
        $symbol = "[OK]"
        $color = "Green"
    } else {
        $symbol = "[FAIL]"
        $color = "Red"
    }

    $output = "  $symbol $Item"
    if ($Message) {
        $output += " - $Message"
    }

    Write-Host $output -ForegroundColor $color
}

# Exporter les fonctions
Export-ModuleMember -Function Write-Log, Write-LogDebug, Write-LogInfo, Write-LogSuccess, `
    Write-LogWarning, Write-LogError, Set-LogConfig, Write-Banner, Write-Section, Write-Result `
    -ErrorAction SilentlyContinue

# Initialiser le répertoire de logs au chargement
Initialize-LogDirectory
