# Dizel desktop startup spinner.
#
# The batch launcher cannot animate while `npm run dev` blocks, so it hands the
# startup to this helper. The helper starts the dev server with output going to
# the launcher log, animates one line while it boots, and turns the line into a
# check once Vite reports readiness. It then waits silently for the window to
# close and returns the dev server's exit code to dizel.cmd.
param(
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [string]$Command = "npm run dev"
)

$ErrorActionPreference = "SilentlyContinue"
try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
} catch {}

$esc = [char]27
$check = [char]0x2713
# ASCII frames render in every console host; the rest of the launcher already
# opts into UTF-8 for its brand glyphs.
$frames = @("|", "/", "-", "\")

$started = Get-Date
# A previous run's "ready in" line must not mark this launch ready before Vite
# has even started. The dev server truncates this file too; clear it first.
if (Test-Path -LiteralPath $LogPath) {
    Remove-Item -LiteralPath $LogPath -Force -ErrorAction SilentlyContinue
}
$cmdLine = '/c ' + $Command + ' > "' + $LogPath + '" 2>&1'
$proc = Start-Process -FilePath $env:ComSpec -ArgumentList $cmdLine `
    -WorkingDirectory $WorkDir -NoNewWindow -PassThru

$ready = $false
$nextProbe = 0
$frame = 0

try {
    while ($proc -and -not $proc.HasExited) {
        $elapsed = [int]((Get-Date) - $started).TotalSeconds
        if (-not $ready -and $elapsed -ge $nextProbe) {
            $nextProbe = $elapsed + 1
            $tail = ""
            if (Test-Path -LiteralPath $LogPath) {
                $tail = (Get-Content -LiteralPath $LogPath -Tail 25 -ErrorAction SilentlyContinue | Out-String)
            }
            if ($tail -match "ready in|Local:\s+http") {
                $ready = $true
                Write-Host ("`r  " + $esc + "[38;5;78m" + $check + $esc + "[0m  " +
                    $esc + "[97mvite ready" + $esc + "[0m" +
                    $esc + "[90m · opening window" + $esc + "[0m" + $esc + "[K")
            }
        }

        if (-not $ready) {
            $line = "`r  " + $frames[$frame % $frames.Count] + "  " +
                $esc + "[90mstarting vite + electron · " + $elapsed + "s" + $esc + "[0m" + $esc + "[K"
            Write-Host -NoNewline $line
            $frame++
            Start-Sleep -Milliseconds 90
        } else {
            Start-Sleep -Milliseconds 200
        }
        $proc.Refresh()
    }
} finally {
    if ($proc -and -not $proc.HasExited) {
        & taskkill.exe /PID $proc.Id /T /F *> $null
    }
}

if (-not $ready) {
    Write-Host ("`r" + $esc + "[K") -NoNewline
}
exit $proc.ExitCode
