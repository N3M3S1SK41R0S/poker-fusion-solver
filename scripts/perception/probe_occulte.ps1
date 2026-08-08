# Test go/no-go Phase 1 — capture d'une fenêtre OCCULTÉE, latence p95.
#
# Orchestration :
#   1. lance la cible animée (PFS-CIBLE) et l'occulteur topmost qui la
#      recouvre entièrement (PFS-OCCULTEUR) ;
#   2. exécute la sonde Rust (probe.exe) contre la cible occultée ;
#   3. nettoie, puis rend le verdict sur trois critères :
#        frames reçues  → WGC capture bien une fenêtre occultée ;
#        mean_px ≥ 150  → c'est le CONTENU de la cible (l'occulteur ≈ 10) ;
#        copy p95 < 1000 µs → coût de lecture par frame sous la milliseconde.
#
# Usage :  pwsh scripts/perception/probe_occulte.ps1 [-Frames 300]
param([int]$Frames = 300)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$probe = Join-Path $here '..\..\rust\target\release\probe.exe'
if (-not (Test-Path $probe)) {
    Write-Error "probe.exe introuvable — compiler d'abord : cargo build --release -p pfs-capture --bin probe"
}

# 1. cible + occulteur (Windows PowerShell 5.1, STA requis par WinForms).
# ATTENTION : pas de -WindowStyle Hidden ici — ce flag s'applique à la
# PREMIÈRE fenêtre du processus, donc au formulaire lui-même, qui
# démarrerait invisible (et une fenêtre cachée/minimisée n'est pas
# capturable par WGC).
$target = Start-Process powershell -PassThru -ArgumentList `
    '-NoProfile', '-STA', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $here 'cible.ps1')
$occluder = Start-Process powershell -PassThru -ArgumentList `
    '-NoProfile', '-STA', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $here 'occulteur.ps1')

try {
    # attendre que les deux fenêtres existent réellement
    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline) {
        $titles = (Get-Process | Where-Object MainWindowTitle).MainWindowTitle
        if (($titles -contains 'PFS-CIBLE') -and ($titles -contains 'PFS-OCCULTEUR')) { break }
        Start-Sleep -Milliseconds 200
    }
    if (-not ((Get-Process | Where-Object MainWindowTitle).MainWindowTitle -contains 'PFS-CIBLE')) {
        Write-Error 'la fenêtre cible ne s''est pas ouverte en 20 s'
    }
    Start-Sleep -Seconds 1   # laisser l'occulteur passer réellement au-dessus

    # 2. la sonde, contre la cible OCCULTÉE
    $json = & $probe 'PFS-CIBLE' $Frames
    if ($LASTEXITCODE -ne 0) { Write-Error "sonde en échec (code $LASTEXITCODE)" }
    $r = $json | ConvertFrom-Json

    # 3. verdict
    $okFrames = $r.frames -ge $Frames
    $okContent = $r.mean_px -ge 150
    # Le critère de latence porte sur la ROI (le scraper lit des régions
    # d'intérêt ~200×100, jamais le buffer complet) ; le plein buffer est
    # rapporté à titre informatif.
    $okLatency = $r.crop_us.p95 -lt 1000
    Write-Output $json
    Write-Output ("occlusion    : {0} frames capturées, mean_px={1} (cible claire, occulteur=10) → {2}" -f
        $r.frames, $r.mean_px, $(if ($okContent) { 'CONTENU DE LA CIBLE' } else { 'ÉCHEC — pas le contenu' }))
    Write-Output ("ROI 200×100  : p50={0}µs p95={1}µs p99={2}µs → {3}" -f
        $r.crop_us.p50, $r.crop_us.p95, $r.crop_us.p99, $(if ($okLatency) { 'p95 < 1 ms' } else { 'AU-DESSUS de 1 ms' }))
    Write-Output ("plein buffer : p50={0}µs p95={1}µs (informatif — {2}×{3}×4 octets)" -f
        $r.copy_us.p50, $r.copy_us.p95, $r.width, $r.height)
    if ($okFrames -and $okContent -and $okLatency) {
        Write-Output 'VERDICT : GO — la perception Phase 1 est faisable sur cette machine.'
        exit 0
    } else {
        Write-Output 'VERDICT : NO-GO — voir les critères en échec ci-dessus.'
        exit 1
    }
}
finally {
    foreach ($p in @($target, $occluder)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
}
