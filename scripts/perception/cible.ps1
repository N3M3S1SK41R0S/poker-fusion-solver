# Fenêtre CIBLE du test d'occultation Phase 1 — animée en permanence
# (WGC n'émet des frames que sur mise à jour : une fenêtre statique ne
# produirait rien et la sonde attendrait indéfiniment).
# Couleurs volontairement CLAIRES : la moyenne des octets capturés doit
# rester haute (≥ 150) si et seulement si WGC lit bien le contenu de la
# fenêtre, pas l'écran (l'occulteur est quasi noir).
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

$script:f = New-Object System.Windows.Forms.Form
$script:f.Text = 'PFS-CIBLE'
$script:f.StartPosition = 'Manual'
$script:f.Location = New-Object System.Drawing.Point(100, 100)
$script:f.Size = New-Object System.Drawing.Size(800, 600)

$script:i = 0
$tick = New-Object System.Windows.Forms.Timer
$tick.Interval = 15
$tick.Add_Tick({
    $script:i = ($script:i + 7) % 60
    $v = 195 + $script:i
    $script:f.BackColor = [System.Drawing.Color]::FromArgb($v, 255 - $script:i, $v)
})
$tick.Start()

# garde-fou : fermeture automatique après 120 s
$kill = New-Object System.Windows.Forms.Timer
$kill.Interval = 120000
$kill.Add_Tick({ $script:f.Close() })
$kill.Start()

[System.Windows.Forms.Application]::Run($script:f)
