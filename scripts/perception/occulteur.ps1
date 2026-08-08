# Fenêtre OCCULTEUR du test Phase 1 : topmost, quasi noire (RGB 10,10,10),
# recouvre ENTIÈREMENT la cible (marge de 50 px de chaque côté).
# Si la capture rendait une moyenne d'octets ≈ 10, c'est qu'on capture
# l'écran et non la fenêtre — l'API serait disqualifiée.
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

$script:f = New-Object System.Windows.Forms.Form
$script:f.Text = 'PFS-OCCULTEUR'
$script:f.StartPosition = 'Manual'
$script:f.Location = New-Object System.Drawing.Point(50, 50)
$script:f.Size = New-Object System.Drawing.Size(900, 700)
$script:f.TopMost = $true
$script:f.BackColor = [System.Drawing.Color]::FromArgb(10, 10, 10)

$kill = New-Object System.Windows.Forms.Timer
$kill.Interval = 120000
$kill.Add_Tick({ $script:f.Close() })
$kill.Start()

[System.Windows.Forms.Application]::Run($script:f)
