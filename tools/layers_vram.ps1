# Video memory a combined scene holds: starts the program on a long run of the combining step and reads
# the process's dedicated video memory from Windows while it runs.
#
# usage: pwsh tools/layers_vram.ps1 <voxels a side> <layers 1..4> [apart]
#   apart: every layer kept apart from the others (as many bodies as layers, so a bodies' volume as well)
param([int]$res = 768, [int]$layers = 2, [string]$how = 'joined')
$exe = Join-Path (Split-Path $PSScriptRoot -Parent) 'target\release\glassforms.exe'
$scene = @('scene=attractors', 'preset=thomas', "res=$res")
$more = @('layer=surfaces:schwarzp', 'L2.size=0.6', "L2.res=$res"), @('layer=knots:trefoil', 'L3.size=0.8', "L3.res=$res"), @('layer=algebraic:tanglecube', 'L4.size=0.5', 'L4.x=0.5', "L4.res=$res")
for ($i = 0; $i -lt $layers - 1; $i++) { $scene += $more[$i] }
if ($how -eq 'apart') {
    foreach ($pair in '12', '13', '14', '23', '24', '34') {
        if ([int]"$($pair[1])" -le $layers) { $scene += "J$pair.op=apart" }
    }
}
if ($layers -eq 1) {
    # one plain layer: a long benchmark keeps it busy long enough to be read
    $run = @('--bench', '4000', 'w=640', 'h=360') + $scene
} else {
    $run = @('--bench-combine', '400', 'w=64', 'h=64') + $scene
}
$p = Start-Process -FilePath $exe -ArgumentList $run -PassThru -WindowStyle Hidden -RedirectStandardOutput "$env:TEMP\layers_vram.txt"
$peak = 0
while (-not $p.HasExited) {
    Start-Sleep -Milliseconds 300
    try {
        $c = Get-Counter "\GPU Process Memory(pid_$($p.Id)*)\Dedicated Usage" -ErrorAction Stop
        $now = ($c.CounterSamples | Measure-Object -Property CookedValue -Sum).Sum
        if ($now -gt $peak) { $peak = $now }
    } catch {}
}
$said = (Get-Content "$env:TEMP\layers_vram.txt" | Select-Object -Last 1)
$volumes = if ($said -match '"volumesMB":(\d+)') { $Matches[1] } else { '?' }
"{0} voxels a side, {1} layer(s) {2}: {3:N0} MB of video memory at most ({4} MB of it volumes)" -f $res, $layers, $how, ($peak / 1MB), $volumes
