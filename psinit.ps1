 $p = $MyInvocation.MyCommand.Path
 Write-Host $p
 $BASEDIR = Split-Path -Path $p

 $env:PYTHONPATH=$BASEDIR

 . $BASEDIR\pyenv\Scripts\Activate.ps1
 
