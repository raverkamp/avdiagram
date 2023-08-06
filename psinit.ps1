 $p = $MyInvocation.MyCommand.Path
 Write-Host $p
 $BASEDIR = Split-Path -Path $p

 . $BASEDIR\pyenv\Scripts\Activate.ps1  

 $env:PYTHONPATH=$BASEDIR
 
