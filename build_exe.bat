@echo off
echo Building SkyGuardAI.exe ...
set PYTHON="C:\Users\shrey\AppData\Local\Programs\Python\Python313\python.exe"
%PYTHON% -m pip install -r requirements.txt 2>nul
%PYTHON% -m PyInstaller --onefile --windowed --name SkyGuardAI ^
  --hidden-import sklearn.ensemble._forest ^
  --hidden-import sklearn.neighbors._base ^
  --collect-all sklearn --collect-all matplotlib --collect-all scipy ^
  --add-data "assets;assets" ^
  main.py
echo.
echo Done: dist\SkyGuardAI.exe
echo You can run it directly — fully offline.
pause
