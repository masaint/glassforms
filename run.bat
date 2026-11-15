@echo off
rem Glass Forms, native. Builds the program the first time (a few minutes), then starts it.
rem The program is kept here, beside this file, so that what it saves (renders, stills, setups and
rem glassforms-ui.json) lands here too and not inside the build folder.
cd /d "%~dp0"
if not exist "target\release\glassforms.exe" (
  echo Building Glass Forms: this takes a few minutes the first time...
  cargo build --release -p glassforms || (echo The build failed. & pause & exit /b 1)
)
rem take the newly built program if it is newer than the one here
xcopy /D /Y /Q "target\release\glassforms.exe" "." >nul
start "" "glassforms.exe" %*
