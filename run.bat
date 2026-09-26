@echo off
echo ====================================================
echo   Launching NEON-OVERDRIVE
echo ====================================================

if not exist "bin" (
    echo Building project first...
    call build.bat
)

java -cp bin com.cybersynth.Main
