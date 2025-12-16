@echo off
echo ================================
echo API Key Setup
echo ================================
echo.


set GEMINI_KEY=
set OPENAI_KEY=
set RAPID_KEY=


echo 1) Set TEMPORARY (current terminal only)
echo 2) Set PERMANENT (survives reboot)
echo 3) DELETE keys
echo.
set /p choice=Choose option (1, 2, or 3): 

if "%choice%"=="1" (
    set GEMINI_API_KEY=%GEMINI_KEY%
    set OPENAI_API_KEY=%OPENAI_KEY%
    set RAPID_API_KEY=%RAPID_KEY%
    echo.
    echo API keys set TEMPORARILY for this session.
)

if "%choice%"=="2" (
    setx GEMINI_API_KEY "%GEMINI_KEY%"
    setx OPENAI_API_KEY "%OPENAI_KEY%"
    setx RAPID_API_KEY "%RAPID_KEY%"
    echo.
    echo API keys set PERMANENTLY.
)

if "%choice%"=="3" (
    set GEMINI_API_KEY=
    set OPENAI_API_KEY=
    set RAPID_API_KEY=

    setx GEMINI_API_KEY ""
    setx OPENAI_API_KEY ""
    setx RAPID_API_KEY ""

    echo.
    echo API keys DELETED.
)

if not "%choice%"=="1" if not "%choice%"=="2" if not "%choice%"=="3" (
    echo Invalid choice.
)

echo.
pause
