@echo off
setlocal enabledelayedexpansion

echo ====================================================================
echo       VibeVoice + SadTalker Talking-Head Avatar Pipeline
echo ====================================================================
echo.

:: 1. Verify Docker is installed and running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not running or not installed.
    echo Please make sure Docker Desktop is launched and WSL2 backend is active.
    pause
    exit /b 1
)

:: 2. Set default parameters or read command line arguments
set "TEXT=%~1"
set "PORTRAIT_PATH=%~2"
set "SPEAKER=%~3"

if "%SPEAKER%"=="" set "SPEAKER=en-Carter_man"

if "%PORTRAIT_PATH%"=="" (
    set "PORTRAIT_PATH=inputs\portrait.png"
)

if not exist "%PORTRAIT_PATH%" (
    echo [WARNING] Portrait image '%PORTRAIT_PATH%' not found!
    echo.
    echo Please provide a front-facing photo in: inputs\portrait.png
    echo Or specify the image path as argument: run_avatar.bat "Text" "path\to\image.png"
    echo.
    set /p USER_IMAGE="Enter image path (or press Enter if you placed portrait.png in inputs): "
    if not "!USER_IMAGE!"=="" set "PORTRAIT_PATH=!USER_IMAGE!"
)

if not exist "%PORTRAIT_PATH%" (
    echo [ERROR] Image file still not found. Aborting.
    pause
    exit /b 1
)

:: Ensure inputs/outputs folders exist
if not exist "inputs" mkdir inputs
if not exist "outputs" mkdir outputs
if not exist "checkpoints\sadtalker" mkdir "checkpoints\sadtalker"

:: Extract filename from path for container referencing
for %%F in ("%PORTRAIT_PATH%") do (
    set "PORTRAIT_FILE=%%~nxF"
    set "PORTRAIT_DIR=%%~dpF"
)

:: If image is outside inputs\, copy it into inputs\
if not exist "inputs\%PORTRAIT_FILE%" (
    echo Copying %PORTRAIT_PATH% to inputs\%PORTRAIT_FILE%...
    copy "%PORTRAIT_PATH%" "inputs\%PORTRAIT_FILE%" >nul
)

:: If text not provided via argument, prompt user
if "%TEXT%"=="" (
    echo.
    set /p TEXT="Enter text for avatar to say [Default: Hello! Welcome to the presentation.]: "
    if "!TEXT!"=="" set "TEXT=Hello! Welcome to the presentation."
)

echo.
echo --------------------------------------------------------------------
echo [Configuration]
echo Text:     "%TEXT%"
echo Portrait: "inputs\%PORTRAIT_FILE%"
echo Speaker:  "%SPEAKER%"
echo --------------------------------------------------------------------
echo.

:: 3. Stage 1: Audio source (custom audio file or VibeVoice generation)
set "IS_AUDIO_FILE=0"
if exist "%TEXT%" (
    for %%A in ("%TEXT%") do (
        set "EXT=%%~xA"
        if /i "!EXT!"==".wav" set "IS_AUDIO_FILE=1"
        if /i "!EXT!"==".mp3" set "IS_AUDIO_FILE=1"
        if /i "!EXT!"==".m4a" set "IS_AUDIO_FILE=1"
        if /i "!EXT!"==".ogg" set "IS_AUDIO_FILE=1"
        if /i "!EXT!"==".flac" set "IS_AUDIO_FILE=1"
    )
)

if "!IS_AUDIO_FILE!"=="1" (
    echo [Stage 1/2] Using provided audio file: %TEXT%
    copy "%TEXT%" "outputs\speech.wav" >nul
    echo [Stage 1/2 Complete] Custom audio copied to outputs\speech.wav
) else (
    echo [Stage 1/2] Generating speech audio with VibeVoice...
    echo Running VibeVoice container...
    docker compose run --rm vibevoice python demo/generate_speech_cli.py --text "%TEXT%" --speaker_name "%SPEAKER%" --output_path /app/outputs/speech.wav
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo [ERROR] VibeVoice audio generation failed. Check logs above.
        pause
        exit /b 1
    )
)

if not exist "outputs\speech.wav" (
    echo [ERROR] Audio file outputs\speech.wav was not created.
    pause
    exit /b 1
)
echo [Stage 1/2 Complete] Audio ready at outputs\speech.wav

echo.
:: 4. Stage 2: Generate Video with SadTalker
echo [Stage 2/2] Generating talking avatar video with SadTalker...
echo Running SadTalker container (wawa9000/sadtalker)...

docker run --gpus all --rm ^
  -v "%cd%\inputs:/host_dir/inputs" ^
  -v "%cd%\outputs:/host_dir/outputs" ^
  wawa9000/sadtalker ^
  --driven_audio /host_dir/outputs/speech.wav ^
  --source_image /host_dir/inputs/%PORTRAIT_FILE% ^
  --enhancer gfpgan ^
  --result_dir /host_dir/outputs

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] SadTalker video generation encountered an error.
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo [SUCCESS] Talking avatar generation completed!
echo Check the video files inside the 'outputs' directory:
echo   outputs\speech.wav
echo   outputs\*.mp4
echo ====================================================================
pause
