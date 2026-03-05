@echo off
echo ========================================
echo ComfyUI-VideoDescription Installer
echo ========================================
echo.

cd /d "%~dp0..\.."

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: ComfyUI virtual environment not found!
    echo Please create .venv first.
    pause
    exit /b 1
)

echo Activating ComfyUI virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Installing base dependencies...
pip install transformers>=4.50.0 accelerate>=0.20.0 qwen-vl-utils segmentation-refinement guided-filter-pytorch

echo.
echo ========================================
echo Installation completed!
echo ========================================
echo.
echo Installed packages:
echo - transformers (upgraded to 4.50.0+)
echo - qwen-vl-utils (for Qwen3-VL model)
echo - segmentation-refinement (CascadePSP mask refinement)
echo - guided-filter-pytorch (DeepGuidedFilter mask refinement)
echo.
echo Your ComfyUI environment is preserved.
echo.
pause
