@echo off
echo Starting Emotion Frontend (Streamlit)...
echo Ensure the BACKEND (app.py) is running in another window!
echo.
cd /d "%~dp0"
python -m streamlit run frontend/streamlit_app.py
pause
