@echo off
echo Starting Outlook PO Email Reader...
cd /d "%~dp0"
streamlit run app.py
pause
