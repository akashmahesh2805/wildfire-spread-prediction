@echo off
REM Batch script to run the complete workflow in venv
call venv\Scripts\activate.bat
python notebooks\complete_workflow_example.py
pause

