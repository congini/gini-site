@echo off
cd /d C:\Users\cmang\Desktop\nfl_estat_dashboard_project

echo ================================================== >> data\live_sources\scheduled_refresh_log.txt
echo Scheduled refresh started at %date% %time% >> data\live_sources\scheduled_refresh_log.txt

echo Running live source refresh... >> data\live_sources\scheduled_refresh_log.txt
python update_live_sources.py >> data\live_sources\scheduled_refresh_log.txt 2>&1

IF %ERRORLEVEL% NEQ 0 (
    echo Local refresh failed; no generated files were staged or committed. >> data\live_sources\scheduled_refresh_log.txt
    exit /b 1
)

echo Local-only refresh completed. Git staging, commits, pulls, and pushes are disabled. >> data\live_sources\scheduled_refresh_log.txt

echo Scheduled refresh finished at %date% %time% >> data\live_sources\scheduled_refresh_log.txt
echo ================================================== >> data\live_sources\scheduled_refresh_log.txt
