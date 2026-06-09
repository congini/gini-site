@echo off
cd /d C:\Users\cmangini\Desktop\nfl_estat_dashboard_project

python update_live_sources.py >> data\live_sources\scheduled_refresh_log.txt 2>&1