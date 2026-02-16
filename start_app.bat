@echo off
echo ========================================================
echo   正在啟動 AI 肺炎診斷系統...
echo   Created by Gemini & User (RTX 4060 Edition)
echo ========================================================

:: 1. 呼叫 Anaconda 環境 (請確認這是您的 Anaconda 安裝路徑)
:: 如果您的 Anaconda 裝在其他地方，請修改下面這行
call "C:\Users\%USERNAME%\anaconda3\Scripts\activate.bat" pneumonia-cls

:: 2. 切換到當前資料夾
cd /d "%~dp0"

:: 3. 啟動 Streamlit
echo 正在喚醒 AI 醫師...
streamlit run app_v2.py

pause