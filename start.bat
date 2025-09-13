@echo off
REM 一键启动项目（Windows）
REM 保存为 start.bat

REM 1. 激活虚拟环境
IF EXIST ".venv\Scripts\activate.bat" (
    echo 激活虚拟环境...
    call .venv\Scripts\activate.bat
) ELSE (
    echo ⚠️ 未找到虚拟环境 .venv，请先创建 Python 3.11 虚拟环境
    exit /b 1
)

REM 2. 启动 Gradio 项目
echo 启动项目...
python app.py

REM 3. 暂停窗口，方便查看日志
pause
