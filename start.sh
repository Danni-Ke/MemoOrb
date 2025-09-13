#!/bin/bash
# 一键启动项目（macOS / Linux）
# 保存为 start.sh

set -e  # 遇到错误就停止执行

# 1. 激活 Python 3.11 虚拟环境
if [ -f ".venv/bin/activate" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
else
    echo "⚠️ 未找到虚拟环境 .venv，请先创建 Python 3.11 虚拟环境"
    exit 1
fi

# 2. 启动 Gradio 项目
echo "启动项目..."
python app.py

# 可选：项目运行结束后自动退出虚拟环境
deactivate
