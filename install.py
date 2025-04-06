#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCP 终端客户端安装脚本

此脚本用于简化 MCP 终端客户端的安装过程，包括：
1. 安装依赖
2. 创建配置文件
3. 设置环境变量
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def print_step(step, message):
    """打印安装步骤"""
    print(f"\n[{step}] {message}")

def install_dependencies():
    """安装依赖"""
    print_step("1", "安装依赖")
    
    # 检查是否有 uv
    has_uv = False
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=False)
        has_uv = True
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    
    if has_uv:
        print("使用 uv 安装依赖（更快）...")
        result = subprocess.run(["uv", "pip", "install", "-r", "requirements.txt"], capture_output=True, text=True)
    else:
        print("使用 pip 安装依赖...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 依赖安装成功")
    else:
        print("❌ 依赖安装失败")
        print(result.stderr)
        return False
    
    return True

def setup_config_files():
    """设置配置文件"""
    print_step("2", "设置配置文件")
    
    # 检查并创建配置文件
    if not os.path.exists("config.py") and os.path.exists("config.py.template"):
        print("创建 config.py 文件...")
        shutil.copy("config.py.template", "config.py")
        print("✅ 已创建 config.py 文件，请编辑此文件并填入您的 API 密钥")
    else:
        print("config.py 文件已存在，跳过创建")
    
    if not os.path.exists("config.json") and os.path.exists("config.json.template"):
        print("创建 config.json 文件...")
        shutil.copy("config.json.template", "config.json")
        print("✅ 已创建 config.json 文件，请编辑此文件并填入您的服务器配置")
    else:
        print("config.json 文件已存在，跳过创建")
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("MCP 终端客户端安装程序")
    print("=" * 60)
    
    # 安装依赖
    if not install_dependencies():
        print("\n❌ 安装失败：无法安装依赖")
        return 1
    
    # 设置配置文件
    if not setup_config_files():
        print("\n❌ 安装失败：无法设置配置文件")
        return 1
    
    # 安装完成
    print("\n" + "=" * 60)
    print("✅ 安装完成！")
    print("=" * 60)
    print("\n使用以下命令启动 MCP 终端客户端：")
    print("  python mcp_terminal.py")
    print("\n请确保编辑 config.py 和 config.json 文件，填入您的 API 密钥和服务器配置。")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
