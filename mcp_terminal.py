#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCP终端 - 基于Rich和Typer的MCP客户端终端应用

这个应用封装了MCP客户端的功能，提供更美观、更易用的命令行界面。
它使用Rich库来格式化输出，使用Typer库来处理命令行参数。
支持从配置文件中读取MCP服务器配置并连接。
"""

import asyncio
import sys
import os
import json
import typer
import pathlib
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.syntax import Syntax
from rich import box
from contextlib import AsyncExitStack
import re

# MCP相关导入
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# 直接使用config.py中的配置
from config import api_key, api_base, model

# 默认配置文件路径
DEFAULT_CONFIG_PATH = os.path.expanduser(os.environ.get("MCP_CONFIG", "~/.mcp/config.json"))

# 如果环境变量没有设置默认配置路径，则使用当前目录下的config.json
if not os.path.exists(DEFAULT_CONFIG_PATH) and os.path.exists("config.json"):
    DEFAULT_CONFIG_PATH = "config.json"

# 单个MCP服务器连接的封装
class MCPServerConnection:
    def __init__(self, name: str, session: ClientSession, stdio, write, tools: List):
        self.name = name
        self.session = session
        self.stdio = stdio
        self.write = write
        self.tools = tools
        self.tool_names = [tool.name for tool in tools]
        self.tool_details = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            } for tool in tools
        ]
        
    async def call_tool(self, tool_name: str, tool_args: dict):
        """调用此服务器上的工具"""
        return await self.session.call_tool(tool_name, tool_args)
    
    async def list_tools(self):
        """获取最新的工具列表"""
        response = await self.session.list_tools()
        self.tools = response.tools
        self.tool_names = [tool.name for tool in self.tools]
        return response
    
    def get_tool_descriptions(self):
        """获取工具描述信息"""
        return "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
    
    def has_tool(self, tool_name: str) -> bool:
        """检查服务器是否有指定的工具"""
        return tool_name in self.tool_names

# 读取配置文件
def read_config(config_path: str) -> Dict[str, Any]:
    """读取配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        return config_data
    except Exception as e:
        console.print(f"[bold red]读取配置文件失败:[/bold red] {str(e)}")
        return {"mcpServers": {}}

# 支持多服务器连接的MCP客户端
class MultiMCPClient:
    def __init__(self, max_history=10, config_path=None):
        # 初始化会话和客户端对象
        self.servers: Dict[str, MCPServerConnection] = {}
        self.exit_stack = AsyncExitStack()
        self.pending_tasks: List[asyncio.Task] = []
        self.conversation_history = []
        self.max_history = max_history
        self.current_server = None
        self.auto_select = True
        self.direct_mode = False  # 直接对话模式开关
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config_data = {}
        
        # 如果配置文件存在，读取配置
        if os.path.exists(self.config_path):
            self.config_data = read_config(self.config_path)

        # 初始化OpenAI客户端
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base
            )
            
            # 使用config.py中定义的模型
            self.model = model
        except Exception as e:
            console.print(f"[bold red]初始化OpenAI客户端失败:[/bold red] {str(e)}")
            # 使用环境变量中的值作为后备
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", api_key),
                base_url=os.environ.get("OPENAI_API_BASE", api_base)
            )
            self.model = os.environ.get("OPENAI_API_MODEL", model)
        
    async def connect_from_config(self, server_id: str) -> str:
        """从配置文件中连接服务器"""
        if not self.config_data or "mcpServers" not in self.config_data:
            raise ValueError("配置文件中没有mcpServers部分")
            
        if server_id not in self.config_data["mcpServers"]:
            raise ValueError(f"配置文件中没有服务器 '{server_id}'")
            
        server_config = self.config_data["mcpServers"][server_id]
        
        # 检查是否禁用
        if server_config.get("disabled", False):
            raise ValueError(f"服务器 '{server_id}' 已禁用")
            
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})
        
        if not command:
            raise ValueError(f"服务器 '{server_id}' 配置缺少command字段")
            
        console.print(f"[dim]从配置文件连接服务器: {server_id}[/dim]")
        console.print(f"[dim]执行命令: {command} {' '.join(args)}[/dim]")
        
        if env:
            env_str = ', '.join([f"{k}={v}" for k, v in env.items()])
            console.print(f"[dim]环境变量: {env_str}[/dim]")
        
        # 创建StdioServerParameters
        # 合并环境变量与当前进程的环境变量
        merged_env = os.environ.copy()
        if env:
            for key, value in env.items():
                merged_env[key] = value
                
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=merged_env
        )
        
        # 创建新的上下文来管理每个服务器的连接
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        
        # 创建会话
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        
        # 获取工具列表
        tools_response = await session.list_tools()
        tools = tools_response.tools
        
        # 创建服务器连接
        server_conn = MCPServerConnection(server_id, session, stdio, write, tools)
        
        # 存储服务器连接
        self.servers[server_id] = server_conn
        
        # 如果这是第一个服务器，则设置为当前服务器
        if self.current_server is None:
            self.current_server = server_id
        
        return server_id
        
    async def connect_to_server(self, server_identifier: str, server_name: str = None) -> str:
        """连接到MCP服务器，支持npm包形式和脚本文件"""
        # 如果未提供服务器名称，使用标识符作为名称
        if not server_name:
            # 为多词命令创建一个简单的名称
            if ' ' in server_identifier:
                parts = server_identifier.split()
                if parts[0] in ["uvx", "npx"]:
                    # 如果是 uvx mcp-server-fetch 这样的多词命令
                    # 使用最后一个部分作为服务器名称
                    server_name = parts[-1].split('/')[-1]
                else:
                    server_name = server_identifier
            else:
                server_name = server_identifier
        
        print(f"尝试连接到服务器: {server_identifier}, 名称: {server_name}")
        
        # 检查是否是npx或uvx命令格式
        args = server_identifier.split()
        if args and args[0] in ['npx', 'uvx']:
            # 直接拆分命令和参数
            command = args[0]  # uvx 或 npx
            command_args = args[1:] if len(args) > 1 else []
            
            print(f"执行命令: {command} {' '.join(command_args)}")
            
            server_params = StdioServerParameters(
                command=command,
                args=command_args,
                env=None
            )
        else:
            # 原有逻辑：处理.py或.js文件
            is_python = server_identifier.endswith('.py')
            is_js = server_identifier.endswith('.js')
            
            if not (is_python or is_js):
                raise ValueError("服务器必须是.py或.js文件，或以'npx/uvx'开头的命令")
                
            command = "python" if is_python else "node"
            args = [server_identifier]
            
            print(f"执行命令: {command} {args[0]}")
            
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=None
            )

        # 创建新的上下文来管理每个服务器的连接
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        
        # 创建会话
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        
        # 获取工具列表
        tools_response = await session.list_tools()
        tools = tools_response.tools
        
        # 创建服务器连接
        server_conn = MCPServerConnection(server_name, session, stdio, write, tools)
        
        # 存储服务器连接
        self.servers[server_name] = server_conn
        
        # 如果这是第一个服务器，则设置为当前服务器
        if self.current_server is None:
            self.current_server = server_name
        
        return server_name

    async def select_best_server(self, query: str) -> str:
        """根据查询内容选择最合适的服务器"""
        if not self.servers:
            raise ValueError("没有连接的服务器")
        
        # 如果只有一个服务器，直接返回
        if len(self.servers) == 1:
            return next(iter(self.servers.keys()))
        
        # 如果已经指定了当前服务器且不是自动选择模式，则使用当前服务器
        if self.current_server and not self.auto_select:
            return self.current_server
        
        # 使用OpenAI API进行服务器选择
        try:
            # 准备服务器信息
            server_info = ""
            for name, server in self.servers.items():
                server_info += f"\n{name}: {server.get_tool_descriptions()}\n"
            
            # 准备提示
            prompt = f"""根据用户的查询，选择最合适的MCP服务器。

用户查询: {query}

可用的服务器及其工具:
{server_info}

请只返回最合适的服务器名称，不要添加任何其他文字。"""
            
            # 发送请求
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个服务器选择助手，根据用户查询选择最合适的MCP服务器。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=50
            )
            
            # 解析响应
            server_name = response.choices[0].message.content.strip()
            
            # 检查服务器是否存在
            if server_name in self.servers:
                return server_name
            else:
                # 如果返回的服务器不存在，使用第一个服务器
                return next(iter(self.servers.keys()))
                
        except Exception as e:
            print(f"服务器选择失败: {str(e)}")
            # 如果出错，使用当前服务器或第一个服务器
            return self.current_server or next(iter(self.servers.keys()))

    async def direct_chat(self, query: str):
        """直接与AI对话，不使用任何MCP服务器或工具"""
        # 添加到对话历史
        self.conversation_history.append({"role": "user", "content": query})
        
        # 如果历史记录超过最大值，删除最早的记录
        while len(self.conversation_history) > self.max_history * 2:
            self.conversation_history.pop(0)
        
        try:
            # 发送请求
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=2000
            )
            
            # 获取响应文本
            response_text = response.choices[0].message.content
            
            # 添加到对话历史
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response_text
        except Exception as e:
            return f"错误: {str(e)}"

    async def process_query(self, query: str):
        """使用OpenAI兼容API和可用工具处理查询"""
        if not self.servers:
            raise ValueError("没有连接的服务器")
        
        # 添加到对话历史
        self.conversation_history.append({"role": "user", "content": query})
        
        # 如果历史记录超过最大值，删除最早的记录
        while len(self.conversation_history) > self.max_history * 2:
            self.conversation_history.pop(0)
        
        # 选择服务器
        if self.auto_select:
            server_name = await self.select_best_server(query)
        else:
            server_name = self.current_server or next(iter(self.servers.keys()))
        
        server = self.servers[server_name]
        
        # 刷新工具列表
        await server.list_tools()
        
        try:
            # 准备工具信息
            tools = []
            for tool in server.tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            
            # 发送初始请求
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=tools,
                tool_choice="auto",  # 允许模型自行选择是否使用工具
                temperature=0.7,
                max_tokens=2000
            )
            
            # 构建最终响应文本
            final_text = []
            final_text.append(f"[使用服务器: {server_name}]")
            
            # 获取响应
            message = response.choices[0].message
            
            # 如果有普通内容，添加到最终文本
            if message.content:
                final_text.append(message.content)
                self.conversation_history.append({"role": "assistant", "content": message.content})
            
            # 如果有工具调用
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # 检查服务器是否有该工具
                    if server.has_tool(function_name):
                        # 打印调试信息
                        final_text.append(f"[调用工具 {function_name}，参数 {function_args}]")
                        
                        # 调用工具
                        result = await server.call_tool(function_name, function_args)
                        
                        # 返回工具调用结果
                        tool_result = str(result.content) if hasattr(result, 'content') else str(result)
                        
                        # 创建新的消息列表，包含历史记录和当前工具调用
                        messages = self.conversation_history + [
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {
                                            "name": function_name,
                                            "arguments": tool_call.function.arguments
                                        }
                                    }
                                ]
                            }
                        ]
                        
                        # 添加工具响应
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result  # 使用字符串
                        })
                        
                        # 获取下一个响应
                        try:
                            second_response = self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                tools=tools,
                                tool_choice="auto",
                                temperature=0.7,
                                max_tokens=2000
                            )
                            
                            next_message = second_response.choices[0].message
                            if next_message.content:
                                final_text.append(next_message.content)
                                # 更新对话历史
                                self.conversation_history = messages + [
                                    {"role": "assistant", "content": next_message.content}
                                ]
                            
                            # 处理可能的级联工具调用
                            if hasattr(next_message, 'tool_calls') and next_message.tool_calls:
                                final_text.append("[检测到级联工具调用，但为简化处理，不再进一步处理]")
                        except Exception as e:
                            error_msg = f"获取最终响应时出错: {str(e)}"
                            final_text.append(error_msg)
                            # 更新对话历史
                            self.conversation_history = messages + [
                                {"role": "assistant", "content": error_msg}
                            ]
                    else:
                        # 如果服务器没有该工具，添加错误信息
                        error_msg = f"[错误: 服务器 {server_name} 没有工具 {function_name}]"
                        final_text.append(error_msg)
                        # 更新对话历史
                        self.conversation_history.append({"role": "assistant", "content": error_msg})
            
            # 返回最终文本
            return "\n".join(final_text)
        except Exception as e:
            error_message = f"错误: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_message})
            return error_message
    
    async def cleanup(self):
        """清理资源"""
        try:
            # 先关闭所有服务器连接
            for server_name, server in list(self.servers.items()):
                try:
                    # 关闭单个服务器连接
                    if hasattr(server.session, 'close') and callable(server.session.close):
                        await server.session.close()
                    if hasattr(server.stdio, 'close') and callable(server.stdio.close):
                        server.stdio.close()
                    if hasattr(server.write, 'close') and callable(server.write.close):
                        server.write.close()
                except Exception as e:
                    # 捕获单个服务器关闭异常
                    pass
            
            # 清空服务器列表
            self.servers.clear()
            
            # 取消所有待处理的任务
            for task in self.pending_tasks:
                if not task.done():
                    task.cancel()
            
            # 等待所有任务完成
            if self.pending_tasks:
                try:
                    await asyncio.gather(*self.pending_tasks, return_exceptions=True)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            
            # 清空任务列表
            self.pending_tasks.clear()
            
            # 最后关闭退出栈
            await self.exit_stack.aclose()
        except Exception as e:
            # 捕获但不重新抛出异常
            pass

# 创建Rich控制台和Typer应用
console = Console()
app = typer.Typer(help="MCP终端 - 与MCP服务器交互的终端应用")

# 全局客户端实例
client = None


@app.command()
def config(
    list_servers: bool = typer.Option(False, "--list", "-l", help="列出配置文件中的服务器"),
    config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="配置文件路径")
):
    """管理MCP配置文件"""
    if not os.path.exists(config_path):
        console.print(f"[bold red]错误:[/bold red] 配置文件 '{config_path}' 不存在")
        return
        
    config_data = read_config(config_path)
    
    if "mcpServers" not in config_data:
        console.print("[bold yellow]警告:[/bold yellow] 配置文件中没有mcpServers部分")
        return
        
    if list_servers:
        servers = config_data["mcpServers"]
        
        if not servers:
            console.print("[yellow]配置文件中没有定义服务器[/yellow]")
            return
        
        # 直接打印服务器信息，不使用表格
        console.print(f"[bold]配置文件中的服务器 ({config_path}):[/bold]")
        
        for server_id, server_config in servers.items():
            command = server_config.get("command", "")
            args = " ".join(server_config.get("args", []))
            env_dict = server_config.get("env", {})
            env = ", ".join([f"{k}={v[:5]}..." if len(v) > 8 else f"{k}={v}" for k, v in env_dict.items()])
            status = "[red]禁用[/red]" if server_config.get("disabled", False) else "[green]启用[/green]"
            
            console.print(f"[cyan]服务器ID:[/cyan] {server_id}")
            console.print(f"  [green]命令:[/green] {command}")
            console.print(f"  [blue]参数:[/blue] {args}")
            console.print(f"  [yellow]环境变量:[/yellow] {env}")
            console.print(f"  [magenta]状态:[/magenta] {status}")
            console.print("")
            
        console.print("\n使用 [bold]connect --from-config <服务器ID>[/bold] 来连接配置文件中的服务器")

@app.command()
def connect(
    server: Optional[str] = typer.Argument(None, help="服务器标识符，可以是脚本路径或npm/uvx命令"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="服务器名称，默认使用标识符"),
    from_config: Optional[str] = typer.Option(None, "--from-config", "-f", help="从配置文件中连接服务器"),
    config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="配置文件路径"),
    keep_alive: bool = typer.Option(False, "--keep-alive", "-k", help="连接后保持服务器运行，不自动清理资源")
):
    """连接到MCP服务器"""
    # 如果使用从配置文件连接的选项
    if from_config:
        console.print(Panel(
            f"[bold]从配置文件连接到MCP服务器[/bold]: {from_config}",
            border_style="blue",
            expand=False
        ))
        
        async def connect_from_config_async():
            global client
            if client is None:
                client = MultiMCPClient(config_path=config_path)
            else:
                # 如果已经创建了客户端，确保使用正确的配置文件
                if client.config_path != config_path:
                    client.config_path = config_path
                    client.config_data = read_config(config_path)
            
            try:
                # 使用从配置文件连接的方法
                server_name = await client.connect_from_config(from_config)
                
                # 显示成功消息
                console.print(f"[bold green]成功连接到服务器:[/bold green] {server_name}")
                
                # 显示可用工具
                if server_name in client.servers:
                    tools = client.servers[server_name].tools
                    
                    if tools:
                        table = Table(title="可用工具", box=box.ROUNDED)
                        table.add_column("工具名称", style="cyan")
                        table.add_column("描述", style="green")
                        
                        for tool in tools:
                            table.add_row(tool.name, tool.description)
                        
                        console.print(table)
                    else:
                        console.print("[yellow]此服务器没有提供工具[/yellow]")
                    
                    # 如果不是在交互模式下，将服务器保存到客户端实例中
                    if len(sys.argv) > 1:  # 如果是命令行模式
                        # 在命令行模式下连接完成后，将服务器保存到客户端实例中并退出
                        # 不进行清理，因为用户可能会在之后的命令中使用这个服务器
                        pass
            except Exception as e:
                console.print(f"[bold red]连接失败:[/bold red] {str(e)}")
        
        # 执行异步连接
        asyncio.run(connect_from_config_async())
        return
    
    # 如果没有使用 --from-config 选项，则需要 server 参数
    if server is None:
        console.print("[bold red]错误:[/bold red] 需要指定服务器ID")
        return
    
    # 输出调试信息
    console.print(f"[dim]收到server参数: '{server}'[/dim]")

    # 初始化server_args
    server_args = []
    
    # 命令行模式下的多参数命令处理
    # 处理环境变量（如 TAVILY_API_KEY=xxx uvx mcp-server-fetch）
    env_vars = {}
    env_var_pattern = re.compile(r'^([A-Za-z0-9_]+)=([^\s]+)')  # 匹配环境变量赋值
    
    if ' ' in server:
        parts = server.split()
        clean_parts = []
        
        for part in parts:
            env_match = env_var_pattern.match(part)
            if env_match:
                env_vars[env_match.group(1)] = env_match.group(2)
            else:
                clean_parts.append(part)
        
        # 如果有环境变量，重构server为不包含环境变量的命令
        if env_vars:
            server = " ".join(clean_parts)
            console.print(f"[dim]检测到环境变量: {list(env_vars.keys())}[/dim]")
    
    # 特殊处理MCP服务器命令
    if ' ' in server:
        # 原始命令中如果有空格，则是一个包含空格的完整命令，例如 "uvx mcp-server-fetch"
        # 仅记录检测到的复合命令
        console.print(f"[dim]检测到复合命令: '{server}'[/dim]")
        
    # 将检测到的环境变量添加到环境中
    os.environ.update(env_vars)
    
    # 原有的连接逻辑
    console.print(Panel(
        f"[bold]连接到MCP服务器[/bold]: {server} {' '.join(server_args) if server_args else ''}",
        border_style="blue",
        expand=False
    ))
    
    async def connect_async():
        global client
        if client is None:
            client = MultiMCPClient(config_path=config_path)
        
        try:
            # 使用内置的connect_to_server方法
            server_name = await client.connect_to_server(server, name)
            console.print(f"[bold green]成功连接到服务器:[/bold green] {server_name}")
            
            # 显示可用工具
            if server_name in client.servers:
                tools = client.servers[server_name].tools
                
                if tools:
                    table = Table(title="可用工具", box=box.ROUNDED)
                    table.add_column("工具名称", style="cyan")
                    table.add_column("描述", style="green")
                    
                    for tool in tools:
                        table.add_row(tool.name, tool.description)
                    
                    console.print(table)
                else:
                    console.print("[yellow]此服务器没有提供工具[/yellow]")
            
            return True
        except Exception as e:
            console.print(f"[bold red]连接失败:[/bold red] {str(e)}")
            # 只有在连接失败时才清理资源
            await client.cleanup()
            return False
    
    try:
        asyncio.run(connect_async())
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断，程序退出[/yellow]")


@app.command()
def chat(
    auto_select: bool = typer.Option(True, "--auto-select/--no-auto-select", help="是否自动选择最合适的服务器"),
    direct_mode: bool = typer.Option(False, "--direct", help="直接对话模式，不使用MCP服务器")
):
    """启动交互式聊天会话"""
    console.print(Panel(
        "[bold blue]MCP终端[/bold blue] - [italic]交互式聊天模式[/italic]",
        subtitle="输入'quit'退出，'help'查看帮助",
        border_style="blue",
        expand=False
    ))
    
    # 显示可用命令
    help_panel = Panel(
        "可用命令:\n"
        "- [bold]!quit[/bold]: 退出程序\n"
        "- [bold]!list[/bold] 或 [bold]!servers[/bold]: 显示已连接的服务器\n"
        "- [bold]!help[/bold]: 显示帮助信息\n"
        "- [bold]!auto[/bold]: 启用服务器自动选择\n"
        "- [bold]!direct[/bold]: 切换到直接对话模式\n"
        "- [bold]!mcp[/bold]: 切换到MCP服务器模式\n"
        "- [bold]!use <服务器名>[/bold]: 切换到指定服务器\n"
        "- [bold]!disconnect <服务器名>[/bold]: 断开指定服务器\n"
        "- [bold]!connect <服务器标识符> [--name <服务器名称>][/bold]: 直接连接服务器\n"
        "  例如: [bold]!connect uvx mcp-server-fetch[/bold] 或 [bold]!connect npx -y @modelcontextprotocol/server-openai[/bold]\n"
        "  也可设置环境变量: [bold]!connect TAVILY_API_KEY=your-key uvx mcp-server-fetch[/bold]\n"
        "- [bold]!connect --from-config/-f <服务器ID> [--config/-c <配置文件路径>][/bold]: 从配置文件连接服务器\n"
        "- [bold]!history [--count/-c <数量>] [--set/-s <最大数量>][/bold]: 管理对话历史记录\n"
        "- [bold]!clear[/bold]: 清除对话历史记录\n"
        "\n注意: \n"
        "1. 所有命令都以感叹号(!)开头，其他输入将被视为发送给模型的对话\n"
        "2. 可以使用 config.json 文件来配置服务器，也可以直接通过命令行连接",
        title="可用命令列表",
        border_style="green"
    )
    console.print(help_panel)
    
    async def chat_async():
        global client
        if client is None:
            client = MultiMCPClient()
            console.print("[yellow]警告: 没有连接的服务器，请使用'connect'命令连接服务器或使用直接对话模式[/yellow]")
        
        # 设置模式
        client.auto_select = auto_select
        client.direct_mode = direct_mode
        
        if auto_select:
            console.print("[dim]已启用服务器自动选择[/dim]")
        if direct_mode:
            console.print("[dim]已启用直接对话模式[/dim]")
        
        # 显示当前连接的服务器
        if client.servers:
            servers_list = ", ".join(client.servers.keys())
            console.print(f"[dim]已连接的服务器: {servers_list}[/dim]")
            if client.current_server:
                console.print(f"[dim]当前服务器: {client.current_server}[/dim]")
        
        # 启动聊天循环
        while True:
            try:
                query = console.input("\n[bold blue]> [/bold blue]")
                
                # 处理特殊命令
                # 检查是否是以感叹号开头的命令
                if query.startswith('!'):
                    cmd = query[1:].lower()  # 去掉感叹号，转为小写
                    
                    # 退出命令
                    if cmd == 'quit':
                        break
                    
                    # 服务器列表命令
                    elif cmd == 'servers' or cmd == 'list':
                        if client.servers:
                            table = Table(title="已连接的服务器", box=box.ROUNDED)
                            table.add_column("名称", style="cyan")
                            table.add_column("工具数量", style="green")
                            table.add_column("当前", style="yellow")
                            
                            for name, server in client.servers.items():
                                is_current = "✓" if name == client.current_server else ""
                                table.add_row(name, str(len(server.tools)), is_current)
                            
                            console.print(table)
                        else:
                            console.print("[yellow]没有连接的服务器[/yellow]")
                        continue
                    
                    # 帮助命令
                    elif cmd == 'help':
                        help_panel = Panel(
                            "可用命令:\n"
                            "- [bold]!quit[/bold]: 退出程序\n"
                            "- [bold]!list[/bold] 或 [bold]!servers[/bold]: 显示已连接的服务器\n"
                            "- [bold]!help[/bold]: 显示帮助信息\n"
                            "- [bold]!auto[/bold]: 启用服务器自动选择\n"
                            "- [bold]!direct[/bold]: 切换到直接对话模式\n"
                            "- [bold]!mcp[/bold]: 切换到MCP服务器模式\n"
                            "- [bold]!use <服务器名>[/bold]: 切换到指定服务器\n"
                            "- [bold]!disconnect <服务器名>[/bold]: 断开指定服务器\n"
                            "- [bold]!connect <服务器标识符> [--name <服务器名称>][/bold]: 直接连接服务器\n"
                            "  例如: [bold]!connect uvx mcp-server-fetch[/bold] 或 [bold]!connect npx -y @modelcontextprotocol/server-openai[/bold]\n"
                            "  也可设置环境变量: [bold]!connect TAVILY_API_KEY=your-key uvx mcp-server-fetch[/bold]\n"
                            "- [bold]!connect --from-config/-f <服务器ID> [--config/-c <配置文件路径>][/bold]: 从配置文件连接服务器\n"
                            "- [bold]!history [--count/-c <数量>] [--set/-s <最大数量>][/bold]: 管理对话历史记录\n"
                            "- [bold]!clear[/bold]: 清除对话历史记录\n"
                            "\n注意: \n"
                            "1. 所有命令都以感叹号(!)开头，其他输入将被视为发送给模型的对话\n"
                            "2. 可以使用 config.json 文件来配置服务器，也可以直接通过命令行连接",
                            title="帮助",
                            border_style="green"
                        )
                        console.print(help_panel)
                        continue
                    
                    # 自动选择命令
                    elif cmd == 'auto':
                        client.auto_select = True
                        client.direct_mode = False
                        console.print("[dim]已启用服务器自动选择[/dim]")
                        continue
                    
                    # 直接对话模式命令
                    elif cmd == 'direct':
                        client.direct_mode = True
                        console.print("[dim]已切换到直接对话模式[/dim]")
                        continue
                    
                    # MCP服务器模式命令
                    elif cmd == 'mcp':
                        client.direct_mode = False
                        console.print(f"[dim]已切换到MCP服务器模式 {'(自动选择)' if client.auto_select else ''}[/dim]")
                        continue
                    
                    # 清除历史记录命令
                    elif cmd == 'clear':
                        client.conversation_history = []
                        console.print("[bold green]已清除对话历史记录[/bold green]")
                        continue
                    
                    # 历史记录命令
                    elif cmd.startswith('history'):
                        # 解析参数
                        parts = cmd.split()
                        count = None
                        max_history = None
                        
                        # 处理参数
                        i = 1  # 跳过'history'
                        while i < len(parts):
                            if parts[i] in ['--count', '-c'] and i + 1 < len(parts):
                                try:
                                    count = int(parts[i+1])
                                    i += 2
                                except ValueError:
                                    console.print(f"[bold red]错误:[/bold red] 无效的数量: {parts[i+1]}")
                                    i += 2
                            elif parts[i] in ['--set', '-s'] and i + 1 < len(parts):
                                try:
                                    max_history = int(parts[i+1])
                                    i += 2
                                except ValueError:
                                    console.print(f"[bold red]错误:[/bold red] 无效的最大数量: {parts[i+1]}")
                                    i += 2
                            else:
                                i += 1
                        
                        # 设置最大历史记录数量
                        if max_history is not None:
                            client.max_history = max_history
                            console.print(f"[bold green]已设置最大历史记录数量为 {max_history}[/bold green]")
                        
                        # 显示历史记录
                        if count is None:
                            count = len(client.conversation_history)
                        
                        # 显示历史记录信息
                        console.print(f"当前历史记录数量: {len(client.conversation_history)}")
                        console.print(f"最大历史记录数量: {client.max_history}")
                        
                        # 显示指定数量的历史记录
                        history_to_show = client.conversation_history[-count:] if count > 0 else []
                        
                        if history_to_show:
                            for i, (user_msg, assistant_msg) in enumerate(history_to_show):
                                console.print(f"\n[bold blue]User ({i//2+1}):[/bold blue]")
                                console.print(user_msg)
                                console.print(f"\n[bold green]Assistant:[/bold green]")
                                console.print(Markdown(assistant_msg))
                                console.print("---")

                    # 使用指定服务器命令
                    elif cmd.startswith('use '):
                        server_name = query[5:].strip()  # 去掉'!use '，长度为5
                        if server_name in client.servers:
                            client.current_server = server_name
                            client.auto_select = False
                            client.direct_mode = False
                            console.print(f"[dim]已切换到服务器: {client.current_server}[/dim]")
                        else:
                            console.print(f"[bold red]错误:[/bold red] 未找到服务器 '{server_name}'")
                        continue
                    
                    # 断开指定服务器命令
                    elif cmd.startswith('disconnect '):
                        server_name = query[12:].strip()  # 去掉'!disconnect '，长度为12
                        if server_name in client.servers:
                            del client.servers[server_name]
                            if client.current_server == server_name:
                                client.current_server = next(iter(client.servers.keys())) if client.servers else None
                                client.auto_select = True
                            console.print(f"[dim]已断开服务器: {server_name}[/dim]")
                        else:
                            console.print(f"[bold red]错误:[/bold red] 未找到服务器 '{server_name}'")
                        continue
                    
                    # 连接到新服务器命令
                    elif cmd.startswith('connect '):
                        # 获取connect之后的全部内容
                        full_cmd = query[9:].strip()  # 去掉'!connect '，长度为9
                        
                        # 检查是否使用从配置文件连接
                        if "--from-config" in full_cmd or "-f" in full_cmd:
                            # 解析参数
                            parts = full_cmd.split()
                            server_id = None
                            config_path = DEFAULT_CONFIG_PATH
                            
                            # 处理参数
                            i = 0
                            while i < len(parts):
                                if parts[i] in ["--from-config", "-f"] and i + 1 < len(parts):
                                    server_id = parts[i+1]
                                    i += 2
                                elif parts[i] in ["--config", "-c"] and i + 1 < len(parts):
                                    config_path = parts[i+1]
                                    i += 2
                                else:
                                    i += 1
                            
                            if not server_id:
                                console.print("[bold red]错误:[/bold red] 需要指定服务器ID")
                                continue
                            
                            try:
                                # 确保客户端已初始化
                                if client.config_path != config_path:
                                    client.config_path = config_path
                                    client.config_data = read_config(config_path)
                                
                                # 从配置文件连接服务器
                                with Progress(
                                    SpinnerColumn(),
                                    TextColumn("[bold green]正在连接到服务器..."),
                                    transient=True,
                                ) as progress:
                                    progress.add_task("connecting", total=None)
                                    new_server_name = await client.connect_from_config(server_id)
                                
                                console.print(f"[bold green]成功连接到服务器:[/bold green] {new_server_name}")
                                
                                # 显示可用工具
                                if new_server_name in client.servers:
                                    tools = client.servers[new_server_name].tools
                                    
                                    if tools:
                                        table = Table(title="可用工具", box=box.ROUNDED)
                                        table.add_column("工具名称", style="cyan")
                                        table.add_column("描述", style="green")
                                        
                                        for tool in tools:
                                            table.add_row(tool.name, tool.description)
                                        
                                        console.print(table)
                                    else:
                                        console.print("[yellow]此服务器没有提供工具[/yellow]")
                                    
                                    # 自动切换到新连接的服务器
                                    client.current_server = new_server_name
                                    client.auto_select = False
                                    client.direct_mode = False
                                    console.print(f"[dim]已切换到服务器: {new_server_name}[/dim]")
                            except Exception as e:
                                console.print(f"[bold red]连接失败:[/bold red] {str(e)}")
                            
                            continue
                        
                        # 这是在聊天模式下使用 !connect 命令
                        console.print(f"[dim]处理聊天模式的connect命令: '{full_cmd}'[/dim]")
                        
                        # 检查是否输入了服务器ID
                        if not full_cmd.strip():
                            console.print("[bold red]错误:[/bold red] 需要指定服务器ID")
                            continue
                        
                        # 处理引号
                        if full_cmd.startswith('"') and full_cmd.endswith('"'):
                            full_cmd = full_cmd[1:-1]
                        elif full_cmd.startswith("'") and full_cmd.endswith("'"):
                            full_cmd = full_cmd[1:-1]
                        
                        # 默认服务器名称为 None
                        server_name = None
                        
                        # 特殊处理常见的MCP服务器命令格式，如 uvx mcp-server-fetch 或 npx package-name
                        server_id = full_cmd
                        
                        # 检查是否有 --name 参数
                        name_match = re.search(r'\s--name\s+([\w-]+)', full_cmd)
                        if name_match:
                            server_name = name_match.group(1)
                            # 从 server_id 中移除 --name 及其参数
                            server_id = full_cmd[:name_match.start()].strip()
                        
                        # 处理包含环境变量的情况 (如 TAVILY_API_KEY=xxx uvx mcp-server-fetch)
                        env_vars = {}
                        env_var_pattern = re.compile(r'^([A-Za-z0-9_]+)=([^\s]+)')  # 匹配环境变量赋值
                        parts = server_id.split()
                        clean_parts = []
                        
                        for part in parts:
                            env_match = env_var_pattern.match(part)
                            if env_match:
                                env_vars[env_match.group(1)] = env_match.group(2)
                            else:
                                clean_parts.append(part)
                        
                        # 如果有环境变量，重构server_id为不包含环境变量的命令
                        if env_vars:
                            server_id = " ".join(clean_parts)
                            console.print(f"[dim]检测到环境变量: {list(env_vars.keys())}[/dim]")
                        
                        # 根据命令形式自动生成服务器名称
                        if not server_name and ' ' in server_id:
                            parts = server_id.split()
                            if parts[0] in ["uvx", "npx"] and len(parts) > 1:
                                # 从最后一个参数生成服务器名称
                                server_name = parts[-1].split('/')[-1]
                                # 移除版本号 (如 @0.1.4)
                                server_name = re.sub(r'@[0-9.]+$', '', server_name)
                        
                        console.print(f"[dim]将连接到: server_id='{server_id}', server_name='{server_name}'[/dim]")

                        if not server_id:
                            console.print("[bold red]错误:[/bold red] 无法确定服务器标识符")
                            continue

                        console.print(f"[dim]正在连接到服务器: {server_id}[/dim]")
                        
                        try:
                            # 添加任何检测到的环境变量
                            os.environ.update(env_vars)
                            
                            # 连接到服务器
                            new_server_name = await client.connect_to_server(server_id, server_name)
                            console.print(f"[bold green]成功连接到服务器:[/bold green] {new_server_name}")
                            
                            # 显示可用工具
                            if new_server_name in client.servers:
                                tools = client.servers[new_server_name].tools
                                
                                if tools:
                                    table = Table(title="可用工具", box=box.ROUNDED)
                                    table.add_column("工具名称", style="cyan")
                                    table.add_column("描述", style="green")
                                    
                                    for tool in tools:
                                        table.add_row(tool.name, tool.description)
                                    
                                    console.print(table)
                                else:
                                    console.print("[yellow]此服务器没有提供工具[/yellow]")
                                
                                # 自动切换到新连接的服务器
                                client.current_server = new_server_name
                                client.auto_select = False
                                client.direct_mode = False
                                console.print(f"[dim]已切换到服务器: {new_server_name}[/dim]")
                        except Exception as e:
                            console.print(f"[bold red]连接失败:[/bold red] {str(e)}")
                        
                        continue
                
                # 处理普通查询
                if not query.strip():
                    continue
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]处理中...[/bold blue]"),
                    transient=True
                ) as progress:
                    task = progress.add_task("处理中", total=None)
                    
                    try:
                        if client.direct_mode:
                            response = await client.direct_chat(query)
                        else:
                            if not client.servers:
                                console.print("[bold red]错误:[/bold red] 没有连接的服务器，请先连接服务器或切换到直接对话模式")
                                continue
                            response = await client.process_query(query)
                        
                        # 显示响应
                        md = Markdown(response)
                        console.print(md)
                    except Exception as e:
                        console.print(f"[bold red]错误:[/bold red] {str(e)}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[bold red]未预期错误:[/bold red] {str(e)}")
    
    try:
        asyncio.run(chat_async())
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断，程序退出[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]程序异常:[/bold red] {str(e)}")


@app.command()
def query(
    question: str = typer.Argument(..., help="要询问的问题"),
    server: Optional[str] = typer.Option(None, "--server", "-s", help="使用指定的服务器，不指定则自动选择"),
    direct: bool = typer.Option(False, "--direct", "-d", help="直接对话模式，不使用MCP服务器")
):
    """发送单个查询并显示结果"""
    console.print(Panel(
        f"[bold]查询[/bold]: {question}",
        border_style="blue",
        expand=False
    ))
    
    async def query_async():
        global client
        if client is None:
            client = MultiMCPClient()
        
        # 设置模式
        client.direct_mode = direct
        
        if server:
            if server in client.servers:
                client.current_server = server
                client.auto_select = False
                console.print(f"[dim]使用服务器: {server}[/dim]")
            else:
                # 尝试连接
                try:
                    await client.connect_to_server(server)
                    client.current_server = server
                    client.auto_select = False
                    console.print(f"[dim]已连接并使用服务器: {server}[/dim]")
                except Exception as e:
                    console.print(f"[bold red]连接服务器失败:[/bold red] {str(e)}")
                    return
        
        # 处理查询
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]处理中...[/bold blue]"),
            transient=True
        ) as progress:
            task = progress.add_task("处理中", total=None)
            
            try:
                if direct:
                    response = await client.direct_chat(question)
                else:
                    if not client.servers:
                        console.print("[bold red]错误:[/bold red] 没有连接的服务器，请先连接服务器或使用--direct选项")
                        return
                    response = await client.process_query(question)
                
                # 显示响应
                md = Markdown(response)
                console.print(md)
            except Exception as e:
                console.print(f"[bold red]错误:[/bold red] {str(e)}")
        
        # 不清理资源，让用户可以继续使用已连接的服务器
    
    try:
        asyncio.run(query_async())
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断，程序退出[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]程序异常:[/bold red] {str(e)}")


@app.command()
def tools(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="显示指定服务器的工具，不指定则显示所有服务器的工具")
):
    """显示可用的MCP工具"""
    
    async def tools_async():
        global client
        if client is None:
            client = MultiMCPClient()
            console.print("[yellow]警告: 没有连接的服务器[/yellow]")
            return
        
        if not client.servers:
            console.print("[bold red]错误:[/bold red] 没有连接的服务器")
            return
        
        if server:
            # 显示指定服务器的工具
            if server in client.servers:
                tools = client.servers[server].tools
                console.print(Panel(f"[bold]{server}[/bold] 的可用工具", border_style="blue"))
                
                if tools:
                    table = Table(box=box.ROUNDED)
                    table.add_column("工具名称", style="cyan")
                    table.add_column("描述", style="green")
                    
                    for tool in tools:
                        table.add_row(tool.name, tool.description)
                    
                    console.print(table)
                else:
                    console.print("[yellow]此服务器没有提供工具[/yellow]")
            else:
                console.print(f"[bold red]错误:[/bold red] 未找到服务器 '{server}'")
        else:
            # 显示所有服务器的工具
            for server_name, server_conn in client.servers.items():
                console.print(Panel(f"[bold]{server_name}[/bold] 的可用工具", border_style="blue"))
                
                tools = server_conn.tools
                if tools:
                    table = Table(box=box.ROUNDED)
                    table.add_column("工具名称", style="cyan")
                    table.add_column("描述", style="green")
                    
                    for tool in tools:
                        table.add_row(tool.name, tool.description)
                    
                    console.print(table)
                else:
                    console.print("[yellow]此服务器没有提供工具[/yellow]")
                
                console.print("\n")
        
        # 不清理资源，让用户可以继续使用已连接的服务器
    
    try:
        asyncio.run(tools_async())
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断，程序退出[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]程序异常:[/bold red] {str(e)}")


@app.command()
def history(
    count: int = typer.Option(None, "--count", "-c", help="显示指定数量的历史记录"),
    set_max: int = typer.Option(None, "--set", "-s", help="设置最大历史记录数量")
):
    """管理对话历史记录"""
    global client
    if client is None:
        client = MultiMCPClient()
    
    if set_max is not None:
        if set_max < 1:
            console.print("[bold red]错误:[/bold red] 最大历史记录数量必须大于0")
            return
        client.max_history = set_max
        console.print(f"[bold green]已设置最大历史记录数量为:[/bold green] {set_max}")
    
    # 显示历史记录
    history = client.conversation_history
    if not history:
        console.print("[yellow]没有历史记录[/yellow]")
        return
    
    # 如果指定了数量，只显示指定数量的记录
    if count is not None:
        history = history[-count*2:] if count > 0 else []
    
    console.print(Panel(
        f"[bold]当前历史记录数量:[/bold] {len(history)//2}",
        subtitle=f"最大历史记录数量: {client.max_history}",
        border_style="blue",
        expand=False
    ))
    
    # 按对话对显示
    for i in range(0, len(history), 2):
        if i+1 < len(history):
            user_msg = history[i].get("content", "")
            ai_msg = history[i+1].get("content", "")
            
            console.print(f"[bold blue]User ({i//2+1}):[/bold blue]")
            console.print(user_msg)
            console.print(f"[bold green]Assistant:[/bold green]")
            console.print(Markdown(ai_msg))
            console.print("---")

@app.command()
def clear():
    """清除对话历史记录"""
    global client
    if client is None:
        client = MultiMCPClient()
    
    client.conversation_history = []
    console.print("[bold green]已清除对话历史记录[/bold green]")

# 全局选项
@app.callback()
def global_options(config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="配置文件路径")):
    """全局选项"""
    # 将配置文件路径存储在环境变量中，供其他命令使用
    os.environ["MCP_CONFIG"] = config_path

# 添加信号处理，确保程序退出时资源正确清理
def cleanup_on_exit():
    """程序退出时清理资源"""
    global client
    if client:
        try:
            # 使用新的事件循环来清理资源
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(client.cleanup())
            loop.close()
        except Exception:
            pass

def register_signal_handlers():
    """注册信号处理函数"""
    import atexit
    import signal
    
    # 注册退出处理
    atexit.register(cleanup_on_exit)
    
    # 注册信号处理
    if sys.platform != 'win32':
        # Unix系统上的信号处理
        signal.signal(signal.SIGTERM, lambda sig, frame: cleanup_on_exit())
        signal.signal(signal.SIGINT, lambda sig, frame: cleanup_on_exit())
    else:
        # Windows上的信号处理
        try:
            signal.signal(signal.SIGTERM, lambda sig, frame: cleanup_on_exit())
        except (AttributeError, ValueError):
            pass

def main():
    """
    主函数
    
    1. 检查是否有命令行参数
    2. 如果没有参数，默认进入聊天模式
    3. 如果有参数，正常执行
    4. 如果发生异常，打印错误信息
    """
    try:
        # 注册信号处理
        register_signal_handlers()
        
        # 检查是否有命令行参数
        if len(sys.argv) <= 1:
            # 没有参数，默认进入聊天模式
            chat()
        else:
            # 有参数，正常执行
            app()
    except Exception as e:
        console.print(f"\n[bold red]程序异常:[/bold red] {str(e)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
