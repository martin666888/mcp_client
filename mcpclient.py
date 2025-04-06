import asyncio
import sys
import json
from typing import Dict, List, Optional
from contextlib import AsyncExitStack

import config
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI


class MCPServerConnection:
    """单个MCP服务器连接的封装"""
    
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


class MultiMCPClient:
    """支持多服务器连接的MCP客户端"""
    
    def __init__(self, max_history=10):
        # 初始化会话和客户端对象
        self.servers: Dict[str, MCPServerConnection] = {}
        self.exit_stack = AsyncExitStack()
        self.pending_tasks: List[asyncio.Task] = []
        self.conversation_history = []
        self.max_history = max_history
        self.current_server = None
        self.auto_select = True
        self.direct_mode = False  # 新增：直接对话模式开关

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base
        )

    async def connect_to_server(self, server_identifier: str, server_name: str = None) -> str:
        """连接到MCP服务器，支持npm包形式和脚本文件"""
        # 如果未提供服务器名称，使用标识符作为名称
        server_name = server_name or server_identifier
        
        # 检查是否是npx或uvx命令格式
        if server_identifier.startswith('npx ') or server_identifier.startswith('uvx '):
            args = server_identifier.split()
            # 如果第一个参数不是npx/uvx，则添加对应的前缀
            if args[0] not in ["npx", "uvx"]:
                args = ["uvx" if "uvx" in args[0] else "npx"] + args
                
            print(f"执行命令: {' '.join(args)}")
            
            server_params = StdioServerParameters(
                command=args[0],
                args=args[1:],
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
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        
        # 获取工具列表
        response = await session.list_tools()
        tools = response.tools
        
        # 创建服务器连接对象
        server_conn = MCPServerConnection(server_name, session, stdio, write, tools)
        
        # 存储到服务器字典中
        self.servers[server_name] = server_conn

        print(f"\n已连接到服务器 '{server_name}'，可用工具: {[tool.name for tool in tools]}")
        
        # 如果这是第一个服务器，将其设为当前服务器
        if len(self.servers) == 1:
            self.current_server = server_name
            
        # 如果连接了服务器，自动关闭直接对话模式
        if self.direct_mode:
            self.direct_mode = False
            print("已自动切换到MCP服务器模式")
            
        return server_name

    async def select_best_server(self, query: str) -> str:
        """根据查询内容选择最合适的服务器"""
        # 如果只有一个服务器，直接返回
        if len(self.servers) == 1:
            return list(self.servers.keys())[0]
        
        # 如果指定了当前服务器且不是自动选择模式，则使用当前服务器
        if not self.auto_select and self.current_server:
            return self.current_server
            
        # 创建选择提示
        prompt = f"""
        你是一个智能的服务器选择助手。你的任务是为给定的用户查询选择最合适的MCP服务器。
        
        用户查询:
        "{query}"
        
        可用的MCP服务器及其工具:
        """
            
        for name, server in self.servers.items():
            prompt += f"\n服务器名称: {name}\n可用工具:\n{server.get_tool_descriptions()}\n"
            
        prompt += """
        分析这个查询，确定哪个服务器最适合处理它。考虑以下因素:
        1. 查询需要什么类型的操作（如网络搜索、文件处理、分步推理等）
        2. 哪个服务器的工具集最符合这些需求
        3. 工具描述与查询内容的匹配程度
        
        只返回最合适的服务器名称，不要添加任何解释或额外文字。
        """
        
        # 调用API决定最佳服务器
        try:
            response = self.client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # 低温度以获得更确定的结果
            )
            
            chosen_server = response.choices[0].message.content.strip()
            
            # 确保选择的服务器存在
            if chosen_server in self.servers:
                print(f"为查询自动选择服务器: {chosen_server}")
                return chosen_server
            else:
                # 如果无法识别响应，默认使用第一个服务器
                default_server = list(self.servers.keys())[0]
                print(f"无法识别服务器选择结果，默认使用: {default_server}")
                return default_server
        except Exception as e:
            # 出错时默认使用第一个服务器
            default_server = list(self.servers.keys())[0]
            print(f"选择服务器时出错: {str(e)}，默认使用: {default_server}")
            return default_server

    async def direct_chat(self, query: str) -> str:
        """直接与AI对话，不使用任何MCP服务器或工具"""
        # 添加用户消息到历史记录
        self.conversation_history.append({"role": "user", "content": query})
        
        # 限制历史记录长度
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        try:
            # 不提供任何工具，直接与AI对话
            response = self.client.chat.completions.create(
                model=config.model,
                messages=self.conversation_history
            )
            
            # 获取模型的响应
            message = response.choices[0].message
            if message.content:
                self.conversation_history.append({"role": "assistant", "content": message.content})
                return message.content
            else:
                return "AI没有返回任何内容"
        
        except Exception as e:
            return f"API调用出错: {str(e)}"

    async def process_query(self, query: str) -> str:
        """使用OpenAI兼容API和可用工具处理查询"""
        if not self.servers:
            return "错误: 没有连接的服务器，请先连接至少一个服务器"
            
        # 添加用户消息到历史记录
        self.conversation_history.append({"role": "user", "content": query})
        
        # 限制历史记录长度
        if len(self.conversation_history) > self.max_history * 2:  # 乘以2因为每条用户消息对应一条助手消息
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        # 选择最合适的服务器
        server_name = await self.select_best_server(query)
        server = self.servers[server_name]
        
        # 获取可用工具
        await server.list_tools()  # 刷新工具列表
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool_detail["name"],
                "description": tool_detail["description"],
                "parameters": tool_detail["parameters"]
            }
        } for tool_detail in server.tool_details]
        
        # 使用完整对话历史进行API调用
        try:
            response = self.client.chat.completions.create(
                model=config.model,
                messages=self.conversation_history,
                tools=available_tools,
                tool_choice="auto"
            )

            # 处理响应
            final_text = []
            final_text.append(f"[使用服务器: {server_name}]")

            # 获取模型的响应
            message = response.choices[0].message
            if message.content:
                final_text.append(message.content)
                self.conversation_history.append({"role": "assistant", "content": message.content})

            # 处理工具调用
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    # 检查所选服务器是否有该工具
                    if not server.has_tool(tool_name):
                        # 查找具有该工具的其他服务器
                        alternative_server = None
                        for s_name, s in self.servers.items():
                            if s.has_tool(tool_name):
                                alternative_server = s_name
                                break
                                
                        if alternative_server:
                            server_name = alternative_server
                            server = self.servers[server_name]
                            final_text.append(f"[切换到服务器: {server_name}，因为它提供了工具: {tool_name}]")
                        else:
                            final_text.append(f"[错误: 没有服务器提供工具 {tool_name}]")
                            continue
                    
                    # 打印调试信息
                    print(f"工具调用: {tool_name}, 参数: {tool_args}")
                    
                    # 执行工具调用
                    result = await server.call_tool(tool_name, tool_args)
                    final_text.append(f"[调用工具 {tool_name}，参数 {tool_args}]")
                    
                    # 返回工具调用结果
                    tool_result = str(result.content)
                    
                    # 添加助手消息和工具结果到上下文
                    messages = self.conversation_history + [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }
                            ]
                        }
                    ]
                    
                    # 确保工具响应格式正确
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result  # 使用字符串
                    })
                    
                    # 打印调试信息
                    print("发送到API的消息结构:")
                    print(json.dumps(messages, indent=2))
                    
                    # 获取下一个响应 - 使用相同的模型
                    try:
                        second_response = self.client.chat.completions.create(
                            model=config.model,
                            messages=messages,
                            tools=available_tools,
                            tool_choice="auto"
                        )
                        
                        next_message = second_response.choices[0].message
                        if next_message.content:
                            final_text.append(next_message.content)
                            self.conversation_history.append({"role": "assistant", "content": next_message.content})
                            
                        # 处理可能的级联工具调用
                        if hasattr(next_message, 'tool_calls') and next_message.tool_calls:
                            final_text.append("[检测到级联工具调用，但为简化处理，不再进一步处理]")
                    except Exception as e:
                        final_text.append(f"获取最终响应时出错: {str(e)}")
                        print(f"获取最终响应时出错: {str(e)}")
            
            return "\n".join(final_text)
        
        except Exception as e:
            return f"API调用出错: {str(e)}"
    
    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\n多服务器MCP客户端已启动！")
        print("\n可用命令:")
        print("- connect <服务器标识符> [服务器名称]: 连接新服务器")
        print("- list: 列出已连接的服务器")
        print("- use <服务器名称>: 指定使用的服务器")
        print("- auto: 让AI自动选择服务器")
        print("- disconnect <服务器名称>: 断开指定服务器的连接")
        print("- direct: 切换到直接对话模式（不使用MCP服务器）")
        print("- mcp: 切换到MCP服务器模式")
        print("- quit: 退出程序")
        print("\n输入您的查询或上述命令:")
        
        while True:
            try:
                query = input("\n> ").strip()
                
                # 处理命令
                if query.lower() == 'quit':
                    print("正在清理并退出...")
                    break
                    
                elif query.lower() == 'list':
                    if not self.servers:
                        print("没有连接的服务器")
                    else:
                        print("\n已连接服务器:")
                        for name, server in self.servers.items():
                            status = "当前使用" if name == self.current_server and not self.auto_select else ""
                            print(f"- {name}: {server.tool_names} {status}")
                    print(f"\n当前模式: {'直接对话' if self.direct_mode else 'MCP服务器'} {'(自动选择)' if not self.direct_mode and self.auto_select else ''}")
                    continue

                elif query.lower().startswith('connect '):
                    parts = query[8:].split(maxsplit=1)
                    server_id = parts[0]
                    server_name = parts[1] if len(parts) > 1 else None
                    
                    # 检查完整的服务器标识符，而不仅仅是第一个部分
                    full_identifier = query[8:].strip()
                    
                    # 查看完整的标识符是否以npx或uvx开头
                    if full_identifier.startswith('npx ') or full_identifier.startswith('uvx '):
                        try:
                            name = await self.connect_to_server(full_identifier, server_name)
                            print(f"已连接到服务器: {name}")
                        except Exception as e:
                            print(f"连接服务器失败: {str(e)}")
                    else:
                        # 原来的处理方式
                        try:
                            name = await self.connect_to_server(server_id, server_name)
                            print(f"已连接到服务器: {name}")
                        except Exception as e:
                            print(f"连接服务器失败: {str(e)}")
                    continue
                    
                # elif query.lower().startswith('connect '):
                #     parts = query[8:].split(maxsplit=1)
                #     server_id = parts[0]
                #     server_name = parts[1] if len(parts) > 1 else None
                    
                #     try:
                #         name = await self.connect_to_server(server_id, server_name)
                #         print(f"已连接到服务器: {name}")
                #     except Exception as e:
                #         print(f"连接服务器失败: {str(e)}")
                #     continue
                    
                elif query.lower().startswith('use '):
                    server_name = query[4:].strip()
                    if server_name in self.servers:
                        self.current_server = server_name
                        self.auto_select = False
                        self.direct_mode = False
                        print(f"已切换到服务器: {self.current_server}")
                    else:
                        print(f"错误: 未找到服务器 '{server_name}'")
                    continue
                    
                elif query.lower() == 'auto':
                    self.auto_select = True
                    self.direct_mode = False
                    print("已启用服务器自动选择")
                    continue

                elif query.lower() == 'direct':
                    self.direct_mode = True
                    print("已切换到直接对话模式，不使用MCP服务器")
                    continue
                    
                elif query.lower() == 'mcp':
                    self.direct_mode = False
                    print(f"已切换到MCP服务器模式 {'(自动选择)' if self.auto_select else ''}")
                    continue
                    
                elif query.lower().startswith('disconnect '):
                    server_name = query[11:].strip()
                    if server_name in self.servers:
                        # 简单实现，不实际清理资源
                        del self.servers[server_name]
                        if self.current_server == server_name:
                            self.current_server = next(iter(self.servers.keys())) if self.servers else None
                            self.auto_select = True
                        print(f"已断开服务器: {server_name}")
                    else:
                        print(f"错误: 未找到服务器 '{server_name}'")
                    continue
                
                # 处理普通查询
                if self.direct_mode:
                    # 直接对话模式
                    response = await self.direct_chat(query)
                else:
                    # MCP服务器模式
                    if not self.servers:
                        print("错误: 没有连接的服务器，请先连接服务器或切换到直接对话模式")
                        continue
                    
                    response = await self.process_query(query)
                
                print("\n" + response)
                
            except Exception as e:
                print(f"\n错误: {str(e)}")
                import traceback
                traceback.print_exc()
    
    async def cleanup(self):
        """清理资源"""
        try:
            # 关闭退出栈，这将清理所有资源
            await self.exit_stack.aclose()
        except Exception as e:
            # 捕获但不重新抛出异常，只是记录它们
            print(f"清理过程中出现异常 (可以忽略): {str(e)}")


async def main():
    if len(sys.argv) < 2:
        print("用法: python mcpclient.py <服务器脚本路径或npm包命令>")
        print("例如: python mcpclient.py server.py")
        print("或:   python mcpclient.py npx -y @modelcontextprotocol/server-sequential-thinking")
        print("或:   python mcpclient.py uvx mcp-server-fetch")
        sys.exit(1)
    
    # 合并所有命令行参数为一个字符串，以便处理npm命令
    server_identifier = " ".join(sys.argv[1:])
    
    client = MultiMCPClient()
    
    try:
        # 连接到初始服务器
        await client.connect_to_server(server_identifier)
        # 启动交互式循环
        await client.chat_loop()
    finally:
        # 清理资源
        await client.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
    except asyncio.CancelledError:
        # 捕获并忽略取消错误
        print("程序被取消，正常退出")
    except Exception as e:
        print(f"程序异常退出: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细的异常堆栈，便于调试