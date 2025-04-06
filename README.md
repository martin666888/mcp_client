# MCP 客户端详解

## 简介

这是一个基于 MCP（Model-Client-Provider）协议的客户端实现，允许您通过命令行与大型语言模型（例如 DeepSeek）交互，并让模型使用工具来执行各种任务。MCP 协议是一种标准化的方式，用于使 AI 模型能够调用外部工具和服务，从而扩展其能力。

本客户端特点：
- 支持与 DeepSeek API 集成
- 工具调用功能
- 交互式命令行界面
- 支持 Python 和 JavaScript 编写的 MCP 服务器

## 安装指南

### 前提条件

- Python 3.10 或更高版本
- 虚拟环境管理工具（如 uv 或 venv）

### 安装步骤

1. 克隆或下载本项目
2. 创建并激活虚拟环境：

```bash
cd mcpclient
uv venv  # 或使用 python -m venv .venv
# 激活虚拟环境
# Windows:
.venv\Scripts\activate  # 如果使用venv
# 使用uv时不需要激活
```

3. 安装依赖：

```bash
uv add mcp openai anthropic  #或者其他相关依赖
```

## 使用方法

### 配置 API

首先，在 `mcpclient.py` 文件中配置您的 API 密钥和相关设置：

```python
# 全局 API 配置
API_CONFIG = {
    "api_key": "your-api-key-here",
    "api_base": "API base url",  # 或其他支持OpenAI兼容API的服务
    "model": "模型名称"  # 根据您使用的API提供商进行调整
}
```

### 运行客户端

要启动客户端，您需要提供一个 MCP 服务器脚本路径：

```bash
uv run python mcpclient.py path/to/your/mcp_server_script.py
# 或
python mcpclient.py path/to/your/mcp_server_script.js
```

### 交互方式

启动后，您将看到提示符：

```
MCP客户端已启动！
输入您的查询或'quit'退出。

查询: 
```

输入您的问题或指令，AI 会回复并根据需要调用相应的工具。
输入 `quit` 退出程序。

## 设计你自己的 MCP 客户端

本节将指导您如何从头开始设计自己的 MCP 客户端。

### 1. 基本架构

MCP 客户端的核心架构包括以下组件：

- **连接管理**：与 MCP 服务器建立连接
- **消息处理**：发送请求和接收响应
- **工具管理**：列出可用工具并处理工具调用
- **AI 交互**：与 AI 模型通信
- **用户界面**：提供与用户交互的方式

### 2. 关键依赖

```
mcp==0.1.0  # MCP 客户端库
openai==1.0.0以上  # 支持 OpenAI 兼容API的客户端
```

### 3. 核心代码结构

一个最小化的 MCP 客户端应包括：

#### 初始化和配置

```python
class MCPClient:
    def __init__(self):
        # 配置客户端
        self.session = None  # MCP会话
        self.exit_stack = AsyncExitStack()  # 用于资源管理
        
        # 初始化API客户端
        self.client = OpenAI(
            api_key="your-api-key",
            base_url="https://api.provider.com"
        )
```

#### 连接到服务器

```python
async def connect_to_server(self, server_script_path):
    # 确定脚本类型和启动命令
    command = "python" if server_script_path.endswith('.py') else "node"
    
    # 设置服务器参数
    server_params = StdioServerParameters(
        command=command,
        args=[server_script_path],
        env=None
    )
    
    # 建立连接
    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
    self.stdio, self.write = stdio_transport
    self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
    await self.session.initialize()
    
    # 列出可用工具
    response = await self.session.list_tools()
    return response.tools
```

#### 处理查询

```python
async def process_query(self, query):
    # 准备消息
    messages = [{"role": "user", "content": query}]
    
    # 获取可用工具
    response = await self.session.list_tools()
    available_tools = [...]  # 转换为API所需格式
    
    # 调用AI
    response = self.client.chat.completions.create(
        model="model-name",
        messages=messages,
        tools=available_tools,
        tool_choice="auto"
    )
    
    # 处理响应和工具调用
    # 如果工具被调用，执行工具并将结果返回给AI
    # ...
```

### 4. 资源清理

确保正确清理资源是非常重要的：

```python
async def cleanup(self):
    try:
        await self.exit_stack.aclose()
    except Exception as e:
        print(f"清理过程中出现异常 (可以忽略): {str(e)}")
```

### 5. 错误处理最佳实践

- 捕获并处理异步错误
- 优雅处理连接问题
- 安全地关闭资源
- 为用户提供有意义的错误消息

### 6. 高级功能

一旦掌握了基本实现，您可以添加以下高级功能：

- **会话管理**：维护对话历史
- **流式响应**：使用流式API获取更快的响应
- **多模型支持**：轻松切换不同的模型
- **用户认证**：添加用户验证层
- **工具自定义**：允许用户添加自定义工具

## 工具开发指南

### 创建 MCP 服务器

MCP 服务器定义了客户端可以调用的工具。一个简单的 MCP 服务器如下：

```python
from mcp import Server, ToolCall

server = Server()

@server.tool("calculator", "A simple calculator")
async def calculator(a: int, b: int, operation: str):
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    # 其他操作...

if __name__ == "__main__":
    server.start()
```

### 连接客户端与服务器

MCP 客户端与服务器之间的通信使用标准输入/输出进行，这使得它们可以用不同的语言实现，只要它们遵循相同的协议。

## 故障排除

### 常见问题

1. **连接错误**：确保服务器脚本路径正确且可执行
2. **API 错误**：验证您的 API 密钥和基础 URL 是否正确
3. **工具调用失败**：检查工具参数是否符合服务器期望

### 退出时的错误信息

如果您在使用 `quit` 退出时看到 asyncio 相关的错误，这通常与资源清理有关。最新版本已经优化了退出过程，但您可能仍会看到一些警告消息，这些通常可以忽略。

## 未来改进

- 添加图形用户界面
- 支持更多类型的 MCP 服务器连接
- 实现多会话管理
- 添加插件系统以扩展功能

## 贡献

欢迎对此项目做出贡献！请通过提交 Pull Request 或开设 Issue 来帮助改进。
