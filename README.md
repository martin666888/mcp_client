# MCP 终端客户端

基于 Rich 和 Typer 的 MCP（Model Context Protocol）客户端终端应用，支持多服务器连接和配置文件管理。

## 功能特点

- 美观的命令行界面（使用 Rich 库）
- 支持从配置文件中读取 MCP 服务器配置
- 支持连接多个 MCP 服务器
- 支持与 OpenAI 兼容的 API 交互
- 支持对话历史记录管理
- 支持自动选择最佳服务器

## 安装

### 前提条件

- Python 3.8 或更高版本
- pip 或 uv 包管理器

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/martin666888/mcp_client.git
cd mcp_client
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

或使用 uv（更快的包管理器）：

```bash
uv pip install -r requirements.txt
```

3. 创建配置文件

将 `config.json.template` 复制为 `config.json` 并填入您的 API 密钥和其他配置信息：

```bash
cp config.json.template config.json
```

将 `config.py.template` 复制为 `config.py` 并填入您的 OpenAI API 密钥：

```bash
cp config.py.template config.py
```

## 使用方法

### 基本命令

- **启动交互式聊天**：
  ```bash
  python mcp_terminal.py
  ```
  或
  ```bash
  python mcp_terminal.py chat
  ```

- **连接到 MCP 服务器**：
  ```bash
  python mcp_terminal.py connect --from-config <服务器ID>
  ```
  或在聊天模式中：
  ```
  !connect --from-config <服务器ID>
  ```

- **查看配置文件中的服务器**：
  ```bash
  python mcp_terminal.py config --list
  ```
  或在聊天模式中：
  ```
  !config --list
  ```

- **管理对话历史记录**：
  ```bash
  python mcp_terminal.py history
  ```
  或在聊天模式中：
  ```
  !history
  ```

- **清除对话历史记录**：
  ```bash
  python mcp_terminal.py clear
  ```
  或在聊天模式中：
  ```
  !clear
  ```

### 配置文件格式

`config.json` 文件格式如下：

```json
{
  "mcpServers": {
    "server-id": {
      "command": "命令",
      "args": ["参数1", "参数2"],
      "env": {
        "ENV_VAR": "值"
      }
    }
  }
}
```

## 常见问题

### 连接服务器时出现错误

- 确保您已安装所需的 npm 包或其他依赖
- 检查环境变量是否正确设置
- 使用 `--keep-alive` 选项保持服务器连接

### API 密钥问题

- 确保 `config.py` 中包含正确的 API 密钥
- 对于特定服务器，确保在 `config.json` 中设置了正确的环境变量

## 贡献

欢迎提交 Pull Request 或创建 Issue 来改进这个项目！

## 许可证

[MIT](LICENSE)
