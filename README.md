# ChatGPT 自动注册工具 v2.0

基于 CloudMail 临时邮箱服务的 ChatGPT 自动注册与 OAuth Token 生成工具。

## ✨ v2.0 架构重构

项目已完成深度重构，采用模块化架构，代码更清晰、更易维护：

### 核心改进

- ✅ **模块化架构**：lib/ 目录精简为 3 个核心文件，职责清晰
- ✅ **逻辑整合优化**：操作类统一管理，减少代码冗余
- ✅ **完整注册流程**：支持新账号注册 + 已有账号登录
- ✅ **OAuth Token 获取**：自动获取 Access Token / Refresh Token / Session Token
- ✅ **智能重试机制**：密码注册、验证码获取、登录流程均支持重试
- ✅ **高并发支持**：支持多线程并发注册

## 📦 项目结构

```
.
├── lib/                          # 核心库模块（3 个文件）
│   ├── clients.py                # 所有客户端类（HTTP、OpenAI、CloudMail、OAuth、Token 管理器）
│   ├── utils.py                  # 工具函数（密码生成、Cookie 提取、日志格式化等）
│   └── core.py                   # 注册引擎 + 所有操作类
├── chatgpt_register_v2.py        # 主程序入口
├── config.json                   # 配置文件
├── .env                          # 环境变量（可选）
└── README.md                     # 本文档
```

### 核心模块说明

#### lib/clients.py
所有客户端和管理器的集合：
- `SentinelTokenGenerator`: Sentinel Token 生成器（纯 Python 实现）
- `HTTPClient`: 基础 HTTP 客户端封装
- `OpenAIHTTPClient`: OpenAI 专用 HTTP 客户端（IP 检查、Sentinel Token）
- `CloudMailService`: CloudMail 邮箱服务客户端
- `OAuthManager`: OAuth 流程管理器（授权、Token 交换）
- `TokenManager`: Token 管理器（保存和加载 Token）
- 数据模型：`RegistrationResult`, `SignupFormResult`, `OAuthStart`
- 常量配置：`OPENAI_API_ENDPOINTS`, `OPENAI_PAGE_TYPES`, `OTP_CODE_PATTERN`
- 工具函数：`generate_oauth_url()`, `submit_callback_url()`, `load_config()`

#### lib/utils.py
通用工具函数：
- `generate_password()`: 随机密码生成
- `generate_random_user_info()`: 随机用户信息生成
- `extract_session_token_from_cookie_jar()`: 从 Cookie Jar 提取 Session Token
- `extract_session_token_from_cookie_text()`: 从 Cookie 文本提取 Session Token
- `flatten_set_cookie_headers()`: 扁平化 Set-Cookie 头
- `extract_request_cookie_header()`: 提取请求 Cookie 头
- `dump_session_cookies()`: 导出会话 Cookie
- `format_log_message()`: 日志消息格式化

#### lib/core.py
注册引擎和所有操作类：
- `RegistrationEngine`: 注册引擎主类（协调整个注册流程）
- `EmailOperations`: 邮箱操作（创建临时邮箱）
- `OTPOperations`: 验证码操作（发送、获取、验证 OTP）
- `AuthOperations`: 认证操作（OAuth 启动、Device ID、Sentinel 验证）
- `LoginOperations`: 登录操作（提交邮箱、密码、重触发 OTP）
- `AccountOperations`: 账户创建操作（注册密码、创建用户账户）
- `WorkspaceOperations`: Workspace 操作（获取、选择 Workspace）
- `RedirectOperations`: 重定向处理（跟随 OAuth 重定向链）
- `TokenOperations`: Token 获取操作（捕获 Session Token、处理 OAuth 回调）

## 功能特性

- 🚀 使用 CloudMail 临时邮箱服务自动创建邮箱
- 🌐 支持自定义域名配置
- 🤖 自动注册 ChatGPT 账号并获取验证码
- 🔑 自动生成 OAuth Token（Access Token / Refresh Token / Session Token）
- ⚡ 支持高并发注册（推荐 5-8 线程）
- 🔄 智能重试机制（密码注册、验证码、登录流程）
- 💾 自动保存账号信息和 Token 到文件
- 📊 实时显示注册进度和成功率
- 🔐 支持已有账号自动登录获取 Token

## 环境要求

- Python 3.7+
- CloudMail 临时邮箱服务（需要管理员账号）
- 代理（可选，用于访问 OpenAI 服务）

## 安装依赖

```bash
pip install curl_cffi
```

## 配置说明

复制 `config.example.json` 为 `config.json` 并修改配置：

```json
{
    "cloudmail_url": "https://your-cloudmail-api.com",
    "cloudmail_admin_email": "admin@example.com",
    "cloudmail_admin_password": "your_password",
    "cloudmail_domains": ["domain1.com", "domain2.com"],
    "cloudmail_subdomain": "",
    "proxy": "http://127.0.0.1:7890",
    "output_file": "registered_accounts.txt",
    "enable_oauth": true,
    "oauth_required": true,
    "token_json_dir": "tokens",
    "timeout": 30
}
```

### 重要配置项说明

1. **cloudmail_url**：CloudMail API 地址
2. **cloudmail_admin_email** 和 **cloudmail_admin_password**：管理员账号
3. **cloudmail_domains**：可用域名列表
4. **proxy**：代理地址（格式：`http://host:port` 或 `socks5://host:port`）
5. **enable_oauth**：是否启用 OAuth 登录
6. **oauth_required**：OAuth 失败时是否视为注册失败
7. **token_json_dir**：Token JSON 文件保存目录

## 使用方法

```bash
# 注册 1 个账号（默认）
python chatgpt_register_v2.py

# 注册 5 个账号，使用 3 个线程
python chatgpt_register_v2.py -n 5 -w 3

# 注册 10 个账号，使用 5 个线程，不启用 OAuth
python chatgpt_register_v2.py -n 10 -w 5 --no-oauth
```

### 命令行参数

- `-n, --num`: 注册账号数量（默认: 1）
- `-w, --workers`: 并发线程数（默认: 1）
- `--no-oauth`: 禁用 OAuth 登录

### 推荐配置

| 场景 | 线程数 | 说明 |
|------|--------|------|
| 稳定优先 | 1 | 最稳定，速度较慢 |
| 平衡模式 | 2-3 | 稳定性好，速度适中 |
| 速度优先 | 4-5 | 速度快，需要稳定网络 |

### 输出文件

- `registered_accounts.txt`：账号密码列表（格式：`email----password----oauth=ok/failed`）
- `tokens/`：每个账号的完整 Token JSON 文件
  - `access_token`: 访问令牌
  - `refresh_token`: 刷新令牌
  - `id_token`: ID 令牌
  - `session_token`: 会话令牌

## 工作原理

### 完整注册流程

1. **IP 检查**：验证客户端 IP 地理位置
2. **创建邮箱**：通过 CloudMail 创建临时邮箱
3. **OAuth 初始化**：启动 OAuth 授权流程，获取 Device ID
4. **Sentinel 验证**：获取并验证 Sentinel POW Token
5. **提交注册**：提交邮箱到注册入口
6. **设置密码**：生成并提交随机密码（带重试）
7. **发送验证码**：请求发送邮箱验证码
8. **验证邮箱**：获取并验证邮箱验证码（带重试）
9. **创建账户**：提交用户信息完成账户创建
10. **重新登录**：新账号需重新登录获取 Token
11. **获取 Workspace**：获取并选择 Workspace ID
12. **跟随重定向**：跟随 OAuth 重定向链获取回调 URL
13. **Token 交换**：通过回调 URL 获取 Access Token / Refresh Token
14. **保存结果**：保存账号信息和 Token 到文件

### 智能重试机制

- **密码注册重试**：遇到 400 错误自动重试（最多 3 次）
- **验证码获取重试**：支持多次尝试获取验证码
- **登录流程重试**：登录失败时自动重新触发 OTP
- **OAuth 回调兜底**：回调失败时通过 `/api/auth/session` 兜底获取 Token

### 已有账号处理

- 自动检测账号是否已注册
- 切换到登录流程
- 自动获取并验证登录验证码
- 获取 Token 并保存

## 注意事项

### 1. CloudMail 服务
- 需要自己搭建或使用现有的 CloudMail 服务
- 确保 CloudMail 服务可正常访问
- 管理员账号需要有创建邮箱的权限

### 2. 代理设置
- 如果在国内使用，必须配置代理
- 确保代理可以访问 OpenAI 服务
- 推荐使用稳定的代理服务

### 3. 并发控制
- **推荐并发数**：1-5 线程
- **不推荐**：超过 5 线程（可能导致 IP 限制或服务不稳定）
- 首次使用建议从 1 线程开始测试

### 4. Token 有效期
- Access Token 有效期较短（通常几小时）
- Refresh Token 可用于刷新 Access Token
- Session Token 用于浏览器会话
- 建议定期备份 Token 文件

### 5. 错误处理
- 程序会自动处理大部分错误并重试
- 如遇到持续失败，检查：
  - 代理是否正常
  - CloudMail 服务是否可用
  - 配置文件是否正确
  - 网络连接是否稳定

## 常见问题

**Q: 注册失败，提示 IP 地理位置不支持？**  
A: 确保使用的代理 IP 位于支持的地区（如美国、欧洲等）。

**Q: 验证码获取失败？**  
A: 检查 CloudMail 服务是否正常，邮件是否能正常接收。

**Q: OAuth Token 获取失败？**  
A: 程序会自动重试，如果持续失败，可以设置 `oauth_required: false` 先完成注册。

**Q: 并发注册时成功率下降？**  
A: 降低并发线程数，推荐使用 2-3 个线程。

## 开发说明

### 代码结构设计

- **clients.py**：客户端层，封装所有外部服务调用（HTTP、邮箱、OAuth）和数据模型
- **utils.py**：工具层，提供通用工具函数（密码生成、Cookie 处理、日志格式化）
- **core.py**：业务层，实现核心注册逻辑和流程编排

### 扩展开发

如需扩展功能，建议：
1. 在 `clients.py` 中添加新的客户端类或数据模型
2. 在 `utils.py` 中添加新的工具函数
3. 在 `core.py` 中添加新的操作类或修改注册流程
4. 保持模块间的低耦合，遵循单一职责原则

## 许可证

MIT License
