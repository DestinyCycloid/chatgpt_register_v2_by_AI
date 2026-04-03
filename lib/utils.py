"""
基础工具模块 (Constants, Models, Config, Helpers)
"""
import os
import re
import json
import random
import string
import secrets
import logging
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


"""
常量定义
"""











class EmailServiceType(str, Enum):
    """邮箱服务类型"""
    CLOUDMAIL = "cloudmail"






# ============================================================================
# 应用常量
# ============================================================================

APP_NAME = "OpenAI/Codex CLI 自动注册系统"
APP_VERSION = "1.1.2"
APP_DESCRIPTION = "自动注册 OpenAI/Codex CLI 账号的系统"

# ============================================================================
# OpenAI OAuth 相关常量
# ============================================================================

# OAuth 参数
OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OAUTH_AUTH_URL = "https://auth.openai.com/oauth/authorize"
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
OAUTH_REDIRECT_URI = "http://localhost:1455/auth/callback"
OAUTH_SCOPE = "openid email profile offline_access"

# OpenAI API 端点
OPENAI_API_ENDPOINTS = {
    "sentinel": "https://sentinel.openai.com/backend-api/sentinel/req",
    "signup": "https://auth.openai.com/api/accounts/authorize/continue",
    "register": "https://auth.openai.com/api/accounts/user/register",
    "password_verify": "https://auth.openai.com/api/accounts/password/verify",
    "send_otp": "https://auth.openai.com/api/accounts/email-otp/send",
    "validate_otp": "https://auth.openai.com/api/accounts/email-otp/validate",
    "create_account": "https://auth.openai.com/api/accounts/create_account",
    "select_workspace": "https://auth.openai.com/api/accounts/workspace/select",
}

# OpenAI 页面类型（用于判断账号状态）
OPENAI_PAGE_TYPES = {
    "EMAIL_OTP_VERIFICATION": "email_otp_verification",  # 已注册账号，需要 OTP 验证
    "PASSWORD_REGISTRATION": "create_account_password",  # 新账号，需要设置密码
    "LOGIN_PASSWORD": "login_password",  # 登录流程，需要输入密码
}




# ============================================================================
# 注册流程相关常量
# ============================================================================

# 验证码相关
OTP_CODE_PATTERN = r"(?<!\d)(\d{6})(?!\d)"
OTP_MAX_ATTEMPTS = 40  # 最大轮询次数

# 验证码提取正则（增强版）
# 简单匹配：任意 6 位数字
OTP_CODE_SIMPLE_PATTERN = r"(?<!\d)(\d{6})(?!\d)"
# 语义匹配：带上下文的验证码（如 "code is 123456", "验证码 123456"）
OTP_CODE_SEMANTIC_PATTERN = r'(?:code\s+is|验证码[是为]?\s*[:：]?\s*)(\d{6})'

# OpenAI 验证邮件发件人
OPENAI_EMAIL_SENDERS = [
    "noreply@openai.com",
    "no-reply@openai.com",
    "@openai.com",     # 精确域名匹配
    ".openai.com",     # 子域名匹配（如 otp@tm1.openai.com）
]

# OpenAI 验证邮件关键词
OPENAI_VERIFICATION_KEYWORDS = [
    "verify your email",
    "verification code",
    "验证码",
    "your openai code",
    "code is",
    "one-time code",
]

# 密码生成
PASSWORD_SPECIAL_CHARSET = "!@#$%^&*_-+="
PASSWORD_CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" + PASSWORD_SPECIAL_CHARSET
DEFAULT_PASSWORD_LENGTH = 12

# 用户信息生成（用于注册）

# 常用英文名
FIRST_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles",
    "Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia", "Harper", "Evelyn",
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn", "Skyler",
    "Liam", "Noah", "Ethan", "Lucas", "Mason", "Oliver", "Elijah", "Aiden", "Henry", "Sebastian",
    "Grace", "Lily", "Chloe", "Zoey", "Nora", "Aria", "Hazel", "Aurora", "Stella", "Ivy"
]

def generate_random_user_info() -> dict:
    """
    生成随机用户信息

    Returns:
        包含 name 和 birthdate 的字典
    """
    # 随机选择名字
    name = random.choice(FIRST_NAMES)

    # 生成随机生日（18-45岁）
    current_year = datetime.now().year
    birth_year = random.randint(current_year - 45, current_year - 18)
    birth_month = random.randint(1, 12)
    # 根据月份确定天数
    if birth_month in [1, 3, 5, 7, 8, 10, 12]:
        birth_day = random.randint(1, 31)
    elif birth_month in [4, 6, 9, 11]:
        birth_day = random.randint(1, 30)
    else:
        # 2月，简化处理
        birth_day = random.randint(1, 28)

    birthdate = f"{birth_year}-{birth_month:02d}-{birth_day:02d}"

    return {
        "name": name,
        "birthdate": birthdate
    }

# 保留默认值供兼容
DEFAULT_USER_INFO = {
    "name": "Neo",
    "birthdate": "2000-02-20",
}
"""
数据模型
"""





@dataclass
class RegistrationResult:
    """注册结果"""
    success: bool
    email: str = ""
    password: str = ""  # 注册密码
    account_id: str = ""
    workspace_id: str = ""
    access_token: str = ""
    refresh_token: str = ""
    id_token: str = ""
    session_token: str = ""  # 会话令牌
    device_id: str = ""  # oai-did
    error_message: str = ""
    logs: list = None
    metadata: dict = None
    source: str = "register"  # 'register' 或 'login'，区分账号来源




@dataclass
class SignupFormResult:
    """提交注册表单的结果"""
    success: bool
    page_type: str = ""  # 响应中的 page.type 字段
    is_existing_account: bool = False  # 是否为已注册账号
    response_data: Optional[Dict[str, Any]] = None  # 完整的响应数据
    error_message: str = ""
"""
配置管理模块
"""






def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """加载配置文件"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def as_bool(value: Any) -> bool:
    """将配置值转换为布尔值"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)
"""
注册辅助函数
"""











logger = logging.getLogger(__name__)


def generate_password(length: int = DEFAULT_PASSWORD_LENGTH) -> str:
    """生成随机密码"""
    length = max(8, int(length or DEFAULT_PASSWORD_LENGTH))
    password_chars = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice(PASSWORD_SPECIAL_CHARSET),
    ]
    password_chars.extend(secrets.choice(PASSWORD_CHARSET) for _ in range(length - len(password_chars)))
    secrets.SystemRandom().shuffle(password_chars)
    return ''.join(password_chars)


def extract_session_token_from_cookie_jar(cookie_jar) -> str:
    """
    从 CookieJar 中提取 next-auth session token（兼容分片 + 重复域名）。
    """
    if not cookie_jar:
        return ""

    entries: list[tuple[str, str]] = []
    try:
        for key, value in cookie_jar.items():
            entries.append((str(key or "").strip(), str(value or "").strip()))
    except Exception:
        pass

    try:
        jar = getattr(cookie_jar, "jar", None)
        if jar is not None:
            for cookie in jar:
                entries.append(
                    (
                        str(getattr(cookie, "name", "") or "").strip(),
                        str(getattr(cookie, "value", "") or "").strip(),
                    )
                )
    except Exception:
        pass

    direct_candidates = [
        val
        for name, val in entries
        if name in ("__Secure-next-auth.session-token", "_Secure-next-auth.session-token") and val
    ]
    if direct_candidates:
        return max(direct_candidates, key=len)

    chunk_map: dict[int, str] = {}
    for name, value in entries:
        if not (
            name.startswith("__Secure-next-auth.session-token.")
            or name.startswith("_Secure-next-auth.session-token.")
        ):
            continue
        if not value:
            continue
        try:
            idx = int(name.rsplit(".", 1)[-1])
        except Exception:
            continue
        prev = chunk_map.get(idx, "")
        if not prev or len(value) > len(prev):
            chunk_map[idx] = value

    if chunk_map:
        return "".join(chunk_map[i] for i in sorted(chunk_map.keys()))
    return ""


def extract_session_token_from_cookie_text(cookie_text: str) -> str:
    """从 Cookie 文本中提取 next-auth session token（兼容分片）。"""
    text = str(cookie_text or "")
    if not text:
        return ""

    # 直接匹配完整 token
    for prefix in ("__Secure-next-auth.session-token=", "_Secure-next-auth.session-token="):
        if prefix in text:
            start = text.find(prefix) + len(prefix)
            end = text.find(";", start)
            if end == -1:
                end = len(text)
            token = text[start:end].strip()
            if token and "." not in token.split("=")[0]:
                return token

    # 分片 token
    chunk_map: dict[int, str] = {}
    for prefix in ("__Secure-next-auth.session-token.", "_Secure-next-auth.session-token."):
        pattern = re.escape(prefix) + r"(\d+)=([^;]+)"
        for match in re.finditer(pattern, text):
            idx = int(match.group(1))
            value = match.group(2).strip()
            if value:
                prev = chunk_map.get(idx, "")
                if not prev or len(value) > len(prev):
                    chunk_map[idx] = value

    if chunk_map:
        return "".join(chunk_map[i] for i in sorted(chunk_map.keys()))
    return ""


def dump_session_cookies(session) -> str:
    """导出当前会话 cookies（用于后续支付/绑卡自动化）。"""
    if not session:
        return ""
    try:
        cookie_map: dict[str, str] = {}
        order: list[str] = []

        def _push(name: Optional[str], value: Optional[str]):
            key = str(name or "").strip()
            val = str(value or "").strip()
            if not key:
                return
            if key not in cookie_map:
                cookie_map[key] = val
                order.append(key)
                return
            prev = str(cookie_map.get(key) or "").strip()
            if (not prev and val) or (val and len(val) > len(prev)):
                cookie_map[key] = val

        # 1) 常规 requests/curl_cffi 字典接口
        try:
            for key, value in session.cookies.items():
                _push(key, value)
        except Exception:
            pass

        # 2) CookieJar 接口（可拿到分片 cookie）
        try:
            jar = getattr(session.cookies, "jar", None)
            if jar is not None:
                for cookie in jar:
                    _push(getattr(cookie, "name", ""), getattr(cookie, "value", ""))
        except Exception:
            pass

        # 3) 关键 cookie 兜底读取
        for key in (
            "oai-did",
            "oai-client-auth-session",
            "__Secure-next-auth.session-token",
            "_Secure-next-auth.session-token",
        ):
            try:
                _push(key, session.cookies.get(key))
            except Exception:
                continue

        pairs = [(k, cookie_map.get(k, "")) for k in order if k]
        return "; ".join(f"{k}={v}" for k, v in pairs if k)
    except Exception:
        return ""


def flatten_set_cookie_headers(response) -> str:
    """合并多条 Set-Cookie（包含分片 cookie）。"""
    try:
        headers = getattr(response, "headers", None)
        if headers is None:
            return ""
        if hasattr(headers, "get_list"):
            values = headers.get_list("set-cookie")
            if values:
                return " | ".join(str(v or "") for v in values if v is not None)
        if hasattr(headers, "get_all"):
            values = headers.get_all("set-cookie")
            if values:
                return " | ".join(str(v or "") for v in values if v is not None)
        return str(headers.get("set-cookie") or "")
    except Exception:
        return ""


def extract_request_cookie_header(response) -> str:
    """从响应对象关联的请求头中提取 Cookie。"""
    try:
        request_obj = getattr(response, "request", None)
        if request_obj is None:
            return ""
        headers = getattr(request_obj, "headers", None)
        if headers is None:
            return ""

        if hasattr(headers, "get"):
            value = headers.get("cookie") or headers.get("Cookie")
            if value:
                return str(value)

        try:
            for key, value in dict(headers).items():
                if str(key or "").strip().lower() == "cookie" and value:
                    return str(value)
        except Exception:
            pass
    except Exception:
        pass
    return ""


def format_log_message(message: str) -> str:
    """格式化日志消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    return f"[{timestamp}] {message}"
