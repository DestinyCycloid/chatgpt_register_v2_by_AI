"""
客户端模块 (Sentinel, HTTPClient, CloudMailService, OAuthManager, TokenManager)
"""
import os
import re
import time
import json
import uuid
import base64
import random
import string
import hashlib
import secrets
import logging
import threading
import urllib.parse

# 文件写入锁
_file_lock = threading.Lock()

from enum import Enum
from typing import Optional, Dict, Any, Union, Tuple, List, Set
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import requests
from curl_cffi import requests as cffi_requests
from curl_cffi.requests import Session, Response

from .utils import (
    EmailServiceType, OTP_CODE_PATTERN,
    OAUTH_CLIENT_ID, OAUTH_AUTH_URL, OAUTH_TOKEN_URL, OAUTH_REDIRECT_URI, OAUTH_SCOPE,
    OPENAI_API_ENDPOINTS
)

# Move EmailServiceStatus and EmailServiceError here because they were in email.py originally
class EmailServiceError(Exception):
    """邮箱服务异常"""
    pass

class EmailServiceStatus(Enum):
    """邮箱服务状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"

logger = logging.getLogger(__name__)


"""
Sentinel Token 生成器模块
基于对 sentinel.openai.com SDK 的逆向分析
"""








class SentinelTokenGenerator:
    """
    Sentinel Token 纯 Python 生成器
    
    通过逆向 sentinel SDK 的 PoW 算法，纯 Python 构造合法的 openai-sentinel-token。
    """

    def __init__(self, device_id=None, user_agent=None):

        self.device_id = device_id or str(uuid.uuid4())
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        )
        self.requirements_seed = str(random.random())
        self.sid = str(uuid.uuid4())

    @staticmethod
    def _fnv1a_32(text):
        """
        FNV-1a 32位哈希算法（从 SDK JS 逆向还原）
        """
        h = 2166136261  # FNV offset basis
        for ch in text:
            code = ord(ch)
            h ^= code
            h = (h * 16777619) & 0xFFFFFFFF

        # xorshift 混合（murmurhash3 finalizer）
        h ^= h >> 16
        h = (h * 2246822507) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 3266489909) & 0xFFFFFFFF
        h ^= h >> 16
        h = h & 0xFFFFFFFF

        return format(h, "08x")

    def _get_config(self):
        """构造浏览器环境数据数组"""

        
        screen_info = "1920x1080"
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)")
        js_heap_limit = 4294705152
        nav_random1 = random.random()
        ua = self.user_agent
        script_src = "https://sentinel.openai.com/sentinel/20260124ceb8/sdk.js"
        script_version = None
        data_build = None
        language = "en-US"
        languages = "en-US,en"
        nav_random2 = random.random()
        
        nav_props = [
            "vendorSub", "productSub", "vendor", "maxTouchPoints",
            "scheduling", "userActivation", "doNotTrack", "geolocation",
            "connection", "plugins", "mimeTypes", "pdfViewerEnabled",
            "webkitTemporaryStorage", "webkitPersistentStorage",
            "hardwareConcurrency", "cookieEnabled", "credentials",
            "mediaDevices", "permissions", "locks", "ink",
        ]
        nav_prop = random.choice(nav_props)
        nav_val = f"{nav_prop}−undefined"
        
        doc_key = random.choice(["location", "implementation", "URL", "documentURI", "compatMode"])
        win_key = random.choice(["Object", "Function", "Array", "Number", "parseFloat", "undefined"])
        perf_now = random.uniform(1000, 50000)
        hardware_concurrency = random.choice([4, 8, 12, 16])
        time_origin = time.time() * 1000 - perf_now

        config = [
            screen_info, date_str, js_heap_limit, nav_random1, ua,
            script_src, script_version, data_build, language, languages,
            nav_random2, nav_val, doc_key, win_key, perf_now,
            self.sid, "", hardware_concurrency, time_origin,
        ]
        return config

    @staticmethod
    def _base64_encode(data):
        """模拟 SDK 的 E() 函数：JSON.stringify → TextEncoder.encode → btoa"""
        json_str = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        encoded = json_str.encode("utf-8")
        return base64.b64encode(encoded).decode("ascii")



    def generate_requirements_token(self):
        """生成 requirements token（不需要服务端参数）"""
        config = self._get_config()
        config[3] = 1
        config[9] = round(random.uniform(5, 50))
        data = self._base64_encode(config)
        return "gAAAAAC" + data

"""
HTTP 客户端封装
基于 curl_cffi 的 HTTP 请求封装，支持代理和错误处理
"""













logger = logging.getLogger(__name__)


@dataclass
class RequestConfig:
    """HTTP 请求配置"""
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    impersonate: str = "chrome"
    verify_ssl: bool = True
    follow_redirects: bool = True


class HTTPClientError(Exception):
    """HTTP 客户端异常"""
    pass


class HTTPClient:
    """
    HTTP 客户端封装
    支持代理、重试、错误处理和会话管理
    """

    def __init__(
        self,
        proxy_url: Optional[str] = None,
        config: Optional[RequestConfig] = None,
        session: Optional[Session] = None
    ):
        """
        初始化 HTTP 客户端

        Args:
            proxy_url: 代理 URL，如 "http://127.0.0.1:7890"
            config: 请求配置
            session: 可重用的会话对象
        """
        self.proxy_url = proxy_url
        self.config = config or RequestConfig()
        self._session = session

    @property
    def proxies(self) -> Optional[Dict[str, str]]:
        """获取代理配置"""
        if not self.proxy_url:
            return None
        return {
            "http": self.proxy_url,
            "https": self.proxy_url,
        }

    @property
    def session(self) -> Session:
        """获取会话对象（单例）"""
        if self._session is None:
            self._session = Session(
                proxies=self.proxies,
                impersonate=self.config.impersonate,
                verify=self.config.verify_ssl,
                timeout=self.config.timeout
            )
        return self._session

    def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> Response:
        """
        发送 HTTP 请求

        Args:
            method: HTTP 方法 (GET, POST, PUT, DELETE, etc.)
            url: 请求 URL
            **kwargs: 其他请求参数

        Returns:
            Response 对象

        Raises:
            HTTPClientError: 请求失败
        """
        # 设置默认参数
        kwargs.setdefault("timeout", self.config.timeout)
        kwargs.setdefault("allow_redirects", self.config.follow_redirects)

        # 添加代理配置
        if self.proxies and "proxies" not in kwargs:
            kwargs["proxies"] = self.proxies

        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)

                # 检查响应状态码
                if response.status_code >= 400:
                    logger.warning(
                        f"HTTP {response.status_code} for {method} {url}"
                        f" (attempt {attempt + 1}/{self.config.max_retries})"
                    )

                    # 如果是服务器错误，重试
                    if response.status_code >= 500 and attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                        continue

                return response

            except (cffi_requests.RequestsError, ConnectionError, TimeoutError) as e:
                last_exception = e
                logger.warning(
                    f"请求失败: {method} {url} (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )

                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    break

        raise HTTPClientError(
            f"请求失败，最大重试次数已达: {method} {url} - {last_exception}"
        )

    def get(self, url: str, **kwargs) -> Response:
        """发送 GET 请求"""
        return self.request("GET", url, **kwargs)

    def post(self, url: str, data: Any = None, json: Any = None, **kwargs) -> Response:
        """发送 POST 请求"""
        return self.request("POST", url, data=data, json=json, **kwargs)

    def close(self):
        """关闭会话"""
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class OpenAIHTTPClient(HTTPClient):
    """
    OpenAI 专用 HTTP 客户端
    包含 OpenAI API 特定的请求方法
    """

    def __init__(
        self,
        proxy_url: Optional[str] = None,
        config: Optional[RequestConfig] = None
    ):
        """
        初始化 OpenAI HTTP 客户端

        Args:
            proxy_url: 代理 URL
            config: 请求配置
        """
        super().__init__(proxy_url, config)

        # OpenAI 特定的默认配置
        if config is None:
            self.config.timeout = 30
            self.config.max_retries = 3

        # 默认请求头
        self.default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
        }

    def check_ip_location(self) -> Tuple[bool, Optional[str]]:
        """
        检查 IP 地理位置

        Returns:
            Tuple[是否支持, 位置信息]
        """
        try:
            response = self.get("https://cloudflare.com/cdn-cgi/trace", timeout=10)
            trace_text = response.text

            # 解析位置信息

            loc_match = re.search(r"loc=([A-Z]+)", trace_text)
            loc = loc_match.group(1) if loc_match else None

            # 检查是否支持
            if loc in ["CN", "HK", "MO", "TW"]:
                return False, loc
            return True, loc

        except Exception as e:
            logger.error(f"检查 IP 地理位置失败: {e}")
            return False, None

    def check_sentinel(self, did: str, flow: str = "authorize_continue", proxies: Optional[Dict] = None) -> Optional[str]:
        """
        检查 Sentinel 拦截

        Args:
            did: Device ID
            flow: 流程类型，如 "authorize_continue", "oauth_create_account"
            proxies: 代理配置

        Returns:
            Sentinel token 或 None
        """


        try:
            pow_token = SentinelTokenGenerator(user_agent=self.default_headers.get("User-Agent", "")).generate_requirements_token()
            sen_req_body = json.dumps({
                "p": pow_token,
                "id": did,
                "flow": flow,
            }, separators=(",", ":"))

            response = self.post(
                OPENAI_API_ENDPOINTS["sentinel"],
                headers={
                    "origin": "https://sentinel.openai.com",
                    "referer": "https://sentinel.openai.com/backend-api/sentinel/frame.html?sv=20260219f9f6",
                    "content-type": "text/plain;charset=UTF-8",
                },
                data=sen_req_body,
            )

            if response.status_code == 200:
                return response.json().get("token")
            else:
                logger.warning(f"Sentinel 检查失败: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Sentinel 检查异常: {e}")
            return None

"""
"""



















logger = logging.getLogger(__name__)


class CloudMailService:
    """
    Cloud Mail 邮箱服务
    """
    
    # 类变量：所有实例共享token（按base_url区分）
    _shared_tokens: Dict[str, tuple] = {}  # {base_url: (token, expires_at)}
    _token_lock = None  # 延迟初始化
    _seen_ids_lock = None  # seen_email_ids 的锁
    _shared_seen_email_ids: Dict[str, set] = {}  # 所有实例共享已处理的邮件ID（按邮箱地址区分）

    def __init__(self, config: Dict[str, Any] = None, name: str = None):
        """
        初始化 Cloud Mail 服务

        Args:
            config: 配置字典，支持以下键:
                - base_url: API 基础地址 (必需)
                - admin_email: 管理员邮箱 (可选，为空时自动生成 admin@域名)
                - admin_password: 管理员密码 (必需)
                - domain: 邮箱域名 (可选，为空时从 base_url 提取)
                - subdomain: 子域名 (可选)，会插入到 @ 和域名之间，例如 subdomain="test" 会生成 xxx@test.example.com
                - timeout: 请求超时时间，默认 30
                - max_retries: 最大重试次数，默认 3
                - proxy_url: 代理地址 (可选)
            name: 服务名称
        """
        self.service_type = EmailServiceType.CLOUDMAIL
        self.name = name or f"{EmailServiceType.CLOUDMAIL.value}_service"
        self._status = EmailServiceStatus.HEALTHY
        self._last_error = None

        required_keys = ["base_url", "admin_password"]
        missing_keys = [key for key in required_keys if not (config or {}).get(key)]
        if missing_keys:
            raise ValueError(f"缺少必需配置: {missing_keys}")

        default_config = {
            "timeout": 30,
            "max_retries": 3,
            "proxy_url": None,
        }
        self.config = {**default_config, **(config or {})}
        self.config["base_url"] = self.config["base_url"].rstrip("/")
        
        # 如果没有提供 admin_email，自动生成
        if not self.config.get("admin_email"):
            domain = self._extract_domain_from_url(self.config["base_url"])
            self.config["admin_email"] = f"admin@{domain}"

        # 创建 requests session
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        
        # 初始化类级别的锁（线程安全）
        if CloudMailService._token_lock is None:

            CloudMailService._token_lock = threading.Lock()
            CloudMailService._seen_ids_lock = threading.Lock()



    def _extract_domain_from_url(self, url: str) -> str:
        """
        从 URL 中提取域名
        
        Args:
            url: URL 地址，如 https://ukumbuko.us.ci
            
        Returns:
            提取的域名，如 ukumbuko.us.ci
        """

        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        if not domain:
            raise ValueError(f"无法从 URL 提取域名: {url}")
        return domain

    def _generate_token(self) -> str:
        """
        生成身份令牌

        Returns:
            token 字符串

        Raises:
            EmailServiceError: 生成失败
        """
        url = f"{self.config['base_url']}/api/public/genToken"
        payload = {
            "email": self.config["admin_email"],
            "password": self.config["admin_password"]
        }

        try:
            response = self.session.post(
                url, 
                json=payload, 
                timeout=self.config["timeout"]
            )

            if response.status_code >= 400:
                error_msg = f"生成 token 失败: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = f"{error_msg} - {error_data}"
                except Exception:
                    error_msg = f"{error_msg} - {response.text[:200]}"
                raise EmailServiceError(error_msg)

            data = response.json()
            if data.get("code") != 200:
                raise EmailServiceError(f"生成 token 失败: {data.get('message', 'Unknown error')}")

            token = data.get("data", {}).get("token")
            if not token:
                raise EmailServiceError("生成 token 失败: 未返回 token")

            return token

        except requests.RequestException as e:
            self.update_status(False, e)
            raise EmailServiceError(f"生成 token 失败: {e}")
        except Exception as e:
            self.update_status(False, e)
            if isinstance(e, EmailServiceError):
                raise
            raise EmailServiceError(f"生成 token 失败: {e}")

    def _get_token(self, force_refresh: bool = False) -> str:
        """
        获取有效的 token（带缓存，所有实例共享）

        Args:
            force_refresh: 是否强制刷新

        Returns:
            token 字符串
        """
        base_url = self.config["base_url"]
        
        with CloudMailService._token_lock:
            # 检查共享缓存（token 有效期设为 1 小时）
            if not force_refresh and base_url in CloudMailService._shared_tokens:
                token, expires_at = CloudMailService._shared_tokens[base_url]
                if time.time() < expires_at:
                    return token

            # 生成新 token
            token = self._generate_token()
            expires_at = time.time() + 3600  # 1 小时后过期
            CloudMailService._shared_tokens[base_url] = (token, expires_at)
            return token

    def _get_headers(self, token: Optional[str] = None) -> Dict[str, str]:
        """构造请求头"""
        if token is None:
            token = self._get_token()

        return {
            "Authorization": token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _make_request(
        self,
        method: str,
        path: str,
        retry_on_auth_error: bool = True,
        **kwargs
    ) -> Any:
        """
        发送请求并返回 JSON 数据

        Args:
            method: HTTP 方法
            path: 请求路径（以 / 开头）
            retry_on_auth_error: 认证失败时是否重试
            **kwargs: 传递给 requests 的额外参数

        Returns:
            响应 JSON 数据

        Raises:
            EmailServiceError: 请求失败
        """
        url = f"{self.config['base_url']}{path}"
        kwargs.setdefault("headers", {})
        kwargs["headers"].update(self._get_headers())
        kwargs.setdefault("timeout", self.config["timeout"])

        try:
            response = self.session.request(method, url, **kwargs)

            if response.status_code >= 400:
                # 如果是认证错误且允许重试，刷新 token 后重试一次
                if response.status_code == 401 and retry_on_auth_error:
                    logger.warning("Cloud Mail 认证失败，尝试刷新 token")
                    kwargs["headers"].update(self._get_headers(self._get_token(force_refresh=True)))
                    response = self.session.request(method, url, **kwargs)

                if response.status_code >= 400:
                    error_msg = f"请求失败: {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg = f"{error_msg} - {error_data}"
                    except Exception:
                        error_msg = f"{error_msg} - {response.text[:200]}"
                    self.update_status(False, EmailServiceError(error_msg))
                    raise EmailServiceError(error_msg)

            try:
                return response.json()
            except Exception:
                return {"raw_response": response.text}

        except requests.RequestException as e:
            self.update_status(False, e)
            raise EmailServiceError(f"请求失败: {method} {path} - {e}")
        except Exception as e:
            self.update_status(False, e)
            if isinstance(e, EmailServiceError):
                raise
            raise EmailServiceError(f"请求失败: {method} {path} - {e}")

    def _generate_email_address(self, prefix: Optional[str] = None, domain: Optional[str] = None, subdomain: Optional[str] = None) -> str:
        """
        生成邮箱地址

        Args:
            prefix: 邮箱前缀，如果不提供则随机生成
            domain: 指定域名，如果不提供则从配置中选择
            subdomain: 子域名，可选参数，会插入到 @ 和域名之间

        Returns:
            完整的邮箱地址
        """
        if not prefix:
            # 生成随机前缀：首字母 + 9位随机字符（共10位）
            first = random.choice(string.ascii_lowercase)
            rest = "".join(random.choices(string.ascii_lowercase + string.digits, k=9))
            prefix = f"{first}{rest}"

        # 如果没有指定域名，从配置中获取
        if not domain:
            domain_config = self.config.get("domain")
            if not domain_config:
                # 如果没有配置域名，从 base_url 提取
                base_url = self.config.get("base_url")
                if base_url:
                    domain = self._extract_domain_from_url(base_url)
                else:
                    raise EmailServiceError("未配置邮箱域名，且无法从 API 地址提取域名")
            else:
                # 支持多个域名（列表）或单个域名（字符串）
                if isinstance(domain_config, list):
                    if not domain_config:
                        raise EmailServiceError("域名列表为空")
                    # 随机选择一个域名
                    domain = random.choice(domain_config)
                else:
                    domain = domain_config

        # 如果提供了子域，插入到域名前面
        if subdomain:
            domain = f"{subdomain}.{domain}"

        return f"{prefix}@{domain}"

    def create_email(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        创建新邮箱地址

        Args:
            config: 配置参数:
                - name: 邮箱前缀（可选）
                - domain: 邮箱域名（可选，覆盖默认域名）
                - subdomain: 子域名（可选），会插入到 @ 和域名之间，例如 subdomain="test" 会生成 xxx@test.example.com

        Returns:
            包含邮箱信息的字典:
            - email: 邮箱地址
            - service_id: 邮箱地址（用作标识）
        """
        req_config = config or {}

        # 生成邮箱地址
        prefix = req_config.get("name")
        specified_domain = req_config.get("domain")
        subdomain = req_config.get("subdomain") or self.config.get("subdomain")
        
        if specified_domain:
            email_address = self._generate_email_address(prefix, specified_domain, subdomain)
        else:
            email_address = self._generate_email_address(prefix, subdomain=subdomain)

        email_info = {
            "email": email_address,
            "service_id": email_address,
            "id": email_address,
            "created_at": time.time(),
        }

        self.update_status(True)
        logger.info(f"生成 CloudMail 邮箱: {email_address}")
        return email_info


    def get_verification_code(
        self,
        email: str,
        email_id: str = None,
        timeout: int = 120,
        pattern: str = OTP_CODE_PATTERN,
        otp_sent_at: Optional[float] = None,
    ) -> Optional[str]:
        """
        从 Cloud Mail 邮箱获取验证码

        Args:
            email: 邮箱地址
            email_id: 未使用，保留接口兼容
            timeout: 超时时间（秒）
            pattern: 验证码正则
            otp_sent_at: OTP 发送时间戳

        Returns:
            验证码字符串，超时返回 None
        """
        start_time = time.time()
        
        # 每次调用时，记录本次查询开始前已存在的邮件ID
        # 这样可以支持同一个邮箱多次接收验证码（注册+OAuth）
        initial_seen_ids = set()
        with CloudMailService._seen_ids_lock:
            if email not in CloudMailService._shared_seen_email_ids:
                CloudMailService._shared_seen_email_ids[email] = set()
            else:
                # 记录本次查询开始前的已处理邮件
                initial_seen_ids = CloudMailService._shared_seen_email_ids[email].copy()
        
        # 本次查询中新处理的邮件ID（仅在本次查询中有效）
        current_seen_ids = set()
        
        check_count = 0

        while time.time() - start_time < timeout:
            try:
                check_count += 1
                
                # 查询邮件列表
                url_path = "/api/public/emailList"
                payload = {
                    "toEmail": email,
                    "timeSort": "desc"  # 最新的邮件优先
                }

                result = self._make_request("POST", url_path, json=payload)

                if result.get("code") != 200:
                    time.sleep(3)
                    continue

                emails = result.get("data", [])
                if not isinstance(emails, list):
                    time.sleep(3)
                    continue

                for email_item in emails:
                    email_id = email_item.get("emailId")
                    
                    if not email_id:
                        continue
                    
                    # 跳过本次查询开始前已存在的邮件
                    if email_id in initial_seen_ids:
                        continue
                    
                    # 跳过本次查询中已处理的邮件（防止同一轮查询重复处理）
                    if email_id in current_seen_ids:
                        continue
                    
                    # 标记为本次已处理
                    current_seen_ids.add(email_id)
                    
                    # 同时更新全局已处理列表（防止其他并发任务重复处理）
                    with CloudMailService._seen_ids_lock:
                        CloudMailService._shared_seen_email_ids[email].add(email_id)
                    
                    sender_email = str(email_item.get("sendEmail", "")).lower()
                    sender_name = str(email_item.get("sendName", "")).lower()
                    subject = str(email_item.get("subject", ""))
                    to_email = email_item.get("toEmail", "")
                    
                    # 检查收件人是否匹配
                    if to_email != email:
                        continue
                    
                    if "openai" not in sender_email and "openai" not in sender_name:
                        continue

                    # 从主题提取
                    match = re.search(pattern, subject)
                    if match:
                        code = match.group(1)
                        self.update_status(True)
                        return code

                    # 从内容提取
                    content = str(email_item.get("content", ""))
                    if content:
                        clean_content = re.sub(r"<[^>]+>", " ", content)
                        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
                        clean_content = re.sub(email_pattern, "", clean_content)
                        
                        match = re.search(pattern, clean_content)
                        if match:
                            code = match.group(1)
                            self.update_status(True)
                            return code

            except Exception as e:
                # 如果是认证错误，强制刷新token
                if "401" in str(e) or "认证" in str(e):
                    try:
                        self._get_token(force_refresh=True)
                    except Exception:
                        pass
                logger.error(f"检查邮件时出错: {e}", exc_info=True)

            time.sleep(3)

        # 超时
        logger.warning(f"等待验证码超时: {email}")
        return None








    @property
    def status(self) -> EmailServiceStatus:
        """获取服务状态"""
        return self._status

    @property
    def last_error(self) -> Optional[str]:
        """获取最后一次错误信息"""
        return self._last_error

    def update_status(self, success: bool, error: Exception = None):
        """
        更新服务状态
        """
        if success:
            self._status = EmailServiceStatus.HEALTHY
            self._last_error = None
        else:
            self._status = EmailServiceStatus.DEGRADED
            if error:
                self._last_error = str(error)

    def __str__(self) -> str:
        return f"{self.name} ({self.service_type.value})"


"""
注册流程 - 邮箱与验证码操作
"""








logger = logging.getLogger(__name__)



"""
OpenAI OAuth 授权模块
从 main.py 中提取的 OAuth 相关函数
"""





















def _b64url_no_pad(raw: bytes) -> str:
    """Base64 URL 编码（无填充）"""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sha256_b64url_no_pad(s: str) -> str:
    """SHA256 哈希后 Base64 URL 编码"""
    return _b64url_no_pad(hashlib.sha256(s.encode("ascii")).digest())


def _random_state(nbytes: int = 16) -> str:
    """生成随机 state"""
    return secrets.token_urlsafe(nbytes)


def _pkce_verifier() -> str:
    """生成 PKCE code_verifier"""
    return secrets.token_urlsafe(64)


def _parse_callback_url(callback_url: str) -> Dict[str, str]:
    """解析回调 URL"""
    candidate = callback_url.strip()
    if not candidate:
        return {"code": "", "state": "", "error": "", "error_description": ""}

    if "://" not in candidate:
        if candidate.startswith("?"):
            candidate = f"http://localhost{candidate}"
        elif any(ch in candidate for ch in "/?#") or ":" in candidate:
            candidate = f"http://{candidate}"
        elif "=" in candidate:
            candidate = f"http://localhost/?{candidate}"

    parsed = urllib.parse.urlparse(candidate)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    fragment = urllib.parse.parse_qs(parsed.fragment, keep_blank_values=True)

    for key, values in fragment.items():
        if key not in query or not query[key] or not (query[key][0] or "").strip():
            query[key] = values

    def get1(k: str) -> str:
        v = query.get(k, [""])
        return (v[0] or "").strip()

    code = get1("code")
    state = get1("state")
    error = get1("error")
    error_description = get1("error_description")

    if code and not state and "#" in code:
        code, state = code.split("#", 1)

    if not error and error_description:
        error, error_description = error_description, ""

    return {
        "code": code,
        "state": state,
        "error": error,
        "error_description": error_description,
    }


def _jwt_claims_no_verify(id_token: str) -> Dict[str, Any]:
    """解析 JWT ID Token（不验证签名）"""
    if not id_token or id_token.count(".") < 2:
        return {}
    payload_b64 = id_token.split(".")[1]
    pad = "=" * ((4 - (len(payload_b64) % 4)) % 4)
    try:
        payload = base64.urlsafe_b64decode((payload_b64 + pad).encode("ascii"))
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return {}






def _to_int(v: Any) -> int:
    """转换为整数"""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _post_form(
    url: str,
    data: Dict[str, str],
    timeout: int = 30,
    proxy_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    发送 POST 表单请求

    Args:
        url: 请求 URL
        data: 表单数据
        timeout: 超时时间
        proxy_url: 代理 URL

    Returns:
        响应 JSON 数据
    """
    # 构建代理配置
    proxies = None
    if proxy_url:
        proxies = {
            "http": proxy_url,
            "https": proxy_url,
        }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    try:
        # 使用 curl_cffi 发送请求，支持代理和浏览器指纹
        response = cffi_requests.post(
            url,
            data=data,
            headers=headers,
            timeout=timeout,
            proxies=proxies,
            impersonate="chrome"
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"token exchange failed: {response.status_code}: {response.text}"
            )

        return response.json()

    except cffi_requests.RequestsError as e:
        raise RuntimeError(f"token exchange failed: network error: {e}") from e


@dataclass(frozen=True)
class OAuthStart:
    """OAuth 开始信息"""
    auth_url: str
    state: str
    code_verifier: str
    redirect_uri: str


def generate_oauth_url(
    *,
    redirect_uri: str = OAUTH_REDIRECT_URI,
    scope: str = OAUTH_SCOPE,
    client_id: str = OAUTH_CLIENT_ID
) -> OAuthStart:
    """
    生成 OAuth 授权 URL

    Args:
        redirect_uri: 回调地址
        scope: 权限范围
        client_id: OpenAI Client ID

    Returns:
        OAuthStart 对象，包含授权 URL 和必要参数
    """
    state = _random_state()
    code_verifier = _pkce_verifier()
    code_challenge = _sha256_b64url_no_pad(code_verifier)

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "prompt": "login",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    auth_url = f"{OAUTH_AUTH_URL}?{urllib.parse.urlencode(params)}"
    return OAuthStart(
        auth_url=auth_url,
        state=state,
        code_verifier=code_verifier,
        redirect_uri=redirect_uri,
    )


def submit_callback_url(
    *,
    callback_url: str,
    expected_state: str,
    code_verifier: str,
    redirect_uri: str = OAUTH_REDIRECT_URI,
    client_id: str = OAUTH_CLIENT_ID,
    token_url: str = OAUTH_TOKEN_URL,
    proxy_url: Optional[str] = None
) -> str:
    """
    处理 OAuth 回调 URL，获取访问令牌

    Args:
        callback_url: 回调 URL
        expected_state: 预期的 state 值
        code_verifier: PKCE code_verifier
        redirect_uri: 回调地址
        client_id: OpenAI Client ID
        token_url: Token 交换地址
        proxy_url: 代理 URL

    Returns:
        包含访问令牌等信息的 JSON 字符串

    Raises:
        RuntimeError: OAuth 错误
        ValueError: 缺少必要参数或 state 不匹配
    """
    cb = _parse_callback_url(callback_url)
    if cb["error"]:
        desc = cb["error_description"]
        raise RuntimeError(f"oauth error: {cb['error']}: {desc}".strip())

    if not cb["code"]:
        raise ValueError("callback url missing ?code=")
    if not cb["state"]:
        raise ValueError("callback url missing ?state=")
    if cb["state"] != expected_state:
        raise ValueError("state mismatch")

    token_resp = _post_form(
        token_url,
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": cb["code"],
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        },
        proxy_url=proxy_url
    )

    access_token = (token_resp.get("access_token") or "").strip()
    refresh_token = (token_resp.get("refresh_token") or "").strip()
    id_token = (token_resp.get("id_token") or "").strip()
    expires_in = _to_int(token_resp.get("expires_in"))

    claims = _jwt_claims_no_verify(id_token)
    email = str(claims.get("email") or "").strip()
    auth_claims = claims.get("https://api.openai.com/auth") or {}
    account_id = str(auth_claims.get("chatgpt_account_id") or "").strip()

    now = int(time.time())
    expired_rfc3339 = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + max(expires_in, 0))
    )
    now_rfc3339 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

    config = {
        "id_token": id_token,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "account_id": account_id,
        "last_refresh": now_rfc3339,
        "email": email,
        "type": "codex",
        "expired": expired_rfc3339,
    }

    return json.dumps(config, ensure_ascii=False, separators=(",", ":"))


class OAuthManager:
    """OAuth 管理器"""

    def __init__(
        self,
        client_id: str = OAUTH_CLIENT_ID,
        auth_url: str = OAUTH_AUTH_URL,
        token_url: str = OAUTH_TOKEN_URL,
        redirect_uri: str = OAUTH_REDIRECT_URI,
        scope: str = OAUTH_SCOPE,
        proxy_url: Optional[str] = None
    ):
        self.client_id = client_id
        self.auth_url = auth_url
        self.token_url = token_url
        self.redirect_uri = redirect_uri
        self.scope = scope
        self.proxy_url = proxy_url

    def start_oauth(self) -> OAuthStart:
        """开始 OAuth 流程"""
        return generate_oauth_url(
            redirect_uri=self.redirect_uri,
            scope=self.scope,
            client_id=self.client_id
        )

    def handle_callback(
        self,
        callback_url: str,
        expected_state: str,
        code_verifier: str
    ) -> Dict[str, Any]:
        """处理 OAuth 回调"""
        result_json = submit_callback_url(
            callback_url=callback_url,
            expected_state=expected_state,
            code_verifier=code_verifier,
            redirect_uri=self.redirect_uri,
            client_id=self.client_id,
            token_url=self.token_url,
            proxy_url=self.proxy_url
        )
        return json.loads(result_json)




"""
注册流程 - 认证与登录操作
"""











logger = logging.getLogger(__name__)


class TokenManager:
    """Token 管理器"""

    def __init__(self, config):
        """
        初始化 Token 管理器
        
        Args:
            config: 配置字典
        """
        self.ak_file = config.get("ak_file", "ak.txt")
        self.rk_file = config.get("rk_file", "rk.txt")
        self.token_json_dir = config.get("token_json_dir", "tokens")
        self.upload_api_url = config.get("upload_api_url", "")
        self.upload_api_token = config.get("upload_api_token", "")
        
        # 确保 token 目录存在
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.token_dir = self.token_json_dir if os.path.isabs(self.token_json_dir) else os.path.join(base_dir, self.token_json_dir)
        os.makedirs(self.token_dir, exist_ok=True)

    def save_tokens(self, email, tokens):
        """
        保存 tokens 到所有目标（txt + JSON + 上传）
        
        Args:
            email: 邮箱地址
            tokens: token 字典，包含 access_token, refresh_token, id_token
        """
        access_token = tokens.get("access_token", "")
        refresh_token = tokens.get("refresh_token", "")
        id_token = tokens.get("id_token", "")

        # 保存到 ak.txt
        if access_token:
            with _file_lock:
                with open(self.ak_file, "a", encoding="utf-8") as f:
                    f.write(f"{access_token}\n")

        # 保存到 rk.txt
        if refresh_token:
            with _file_lock:
                with open(self.rk_file, "a", encoding="utf-8") as f:
                    f.write(f"{refresh_token}\n")

        if not access_token:
            return

        # 解析 JWT payload
        payload = _jwt_claims_no_verify(access_token)
        auth_info = payload.get("https://api.openai.com/auth", {})
        account_id = auth_info.get("chatgpt_account_id", "")

        # 计算过期时间
        exp_timestamp = payload.get("exp")
        expired_str = ""
        if isinstance(exp_timestamp, int) and exp_timestamp > 0:
            exp_dt = datetime.fromtimestamp(exp_timestamp, tz=timezone(timedelta(hours=8)))
            expired_str = exp_dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")

        # 构造 token 数据
        now = datetime.now(tz=timezone(timedelta(hours=8)))
        token_data = {
            "type": "codex",
            "email": email,
            "expired": expired_str,
            "id_token": id_token,
            "account_id": account_id,
            "access_token": access_token,
            "last_refresh": now.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
            "refresh_token": refresh_token,
        }

        # 保存 JSON 文件
        token_path = os.path.join(self.token_dir, f"{email}.json")
        with _file_lock:
            with open(token_path, "w", encoding="utf-8") as f:
                json.dump(token_data, f, ensure_ascii=False, indent=2)

        # 上传到 CPA 管理平台（如果配置了）
        if self.upload_api_url:
            self._upload_token_json(token_path)

    def _upload_token_json(self, filepath):
        """上传 Token JSON 文件到 CPA 管理平台"""
        try:
            # 尝试使用 curl_cffi
            try:


                
                filename = os.path.basename(filepath)
                mp = CurlMime()
                mp.addpart(
                    name="file",
                    content_type="application/json",
                    filename=filename,
                    local_path=filepath,
                )

                session = curl_requests.Session()
                resp = session.post(
                    self.upload_api_url,
                    multipart=mp,
                    headers={"Authorization": f"Bearer {self.upload_api_token}"},
                    verify=False,
                    timeout=30,
                )

                if resp.status_code == 200:
                    print(f"  [CPA] Token JSON 已上传到 CPA 管理平台")
                else:
                    print(f"  [CPA] 上传失败: {resp.status_code} - {resp.text[:200]}")
                
                mp.close()
                
            except ImportError:
                # 如果没有 curl_cffi，使用标准 requests

                
                with open(filepath, 'rb') as f:
                    files = {'file': (os.path.basename(filepath), f, 'application/json')}
                    headers = {"Authorization": f"Bearer {self.upload_api_token}"}
                    
                    resp = requests.post(
                        self.upload_api_url,
                        files=files,
                        headers=headers,
                        verify=False,
                        timeout=30,
                    )
                    
                    if resp.status_code == 200:
                        print(f"  [CPA] Token JSON 已上传到 CPA 管理平台")
                    else:
                        print(f"  [CPA] 上传失败: {resp.status_code} - {resp.text[:200]}")
                        
        except Exception as e:
            print(f"  [CPA] 上传异常: {e}")
