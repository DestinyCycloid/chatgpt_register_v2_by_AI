"""
核心业务逻辑模块 (RegistrationEngine & Operations)
"""
import re
import json
import time
import base64
import uuid
import logging
import threading
import urllib.parse
from typing import Optional, Tuple, Any, List, Set, Dict, Callable
from datetime import datetime, timezone, timedelta

from .utils import *
from .clients import *

logger = logging.getLogger(__name__)

# ==========================================
class EmailOperations:
    """邮箱操作类"""
    
    def __init__(self, email_service: CloudMailService, log_callback):
        self.email_service = email_service
        self._log = log_callback
        self.email: Optional[str] = None
        self.inbox_email: Optional[str] = None
        self.email_info: Optional[dict] = None
    
    def create_email(self) -> bool:
        """创建邮箱"""
        try:
            self._log(f"正在初始化 {self.email_service.service_type.value} 临时邮箱账户...")
            self.email_info = self.email_service.create_email()

            if not self.email_info or "email" not in self.email_info:
                self._log("创建邮箱失败: 返回信息不完整", "error")
                return False

            raw_email = str(self.email_info["email"] or "").strip()
            normalized_email = raw_email.lower()

            self.inbox_email = raw_email
            self.email = normalized_email
            self.email_info["email"] = normalized_email

            if raw_email and raw_email != normalized_email:
                self._log(f"邮箱规范化: {raw_email} -> {normalized_email}")

            self._log(f"邮箱创建成功并分配地址: {self.email}")
            return True

        except Exception as e:
            self._log(f"创建邮箱失败: {e}", "error")
            return False


class OTPOperations:
    """OTP 验证码操作类"""
    
    def __init__(self, engine, email_service: CloudMailService, log_callback):
        self.engine = engine  # 主引擎引用
        self.email_service = email_service
        self._log = log_callback
        self._otp_sent_at: Optional[float] = None
        self._last_otp_validation_code: Optional[str] = None
        self._last_otp_validation_status_code: Optional[int] = None
        self._last_otp_validation_outcome: str = ""
        self._last_validate_otp_continue_url: Optional[str] = None
        self._last_validate_otp_workspace_id: Optional[str] = None
    
    @property
    def session(self):
        """获取主引擎的唯一 session"""
        return self.engine.session
    
    def send_verification_code(self, referer: Optional[str] = None) -> bool:
        """发送验证码"""
        try:
            self._otp_sent_at = time.time()
            send_referer = str(referer or "https://auth.openai.com/email-verification").strip()
            
            response = self.session.get(
                OPENAI_API_ENDPOINTS["send_otp"],
                headers={
                    "referer": send_referer,
                    "accept": "application/json",
                },
            )

            self._log(f"验证码发送状态: {response.status_code}")
            return response.status_code == 200

        except Exception as e:
            self._log(f"发送验证码失败: {e}", "error")
            return False
    
    def get_verification_code(
        self,
        email: str,
        email_id: Optional[str],
        timeout: Optional[int] = None
    ) -> Optional[str]:
        """获取验证码"""
        try:
            self._log(f"正在等待邮箱 {email} 的验证码...")

            fetch_timeout = int(timeout) if timeout and int(timeout) > 0 else 120
            code = self.email_service.get_verification_code(
                email=email,
                email_id=email_id,
                timeout=fetch_timeout,
                pattern=OTP_CODE_PATTERN,
                otp_sent_at=self._otp_sent_at,
            )

            if code:
                self._log(f"成功获取验证码: {code}")
                return code
            else:
                self._log("等待验证码超时", "error")
                return None

        except Exception as e:
            self._log(f"获取验证码失败: {e}", "error")
            return None
    
    def validate_verification_code(self, code: str) -> bool:
        """验证验证码"""
        try:
            import json
            import urllib.parse as urlparse
            from typing import List, Dict, Any
            
            self._last_otp_validation_code = str(code or "").strip()
            self._last_otp_validation_status_code = None
            self._last_otp_validation_outcome = ""
            code_body = f'{{"code":"{code}"}}'

            response = self.session.post(
                OPENAI_API_ENDPOINTS["validate_otp"],
                headers={
                    "referer": "https://auth.openai.com/email-verification",
                    "accept": "application/json",
                    "content-type": "application/json",
                },
                data=code_body,
            )

            self._log(f"验证码校验状态: {response.status_code}")
            self._last_otp_validation_status_code = int(response.status_code)
            self._last_otp_validation_outcome = "success" if response.status_code == 200 else "http_non_200"
            
            if response.status_code == 200:
                try:
                    payload = response.json() or {}
                    candidates: List[Dict[str, Any]] = []
                    if isinstance(payload, dict):
                        candidates.append(payload)
                        for key in ("data", "result", "next", "payload"):
                            value = payload.get(key)
                            if isinstance(value, dict):
                                candidates.append(value)

                    found_continue = ""
                    found_workspace = ""
                    for item in candidates:
                        if not isinstance(item, dict):
                            continue
                        if not found_workspace:
                            found_workspace = str(
                                item.get("workspace_id")
                                or item.get("workspaceId")
                                or item.get("default_workspace_id")
                                or ((item.get("workspace") or {}).get("id") if isinstance(item.get("workspace"), dict) else "")
                                or ""
                            ).strip()
                        if not found_continue:
                            for key in ("continue_url", "continueUrl", "next_url", "nextUrl", "redirect_url", "redirectUrl", "url"):
                                candidate = str(item.get(key) or "").strip()
                                if not candidate:
                                    continue
                                if candidate.startswith("/"):
                                    candidate = urlparse.urljoin(OPENAI_API_ENDPOINTS["validate_otp"], candidate)
                                found_continue = candidate
                                break
                        if found_workspace and found_continue:
                            break

                    if found_workspace:
                        self._last_validate_otp_workspace_id = found_workspace
                        self._log(f"OTP 校验返回 Workspace ID: {found_workspace}")
                    if found_continue:
                        self._last_validate_otp_continue_url = found_continue
                        self._log(f"OTP 校验返回 continue_url: {found_continue[:100]}...")
                except Exception as parse_err:
                    self._log(f"解析 OTP 校验返回信息失败: {parse_err}", "warning")

            return response.status_code == 200

        except Exception as e:
            err_text = str(e or "").lower()
            if (
                "timed out" in err_text
                or "timeout" in err_text
                or "curl: (28)" in err_text
                or "operation timed out" in err_text
            ):
                self._last_otp_validation_outcome = "network_timeout"
            else:
                self._last_otp_validation_outcome = "network_error"
            self._log(f"验证验证码失败: {e}", "error")
            return False
    
    def verify_email_otp_with_retry(
        self,
        email: str,
        email_id: Optional[str],
        stage_label: str = "验证码",
        max_attempts: int = 3,
        fetch_timeout: Optional[int] = None,
        attempted_codes: Optional[Set[str]] = None,
    ) -> bool:
        """获取并校验验证码（带重试）"""
        self._last_validate_otp_continue_url = None
        self._last_validate_otp_workspace_id = None
        
        if attempted_codes is None:
            attempted_codes = set()
            
        for attempt in range(1, max_attempts + 1):
            code = self.get_verification_code(email, email_id, timeout=fetch_timeout)
            if not code:
                if attempt < max_attempts:
                    self._log(
                        f"{stage_label}第 {attempt}/{max_attempts} 次未取到验证码，稍后重试...",
                        "warning",
                    )
                    time.sleep(2)
                    continue
                return False

            if code in attempted_codes:
                allow_same_code_retry = (
                    self._last_otp_validation_code == code
                    and self._last_otp_validation_outcome in {"network_timeout", "network_error"}
                )
                if allow_same_code_retry:
                    self._log(
                        f"{stage_label}第 {attempt}/{max_attempts} 次命中重复验证码 {code}，"
                        f"但上次校验为网络异常（{self._last_otp_validation_outcome}），重试同码...",
                        "warning",
                    )
                    if self.validate_verification_code(code):
                        return True
                    if attempt < max_attempts:
                        time.sleep(2)
                        continue
                    return False

                if attempt < max_attempts:
                    self._log(
                        f"{stage_label}第 {attempt}/{max_attempts} 次命中重复验证码 {code}，等待新邮件...",
                        "warning",
                    )
                    time.sleep(2)
                    continue
                return False

            attempted_codes.add(code)

            if self.validate_verification_code(code):
                return True

            if attempt < max_attempts:
                self._log(
                    f"{stage_label}第 {attempt}/{max_attempts} 次校验未通过，疑似旧验证码，自动重试下一封...",
                    "warning",
                )
                time.sleep(2)

        return False


class AuthOperations:
    """认证操作类"""
    
    def __init__(
        self,
        engine,
        oauth_manager: OAuthManager,
        log_callback
    ):
        self.engine = engine  # 主引擎引用
        self.oauth_manager = oauth_manager
        self._log = log_callback
        
        self.oauth_start: Optional[OAuthStart] = None
        self.device_id: Optional[str] = None
        self._otp_sent_at: Optional[float] = None
    
    @property
    def session(self):
        """获取主引擎的唯一 session"""
        return self.engine.session
    
    @property
    def http_client(self):
        """获取主引擎的 http_client"""
        return self.engine.http_client
    
    def start_oauth(self) -> bool:
        """开始 OAuth 流程"""
        try:
            self._log("步骤: 启动 OAuth 登录入口流程...")
            self.oauth_start = self.oauth_manager.start_oauth()
            self._log(f"成功生成 OAuth 授权链接: {self.oauth_start.auth_url[:80]}...")
            return True
        except Exception as e:
            self._log(f"生成 OAuth URL 失败: {e}", "error")
            return False
    
    def init_session(self) -> bool:
        """初始化会话"""
        try:
            self.session = self.http_client.session
            return True
        except Exception as e:
            self._log(f"初始化会话失败: {e}", "error")
            return False
    
    def get_device_id(self) -> Optional[str]:
        """获取 Device ID"""
        if not self.oauth_start:
            return None

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                if not self.session:
                    self.engine.session = self.http_client.session

                response = self.session.get(
                    self.oauth_start.auth_url,
                    timeout=20
                )
                did = self.session.cookies.get("oai-did")

                if not did:
                    try:
                        import re
                        m = re.search(r'oai-did["\s:=]+([a-f0-9-]{36})', str(response.text or ""), re.IGNORECASE)
                        if m:
                            did = str(m.group(1) or "").strip()
                            if did:
                                try:
                                    self.session.cookies.set("oai-did", did, domain=".chatgpt.com", path="/")
                                except Exception:
                                    pass
                    except Exception:
                        pass

                if did:
                    self._log(f"Device ID: {did}")
                    self.device_id = did
                    return did

                self._log(
                    f"获取 Device ID 失败: 未返回 oai-did Cookie (HTTP {response.status_code}, 第 {attempt}/{max_attempts} 次)",
                    "warning" if attempt < max_attempts else "error"
                )
            except Exception as e:
                self._log(
                    f"获取 Device ID 失败: {e} (第 {attempt}/{max_attempts} 次)",
                    "warning" if attempt < max_attempts else "error"
                )

            if attempt < max_attempts:
                time.sleep(attempt)
                self.http_client.close()
                self.engine.session = self.http_client.session

        fallback_did = str(self.device_id or "").strip() or str(uuid.uuid4())
        try:
            if self.session:
                self.session.cookies.set("oai-did", fallback_did, domain=".chatgpt.com", path="/")
        except Exception:
            pass
        self._log(f"未获取到 oai-did，使用兜底 Device ID: {fallback_did}", "warning")
        self.device_id = fallback_did
        return fallback_did
    
    def check_sentinel(self, did: str, flow: str = "authorize_continue") -> Optional[str]:
        """检查 Sentinel 拦截"""
        try:
            sen_token = self.http_client.check_sentinel(did, flow=flow)
            if sen_token:
                self._log(f"Sentinel token 获取成功")
                return sen_token
            self._log("Sentinel 检查失败: 未获取到 token", "warning")
            return None
        except Exception as e:
            self._log(f"Sentinel 检查异常: {e}", "warning")
            return None
    
    def submit_auth_start(
        self,
        email: str,
        did: str,
        sen_token: Optional[str],
        *,
        screen_hint: str,
        referer: str,
        log_label: str,
        record_existing_account: bool = True,
    ) -> SignupFormResult:
        """提交授权入口表单"""
        max_attempts = 3
        current_did = str(did or "").strip()
        current_sen_token = str(sen_token or "").strip() if sen_token else None
        
        for attempt in range(1, max_attempts + 1):
            try:
                request_body = json.dumps({
                    "username": {
                        "value": email,
                        "kind": "email",
                    },
                    "screen_hint": screen_hint,
                })

                headers = {
                    "referer": referer,
                    "accept": "application/json",
                    "content-type": "application/json",
                }

                if current_sen_token:
                    sentinel = json.dumps({
                        "p": "",
                        "t": "",
                        "c": current_sen_token,
                        "id": current_did,
                        "flow": "authorize_continue",
                    })
                    headers["openai-sentinel-token"] = sentinel

                response = self.session.post(
                    OPENAI_API_ENDPOINTS["signup"],
                    headers=headers,
                    data=request_body,
                )

                self._log(f"{log_label}状态: {response.status_code}")

                if response.status_code == 429 and attempt < max_attempts:
                    wait_seconds = min(18, 5 * attempt)
                    self._log(
                        f"{log_label}命中限流 429（第 {attempt}/{max_attempts} 次），{wait_seconds}s 后自动重试...",
                        "warning",
                    )
                    time.sleep(wait_seconds)
                    continue

                if response.status_code == 409 and attempt < max_attempts:
                    wait_seconds = min(10, 2 * attempt)
                    self._log(
                        f"{log_label}命中 409（第 {attempt}/{max_attempts} 次），"
                        f"会话上下文可能冲突，{wait_seconds}s 后自动重试...",
                        "warning",
                    )
                    try:
                        refreshed = self.check_sentinel(current_did)
                        if refreshed:
                            current_sen_token = refreshed
                    except Exception:
                        pass
                    try:
                        if self.oauth_start and getattr(self.oauth_start, "auth_url", None):
                            self.session.get(str(self.oauth_start.auth_url), timeout=12)
                    except Exception:
                        pass
                    time.sleep(wait_seconds)
                    continue

                if response.status_code != 200:
                    return SignupFormResult(
                        success=False,
                        error_message=f"HTTP {response.status_code}: {response.text[:200]}"
                    )

                try:
                    response_data = response.json()
                    page_type = response_data.get("page", {}).get("type", "")
                    self._log(f"响应页面类型: {page_type}")

                    is_existing = page_type == OPENAI_PAGE_TYPES["EMAIL_OTP_VERIFICATION"]

                    if is_existing:
                        self._otp_sent_at = time.time()
                        if record_existing_account:
                            self._log(f"检测到已注册账号，将自动切换到登录流程")
                        else:
                            self._log("登录流程已触发，等待系统自动发送的验证码")

                    return SignupFormResult(
                        success=True,
                        page_type=page_type,
                        is_existing_account=is_existing,
                        response_data=response_data
                    )

                except Exception as parse_error:
                    self._log(f"解析响应失败: {parse_error}", "warning")
                    return SignupFormResult(success=True)

            except Exception as e:
                if attempt < max_attempts:
                    self._log(
                        f"{log_label}异常（第 {attempt}/{max_attempts} 次）: {e}，准备重试...",
                        "warning",
                    )
                    time.sleep(2 * attempt)
                    continue
                self._log(f"{log_label}失败: {e}", "error")
                return SignupFormResult(success=False, error_message=str(e))

        return SignupFormResult(success=False, error_message=f"{log_label}失败: 超过最大重试次数")
    
    def reset_auth_flow(self) -> None:
        """重置会话，准备重新发起 OAuth 流程"""
        self.http_client.close()
        self.engine.session = None
        self.oauth_start = None
        self._otp_sent_at = None
    
    def check_ip_location(self) -> Tuple[bool, Optional[str]]:
        """检查 IP 地理位置"""
        try:
            return self.http_client.check_ip_location()
        except Exception as e:
            self._log(f"检查 IP 地理位置失败: {e}", "error")
            return False, None


class LoginOperations:
    """登录专用操作类"""
    
    def __init__(self, engine, log_callback):
        self.engine = engine  # 主引擎引用
        self._log = log_callback
        self.device_id: Optional[str] = None
    
    @property
    def session(self):
        """获取主引擎的唯一 session"""
        return self.engine.session
    
    @property
    def http_client(self):
        """获取主引擎的 http_client"""
        return self.engine.http_client
    
    def submit_login_start(
        self,
        email: str,
        did: str,
        sen_token: Optional[str]
    ) -> SignupFormResult:
        """提交登录入口表单（代理到 AuthOperations 的通用方法）"""
        return self.engine.auth_ops.submit_auth_start(
            email=email,
            did=did,
            sen_token=sen_token,
            screen_hint="login",
            referer="https://auth.openai.com/log-in",
            log_label="提交登录入口",
            record_existing_account=False
        )
    
    def submit_login_password(self, password: str) -> SignupFormResult:
        """提交登录密码"""
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.session.post(
                    OPENAI_API_ENDPOINTS["password_verify"],
                    headers={
                        "referer": "https://auth.openai.com/log-in/password",
                        "accept": "application/json",
                        "content-type": "application/json",
                    },
                    data=json.dumps({"password": password}),
                )

                self._log(f"提交登录密码状态: {response.status_code}")

                if response.status_code == 429 and attempt < max_attempts:
                    wait_seconds = min(18, 5 * attempt)
                    self._log(
                        f"提交登录密码命中限流 429（第 {attempt}/{max_attempts} 次），{wait_seconds}s 后自动重试...",
                        "warning",
                    )
                    time.sleep(wait_seconds)
                    continue

                if response.status_code == 401 and attempt < max_attempts:
                    body = str(response.text or "")
                    if "invalid_username_or_password" in body:
                        wait_seconds = min(12, 3 * attempt)
                        self._log(
                            f"提交登录密码命中 401（第 {attempt}/{max_attempts} 次），"
                            f"疑似密码尚未生效或历史账号密码不一致，{wait_seconds}s 后自动重试...",
                            "warning",
                        )
                        time.sleep(wait_seconds)
                        continue

                if response.status_code != 200:
                    return SignupFormResult(
                        success=False,
                        error_message=f"HTTP {response.status_code}: {response.text[:200]}"
                    )

                response_data = response.json()
                page_type = response_data.get("page", {}).get("type", "")
                self._log(f"登录密码响应页面类型: {page_type}")

                is_existing = page_type == OPENAI_PAGE_TYPES["EMAIL_OTP_VERIFICATION"]
                if is_existing:
                    self._log("登录密码校验通过，等待系统自动发送的验证码")

                return SignupFormResult(
                    success=True,
                    page_type=page_type,
                    is_existing_account=is_existing,
                    response_data=response_data,
                )

            except Exception as e:
                if attempt < max_attempts:
                    self._log(
                        f"提交登录密码异常（第 {attempt}/{max_attempts} 次）: {e}，准备重试...",
                        "warning",
                    )
                    time.sleep(2 * attempt)
                    continue
                self._log(f"提交登录密码失败: {e}", "error")
                return SignupFormResult(success=False, error_message=str(e))

        return SignupFormResult(success=False, error_message="提交登录密码失败: 超过最大重试次数")
    
    def retrigger_login_otp(self, email: str, password: str) -> bool:
        """在登录验证码阶段重触发 OTP 发送"""
        try:
            did = str(self.device_id or self.session.cookies.get("oai-did") or "").strip()
            if not did:
                did = str(uuid.uuid4())
                try:
                    self.session.cookies.set("oai-did", did, domain=".chatgpt.com", path="/")
                except Exception:
                    pass
                self.device_id = did

            sen_token = self.http_client.check_sentinel(did)
            login_start_result = self.submit_login_start(email, did, sen_token)
            if not login_start_result.success:
                self._log(
                    f"重触发登录 OTP 失败：提交登录入口失败: {login_start_result.error_message}",
                    "warning",
                )
                return False

            page_type = str(login_start_result.page_type or "").strip()
            if page_type == OPENAI_PAGE_TYPES["EMAIL_OTP_VERIFICATION"]:
                self._log("重触发登录 OTP 成功：已直达邮箱验证码页")
                return True

            if page_type != OPENAI_PAGE_TYPES["LOGIN_PASSWORD"]:
                self._log(f"重触发登录 OTP 失败：未进入密码页（{page_type or 'unknown'}）", "warning")
                return False

            password_result = self.submit_login_password(password)
            if not password_result.success:
                self._log(f"重触发登录 OTP 失败：提交登录密码失败: {password_result.error_message}", "warning")
                return False
            if not password_result.is_existing_account:
                self._log(
                    f"重触发登录 OTP 失败：密码后未进入验证码页（{password_result.page_type or 'unknown'}）",
                    "warning",
                )
                return False

            self._log("重触发登录 OTP 成功：已进入邮箱验证码页")
            return True
        except Exception as e:
            self._log(f"重触发登录 OTP 异常: {e}", "warning")
            return False


class AccountOperations:
    """账户创建操作类"""
    
    def __init__(self, engine, log_callback):
        self.engine = engine  # 主引擎引用
        self._log = log_callback
        self._last_register_password_error: Optional[str] = None
        self._create_account_continue_url: Optional[str] = None
        self._create_account_workspace_id: Optional[str] = None
        self._create_account_account_id: Optional[str] = None
        self._create_account_refresh_token: Optional[str] = None
    
    @property
    def session(self):
        """获取主引擎的唯一 session"""
        return self.engine.session
    
    @property
    def http_client(self):
        """获取主引擎的 http_client"""
        return self.engine.http_client
    
    def register_password(
        self,
        email: str,
        password: str,
        did: Optional[str] = None,
        sen_token: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """注册密码"""
        try:
            self._last_register_password_error = None
            self._log(f"生成密码: {password}")

            register_body = json.dumps({
                "password": password,
                "username": email
            })

            response = self.session.post(
                OPENAI_API_ENDPOINTS["register"],
                headers={
                    "referer": "https://auth.openai.com/create-account/password",
                    "accept": "application/json",
                    "content-type": "application/json",
                },
                data=register_body,
            )

            self._log(f"提交密码状态: {response.status_code}")

            if response.status_code != 200:
                error_text = response.text[:500]
                self._log(f"密码注册失败: {error_text}", "warning")

                try:
                    error_json = response.json()
                    error_msg = error_json.get("error", {}).get("message", "")
                    error_code = error_json.get("error", {}).get("code", "")
                    normalized_error_msg = str(error_msg or "").strip()
                    normalized_error_code = str(error_code or "").strip()

                    if "already" in normalized_error_msg.lower() or "exists" in normalized_error_msg.lower() or normalized_error_code == "user_exists":
                        self._log(f"邮箱 {email} 可能已在 OpenAI 注册过", "error")
                        self._last_register_password_error = "该邮箱可能已在 OpenAI 注册，建议更换邮箱或改走登录流程"
                    elif "failed to register username" in normalized_error_msg.lower():
                        self._last_register_password_error = (
                            "OpenAI 拒绝当前邮箱用户名（可能已占用或触发风控），建议更换邮箱后重试"
                        )
                    else:
                        self._last_register_password_error = (
                            f"注册密码接口返回异常: {normalized_error_msg or f'HTTP {response.status_code}'}"
                        )
                except Exception:
                    self._last_register_password_error = f"注册密码接口返回异常: HTTP {response.status_code}"

                return False, None

            return True, password

        except Exception as e:
            self._log(f"密码注册失败: {e}", "error")
            self._last_register_password_error = str(e)
            return False, None
    
    def register_password_with_retry(
        self,
        email: str,
        password: str,
        did: Optional[str] = None,
        sen_token: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """带重试的密码注册"""
        max_attempts = 3
        retryable_markers = (
            "failed to create account",
            "create account",
            "invalid_request_error",
            "http 400",
        )

        for attempt in range(1, max_attempts + 1):
            success, pwd = self.register_password(email, password, did, sen_token)
            if success:
                return True, pwd

            error_text = str(self._last_register_password_error or "").strip().lower()
            if attempt >= max_attempts:
                break
            if not any(marker in error_text for marker in retryable_markers):
                break

            self._log(
                f"密码注册命中可重试 400，准备重新生成密码后重试 ({attempt}/{max_attempts})...",
                "warning",
            )
            time.sleep(min(2 * attempt, 4))

        return False, None
    
    def create_user_account(self) -> bool:
        """创建用户账户"""
        try:
            user_info = generate_random_user_info()
            self._log(f"生成用户信息: {user_info['name']}, 生日: {user_info['birthdate']}")
            
            try:
                about_you_resp = self.session.get(
                    "https://auth.openai.com/about-you",
                    headers={
                        "referer": "https://auth.openai.com/email-verification",
                        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    },
                    timeout=30
                )
                self._log(f"访问 about-you 页面: {about_you_resp.status_code}")
            except Exception as e:
                self._log(f"访问 about-you 页面失败: {e}", "warning")
            
            did = self.session.cookies.get("oai-did", "")
            sentinel_token = self.http_client.check_sentinel(did, flow="oauth_create_account") if did else None
            
            headers = {
                "referer": "https://auth.openai.com/about-you",
                "accept": "application/json",
                "content-type": "application/json",
                "origin": "https://auth.openai.com",
            }
            
            if sentinel_token:
                sentinel_header = json.dumps({
                    "p": "",
                    "t": "",
                    "c": sentinel_token,
                    "id": did,
                    "flow": "oauth_create_account"
                })
                headers["openai-sentinel-token"] = sentinel_header
                self._log("已添加 Sentinel Token 到请求头")
            else:
                self._log("⚠️ 警告: 未能获取 Sentinel Token，可能会失败", "warning")
            
            create_account_body = json.dumps(user_info)

            response = self.session.post(
                OPENAI_API_ENDPOINTS["create_account"],
                headers=headers,
                data=create_account_body,
            )

            self._log(f"账户创建状态: {response.status_code}")

            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "")
                    error_code = error_data.get("error", {}).get("code", "")
                    self._log(f"账户创建失败: {response.text}", "warning")
                    self._log(f"错误代码: {error_code}, 错误信息: {error_msg}", "warning")
                except Exception:
                    self._log(f"账户创建失败: {response.text[:500]}", "warning")
                return False

            try:
                data = response.json() or {}
                continue_url = str(data.get("continue_url") or "").strip()
                if continue_url:
                    self._create_account_continue_url = continue_url
                    self._log(f"create_account 返回 continue_url，已缓存: {continue_url[:100]}...")
                    
                account_id = str(
                    data.get("account_id")
                    or data.get("chatgpt_account_id")
                    or (data.get("account") or {}).get("id")
                    or ""
                ).strip()
                if account_id:
                    self._create_account_account_id = account_id
                    self._log(f"create_account 返回 account_id，已缓存: {account_id}")
                    
                workspace_id = str(
                    data.get("workspace_id")
                    or data.get("default_workspace_id")
                    or (data.get("workspace") or {}).get("id")
                    or ""
                ).strip()
                if (not workspace_id) and isinstance(data.get("workspaces"), list) and data.get("workspaces"):
                    workspace_id = str((data.get("workspaces")[0] or {}).get("id") or "").strip()
                if workspace_id:
                    self._create_account_workspace_id = workspace_id
                    self._log(f"create_account 返回 workspace_id，已缓存: {workspace_id}")
                else:
                    self._log("create_account 响应中未包含 workspace_id", "warning")
                    self._log(f"create_account 返回 workspace_id，已缓存: {workspace_id}")
                    
                refresh_token = str(data.get("refresh_token") or "").strip()
                if refresh_token:
                    self._create_account_refresh_token = refresh_token
                    self._log("create_account 返回 refresh_token，已缓存")
            except Exception:
                pass

            return True

        except Exception as e:
            self._log(f"创建账户失败: {e}", "error")
            return False


class WorkspaceOperations:
    """Workspace 操作类"""
    
    def __init__(self, engine, log_callback):
        self.engine = engine  # 主引擎引用
        self._log = log_callback
        self._create_account_workspace_id: Optional[str] = None
    
    @property
    def session(self):
        """获取主引擎的唯一 session"""
        return self.engine.session
    
    def get_workspace_id(self) -> Optional[str]:
        """获取 Workspace ID"""
        try:
            def _extract_workspace_id(payload: Any) -> str:
                if not isinstance(payload, dict):
                    return ""
                workspace_id = str(
                    payload.get("workspace_id")
                    or payload.get("default_workspace_id")
                    or ((payload.get("workspace") or {}).get("id") if isinstance(payload.get("workspace"), dict) else "")
                    or ""
                ).strip()
                if workspace_id:
                    return workspace_id
                workspaces = payload.get("workspaces") or []
                if isinstance(workspaces, list) and workspaces:
                    return str((workspaces[0] or {}).get("id") or "").strip()
                return ""

            auth_cookie = str(self.session.cookies.get("oai-client-auth-session") or "").strip()
            if not auth_cookie:
                self._log("未能获取到授权 Cookie，尝试从 auth-info 里取 workspace", "warning")

            try:
                candidate_payloads: List[str] = []
                if auth_cookie:
                    segments = auth_cookie.split(".")
                    if len(segments) >= 2 and segments[1]:
                        candidate_payloads.append(segments[1])
                    if segments and segments[0]:
                        candidate_payloads.append(segments[0])
                    candidate_payloads.append(auth_cookie)

                for payload in candidate_payloads:
                    raw = str(payload or "").strip()
                    if not raw:
                        continue
                    auth_json = None
                    try:
                        pad = "=" * ((4 - (len(raw) % 4)) % 4)
                        decoded = base64.urlsafe_b64decode((raw + pad).encode("ascii"))
                        auth_json = json.loads(decoded.decode("utf-8"))
                    except Exception:
                        try:
                            auth_json = json.loads(raw)
                        except Exception:
                            auth_json = None

                    workspace_id = _extract_workspace_id(auth_json)
                    if workspace_id:
                        self._log(f"Workspace ID: {workspace_id}")
                        return workspace_id

                auth_info_raw = str(self.session.cookies.get("oai-client-auth-info") or "").strip()
                if auth_info_raw:
                    auth_info_text = auth_info_raw
                    for _ in range(2):
                        decoded = urlparse.unquote(auth_info_text)
                        if decoded == auth_info_text:
                            break
                        auth_info_text = decoded
                    try:
                        auth_info_json = json.loads(auth_info_text)
                        workspace_id = _extract_workspace_id(auth_info_json)
                        if workspace_id:
                            self._log(f"Workspace ID (auth-info): {workspace_id}")
                            return workspace_id
                    except Exception as auth_info_err:
                        self._log(f"解析 auth-info Cookie失败: {auth_info_err}", "warning")

                cached_workspace = str(self._create_account_workspace_id or "").strip()
                if cached_workspace:
                    self._log(f"Workspace ID (create_account缓存): {cached_workspace}")
                    return cached_workspace

                self._log("授权 Cookie 里没有 workspace 信息", "warning")
                return None

            except Exception as e:
                self._log(f"解析授权 Cookie 失败: {e}", "warning")
                return None

        except Exception as e:
            self._log(f"获取 Workspace ID 失败: {e}", "error")
            return None
    
    def select_workspace(self, workspace_id: str) -> Optional[str]:
        """选择 Workspace"""
        try:
            select_body = f'{{"workspace_id":"{workspace_id}"}}'

            response = self.session.post(
                OPENAI_API_ENDPOINTS["select_workspace"],
                headers={
                    "referer": "https://auth.openai.com/sign-in-with-chatgpt/codex/consent",
                    "content-type": "application/json",
                    "accept": "application/json",
                },
                data=select_body,
                allow_redirects=False,
            )

            location = str(response.headers.get("Location") or "").strip()
            if response.status_code in [301, 302, 303, 307, 308] and location:
                continue_url = urlparse.urljoin(OPENAI_API_ENDPOINTS["select_workspace"], location)
                self._log(f"Continue URL (Location): {continue_url[:100]}...")
                return continue_url

            if response.status_code != 200:
                self._log(f"选择 workspace 失败: {response.status_code}", "error")
                self._log(f"响应: {response.text[:200]}", "warning")
                return None

            continue_url = ""
            try:
                continue_url = str((response.json() or {}).get("continue_url") or "").strip()
            except Exception as json_err:
                body_text = str(response.text or "")
                self._log(f"workspace/select 非 JSON 响应，尝试文本兜底解析: {json_err}", "warning")
                m = re.search(r'"continue_url"\s*:\s*"([^"]+)"', body_text)
                if m:
                    continue_url = str(m.group(1) or "").strip()
                if not continue_url:
                    m2 = re.search(r"https://auth\.openai\.com/[^\s\"'<>]+", body_text)
                    if m2:
                        continue_url = str(m2.group(0) or "").strip()

            if not continue_url:
                if location:
                    continue_url = urlparse.urljoin(OPENAI_API_ENDPOINTS["select_workspace"], location)
                else:
                    self._log("workspace/select 响应里缺少 continue_url", "error")
                    return None

            if continue_url:
                continue_url = continue_url.replace("\\/", "/")
                self._log(f"Continue URL: {continue_url[:100]}...")
                return continue_url

            return None

        except Exception as e:
            self._log(f"选择 Workspace 失败: {e}", "error")
            return None


class RedirectOperations:
    """重定向处理类"""
    
    def __init__(self, engine, log_callback):
        self.engine = engine  # 主引擎引用
        self._log = log_callback
    
    @property
    def session(self):
        """获取主引擎的唯一 session"""
        return self.engine.session
    
    def follow_redirects(self, start_url: str) -> Tuple[Optional[str], str]:
        """手动跟随重定向链，返回 (callback_url, final_url)"""
        try:
            def _is_oauth_callback(url: str) -> bool:
                try:
                    parsed = urllib.parse.urlparse(url)
                    path = (parsed.path or "").lower()
                    if ("/auth/callback" not in path) and ("/api/auth/callback/openai" not in path):
                        return False
                    query = urllib.parse.parse_qs(parsed.query or "", keep_blank_values=True)
                    return bool(query.get("code") or query.get("error"))
                except Exception:
                    return False

            current_url = start_url
            callback_url: Optional[str] = None
            max_redirects = 12

            for i in range(max_redirects):
                self._log(f"重定向 {i+1}/{max_redirects}: {current_url[:100]}...")
                if _is_oauth_callback(current_url) and not callback_url:
                    callback_url = current_url
                    self._log(f"命中回调 URL: {current_url[:120]}...")
                    break

                response = self.session.get(
                    current_url,
                    allow_redirects=False,
                    timeout=15
                )

                location = response.headers.get("Location") or ""

                if "/api/auth/callback/openai" in current_url and not callback_url:
                    callback_url = current_url

                if response.status_code not in [301, 302, 303, 307, 308]:
                    self._log(f"非重定向状态码: {response.status_code}")
                    break

                if not location:
                    self._log("重定向响应缺少 Location 头")
                    break

                next_url = urllib.parse.urljoin(current_url, location)

                if _is_oauth_callback(next_url) and not callback_url:
                    callback_url = next_url
                    self._log(f"找到回调 URL: {next_url[:100]}...")
                    current_url = next_url
                    break

                current_url = next_url

            try:
                if not current_url.rstrip("/").endswith("chatgpt.com"):
                    self.session.get(
                        "https://chatgpt.com/",
                        headers={
                            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                            "referer": current_url,
                            "user-agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                            ),
                        },
                        timeout=20,
                    )
            except Exception as home_err:
                self._log(f"重定向结束后首页补跳异常: {home_err}", "warning")

            if not callback_url:
                self._log("未能在重定向链中找到回调 URL", "warning")
            return callback_url, current_url

        except Exception as e:
            self._log(f"跟随重定向失败: {e}", "error")
            return None, start_url


class TokenOperations:
    """Token 操作类"""
    
    def __init__(self, engine, oauth_manager: OAuthManager, log_callback):
        self.engine = engine  # 主引擎引用
        self.oauth_manager = oauth_manager
        self._log = log_callback
        self.oauth_start: Optional[OAuthStart] = None
    
    @property
    def session(self):
        """获取主引擎的唯一 session"""
        return self.engine.session
    
    def capture_auth_session_tokens(
        self,
        result: RegistrationResult,
        access_hint: Optional[str] = None
    ) -> bool:
        """通过 /api/auth/session 捕获 session_token + access_token"""
        access_token = str(access_hint or "").strip()
        set_cookie_text = ""
        request_cookie_text = ""
        
        try:
            headers = {
                "accept": "application/json",
                "referer": "https://chatgpt.com/",
                "origin": "https://chatgpt.com",
                "user-agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                "cache-control": "no-cache",
                "pragma": "no-cache",
            }
            if access_token:
                headers["authorization"] = f"Bearer {access_token}"
                
            response = self.session.get(
                "https://chatgpt.com/api/auth/session",
                headers=headers,
                timeout=20,
            )
            
            set_cookie_text = flatten_set_cookie_headers(response)
            request_cookie_text = extract_request_cookie_header(response)
            
            if response.status_code == 200:
                try:
                    data = response.json() or {}
                    access_from_json = str(data.get("accessToken") or "").strip()
                    if access_from_json:
                        access_token = access_from_json
                except Exception:
                    pass
            else:
                self._log(f"/api/auth/session 返回异常状态: {response.status_code}", "warning")
        except Exception as e:
            self._log(f"获取 auth/session 失败: {e}", "warning")

        session_token = extract_session_token_from_cookie_jar(self.session.cookies)

        if not session_token:
            session_token = extract_session_token_from_cookie_text(dump_session_cookies(self.session))

        if not session_token and set_cookie_text:
            session_token = extract_session_token_from_cookie_text(set_cookie_text)

        if not session_token and request_cookie_text:
            session_token = extract_session_token_from_cookie_text(request_cookie_text)

        if (not session_token) and access_token:
            try:
                retry_response = self.session.get(
                    "https://chatgpt.com/api/auth/session",
                    headers={
                        "accept": "application/json",
                        "referer": "https://chatgpt.com/",
                        "origin": "https://chatgpt.com",
                        "user-agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                        ),
                        "authorization": f"Bearer {access_token}",
                        "cache-control": "no-cache",
                        "pragma": "no-cache",
                    },
                    timeout=20,
                )
                retry_set_cookie = flatten_set_cookie_headers(retry_response)
                retry_request_cookie = extract_request_cookie_header(retry_response)
                
                if not session_token:
                    session_token = extract_session_token_from_cookie_jar(self.session.cookies)
                if not session_token:
                    session_token = extract_session_token_from_cookie_text(dump_session_cookies(self.session))
                if not session_token and retry_set_cookie:
                    session_token = extract_session_token_from_cookie_text(retry_set_cookie)
                if not session_token and retry_request_cookie:
                    session_token = extract_session_token_from_cookie_text(retry_request_cookie)
            except Exception as e:
                self._log(f"Bearer 兜底换 session_token 失败: {e}", "warning")

        did = ""
        try:
            did = str(self.session.cookies.get("oai-did") or "").strip()
        except Exception:
            did = ""
        if did:
            result.device_id = did

        if session_token:
            result.session_token = session_token
        if access_token:
            result.access_token = access_token

        self._log(
            "Auth Session 捕获结果: session_token="
            + ("有" if bool(result.session_token) else "无")
            + ", access_token="
            + ("有" if bool(result.access_token) else "无")
        )
        return bool(result.session_token and result.access_token)
    
    def handle_oauth_callback(self, callback_url: str) -> Optional[Dict[str, Any]]:
        """处理 OAuth 回调"""
        try:
            if not self.oauth_start:
                self._log("OAuth 流程未初始化", "error")
                return None

            self._log("正在校验 OAuth Callback 参数状态...")
            token_info = self.oauth_manager.handle_callback(
                callback_url=callback_url,
                expected_state=self.oauth_start.state,
                code_verifier=self.oauth_start.code_verifier
            )

            self._log("OAuth 授权完成，成功提取全部 Token 数据")
            return token_info

        except Exception as e:
            self._log(f"处理 OAuth 回调失败: {e}", "error")
            return None

    



"""
Token 管理模块 - 保存和上传 Token
"""




# 全局文件锁
# (Moved _file_lock to clients.py)




class RegistrationEngine:
    """
    注册引擎 - 重构版
    负责协调各个子模块完成注册流程
    """

    def __init__(
        self,
        email_service: CloudMailService,
        proxy_url: Optional[str] = None,
        callback_logger: Optional[Callable[[str], None]] = None,
        task_uuid: Optional[str] = None
    ):
        """初始化注册引擎"""
        self.email_service = email_service
        self.proxy_url = proxy_url
        self.callback_logger = callback_logger or (lambda msg: logger.info(msg))
        self.task_uuid = task_uuid

        # 创建 HTTP 客户端
        self.http_client = OpenAIHTTPClient(proxy_url=proxy_url)

        # 创建 OAuth 管理器
        self.oauth_manager = OAuthManager(
            client_id='app_EMoamEEZ73f0CkXaXp7hrann',
            auth_url='https://auth.openai.com/oauth/authorize',
            token_url='https://auth.openai.com/oauth/token',
            redirect_uri='http://localhost:1455/auth/callback',
            scope='openid email profile offline_access',
            proxy_url=proxy_url
        )

        # 状态变量
        self.email: Optional[str] = None
        self.password: Optional[str] = None
        self.device_id: Optional[str] = None
        self.session: Optional[object] = None  # 唯一的会话对象
        self.session_token: Optional[str] = None
        self.logs: list = []
        self._is_existing_account: bool = False

        # 创建子模块 - 传入 self 以便访问唯一的 session
        self.email_ops = EmailOperations(email_service, self._log)
        self.auth_ops = AuthOperations(self, self.oauth_manager, self._log)
        self.otp_ops = OTPOperations(self, email_service, self._log)
        self.account_ops = AccountOperations(self, self._log)
        self.workspace_ops = WorkspaceOperations(self, self._log)
        self.redirect_ops = RedirectOperations(self, self._log)
        self.token_ops = TokenOperations(self, self.oauth_manager, self._log)
        self.login_ops = LoginOperations(self, self._log)


    def _log(self, message: str, level: str = "info"):
        """记录日志"""
        log_message = format_log_message(message)
        self.logs.append(log_message)

        if self.callback_logger:
            self.callback_logger(log_message)

        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)

    def _init_session(self):
        """初始化会话 - 对齐原始代码，只维护一个 session"""
        self.session = self.http_client.session
        return self.session

    def _prepare_authorize_flow(self, label: str):
        """初始化授权流程"""
        self._log(f"{label}: 初始化 HTTP 会话...")
        session = self._init_session()
        if not session:
            return None, None

        self._log(f"{label}: 启动 OAuth 授权流程...")
        if not self.auth_ops.start_oauth():
            return None, None

        self._log(f"{label}: 提取并校验 Device ID...")
        did = self.auth_ops.get_device_id()
        if not did:
            return None, None

        self.device_id = did
        self.token_ops.oauth_start = self.auth_ops.oauth_start
        self.login_ops.device_id = did

        self._log(f"{label}: 请求并验证 Sentinel POW Token...")
        sen_token = self.auth_ops.check_sentinel(did)
        if not sen_token:
            return did, None

        self._log(f"{label}: Sentinel POW 验证通过")
        return did, sen_token


    def run(self) -> RegistrationResult:
        """执行完整的注册流程"""
        result = RegistrationResult(success=False, logs=self.logs)

        try:
            self._log("=" * 60)
            self._log("注册流程启动")
            self._log("=" * 60)

            # 1. 检查 IP 地理位置
            self._log("步骤 1: 检查客户端 IP 及地理位置...")
            ip_ok, location = self.auth_ops.check_ip_location()
            if not ip_ok:
                result.error_message = f"IP 地理位置不支持: {location}"
                self._log(f"IP 检查失败: {location}", "error")
                return result
            self._log(f"IP 位置: {location}")

            # 2. 创建邮箱
            self._log("步骤 2: 创建临时邮箱账户...")
            if not self.email_ops.create_email():
                result.error_message = "创建邮箱失败"
                return result
            self.email = self.email_ops.email
            result.email = self.email

            # 3. 准备授权流程
            did, sen_token = self._prepare_authorize_flow("首次授权")
            if not did:
                result.error_message = "获取 Device ID 失败"
                return result
            result.device_id = did
            if not sen_token:
                result.error_message = "Sentinel POW 验证失败"
                return result

            # 4. 提交注册入口邮箱
            self._log("步骤 4: 提交邮箱验证并获取授权状态...")
            signup_result = self.auth_ops.submit_auth_start(
                email=self.email,
                did=did,
                sen_token=sen_token,
                screen_hint="signup",
                referer="https://auth.openai.com/create-account",
                log_label="提交注册表单"
            )
            if not signup_result.success:
                result.error_message = f"提交注册表单失败: {signup_result.error_message}"
                return result

            # 检查是否为已注册账号
            if signup_result.is_existing_account:
                self._is_existing_account = True
                self._log("检测到账号已注册，自动切换至登录流程...")
                return self._handle_existing_account(result, did, sen_token)

            # 5. 新账号注册流程
            return self._handle_new_account_registration(result, did, sen_token)

        except Exception as e:
            self._log(f"注册过程中发生未预期错误: {e}", "error")
            result.error_message = str(e)
            return result


    def _handle_new_account_registration(self, result: RegistrationResult, did: str, sen_token: Optional[str]) -> RegistrationResult:
        """处理新账号注册流程"""
        # 5. 设置密码
        self._log("步骤 5: 设置账户登录密码...")
        self.password = generate_password()
        password_ok, _ = self.account_ops.register_password_with_retry(
            email=self.email,
            password=self.password,
            did=did,
            sen_token=sen_token
        )
        if not password_ok:
            result.error_message = self.account_ops._last_register_password_error or "注册密码失败"
            return result

        # 6. 发送验证码
        self._log("步骤 6: 请求发送注册验证码...")
        if not self.otp_ops.send_verification_code():
            result.error_message = "发送验证码失败"
            return result

        # 7-8. 获取并验证验证码
        self._log("步骤 7: 等待接收验证码...")
        self._log("步骤 8: 校验接收到的验证码...")
        if not self.otp_ops.verify_email_otp_with_retry(
            email=self.email_ops.inbox_email or self.email,
            email_id=self.email_ops.email_info.get("service_id") if self.email_ops.email_info else None,
            stage_label="注册验证码",
            max_attempts=3
        ):
            result.error_message = "验证验证码失败"
            return result

        # 9. 创建用户账户
        self._log("步骤 9: 提交并完成用户注册...")
        if not self.account_ops.create_user_account():
            result.error_message = "创建用户账户失败"
            return result

        # 10. 新账号需要重新登录来获取 token（关键步骤！）
        self._log("注册成功，开始执行自动登录流程获取 Token...")
        login_ready, login_error = self._restart_login_flow()
        if not login_ready:
            result.error_message = login_error
            return result

        # 11. 完成 token 交换
        return self._complete_registration(result)

    def _restart_login_flow(self) -> tuple[bool, str]:
        """新注册账号完成建号后，重新发起一次登录流程拿 token"""
        self._log("重新登录流程开始...")
        
        # 重置会话（reset_auth_flow 内部会调用 http_client.close()）
        self.auth_ops.reset_auth_flow()
        
        # 重新初始化会话，获取新的 session 对象并更新所有操作类
        self._init_session()

        # 准备新的授权流程
        did, sen_token = self._prepare_authorize_flow("重新登录")
        if not did:
            return False, "重新登录时获取 Device ID 失败"
        if not sen_token:
            return False, "重新登录时 Sentinel POW 验证失败"

        # 提交登录入口
        login_start_result = self.login_ops.submit_login_start(self.email, did, sen_token)
        if not login_start_result.success:
            return False, f"重新登录提交邮箱失败: {login_start_result.error_message}"
        
        from .utils import OPENAI_PAGE_TYPES
        if login_start_result.page_type != OPENAI_PAGE_TYPES["LOGIN_PASSWORD"]:
            return False, f"重新登录未进入密码页面: {login_start_result.page_type or 'unknown'}"

        # 提交登录密码
        password_result = self.login_ops.submit_login_password(self.password)
        if not password_result.success:
            return False, f"重新登录提交密码失败: {password_result.error_message}"
        if not password_result.is_existing_account:
            return False, f"重新登录未进入验证码页面: {password_result.page_type or 'unknown'}"
        
        return True, ""

    def _handle_existing_account(self, result: RegistrationResult, did: str, sen_token: Optional[str]) -> RegistrationResult:
        """处理已存在账号的登录流程"""
        # 对于已存在账号，需要提交密码并验证
        self._log("已注册账号流程：提交登录密码...")
        
        # 尝试从数据库获取密码
        if not self.password:
            try:
                with get_db() as db:
                    account = crud.get_account_by_email(db, self.email)
                    if account:
                        self.password = getattr(account, "password", "")
            except Exception:
                pass

        if not self.password:
            result.error_message = "已注册账号但未找到密码"
            return result

        password_result = self.login_ops.submit_login_password(self.password)
        if not password_result.success:
            result.error_message = f"提交登录密码失败: {password_result.error_message}"
            return result

        # 验证登录 OTP
        if not self.otp_ops.verify_email_otp_with_retry(
            email=self.email_ops.inbox_email or self.email,
            email_id=self.email_ops.email_info.get("service_id") if self.email_ops.email_info else None,
            stage_label="登录验证码",
            max_attempts=3
        ):
            result.error_message = "登录验证码校验失败"
            return result

        return self._complete_registration(result)


    def _complete_registration(self, result: RegistrationResult) -> RegistrationResult:
        """完成注册流程，获取 tokens - 对齐原生入口链路"""
        def _is_registration_gate_url(url: str) -> bool:
            """检查是否为注册门页（add-phone/about-you）"""
            u = str(url or "").strip().lower()
            if not u:
                return False
            return ("auth.openai.com/about-you" in u) or ("auth.openai.com/add-phone" in u)

        # 先验证登录 OTP
        self._log("步骤: 阻塞等待登录阶段的验证码...")
        self._log("步骤: 校验获取的登录验证码...")
        login_otp_tried_codes: set = set()
        login_otp_ok = self.otp_ops.verify_email_otp_with_retry(
            email=self.email_ops.inbox_email or self.email,
            email_id=self.email_ops.email_info.get("service_id") if self.email_ops.email_info else None,
            stage_label="登录验证码",
            max_attempts=1,
            fetch_timeout=120,
            attempted_codes=login_otp_tried_codes,
        )
        
        if not login_otp_ok:
            self._log("登录验证码首轮未命中，尝试在当前会话原地重发 OTP 后再校验...", "warning")
            resent = self.otp_ops.send_verification_code(referer="https://auth.openai.com/email-verification")
            if resent:
                login_otp_ok = self.otp_ops.verify_email_otp_with_retry(
                    email=self.email_ops.inbox_email or self.email,
                    email_id=self.email_ops.email_info.get("service_id") if self.email_ops.email_info else None,
                    stage_label="登录验证码(原地重发)",
                    max_attempts=2,
                    fetch_timeout=120,
                    attempted_codes=login_otp_tried_codes,
                )

        if not login_otp_ok:
            self._log("登录验证码仍未命中，尝试重触发登录 OTP 后再校验...", "warning")
            if not self.login_ops.retrigger_login_otp(self.email, self.password):
                self._log("重触发登录 OTP 失败，尝试完整重登链路后再校验一次...", "warning")
                login_ready, login_error = self._restart_login_flow()
                if not login_ready:
                    result.error_message = f"登录验证码重触发失败，且完整重登失败: {login_error}"
                    return result
            login_otp_ok = self.otp_ops.verify_email_otp_with_retry(
                email=self.email_ops.inbox_email or self.email,
                email_id=self.email_ops.email_info.get("service_id") if self.email_ops.email_info else None,
                stage_label="登录验证码(重发)",
                max_attempts=3,
                fetch_timeout=120,
                attempted_codes=login_otp_tried_codes,
            )
            if not login_otp_ok:
                result.error_message = "验证码校验失败"
                return result

        # 获取 Workspace ID
        self._log("步骤: 获取当前账户关联的 Workspace ID...")
        workspace_id = str(self.otp_ops._last_validate_otp_workspace_id or "").strip()
        if workspace_id:
            self._log(f"使用 OTP 返回的 Workspace ID: {workspace_id}")
        if not workspace_id:
            self.workspace_ops._create_account_workspace_id = self.account_ops._create_account_workspace_id
            workspace_id = str(self.workspace_ops.get_workspace_id() or "").strip()
        if workspace_id:
            result.workspace_id = workspace_id

        # 获取 continue_url，过滤注册门页
        continue_url = ""
        otp_continue = str(self.otp_ops._last_validate_otp_continue_url or "").strip()
        if otp_continue and _is_registration_gate_url(otp_continue):
            self._log("OTP 返回 continue_url 指向注册门页（about-you/add-phone），本轮收尾忽略该地址", "warning")
            otp_continue = ""

        cached_continue = str(self.account_ops._create_account_continue_url or "").strip()
        if cached_continue and _is_registration_gate_url(cached_continue):
            self._log("create_account 缓存 continue_url 指向注册门页（about-you/add-phone），本轮收尾忽略该地址", "warning")
            cached_continue = ""

        # 优先使用 workspace/select
        if workspace_id:
            self._log("步骤: 发起 Workspace 选择确认请求...")
            continue_url = str(self.workspace_ops.select_workspace(workspace_id) or "").strip()
            if not continue_url:
                self._log("workspace/select 未返回 continue_url，尝试 OAuth authorize 兜底", "warning")

        # 兜底1: OAuth authorize URL
        if not continue_url:
            oauth_start_url = str(
                (
                    getattr(self.auth_ops.oauth_start, "auth_url", "")
                    or getattr(self.auth_ops.oauth_start, "url", "")
                    if self.auth_ops.oauth_start
                    else ""
                )
                or ""
            ).strip()
            if oauth_start_url:
                continue_url = oauth_start_url
                self._log("使用 OAuth authorize URL 作为兜底 continue_url", "warning")

        # 兜底2: OTP continue_url（已过滤门页）
        if not continue_url and otp_continue:
            continue_url = otp_continue
            self._log("使用 OTP 返回 continue_url 继续授权链路", "warning")

        # 兜底3: create_account continue_url（已过滤门页）
        if not continue_url and cached_continue:
            continue_url = cached_continue
            self._log("使用 create_account 缓存 continue_url 作为兜底", "warning")

        if not continue_url:
            result.error_message = "获取 continue_url 失败"
            return result

        # 跟随重定向
        self._log("步骤: 跟随 OAuth 重定向链条追踪返回结果...")
        callback_url, _final_url = self.redirect_ops.follow_redirects(continue_url)
        
        if not callback_url:
            self._log("未命中 OAuth 回调，尝试 auth/session 兜底抓取 token...", "warning")
            self.token_ops.capture_auth_session_tokens(result, access_hint=result.access_token)
            if not result.account_id:
                result.account_id = str(self.account_ops._create_account_account_id or "").strip()
            if not result.workspace_id:
                result.workspace_id = str(workspace_id or self.account_ops._create_account_workspace_id or "").strip()
            if not result.refresh_token:
                result.refresh_token = str(self.account_ops._create_account_refresh_token or "").strip()
            if result.access_token:
                result.password = self.password or ""
                result.source = "login" if self._is_existing_account else "register"
                result.device_id = result.device_id or str(self.device_id or "")
                self._log("未命中 callback，已通过 auth/session 兜底拿到 Access Token，继续完成注册", "warning")
                self._finalize_result(result)
                return result

            # 对新注册账号放宽：账号已创建成功时允许"注册成功、token 待补"
            if (not self._is_existing_account) and self.account_ops._create_account_account_id:
                result.account_id = result.account_id or str(self.account_ops._create_account_account_id or "").strip()
                result.workspace_id = result.workspace_id or str(workspace_id or self.account_ops._create_account_workspace_id or "").strip()
                result.refresh_token = result.refresh_token or str(self.account_ops._create_account_refresh_token or "").strip()
                result.password = self.password or ""
                result.source = "register"
                result.device_id = result.device_id or str(self.device_id or "")
                self._log("回调链路未命中且未抓到 Access Token，但账号已创建成功；按注册成功收尾（token 待后续补齐）", "warning")
                self._finalize_result(result)
                return result

            result.error_message = "跟随重定向链失败"
            return result

        # 处理 OAuth 回调
        self._log("步骤: 解析 OAuth 回调数据获取 Token...")
        token_info = self.token_ops.handle_oauth_callback(callback_url)
        if not token_info:
            if (not self._is_existing_account) and self.account_ops._create_account_account_id:
                result.account_id = result.account_id or str(self.account_ops._create_account_account_id or "").strip()
                result.workspace_id = result.workspace_id or str(workspace_id or self.account_ops._create_account_workspace_id or "").strip()
                result.refresh_token = result.refresh_token or str(self.account_ops._create_account_refresh_token or "").strip()
                result.password = self.password or ""
                result.source = "register"
                result.device_id = result.device_id or str(self.device_id or "")
                self._log("OAuth 回调处理失败，但账号已创建成功；按注册成功收尾（token 待后续补齐）", "warning")
                self._finalize_result(result)
                return result
            result.error_message = "处理 OAuth 回调失败"
            return result

        result.account_id = token_info.get("account_id", "")
        result.access_token = token_info.get("access_token", "")
        result.refresh_token = token_info.get("refresh_token", "")
        result.id_token = token_info.get("id_token", "")
        result.password = self.password or ""
        result.source = "login" if self._is_existing_account else "register"
        result.device_id = result.device_id or str(self.device_id or "")

        session_cookie = self.http_client.session.cookies.get("__Secure-next-auth.session-token")
        if session_cookie:
            self.session_token = session_cookie
            result.session_token = session_cookie
            self._log("成功提取到 Session Token")

        self._finalize_result(result)
        return result

    def _finalize_result(self, result: RegistrationResult):
        """完成注册结果的最终设置"""
        self._log("=" * 60)
        if self._is_existing_account:
            self._log("登录流程成功完成")
        else:
            self._log("注册流程成功完成")
        self._log(f"邮箱: {result.email}")
        self._log(f"Account ID: {result.account_id}")
        self._log(f"Workspace ID: {result.workspace_id}")
        self._log("=" * 60)

        result.success = True
        result.metadata = {
            "email_service": self.email_service.service_type.value,
            "proxy_used": self.proxy_url,
            "registered_at": datetime.now().isoformat(),
            "is_existing_account": self._is_existing_account,
            "has_session_token": bool(result.session_token),
            "has_access_token": bool(result.access_token),
            "has_refresh_token": bool(result.refresh_token),
        }

