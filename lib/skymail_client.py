"""
Skymail 邮箱客户端模块
"""

import sys
import time
import random
import string
import re


class SkymailClient:
    """Skymail 邮箱服务客户端"""

    def __init__(self, admin_email, admin_password, api_base=None, proxy=None, domains=None):
        """
        初始化 Skymail 客户端
        
        Args:
            admin_email: 管理员邮箱
            admin_password: 管理员密码
            api_base: API 基础地址（可选，默认从邮箱域名提取）
            proxy: 代理地址（可选）
            domains: 可用域名列表（必须）
        """
        self.admin_email = admin_email
        self.admin_password = admin_password
        
        # 从管理员邮箱提取 API 域名
        if api_base:
            self.api_base = api_base.rstrip("/")
        elif admin_email and "@" in admin_email:
            self.api_base = f"https://{admin_email.split('@')[1]}"
        else:
            self.api_base = ""
        
        self.proxy = proxy
        self.api_token = None
        
        # 可用域名列表（必须从配置文件传入）
        if not domains or not isinstance(domains, list) or len(domains) == 0:
            raise Exception("❌ 错误: 未配置 skymail_domains，请在 config.json 中设置域名列表")
        
        self.domains = domains

    def generate_token(self):
        """自动生成 Skymail API Token"""
        if not self.admin_email or not self.admin_password:
            print("⚠️ 未配置 Skymail 管理员账号")
            return None
        
        if not self.api_base:
            print("⚠️ 无法从管理员邮箱提取 API 域名")
            return None
        
        try:
            import requests
            
            session = requests.Session()
            if self.proxy:
                session.proxies = {"http": self.proxy, "https": self.proxy}
            
            res = session.post(
                f"{self.api_base}/api/public/genToken",
                json={
                    "email": self.admin_email,
                    "password": self.admin_password
                },
                headers={"Content-Type": "application/json"},
                timeout=15,
                verify=False
            )
            
            if res.status_code == 200:
                data = res.json()
                if data.get("code") == 200:
                    token = data.get("data", {}).get("token")
                    if token:
                        print(f"✅ 成功生成 Skymail API Token")
                        self.api_token = token
                        return token
            
            print(f"⚠️ 生成 Skymail Token 失败: {res.status_code} - {res.text[:200]}")
        except Exception as e:
            print(f"⚠️ 生成 Skymail Token 异常: {e}")
        
        return None

    def create_temp_email(self):
        """
        创建 Skymail 临时邮箱
        
        Returns:
            tuple: (email, email) - 邮箱地址和 token（在 Skymail 中 token 就是邮箱地址）
        """
        if not self.api_token:
            raise Exception("SKYMAIL_API_TOKEN 未设置，无法创建临时邮箱")

        try:
            # 随机选择一个域名
            domain = random.choice(self.domains)
            
            # 生成随机前缀（6-10位字母数字组合）
            prefix_length = random.randint(6, 10)
            prefix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=prefix_length))
            
            email = f"{prefix}@{domain}"
            
            # Skymail API 不需要预先创建邮箱，直接使用即可
            return email, email

        except Exception as e:
            raise Exception(f"Skymail 创建邮箱失败: {e}")

    def fetch_emails(self, email):
        """
        从 Skymail 获取邮件列表
        
        Args:
            email: 邮箱地址
            
        Returns:
            list: 邮件列表
        """
        try:
            import requests
            
            session = requests.Session()
            if self.proxy:
                session.proxies = {"http": self.proxy, "https": self.proxy}

            res = session.post(
                f"{self.api_base}/api/public/emailList",
                json={
                    "toEmail": email,
                    "timeSort": "desc",
                    "num": 1,
                    "size": 20
                },
                headers={
                    "Authorization": self.api_token,
                    "Content-Type": "application/json"
                },
                timeout=15,
                verify=False
            )

            if res.status_code == 200:
                data = res.json()
                if data.get("code") == 200:
                    return data.get("data", [])
            return []
        except Exception:
            return []

    def extract_verification_code(self, content):
        """从邮件内容提取6位验证码"""
        if not content:
            return None

        patterns = [
            r"Verification code:?\s*(\d{6})",
            r"code is\s*(\d{6})",
            r"代码为[:：]?\s*(\d{6})",
            r"验证码[:：]?\s*(\d{6})",
            r">\s*(\d{6})\s*<",
            r"(?<![#&])\b(\d{6})\b",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for code in matches:
                if code == "177010":  # 已知误判
                    continue
                return code
        return None

    def wait_for_verification_code(self, email, timeout=30, exclude_codes=None):
        """
        等待验证邮件并提取验证码
        
        Args:
            email: 邮箱地址
            timeout: 超时时间（秒）
            exclude_codes: 要排除的验证码集合（避免重复使用）
            
        Returns:
            str: 验证码，失败返回 None
        """
        if exclude_codes is None:
            exclude_codes = set()
        
        # 合并实例级别的已使用验证码
        if not hasattr(self, '_used_codes'):
            self._used_codes = set()
        all_exclude_codes = exclude_codes | self._used_codes
        
        print(f"  ⏳ 等待验证码 (最大 {timeout}s)...")
        
        start = time.time()
        last_email_ids = set()
        
        # 立即开始轮询
        while time.time() - start < timeout:
            emails = self.fetch_emails(email)
            
            if emails:
                for item in emails:
                    if not isinstance(item, dict):
                        continue
                    
                    email_id = item.get("emailId")
                    if not email_id or email_id in last_email_ids:
                        continue
                    
                    # 记录这个邮件 ID
                    last_email_ids.add(email_id)
                    
                    # 提取验证码
                    content = item.get("content") or item.get("text") or ""
                    code = self.extract_verification_code(content)
                    
                    if code and code not in all_exclude_codes:
                        print(f"  ✅ 验证码: {code}")
                        # 记录已使用的验证码
                        self._used_codes.add(code)
                        return code
            
            # 动态等待时间：前10秒快速轮询（0.5秒），之后慢速轮询（2秒）
            elapsed = time.time() - start
            if elapsed < 10:
                time.sleep(0.5)
            else:
                time.sleep(2)
        
        print("  ⏰ 等待验证码超时")
        return None


def init_skymail_client(config):
    """
    初始化 Skymail 客户端并生成 Token
    
    Args:
        config: 配置字典
        
    Returns:
        SkymailClient: 初始化好的客户端实例
    """
    admin_email = config.get("skymail_admin_email", "")
    admin_password = config.get("skymail_admin_password", "")
    proxy = config.get("proxy", "")
    domains = config.get("skymail_domains", None)
    api_base = config.get("skymail_api_base", None)

    if not admin_email or not admin_password:
        print("❌ 错误: 未配置 Skymail 管理员账号")
        print("   请在 config.json 中设置 skymail_admin_email 和 skymail_admin_password")
        sys.exit(1)
    
    if not domains or not isinstance(domains, list) or len(domains) == 0:
        print("❌ 错误: 未配置 skymail_domains")
        print("   请在 config.json 中设置域名列表，例如: \"skymail_domains\": [\"admin.example.com\"]")
        sys.exit(1)
    
    client = SkymailClient(admin_email, admin_password, proxy=proxy, domains=domains, api_base=api_base)
    
    print(f"🔑 正在生成 Skymail API Token (API: {client.api_base})...")
    print(f"📧 可用域名: {', '.join(domains)}")
    token = client.generate_token()
    
    if not token:
        print("❌ Token 生成失败，无法继续")
        sys.exit(1)
    
    return client
