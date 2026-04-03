"""
ChatGPT 批量自动注册工具 v2.0 - 模块化版本
使用 CloudMail 临时邮箱，并发自动注册 ChatGPT 账号
"""

import os
import sys
import time
import threading
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# 强制使用 UTF-8 输出
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 禁用 SSL 警告
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入自定义模块
from lib.utils import load_config, as_bool
from lib.clients import TokenManager

# 导入引擎模块
from lib.core import RegistrationEngine
from lib.clients import CloudMailService

# 全局文件写入锁
_file_lock = threading.Lock()

def init_cloudmail_client(config):
    cloud_mail_config = {
        "base_url": config.get("cloudmail_url", ""),
        "admin_email": config.get("cloudmail_admin_email", ""),
        "admin_password": config.get("cloudmail_admin_password", ""),
        "domain": config.get("cloudmail_domains", []),
        "subdomain": config.get("cloudmail_subdomain", ""),
        "timeout": config.get("timeout", 30),
        "proxy_url": config.get("proxy", "")
    }
    return CloudMailService(config=cloud_mail_config)

def register_one_account(idx, total, cloudmail_client, token_manager, config):
    """
    注册单个账号的完整流程
    
    Args:
        idx: 账号序号
        total: 总账号数
        cloudmail_client: CloudMail 客户端
        token_manager: Token 管理器
        config: 配置字典
        
    Returns:
        tuple: (success, email, password, message)
    """
    tag = f"[{idx}/{total}]"
    print(f"\n{tag} 开始注册...")
    
    try:
        # 1. 准备配置信息
        proxy = config.get("proxy", "")
        enable_oauth = as_bool(config.get("enable_oauth", True))
        oauth_required = as_bool(config.get("oauth_required", True))
        output_file = config.get("output_file", "registered_accounts.txt")
        
        # 定义一个简单的回调日志来将 RegistrationEngine 的输出带上标签
        def callback_logger(msg):
            print(f"{tag} {msg}")

        # 2. 创建开源项目的注册引擎
        engine = RegistrationEngine(
            email_service=cloudmail_client,
            proxy_url=proxy,
            callback_logger=callback_logger
        )
        
        # 3. 执行注册流程
        print(f"{tag} 开始执行开源引擎的注册流程...")
        result = engine.run()
        
        email = result.email
        password = result.password
        
        if not result.success:
            print(f"{tag} ❌ 注册失败: {result.error_message}")
            return False, email, password, result.error_message
            
        print(f"{tag} ✅ 注册成功 (账户: {email})")
        
        # 4. 判断并保存 OAuth Token
        # 开源项目的 RegistrationEngine 在 run() 里已经自带了获取 token 流程（如果成功就会有 access_token）
        has_token = bool(result.access_token)
        
        if enable_oauth:
            if has_token:
                print(f"{tag} ✅ OAuth 成功")
                # 组装 tokens 给 token_manager
                tokens = {
                    "access_token": result.access_token,
                    "refresh_token": getattr(result, "refresh_token", ""),
                    "id_token": getattr(result, "id_token", ""),
                    "session_token": getattr(result, "session_token", "")
                }
                token_manager.save_tokens(email, tokens)
                
                # 保存账号信息
                with _file_lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(f"{email}----{password}----oauth=ok\n")
                
                return True, email, password, "注册成功 + OAuth 成功"
            else:
                print(f"{tag} ⚠️ OAuth 失败或未能获取")
                if oauth_required:
                    return False, email, password, "OAuth 失败（必需）"
                else:
                    # 保存账号信息（无 OAuth）
                    with _file_lock:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(f"{email}----{password}----oauth=failed\n")
                    return True, email, password, "注册成功（OAuth 失败）"
        else:
            # 不启用 OAuth 或者不关心 OAuth 结果，直接保存账号
            with _file_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"{email}----{password}\n")
            return True, email, password, "注册成功"
            
    except Exception as e:
        print(f"{tag} ❌ 注册失败: {e}")
        import traceback
        traceback.print_exc()
        return False, "", "", str(e)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ChatGPT 批量自动注册工具 v2.0')
    parser.add_argument('-n', '--num', type=int, default=1, help='注册账号数量（默认: 1）')
    parser.add_argument('-w', '--workers', type=int, default=1, help='并发线程数（默认: 1）')
    parser.add_argument('--no-oauth', action='store_true', help='禁用 OAuth 登录')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  ChatGPT 批量自动注册工具 v2.0 (模块化版本)")
    print("  使用 CloudMail 临时邮箱")
    print("=" * 60)
    
    # 加载配置
    config = load_config()
    
    # 命令行参数覆盖配置文件
    total_accounts = args.num
    max_workers = args.workers
    if args.no_oauth:
        config['enable_oauth'] = False
    
    # 初始化 CloudMail 客户端
    cloudmail_client = init_cloudmail_client(config)
    
    # 初始化 Token 管理器
    token_manager = TokenManager(config)
    
    # 获取配置参数
    output_file = config.get("output_file", "registered_accounts.txt")
    enable_oauth = as_bool(config.get("enable_oauth", True))
    
    print(f"\n配置信息:")
    print(f"  注册数量: {total_accounts}")
    print(f"  并发数: {max_workers}")
    print(f"  输出文件: {output_file}")
    print(f"  CloudMail API: {cloudmail_client.config.get('base_url', '')}")
    print(f"  Token 目录: {token_manager.token_dir}")
    print(f"  启用 OAuth: {enable_oauth}")
    print()
    
    # 批量注册
    success_count = 0
    failed_count = 0
    start_time = time.time()
    
    if max_workers == 1:
        # 串行执行
        for i in range(1, total_accounts + 1):
            success, email, password, msg = register_one_account(
                i, total_accounts, cloudmail_client, token_manager, config
            )
            if success:
                success_count += 1
            else:
                failed_count += 1
    else:
        # 并发执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(1, total_accounts + 1):
                future = executor.submit(
                    register_one_account,
                    i, total_accounts, cloudmail_client, token_manager, config
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    success, email, password, msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"❌ 任务异常: {e}")
                    failed_count += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 输出统计
    print("\n" + "=" * 60)
    print(f"注册完成！")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"  总计: {total_accounts}")
    print(f"  总耗时: {total_time:.1f}s")
    if success_count > 0:
        print(f"  平均耗时: {total_time/total_accounts:.1f}s/账号")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
