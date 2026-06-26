"""
全自动 Freebuff Token 获取
用法: python tool/auto_token.py [--codebuff] [--write-env] [--headless] [--dry-run]

也可作为模块导入:
    acquire_token(username, password, totp_secret=None, mode="freebuff") -> dict
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("auto_token")

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    sync_playwright = None
    PlaywrightTimeout = Exception

try:
    import pyotp as _pyotp
except ImportError:
    _pyotp = None

try:
    from freebuff2api.upstream_fingerprint import load_upstream_fingerprint_config
    CODEBUFF_JSON_USER_AGENT = load_upstream_fingerprint_config().codebuff_json_user_agent
except ImportError:
    CODEBUFF_JSON_USER_AGENT = "Bun/1.3.14"

BASE_FREEBUFF = "https://freebuff.com"
BASE_CODEBUFF = "https://www.codebuff.com"
VERIFY_URL = "https://www.codebuff.com/api/v1/freebuff/session"

MAX_RETRIES = 3
RETRY_DELAY_BASE = 5  # seconds, exponential backoff


@dataclass
class TokenResult:
    success: bool
    token: str = ""
    user_id: str = ""
    user_name: str = ""
    user_email: str = ""
    user_login: str = ""
    error: str = ""


def _endpoints(mode: str) -> tuple[str, str]:
    base = BASE_CODEBUFF if mode == "codebuff" else BASE_FREEBUFF
    return f"{base}/api/auth/cli/code", f"{base}/api/auth/cli/status"


def _request_code(fingerprint_id: str, code_url: str) -> dict:
    body = json.dumps({"fingerprintId": fingerprint_id}).encode()
    req = urllib.request.Request(
        code_url, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": CODEBUFF_JSON_USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _poll_token(fingerprint_id: str, fingerprint_hash: str, expires_at: int, status_url: str) -> dict | None:
    qs = urllib.parse.urlencode({
        "fingerprintId": fingerprint_id,
        "fingerprintHash": fingerprint_hash,
        "expiresAt": str(expires_at),
    })
    url = f"{status_url}?{qs}"
    deadline = time.monotonic() + 300

    while time.monotonic() < deadline:
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "User-Agent": CODEBUFF_JSON_USER_AGENT,
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                user = data.get("user")
                if user and user.get("authToken"):
                    return user
        except urllib.error.HTTPError:
            pass
        time.sleep(2)
    return None


def _verify_token(token: str) -> tuple[bool, str]:
    req = urllib.request.Request(
        VERIFY_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "*/*",
            "User-Agent": CODEBUFF_JSON_USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        body = e.read()[:200].decode(errors="replace")
        return False, f"HTTP {e.code}: {body}"


def _read_github_login(page) -> str:
    """Read the *actually* authenticated GitHub login from the page.

    GitHub embeds the signed-in user as <meta name="user-login"> (and
    <meta name="octolytics-actor-login">) on authenticated github.com pages,
    including the OAuth authorize screen. This is the source of truth for which
    account is logged in — far more reliable than checking whether the typed
    username merely appears somewhere in the HTML. Returns "" if not present
    (e.g. the page already redirected away from github.com)."""
    for selector in ('meta[name="user-login"]', 'meta[name="octolytics-actor-login"]'):
        try:
            el = page.query_selector(selector)
            if el:
                val = (el.get_attribute("content") or "").strip()
                if val and val.lower() != "anonymous":
                    return val
        except Exception:
            pass
    return ""


def _identity_mismatch(verified_login: str, username: str) -> bool:
    """True if the authenticated GitHub login clearly differs from the requested
    account. Tolerant: an empty verified_login (couldn't read the page) is NOT a
    mismatch, and an email-style username vs handle (one contains the other) is
    NOT a mismatch — only a genuinely different account returns True."""
    if not verified_login:
        return False
    gl, un = verified_login.strip().lower(), (username or "").strip().lower()
    if not un:
        return False
    return gl != un and un not in gl and gl not in un


def _get_proxy_config() -> dict[str, str] | None:
    """Read proxy config from environment for Playwright."""
    if not _env_bool("FREEBUFF_PROXY_ENABLED", False):
        return None
    proxy_url = (os.getenv("FREEBUFF_PROXY_URL") or "").strip()
    if not proxy_url:
        return None
    return {"server": proxy_url}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _write_env(token: str, env_path: Path | None = None) -> None:
    if env_path is None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
    new_token = token.strip()
    if not new_token:
        return
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        found = False
        modified = False
        out = []
        for line in lines:
            if line.startswith("FREEBUFF_TOKEN="):
                raw = line.split("=", 1)[1].strip()
                existing = [t.strip() for t in raw.split(",") if t.strip()]
                # Deduplicate: remove any empty/duplicate entries
                seen = set()
                deduped = []
                for t in existing:
                    if t and t not in seen:
                        seen.add(t)
                        deduped.append(t)
                if new_token in seen:
                    # Token already exists, no change needed
                    deduped_clean = deduped
                else:
                    deduped_clean = deduped + [new_token]
                    modified = True
                out.append(f"FREEBUFF_TOKEN={','.join(deduped_clean)}")
                found = True
            else:
                out.append(line)
        if not found:
            out.append(f"FREEBUFF_TOKEN={new_token}")
            modified = True
        if modified:
            env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
            logger.info("Wrote token to %s (preview: %s...)", env_path, new_token[:12])
        else:
            logger.info("Token already exists in %s, skipped", env_path)
    else:
        env_path.write_text(f"FREEBUFF_TOKEN={new_token}\n", encoding="utf-8")
        logger.info("Created %s with new token", env_path)


def get_existing_token_count() -> int:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return 0
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("FREEBUFF_TOKEN="):
            values = line.split("=", 1)[1].strip()
            return len([t for t in values.split(",") if t.strip()])
    return 0


def acquire_token(
    username: str,
    password: str,
    totp_secret: str = "",
    mode: str = "freebuff",
    headless: bool = True,
) -> TokenResult:
    """使用 GitHub 账号自动获取 Freebuff/Codebuff token.

    支持 Playwright 代理（通过 FREEBUFF_PROXY_URL 环境变量）。
    自动重试最多 MAX_RETRIES 次。
    """
    if sync_playwright is None:
        return TokenResult(success=False, error="playwright 未安装，请运行: pip install playwright && playwright install chromium")

    code_url, status_url = _endpoints(mode)

    last_error = ""
    verified_gh_login = ""  # GitHub login verified from Playwright page
    for attempt in range(1, MAX_RETRIES + 1):
        skip_polling = False  # Set to True to skip to next attempt after Playwright
        verified_gh_login = ""  # Real GitHub login read from the authorize page this attempt
        fingerprint_id = f"fb-{secrets.token_hex(8)}"
        try:
            code = _request_code(fingerprint_id, code_url)
        except Exception as e:
            last_error = f"请求登录码失败 (attempt {attempt}/{MAX_RETRIES}): {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_BASE * attempt)
            continue

        login_url = code.get("loginUrl", "")
        fingerprint_hash = code.get("fingerprintHash", "")
        expires_at = code.get("expiresAt", 0)

        if not login_url or not fingerprint_hash:
            last_error = f"上游返回数据不完整 (attempt {attempt}/{MAX_RETRIES})"
            logger.warning(last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_BASE * attempt)
            continue

        proxy_config = _get_proxy_config()
        browser_launch_kwargs = {"headless": headless}
        if proxy_config:
            browser_launch_kwargs["proxy"] = proxy_config
            logger.info("Using proxy: %s", proxy_config.get("server", "unknown"))

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(**browser_launch_kwargs)
                context_kwargs = {
                    "storage_state": None,  # Explicit clean state, no cached login
                    "no_viewport": True,
                }
                if proxy_config:
                    context_kwargs["proxy"] = proxy_config
                context = browser.new_context(**context_kwargs)
                # Clear all cookies to ensure no session leakage
                context.clear_cookies()
                page = context.new_page()

                # Set reasonable timeouts
                page.set_default_timeout(30000)

                logger.info("[%d/%d] Navigating to login page...", attempt, MAX_RETRIES)
                page.goto(login_url, wait_until="domcontentloaded")

                # Click "Continue with GitHub"
                page.wait_for_selector('button:has-text("Continue with GitHub")', timeout=15000)
                page.click('button:has-text("Continue with GitHub")')

                # Wait for GitHub login page — must show the login form
                try:
                    page.wait_for_selector("#login_field", timeout=15000)
                except Exception:
                    # Already on authorize page = residual GitHub session → fail & retry
                    current = page.url
                    context.close()
                    browser.close()
                    last_error = (
                        f"GitHub 已有残留登录会话 (当前页面: {current[:80]})，"
                        f"跳过以避免错号 (attempt {attempt}/{MAX_RETRIES})"
                    )
                    logger.warning(last_error)
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY_BASE * attempt)
                    continue

                # Fill GitHub credentials and submit login form
                page.fill("#login_field", username)
                page.fill("#password", password)
                page.click('input[type="submit"]')
                page.wait_for_timeout(3000)

                logger.info("After login submit, current URL: %s", page.url[:100])

                # Handle 2FA
                if "two-factor" in page.url:
                    if not totp_secret:
                        context.close()
                        browser.close()
                        return TokenResult(success=False, error="账号开启了 2FA 但未提供 TOTP 密钥")
                    if _pyotp is None:
                        context.close()
                        browser.close()
                        return TokenResult(success=False, error="需要 pyotp 库: pip install pyotp")

                    otp_code = _pyotp.TOTP(totp_secret).now()
                    logger.info("Entering 2FA code...")
                    page.fill("#app_totp", otp_code)
                    # Click submit and wait for navigation OR already-redirected
                    try:
                        with page.expect_navigation(wait_until="domcontentloaded", timeout=15000):
                            page.click('button[type="submit"]')
                    except Exception:
                        # May have already navigated away before click completed
                        pass
                    page.wait_for_timeout(2000)
                    logger.info("After 2FA, current URL: %s", page.url)

                # Handle "trusted device" page — click Skip
                current_url = page.url
                if "trusted-device" in current_url:
                    logger.info("Trusted device page detected, skipping...")
                    try:
                        page.click('button:has-text("Skip"), button:has-text("skip"), '
                                   '[value="skip"], .btn:has-text("Skip")',
                                   timeout=5000)
                        page.wait_for_timeout(2000)
                    except Exception:
                        logger.info("Could not click Skip on trusted-device, continuing...")
                    current_url = page.url

                # Read the *actual* authenticated GitHub login while still on
                # github.com (the OAuth authorize page) so we can confirm the token
                # is tied to the requested account and not a leaked/other session.
                if not verified_gh_login and "github.com" in current_url:
                    for _ in range(3):
                        verified_gh_login = _read_github_login(page)
                        if verified_gh_login:
                            logger.info("Authenticated GitHub login: %s", verified_gh_login)
                            break
                        page.wait_for_timeout(1000)
                        current_url = page.url
                        if "github.com" not in current_url:
                            break

                # Check for auth error page
                logger.info("Post-login URL: %s", current_url[:100])
                if "error" in current_url.lower() and "freebuff" not in current_url.lower():
                    error_text = page.text_content("body")[:500] if page.text_content("body") else "unknown"
                    context.close()
                    browser.close()
                    last_error = f"GitHub 登录后出现错误页面 (attempt {attempt}/{MAX_RETRIES}): {error_text[:200]}"
                    logger.warning(last_error)
                    skip_polling = True

                # Wait for redirect back to freebuff/codebuff
                target_pattern = "**/freebuff.com/**" if mode == "freebuff" else "**/codebuff.com/**"
                if target_pattern.replace("**/", "") not in current_url:
                    try:
                        page.wait_for_url(target_pattern, timeout=30000)
                        logger.info("Redirected to %s", page.url[:100])
                    except Exception:
                        logger.info("Did not redirect to target; current: %s", page.url[:100])

                context.close()
                browser.close()
        except Exception as e:
            last_error = f"浏览器自动化失败 (attempt {attempt}/{MAX_RETRIES}): {e}"
            logger.warning(last_error)
            skip_polling = True

        if skip_polling:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_BASE * attempt)
            continue

        # Poll for token
        user = _poll_token(fingerprint_id, fingerprint_hash, expires_at, status_url)
        if user is None:
            last_error = f"轮询超时 (attempt {attempt}/{MAX_RETRIES})"
            logger.warning(last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_BASE * attempt)
            continue

        token = user.get("authToken", "")
        if not token:
            last_error = f"未返回 authToken (attempt {attempt}/{MAX_RETRIES})"
            logger.warning(last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_BASE * attempt)
            continue

        # Log exactly what the upstream tied this token to. The returned name/email
        # are the GitHub *profile* fields, which legitimately differ from the login
        # handle — surface them so that difference isn't mistaken for a wrong account.
        safe_user = {k: v for k, v in user.items() if k != "authToken"}
        logger.info("Upstream user object: %s", json.dumps(safe_user, ensure_ascii=False))

        # Identity check: the GitHub account actually authenticated (read from the
        # authorize page) must match the requested username, or the token is discarded.
        if _identity_mismatch(verified_gh_login, username):
            last_error = (
                f"账号不匹配！输入 {username}，但 GitHub 实际登入 {verified_gh_login}"
                f" (attempt {attempt}/{MAX_RETRIES})，token 已丢弃"
            )
            logger.error(last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_BASE * attempt)
            continue
        if not verified_gh_login:
            logger.warning(
                "未能从授权页读取 GitHub 登录名（可能已直接跳转回上游）；"
                "依赖登录表单已填入指定凭据来保证账号正确。"
            )

        ok, info = _verify_token(token)
        if not ok:
            last_error = f"Token 验证失败: {info}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_BASE * attempt)
            continue

        logger.info("Token acquired successfully: login=%s name=%s (%s)",
                    verified_gh_login or "?", user.get("name", "?"), user.get("email", "?"))
        return TokenResult(
            success=True,
            token=token,
            user_id=user.get("id", ""),
            user_name=user.get("name", ""),
            user_email=user.get("email", ""),
            user_login=verified_gh_login or user.get("login", ""),
        )

    return TokenResult(success=False, error=last_error or "未知错误")


# ── CLI ───────────────────────────────────────────

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="全自动 Freebuff Token 获取")
    parser.add_argument("--codebuff", action="store_true", help="使用 codebuff.com 模式")
    parser.add_argument("--write-env", action="store_true", help="自动写入 .env 文件")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="无头模式运行浏览器 (默认)")
    parser.add_argument("--no-headless", action="store_true",
                        help="显示浏览器窗口")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅检查环境，不实际执行")
    parser.add_argument("--username", help="GitHub 用户名 (覆盖环境变量)")
    parser.add_argument("--password", help="GitHub 密码 (覆盖环境变量)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dry_run:
        print("=== 环境检查 ===")
        print(f"  playwright: {'OK' if sync_playwright else 'MISSING - pip install playwright && playwright install chromium'}")
        print(f"  pyotp:      {'OK' if _pyotp else 'MISSING - pip install pyotp'}")
        proxy = _get_proxy_config()
        print(f"  proxy:      {proxy.get('server') if proxy else '未配置'}")
        print(f"  .env:       {'存在' if (Path(__file__).resolve().parents[1] / '.env').exists() else '不存在'}")
        print(f"  当前 token 数: {get_existing_token_count()}")
        return 0

    mode = "codebuff" if args.codebuff else "freebuff"
    username = args.username or os.getenv("FREEBUFF_GITHUB_USERNAME")
    password = args.password or os.getenv("FREEBUFF_GITHUB_PASSWORD")
    totp_secret = os.getenv("FREEBUFF_GITHUB_TOTP_SECRET")

    if not username or not password:
        print("[err] 请设置环境变量 FREEBUFF_GITHUB_USERNAME 和 FREEBUFF_GITHUB_PASSWORD")
        print("      或使用 --username 和 --password 参数")
        return 1

    headless = not args.no_headless
    print(f"[mode] {mode}  |  headless={headless}  |  user={username}")
    result = acquire_token(username, password, totp_secret or "", mode, headless)

    if result.success:
        print(f"\n=== 成功 ===")
        print(f"  login  : {result.user_login}")
        print(f"  name   : {result.user_name}")
        print(f"  email  : {result.user_email}")
        print(f"  token  : {result.token}")
        if args.write_env:
            _write_env(result.token)
            print(f"  当前 token 数: {get_existing_token_count()}")
        else:
            print("\n(tip: 用 --write-env 自动写入 .env)")
        return 0
    else:
        print(f"\n[err] {result.error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
