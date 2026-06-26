"""
Freebuff Token Web — 后台管理 + 自动 Token 生成调度器
启动: python tool/web/main.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("freebuff2api.web")

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path for tool/auto_token import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tool.auto_token import acquire_token, get_existing_token_count, _write_env, TokenResult
from freebuff2api.token_rotation import get_rotation_manager

# On startup, ensure the active token group is written to .env
_rotation = get_rotation_manager()
_rotation.ensure_active_group()

load_dotenv()

BASE_FREEBUFF = "https://freebuff.com"
BASE_CODEBUFF = "https://www.codebuff.com"
VERIFY_URL = "https://www.codebuff.com/api/v1/freebuff/session"
MAX_TOKENS = int(os.getenv("FREEBUFF_MAX_TOKENS", "100"))
INTERVAL_SECONDS = int(os.getenv("FREEBUFF_AUTO_INTERVAL", "300"))  # default 5 min
MAX_CONSECUTIVE_FAILURES = int(os.getenv("FREEBUFF_MAX_FAILURES", "10"))
FAILURE_BACKOFF_BASE = int(os.getenv("FREEBUFF_FAILURE_BACKOFF", "30"))  # seconds
THREAD_TIMEOUT = int(os.getenv("FREEBUFF_THREAD_TIMEOUT", "360"))  # 6 min max per attempt

try:
    from freebuff2api.upstream_fingerprint import load_upstream_fingerprint_config
    CODEBUFF_JSON_USER_AGENT = load_upstream_fingerprint_config().codebuff_json_user_agent
except ImportError:
    CODEBUFF_JSON_USER_AGENT = "Bun/1.3.14"

NO_STORE_HEADERS = {"Cache-Control": "no-store"}
ACCOUNTS_FILE = Path(__file__).parent / "accounts.json"
PROJECT_ENV = Path(__file__).resolve().parents[2] / ".env"

SUPPORTED_MODES = {"freebuff", "codebuff"}

app = FastAPI(title="Freebuff Token Manager")
web_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=web_dir / "static"), name="static")

# ── Runtime State ─────────────────────────────────

_sessions: dict[str, dict[str, Any]] = {}  # manual flow sessions

# Auto-generation state
_auto_state: dict[str, Any] = {
    "running": False,
    "total": 0,
    "success": 0,
    "failed": 0,
    "stopped": False,
    "stop_reason": "",
    "current_account": "",
    "current_mode": "freebuff",
    "last_result": None,
    "tokens": [],  # list of {token, name, email, time, status}
    "events": asyncio.Queue(maxsize=500),
}
_auto_task: asyncio.Task | None = None


# ── Helpers ───────────────────────────────────────

def _load_accounts() -> list[dict]:
    if ACCOUNTS_FILE.exists():
        data = json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    return []


def _save_accounts(accounts: list[dict]) -> None:
    ACCOUNTS_FILE.write_text(json.dumps(accounts, ensure_ascii=False, indent=2), encoding="utf-8")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _proxy_url() -> str | None:
    if not _env_bool("FREEBUFF_PROXY_ENABLED", False):
        return None
    return (os.getenv("FREEBUFF_PROXY_URL") or "").strip() or None


def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(proxy=_proxy_url(), trust_env=False)


async def _push_event(event: dict) -> None:
    try:
        _auto_state["events"].put_nowait(event)
    except asyncio.QueueFull:
        pass


# ── Manual Flow APIs ──────────────────────────────

def _endpoints(mode: str) -> tuple[str, str]:
    base = BASE_CODEBUFF if mode == "codebuff" else BASE_FREEBUFF
    return f"{base}/api/auth/cli/code", f"{base}/api/auth/cli/status"


async def _request_code(fingerprint_id: str, code_url: str) -> dict[str, Any]:
    async with _http_client() as client:
        resp = await client.post(
            code_url,
            json={"fingerprintId": fingerprint_id},
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": CODEBUFF_JSON_USER_AGENT,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


async def _poll_status_sse(fingerprint_id: str, fingerprint_hash: str, expires_at: int, status_url: str):
    qs = urllib.parse.urlencode({
        "fingerprintId": fingerprint_id,
        "fingerprintHash": fingerprint_hash,
        "expiresAt": str(expires_at),
    })
    url = f"{status_url}?{qs}"
    deadline = time.monotonic() + 5 * 60
    attempt = 0

    async with _http_client() as client:
        while time.monotonic() < deadline:
            attempt += 1
            try:
                resp = await client.get(
                    url,
                    headers={"Accept": "application/json", "User-Agent": CODEBUFF_JSON_USER_AGENT},
                    timeout=15,
                )
                if resp.status_code == 401:
                    yield f'data: {json.dumps({"event": "pending", "attempt": attempt}, ensure_ascii=False)}\n\n'
                    await asyncio.sleep(2.0)
                    continue
                if resp.status_code >= 400:
                    yield f'data: {json.dumps({"event": "error", "message": f"HTTP {resp.status_code}"}, ensure_ascii=False)}\n\n'
                    return
                try:
                    data = resp.json()
                except ValueError:
                    yield f'data: {json.dumps({"event": "error", "message": "invalid upstream response"}, ensure_ascii=False)}\n\n'
                    return
                user = data.get("user")
                if user and user.get("authToken"):
                    yield f'data: {json.dumps({"event": "success", "user": user}, ensure_ascii=False)}\n\n'
                    return
                yield f'data: {json.dumps({"event": "pending", "attempt": attempt}, ensure_ascii=False)}\n\n'
            except Exception as e:
                yield f'data: {json.dumps({"event": "error", "message": str(e)}, ensure_ascii=False)}\n\n'
                return
            await asyncio.sleep(2.0)
        yield f'data: {json.dumps({"event": "timeout"}, ensure_ascii=False)}\n\n'


# ── Auth APIs ─────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = web_dir / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), headers=NO_STORE_HEADERS)


@app.get("/auto", response_class=HTMLResponse)
async def auto_page():
    html_path = web_dir / "templates" / "auto.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), headers=NO_STORE_HEADERS)


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204, headers=NO_STORE_HEADERS)


@app.post("/api/start")
async def start_auth(request: Request):
    body = await request.json()
    mode = body.get("mode", "freebuff")
    if mode not in SUPPORTED_MODES:
        raise HTTPException(status_code=400, detail="unsupported mode")
    code_url, status_url = _endpoints(mode)
    fingerprint_id = f"fb-{secrets.token_hex(8)}"
    code = await _request_code(fingerprint_id, code_url)
    _sessions[fingerprint_id] = {
        "mode": mode,
        "fingerprint_id": fingerprint_id,
        "fingerprint_hash": code["fingerprintHash"],
        "expires_at": code["expiresAt"],
        "status_url": status_url,
    }
    return {
        "mode": mode,
        "fingerprint_id": fingerprint_id,
        "login_url": code["loginUrl"],
        "expires_at": code["expiresAt"],
    }


@app.get("/api/poll/{fingerprint_id}")
async def poll_auth(fingerprint_id: str):
    session = _sessions.get(fingerprint_id)
    if not session:
        async def _err():
            yield f'data: {json.dumps({"event": "error", "message": "session not found"})}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream", headers=NO_STORE_HEADERS)

    async def event_generator():
        async for event in _poll_status_sse(
            session["fingerprint_id"], session["fingerprint_hash"],
            session["expires_at"], session["status_url"],
        ):
            yield event
        _sessions.pop(fingerprint_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=NO_STORE_HEADERS)


@app.post("/api/verify")
async def verify_token_endpoint(request: Request):
    body = await request.json()
    token = body.get("token", "").strip()
    if not token:
        return {"ok": False, "info": "missing token"}
    async with _http_client() as client:
        try:
            resp = await client.get(
                VERIFY_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "*/*",
                    "User-Agent": CODEBUFF_JSON_USER_AGENT,
                },
                timeout=15,
            )
            ok = 200 <= resp.status_code < 300
            return {"ok": ok, "info": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"ok": False, "info": f"network error: {e}"}


# ── GitHub Account CRUD ───────────────────────────

@app.get("/api/accounts")
async def list_accounts():
    accounts = _load_accounts()
    # Don't return passwords in list view, only mask
    result = []
    for i, a in enumerate(accounts):
        result.append({
            "index": i,
            "username": a["username"],
            "has_password": bool(a.get("password")),
            "has_totp": bool(a.get("totp_secret")),
            "created_at": a.get("created_at", ""),
        })
    return {"accounts": result}


@app.post("/api/accounts")
async def add_account(request: Request):
    body = await request.json()
    username = (body.get("username") or "").strip()
    password = (body.get("password") or "").strip()
    totp_secret = (body.get("totp_secret") or "").strip()

    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    accounts = _load_accounts()
    if any(a["username"] == username for a in accounts):
        raise HTTPException(status_code=409, detail=f"账号 {username} 已存在")

    accounts.append({
        "username": username,
        "password": password,
        "totp_secret": totp_secret,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_accounts(accounts)
    await _push_event({"type": "accounts_updated", "count": len(accounts)})
    return {"ok": True, "count": len(accounts)}


@app.put("/api/accounts/{index}")
async def update_account(index: int, request: Request):
    body = await request.json()
    accounts = _load_accounts()
    if index < 0 or index >= len(accounts):
        raise HTTPException(status_code=404, detail="账号不存在")
    if "username" in body:
        accounts[index]["username"] = (body["username"] or "").strip()
    if "password" in body:
        accounts[index]["password"] = (body["password"] or "").strip()
    if "totp_secret" in body:
        accounts[index]["totp_secret"] = (body["totp_secret"] or "").strip()
    _save_accounts(accounts)
    await _push_event({"type": "accounts_updated", "count": len(accounts)})
    return {"ok": True}


@app.delete("/api/accounts/{index}")
async def delete_account(index: int):
    accounts = _load_accounts()
    if index < 0 or index >= len(accounts):
        raise HTTPException(status_code=404, detail="账号不存在")
    deleted = accounts.pop(index)
    _save_accounts(accounts)
    await _push_event({"type": "accounts_updated", "count": len(accounts)})
    return {"ok": True, "deleted": deleted["username"], "count": len(accounts)}


# ── Auto Generation ───────────────────────────────

async def _auto_generator_loop(mode: str):
    """Background task that cycles through GitHub accounts to generate tokens."""
    state = _auto_state
    token_count_at_start = get_existing_token_count()
    targets = MAX_TOKENS - token_count_at_start
    state["target"] = MAX_TOKENS
    state["start_count"] = token_count_at_start
    consecutive_failures = 0

    logger.info("Auto generator started: target=%s, start_count=%s, mode=%s, interval=%ss",
                MAX_TOKENS, token_count_at_start, mode, INTERVAL_SECONDS)

    while state["running"] and not state["stopped"]:
        accounts = _load_accounts()
        if not accounts:
            state["stopped"] = True
            state["stop_reason"] = "没有可用的 GitHub 账号"
            await _push_event({"type": "error", "message": state["stop_reason"]})
            break

        current_count = get_existing_token_count()
        if current_count >= MAX_TOKENS:
            state["stopped"] = True
            state["stop_reason"] = f"已达到 {MAX_TOKENS} 个 token"
            state["total"] = current_count - token_count_at_start
            await _push_event({"type": "done", "message": state["stop_reason"],
                               "total": state["total"], "progress": current_count,
                               "target": MAX_TOKENS})
            break

        # Check if too many consecutive failures
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            state["stopped"] = True
            state["stop_reason"] = f"连续失败 {consecutive_failures} 次，自动停止"
            await _push_event({"type": "error", "message": state["stop_reason"]})
            break

        # Round-robin through accounts
        account_index = state["total"] % len(accounts)
        account = accounts[account_index]
        state["current_account"] = account["username"]
        state["current_index"] = account_index

        await _push_event({
            "type": "generating",
            "account": account["username"],
            "index": account_index + 1,
            "total_accounts": len(accounts),
            "progress": current_count,
            "target": MAX_TOKENS,
            "session": state["total"] + 1,
        })

        # Run in thread with timeout to prevent hanging
        try:
            result: TokenResult = await asyncio.wait_for(
                asyncio.to_thread(
                    acquire_token,
                    account["username"],
                    account["password"],
                    account.get("totp_secret", ""),
                    mode,
                    True,  # headless
                ),
                timeout=THREAD_TIMEOUT,
            )
        except asyncio.TimeoutError:
            consecutive_failures += 1
            state["failed"] += 1
            state["total"] += 1
            last_error = f"生成超时 ({THREAD_TIMEOUT}s)，可能 GitHub 要求验证码或网络不通"
            state["last_result"] = {"ok": False, "error": last_error}
            await _push_event({
                "type": "token_failed",
                "error": last_error,
                "account": account["username"],
                "total": state["total"],
                "failed": state["failed"],
                "consecutive_failures": consecutive_failures,
            })
            logger.warning(last_error)
            # Skip the normal result handling below
            if state["running"] and not state["stopped"]:
                backoff = min(FAILURE_BACKOFF_BASE * consecutive_failures, INTERVAL_SECONDS)
                await _push_event({"type": "waiting", "seconds": backoff,
                                   "message": f"超时，等待 {backoff}s 后重试..."})
                waited = 0
                while waited < backoff and state["running"] and not state["stopped"]:
                    await asyncio.sleep(min(5, backoff - waited))
                    waited += 5
            continue

        state["total"] += 1
        timestamp = datetime.now(timezone.utc).isoformat()

        if result.success:
            consecutive_failures = 0  # Reset backoff on success
            state["success"] += 1
            _write_env(result.token, PROJECT_ENV)
            token_entry = {
                "token": result.token,
                "name": result.user_name,
                "email": result.user_email,
                "login": result.user_login,
                "time": timestamp,
                "status": "ok",
                "account": account["username"],
            }
            state["tokens"].append(token_entry)
            state["last_result"] = {"ok": True, "token": result.token[:10] + "...",
                                    "name": result.user_name}
            new_count = get_existing_token_count()
            await _push_event({
                "type": "token_generated",
                "token_preview": result.token[:12] + "...",
                "name": result.user_name,
                "email": result.user_email,
                "login": result.user_login,
                "account": account["username"],
                "total": state["total"],
                "success": state["success"],
                "failed": state["failed"],
                "progress": new_count,
                "target": MAX_TOKENS,
            })
            if new_count >= MAX_TOKENS:
                state["stopped"] = True
                state["stop_reason"] = f"已达到 {MAX_TOKENS} 个 token，自动停止"
                await _push_event({"type": "done", "message": state["stop_reason"],
                                   "total": state["total"], "progress": new_count,
                                   "target": MAX_TOKENS})
                break
        else:
            consecutive_failures += 1
            state["failed"] += 1
            token_entry = {
                "token": "",
                "name": "",
                "email": "",
                "time": timestamp,
                "status": "fail",
                "error": result.error,
                "account": account["username"],
            }
            state["tokens"].append(token_entry)
            state["last_result"] = {"ok": False, "error": result.error}
            await _push_event({
                "type": "token_failed",
                "error": result.error,
                "account": account["username"],
                "total": state["total"],
                "failed": state["failed"],
                "consecutive_failures": consecutive_failures,
            })
            logger.warning("Token generation failed (consecutive=%s): %s",
                           consecutive_failures, result.error)

        if state["running"] and not state["stopped"]:
            # Calculate wait time: if failed, use backoff, otherwise normal interval
            if consecutive_failures > 0:
                backoff = min(FAILURE_BACKOFF_BASE * consecutive_failures, INTERVAL_SECONDS)
                wait_msg = f"失败 {consecutive_failures} 次，等待 {backoff}s 后重试..."
            else:
                backoff = INTERVAL_SECONDS
                wait_msg = f"等待 {backoff // 60} 分钟后生成下一个..."
            await _push_event({"type": "waiting", "seconds": backoff, "message": wait_msg})

            waited = 0
            while waited < backoff and state["running"] and not state["stopped"]:
                await asyncio.sleep(min(5, backoff - waited))
                waited += 5

    state["running"] = False
    final_count = get_existing_token_count()
    logger.info("Auto generator stopped: %s, final_count=%s",
                state["stop_reason"] or "已停止", final_count)
    await _push_event({"type": "stopped", "reason": state["stop_reason"] or "已停止",
                       "progress": final_count, "target": MAX_TOKENS,
                       "total": state["total"], "success": state["success"],
                       "failed": state["failed"]})


@app.get("/api/auto/status")
async def auto_status():
    """Get current auto-generation state."""
    current_count = get_existing_token_count()
    accounts = _load_accounts()
    return {
        "running": _auto_state["running"],
        "total": _auto_state["total"],
        "success": _auto_state["success"],
        "failed": _auto_state["failed"],
        "stopped": _auto_state["stopped"],
        "stop_reason": _auto_state["stop_reason"],
        "current_account": _auto_state["current_account"],
        "current_mode": _auto_state["current_mode"],
        "last_result": _auto_state["last_result"],
        "progress": current_count,
        "target": MAX_TOKENS,
        "accounts_count": len(accounts),
        "recent_tokens": _auto_state["tokens"][-30:],
    }


@app.post("/api/auto/start")
async def auto_start(request: Request):
    """Start auto token generation."""
    global _auto_task

    if _auto_state["running"]:
        return {"ok": False, "error": "已经在运行中"}

    accounts = _load_accounts()
    if not accounts:
        return {"ok": False, "error": "请先添加 GitHub 账号"}

    body = await request.json()
    mode = body.get("mode", "freebuff")

    # Reset state
    _auto_state["running"] = True
    _auto_state["total"] = 0
    _auto_state["success"] = 0
    _auto_state["failed"] = 0
    _auto_state["stopped"] = False
    _auto_state["stop_reason"] = ""
    _auto_state["current_account"] = ""
    _auto_state["current_mode"] = mode
    _auto_state["last_result"] = None
    _auto_state["tokens"] = []

    _auto_task = asyncio.create_task(_auto_generator_loop(mode))
    await _push_event({"type": "started", "mode": mode, "accounts": len(accounts)})
    return {"ok": True, "mode": mode, "accounts": len(accounts)}


@app.post("/api/auto/stop")
async def auto_stop():
    """Stop auto token generation."""
    if not _auto_state["running"]:
        return {"ok": False, "error": "没有在运行"}

    _auto_state["stopped"] = True
    _auto_state["stop_reason"] = "手动停止"
    _auto_state["running"] = False
    await _push_event({"type": "stopped", "reason": "手动停止"})
    return {"ok": True}


@app.get("/api/auto/events")
async def auto_events():
    """SSE stream for real-time auto-generation updates."""
    async def generator():
        last_event_id = 0
        while True:
            try:
                event = await asyncio.wait_for(_auto_state["events"].get(), timeout=15)
                last_event_id += 1
                yield f"id: {last_event_id}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            if not _auto_state["running"]:
                break

    return StreamingResponse(generator(), media_type="text/event-stream", headers=NO_STORE_HEADERS)


@app.get("/api/auto/tokens")
async def auto_tokens():
    """Get current token groups from token.json and status."""
    rotation = get_rotation_manager()
    all_tokens = []
    if PROJECT_ENV.exists():
        for line in PROJECT_ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith("FREEBUFF_TOKEN="):
                values = line.split("=", 1)[1].strip()
                all_tokens = [t.strip() for t in values.split(",") if t.strip()]

    # Build group data from token.json groups
    groups_info = []
    for i, tokens in enumerate(rotation._groups):
        is_active = i == rotation.current_index
        is_blocked = rotation.is_group_blocked(i)
        remaining = rotation.group_block_remaining(i)
        h, m = int(remaining // 3600), int((remaining % 3600) // 60)
        s = int(remaining % 60)
        name = rotation._group_names[i] if i < len(rotation._group_names) else f"[{i}]"
        groups_info.append({
            "name": name,
            "index": i,
            "is_active": is_active,
            "is_blocked": is_blocked,
            "block_remaining_str": f"{h}时{m}分{s}秒" if is_blocked else "",
            "total": len(tokens),
            "in_env": sum(1 for t in tokens if t in all_tokens),
            "tokens": [t[:8] + "..." for t in tokens],
        })

    return {
        "count": len(all_tokens),
        "groups": groups_info,
        "active_group": rotation.current_group_name,
    }


@app.get("/api/rotation/status")
async def rotation_status():
    """Get token group rotation status."""
    rotation = get_rotation_manager()
    current_num = 0
    if PROJECT_ENV.exists():
        for line in PROJECT_ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith("CURRENT_TOKENNum="):
                try:
                    current_num = int(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
    # Token file name from env
    token_file = os.getenv("FREEBUFF_TOKENFILE", "token.json")
    return {
        "groups": rotation.group_count,
        "current_group": f"[{rotation.current_index}]",
        "current_index": rotation.current_index,
        "current_token_num": current_num,
        "total_rotations": rotation.total_rotations,
        "current_tokens": len(rotation.current_tokens),
        "token_file": token_file,
        "last_429": rotation.last_429_info,
        "last_429_time": rotation.last_429_time,
        "blocked_groups": {str(k): int(v) for k, v in rotation.blocked_groups.items()},
    }


@app.post("/api/rotation/rotate")
async def rotation_rotate():
    """Manually trigger token group rotation."""
    rotation = get_rotation_manager()
    if not rotation.group_count:
        return {"ok": False, "error": "没有配置 token 组"}
    idx, name, tokens = rotation.rotate(reason="manual")
    return {
        "ok": True,
        "index": idx,
        "group": name,
        "tokens": len(tokens),
        "total_rotations": rotation.total_rotations,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("FREEBUFF_WEB_PORT", "8001"))
    host = os.getenv("FREEBUFF_WEB_HOST", "127.0.0.1")
    print(f"Token Manager starting at http://{host}:{port}/auto")
    uvicorn.run(app, host=host, port=port, log_level="info")
