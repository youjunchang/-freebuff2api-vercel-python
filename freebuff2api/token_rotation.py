"""
Token 组自动轮换 — 从 token.json 读取命名组，429 限流后切换下一组。

数据源: 项目根目录 token.json
  [
    {"name": "github-fownbqu", "tokens": ["tokenA", "tokenB"]},
    {"name": "gmail-acc2",     "tokens": ["tokenC", "tokenD"]},
    ...
  ]

旧格式（二维数组）自动兼容:
  [["tokenA","tokenB"], ["tokenC"]]  →  自动命名为 [0], [1]

状态记录: .env 中的 CURRENT_TOKENNum
  - 启动时读取 CURRENT_TOKENNum，从该组开始
  - 429 后 CURRENT_TOKENNum+1，切换到下一组，写回 .env
  - 最后一组轮完回到 0

触发条件: 上游返回 429 (rate_limited)
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger("freebuff2api.token_rotation")

SHA_TZ = timezone(timedelta(hours=8))  # Asia/Shanghai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ENV = PROJECT_ROOT / ".env"


def _token_file_path() -> Path:
    """Read token file path from FREEBUFF_TOKENFILE env var, default to token.json."""
    import os
    filename = os.getenv("FREEBUFF_TOKENFILE", "token.json").strip()
    if not filename:
        filename = "token.json"
    path = Path(filename)
    if not path.is_absolute():
        path = PROJECT_ROOT / filename
    return path

COOLDOWN_SECONDS = 30


class TokenRotationManager:
    """Manages token group rotation. Reads groups from token.json, tracks position via CURRENT_TOKENNum."""

    def __init__(self) -> None:
        self._groups: list[list[str]] = []   # [[token, token], [token, token], ...]
        self._group_names: list[str] = []     # ["github-xxx", "gmail-xxx", ...]
        self._current_index: int = 0
        self._last_rotation: float = 0.0
        self._total_rotations: int = 0
        self._last_429_info: dict = {}       # Latest 429 rate-limit info
        self._last_429_time: str = ""        # When the last 429 occurred (local time)
        self._group_blocked_until: dict[int, float] = {}  # group_index → epoch seconds
        self._load()

    # ── Load / Save ──────────────────────────────

    def _load(self) -> None:
        """Load token groups from token file (FREEBUFF_TOKENFILE) and restore position."""
        token_path = _token_file_path()
        if not token_path.exists():
            logger.warning("Token file not found: %s (FREEBUFF_TOKENFILE=%s)",
                          token_path, token_path.name)
            return

        try:
            raw = json.loads(token_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read %s: %s", token_path, e)
            return

        if not isinstance(raw, list):
            logger.error("%s must be a JSON array", token_path)
            return

        groups: list[list[str]] = []
        names: list[str] = []
        is_old_format = False
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                # New format: {"name": "xxx", "tokens": ["t1","t2"]}
                name = str(item.get("name", f"[{i}]")).strip() or f"[{i}]"
                token_list = item.get("tokens", [])
                tokens = [t.strip() for t in token_list if isinstance(t, str) and t.strip()]
                if tokens:
                    groups.append(tokens)
                    names.append(name)
            elif isinstance(item, list):
                # Old format: ["tokenA", "tokenB"]
                is_old_format = True
                tokens = [t.strip() for t in item if isinstance(t, str) and t.strip()]
                if tokens:
                    groups.append(tokens)
                    names.append(f"[{i}]")
            elif isinstance(item, str) and item.strip():
                # Old format: single string → one-token group
                is_old_format = True
                groups.append([item.strip()])
                names.append(f"[{i}]")

        self._groups = groups
        self._group_names = names
        if not groups:
            logger.warning("token.json is empty")
            return

        logger.info("Loaded %d token groups from %s (%s format)",
                    len(groups), token_path.name,
                    "legacy array, auto-named" if is_old_format else "named objects")
        for i, tokens in enumerate(groups):
            name = names[i] if i < len(names) else f"[{i}]"
            logger.info("  %s: %d tokens (e.g. %s...)", name, len(tokens), tokens[0][:12])

        # Restore position from CURRENT_TOKENNum
        self._current_index = self._read_current_token_num()

    def _read_current_token_num(self) -> int:
        """Read CURRENT_TOKENNum from .env. Returns 0-based index."""
        if not PROJECT_ENV.exists():
            return 0
        for line in PROJECT_ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith("CURRENT_TOKENNum="):
                try:
                    num = int(line.split("=", 1)[1].strip())
                    idx = num % len(self._groups) if self._groups else 0
                    logger.info("Resumed from CURRENT_TOKENNum=%d → group [%d]", num, idx)
                    return idx
                except ValueError:
                    pass
        return 0

    def _write_current_token_num(self) -> None:
        """Write CURRENT_TOKENNum to .env, preserving other lines."""
        if PROJECT_ENV.exists():
            lines = PROJECT_ENV.read_text(encoding="utf-8").splitlines()
            out = []
            found = False
            for line in lines:
                if line.startswith("CURRENT_TOKENNum="):
                    out.append(f"CURRENT_TOKENNum={self._current_index}")
                    found = True
                else:
                    out.append(line)
            if not found:
                out.append(f"CURRENT_TOKENNum={self._current_index}")
            PROJECT_ENV.write_text("\n".join(out) + "\n", encoding="utf-8")
        else:
            PROJECT_ENV.write_text(f"CURRENT_TOKENNum={self._current_index}\n", encoding="utf-8")

    def _write_env(self, token_value: str) -> None:
        """Write FREEBUFF_TOKEN=<value> to .env, preserving other lines."""
        if not token_value:
            return
        if PROJECT_ENV.exists():
            lines = PROJECT_ENV.read_text(encoding="utf-8").splitlines()
            out = []
            found = False
            for line in lines:
                if line.startswith("FREEBUFF_TOKEN="):
                    out.append(f"FREEBUFF_TOKEN={token_value}")
                    found = True
                else:
                    out.append(line)
            if not found:
                out.append(f"FREEBUFF_TOKEN={token_value}")
            PROJECT_ENV.write_text("\n".join(out) + "\n", encoding="utf-8")
        else:
            PROJECT_ENV.write_text(f"FREEBUFF_TOKEN={token_value}\n", encoding="utf-8")

    # ── Properties ──────────────────────────────

    @property
    def group_count(self) -> int:
        return len(self._groups)

    @property
    def current_group_name(self) -> str:
        if not self._groups:
            return "无"
        if self._current_index < len(self._group_names):
            return self._group_names[self._current_index]
        return f"[{self._current_index}]"

    @property
    def group_names(self) -> list[str]:
        return list(self._group_names)

    @property
    def current_tokens(self) -> list[str]:
        if not self._groups:
            return []
        return self._groups[self._current_index]

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def total_rotations(self) -> int:
        return self._total_rotations

    @property
    def last_429_info(self) -> dict:
        return dict(self._last_429_info)

    @property
    def last_429_time(self) -> str:
        return self._last_429_time

    @property
    def blocked_groups(self) -> dict[int, float]:
        """Return {group_index: blocked_until_epoch} for currently blocked groups."""
        now = time.time()
        return {i: t for i, t in self._group_blocked_until.items() if t > now}

    def is_group_blocked(self, index: int) -> bool:
        """Check if a group is currently rate-limited."""
        until = self._group_blocked_until.get(index, 0)
        return until > time.time()

    def group_block_remaining(self, index: int) -> float:
        """Seconds remaining until group is unblocked. 0 if not blocked."""
        until = self._group_blocked_until.get(index, 0)
        remaining = until - time.time()
        return max(0, remaining)

    def get_next_available_index(self) -> int | None:
        """Find the next unblocked group index, starting from current. Returns None if all blocked."""
        if not self._groups:
            return None
        for offset in range(len(self._groups)):
            idx = (self._current_index + offset) % len(self._groups)
            if not self.is_group_blocked(idx):
                return idx
        return None  # All groups blocked

    # ── Rotation ────────────────────────────────

    def rotate(self, *, reason: str = "", error_message: str = "") -> tuple[int, str, list[str]]:
        """Switch to the next token group. Returns (new_index, group_name, tokens)."""
        is_429 = is_rate_limit_error(error_message or "")
        now = time.monotonic()
        # Cooldown only for non-429 rotations; 429 must always switch immediately
        if not is_429 and now - self._last_rotation < COOLDOWN_SECONDS and self._total_rotations > 0:
            logger.warning("Rotation cooldown (%.1fs < %ds), skipping", now - self._last_rotation, COOLDOWN_SECONDS)
            return (self._current_index, self.current_group_name, self.current_tokens)

        old_index = self._current_index

        # Record 429 info and block the current group
        if is_429:
            self._last_429_info = parse_429_info(error_message)
            self._last_429_time = datetime.now(SHA_TZ).strftime("%Y-%m-%d %H:%M")
            retry_ms = self._last_429_info.get("retry_after_ms", 0)
            if retry_ms > 0:
                self._group_blocked_until[old_index] = time.time() + (retry_ms / 1000)
                logger.warning(
                    "Group %s blocked ~%s (retry_after=%s)",
                    self._group_names[old_index] if old_index < len(self._group_names) else f"[{old_index}]",
                    self._last_429_info.get("retry_after_str", "?"),
                    self._last_429_info.get("reset_at_sha", "?"),
                )

            # Find next unblocked group, skip blocked ones
            skip_count = 0
            for _ in range(len(self._groups)):
                self._current_index = (self._current_index + 1) % len(self._groups)
                if not self.is_group_blocked(self._current_index):
                    break
                skip_count += 1
            if self.is_group_blocked(self._current_index):
                # All blocked → pick earliest unblock
                self._current_index = min(
                    range(len(self._groups)),
                    key=lambda i: self._group_blocked_until.get(i, 0),
                )
            suffix = f" (跳过 {skip_count} 个已限流组)" if skip_count > 0 else ""
        else:
            self._current_index = (self._current_index + 1) % len(self._groups) if self._groups else 0
            suffix = ""

        self._last_rotation = now
        self._total_rotations += 1
        tokens = self.current_tokens

        old_name = self._group_names[old_index] if old_index < len(self._group_names) else f"[{old_index}]"
        new_name = self._group_names[self._current_index] if self._current_index < len(self._group_names) else f"[{self._current_index}]"
        logger.warning(
            "Token rotation #%d: %s → %s (reason: %s, %d tokens)%s",
            self._total_rotations, old_name, new_name, reason, len(tokens), suffix,
        )

        self._write_env(",".join(tokens))
        self._write_current_token_num()

        return (self._current_index, self.current_group_name, tokens)

    def ensure_active_group(self) -> None:
        """On startup, write current group's tokens to .env so the API picks them up."""
        if not self._groups:
            logger.warning("No token groups loaded, cannot activate")
            return

        tokens = self.current_tokens
        token_str = ",".join(tokens)
        self._write_env(token_str)
        self._write_current_token_num()
        name = self._group_names[self._current_index] if self._current_index < len(self._group_names) else f"[{self._current_index}]"
        logger.info(
            "Activated group %s (%d tokens) → .env, CURRENT_TOKENNum=%d",
            name, len(tokens), self._current_index,
        )


# Singleton
_manager: TokenRotationManager | None = None


def get_rotation_manager() -> TokenRotationManager:
    global _manager
    if _manager is None:
        _manager = TokenRotationManager()
    return _manager


def is_rate_limit_error(error_message: str) -> bool:
    """Check if an error message string indicates a 429 rate limit."""
    return "429" in error_message and "rate_limited" in error_message


def parse_429_info(error_message: str) -> dict:
    """Extract rate-limit info from a 429 error message. Returns dict with:
    - reset_at_utc: str    original UTC reset time
    - reset_at_sha: str    Shanghai time formatted
    - retry_after_ms: int  milliseconds until reset
    - retry_after_str: str human-readable duration
    - model: str           the rate-limited model
    - limit: int           daily limit
    """
    info = {"reset_at_utc": "", "reset_at_sha": "", "retry_after_ms": 0,
            "retry_after_str": "", "model": "", "limit": 0}
    try:
        # Extract JSON payload from the error string
        m = re.search(r'429\s+(\{.*"rate_limited".*?\})\s*$', error_message)
        if not m:
            m = re.search(r'429\s+(\{.*\})', error_message)
        if not m:
            return info
        payload = json.loads(m.group(1))
        info["model"] = payload.get("model", "")
        info["limit"] = payload.get("limit", 0)

        # resetAt is UTC (ends with Z)
        reset_at_str = payload.get("resetAt", "")
        if reset_at_str:
            dt_utc = datetime.fromisoformat(reset_at_str.replace("Z", "+00:00"))
            dt_sha = dt_utc.astimezone(SHA_TZ)
            info["reset_at_utc"] = dt_utc.strftime("%Y-%m-%d %H:%M UTC")
            info["reset_at_sha"] = dt_sha.strftime("%Y-%m-%d %H:%M")

        # retryAfterMs to human-readable
        ms = payload.get("retryAfterMs", 0)
        info["retry_after_ms"] = ms
        if ms > 0:
            total_min = ms // 60000
            hours = total_min // 60
            mins = total_min % 60
            if hours > 0:
                info["retry_after_str"] = f"{hours}小时{mins}分钟"
            else:
                info["retry_after_str"] = f"{mins}分钟"
    except Exception:
        pass
    return info
