from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


DEFAULT_CODEBUFF_JSON_USER_AGENT = "Bun/1.3.14"
DEFAULT_FREEBUFF_CLI_USER_AGENT = "Freebuff-CLI/0.0.113"
DEFAULT_CHAT_COMPLETIONS_USER_AGENT = (
    "ai-sdk/openai-compatible/0.0.0-test/codebuff "
    "ai-sdk/provider-utils/3.0.20 runtime/browser"
)
DEFAULT_HAR_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_UPSTREAM_CHAT_KEYS = (
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "metadata",
    "modalities",
    "parallel_tool_calls",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
    "verbosity",
)
DEFAULT_RESERVED_CODEBUFF_METADATA_KEYS = (
    "client_id",
    "cost_mode",
    "freebuff_instance_id",
    "n",
    "run_id",
    "trace_session_id",
)

OFFICIAL_GITHUB_REPO_URL = "https://github.com/CodebuffAI/codebuff"
OFFICIAL_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/CodebuffAI/codebuff/main"


@dataclass(frozen=True)
class UpstreamFingerprint:
    codebuff_json_user_agent: str = DEFAULT_CODEBUFF_JSON_USER_AGENT
    freebuff_cli_user_agent: str = DEFAULT_FREEBUFF_CLI_USER_AGENT
    chat_completions_user_agent: str = DEFAULT_CHAT_COMPLETIONS_USER_AGENT
    har_browser_user_agent: str = DEFAULT_HAR_BROWSER_USER_AGENT
    upstream_chat_keys: tuple[str, ...] = DEFAULT_UPSTREAM_CHAT_KEYS
    reserved_codebuff_metadata_keys: tuple[str, ...] = (
        DEFAULT_RESERVED_CODEBUFF_METADATA_KEYS
    )
    source: str = "code"
    synced_at: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = 1
        return data


def default_fingerprint_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "freebuff_fingerprint.json"


def github_repo_url_to_raw_base(url: str, branch: str = "main") -> str:
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        return OFFICIAL_GITHUB_RAW_BASE
    if "raw.githubusercontent.com" in cleaned:
        return cleaned

    parsed = urlparse(cleaned)
    if parsed.netloc.lower() != "github.com":
        return cleaned
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        return OFFICIAL_GITHUB_RAW_BASE
    owner = parts[0]
    repo = parts[1].removesuffix(".git")
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"


def load_upstream_fingerprint_config(path: Path | None = None) -> UpstreamFingerprint:
    config_path = path or default_fingerprint_config_path()
    try:
        raw = config_path.read_text(encoding="utf-8").strip()
    except OSError:
        return UpstreamFingerprint()
    if not raw:
        return UpstreamFingerprint()
    try:
        data = json.loads(raw)
    except ValueError:
        return UpstreamFingerprint()
    if not isinstance(data, dict):
        return UpstreamFingerprint()
    return _fingerprint_from_mapping(data, source_default="config")


def write_upstream_fingerprint_config(
    fingerprint: UpstreamFingerprint,
    path: Path | None = None,
) -> None:
    config_path = path or default_fingerprint_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(fingerprint.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(config_path)


async def fetch_official_upstream_fingerprint(
    current: UpstreamFingerprint,
    *,
    raw_base_url: str = OFFICIAL_GITHUB_RAW_BASE,
    source_url: str | None = None,
    proxy_url: str | None = None,
    timeout: float = 8.0,
    os_name: str = "windows",
) -> UpstreamFingerprint:
    raw_base = raw_base_url.rstrip("/")
    paths = {
        "package": "package.json",
        "bun_version": ".bun-version",
        "freebuff_release": "freebuff/cli/release/package.json",
        "openai_compatible_version": (
            "packages/llm-providers/src/openai-compatible/version.ts"
        ),
        "openai_compatible_chat": (
            "packages/llm-providers/src/openai-compatible/chat/"
            "openai-compatible-chat-language-model.ts"
        ),
        "gravity_ad": "cli/src/hooks/use-gravity-ad.ts",
    }
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        proxy=proxy_url,
        trust_env=False,
    ) as client:
        fetched = {}
        for name, path in paths.items():
            fetched[name] = await _fetch_text_or_empty(client, f"{raw_base}/{path}")

    package = _json_object(fetched["package"])
    release = _json_object(fetched["freebuff_release"])
    if not package and not release:
        raise RuntimeError("official GitHub fingerprint files unavailable")

    bun_version = (
        _clean_version(fetched["bun_version"])
        or _bun_version_from_package(package)
        or _version_from_user_agent(current.codebuff_json_user_agent, "Bun/")
        or "1.3.14"
    )
    freebuff_version = (
        _string_value(release.get("version"))
        or _version_from_user_agent(current.freebuff_cli_user_agent, "Freebuff-CLI/")
        or "0.0.113"
    )
    provider_utils_version = (
        _package_version(package, "@ai-sdk/provider-utils") or "3.0.20"
    )
    openai_compatible_version = (
        _extract_openai_compatible_fallback(fetched["openai_compatible_version"])
        or _version_from_chat_user_agent(current.chat_completions_user_agent)
        or "0.0.0-test"
    )
    chrome_version = (
        _extract_const_string(fetched["gravity_ad"], "AD_CHROME_VERSION")
        or _chrome_version_from_user_agent(current.har_browser_user_agent)
        or "124.0.0.0"
    )
    chat_keys = tuple(
        sorted(
            set(current.upstream_chat_keys)
            | set(DEFAULT_UPSTREAM_CHAT_KEYS)
            | _extract_openai_compatible_chat_keys(fetched["openai_compatible_chat"])
        )
    )

    return UpstreamFingerprint(
        codebuff_json_user_agent=f"Bun/{bun_version}",
        freebuff_cli_user_agent=f"Freebuff-CLI/{freebuff_version}",
        chat_completions_user_agent=(
            f"ai-sdk/openai-compatible/{openai_compatible_version}/codebuff "
            f"ai-sdk/provider-utils/{provider_utils_version} runtime/browser"
        ),
        har_browser_user_agent=_browser_user_agent(os_name, chrome_version),
        upstream_chat_keys=chat_keys,
        reserved_codebuff_metadata_keys=tuple(
            sorted(
                set(current.reserved_codebuff_metadata_keys)
                | set(DEFAULT_RESERVED_CODEBUFF_METADATA_KEYS)
            )
        ),
        source=source_url or raw_base,
        synced_at=datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    )


async def _fetch_text(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url, headers={"Accept": "text/plain,*/*"})
    response.raise_for_status()
    return response.text


async def _fetch_text_or_empty(client: httpx.AsyncClient, url: str) -> str:
    try:
        return await _fetch_text(client, url)
    except Exception:
        return ""


def _fingerprint_from_mapping(
    data: dict[str, Any],
    *,
    source_default: str,
) -> UpstreamFingerprint:
    defaults = UpstreamFingerprint()
    return UpstreamFingerprint(
        codebuff_json_user_agent=_string_value(
            data.get("codebuff_json_user_agent")
        )
        or defaults.codebuff_json_user_agent,
        freebuff_cli_user_agent=_string_value(data.get("freebuff_cli_user_agent"))
        or defaults.freebuff_cli_user_agent,
        chat_completions_user_agent=_string_value(
            data.get("chat_completions_user_agent")
        )
        or defaults.chat_completions_user_agent,
        har_browser_user_agent=_string_value(data.get("har_browser_user_agent"))
        or defaults.har_browser_user_agent,
        upstream_chat_keys=_string_tuple(
            data.get("upstream_chat_keys"),
            defaults.upstream_chat_keys,
        ),
        reserved_codebuff_metadata_keys=_string_tuple(
            data.get("reserved_codebuff_metadata_keys"),
            defaults.reserved_codebuff_metadata_keys,
        ),
        source=_string_value(data.get("source")) or source_default,
        synced_at=_string_value(data.get("synced_at")),
    )


def _json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _string_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, list):
        return default
    values = sorted({item.strip() for item in value if isinstance(item, str) and item.strip()})
    return tuple(values) or default


def _clean_version(text: str) -> str | None:
    stripped = text.strip()
    return stripped if re.fullmatch(r"\d+\.\d+\.\d+(?:[-.\w]*)?", stripped) else None


def _bun_version_from_package(package: dict[str, Any]) -> str | None:
    engines = package.get("engines")
    if isinstance(engines, dict):
        version = _string_value(engines.get("bun"))
        if version:
            return version
    manager = _string_value(package.get("packageManager"))
    if manager and manager.startswith("bun@"):
        return manager.split("@", 1)[1]
    return None


def _package_version(package: dict[str, Any], name: str) -> str | None:
    for section in ("overrides", "dependencies", "devDependencies"):
        values = package.get(section)
        if isinstance(values, dict):
            version = _string_value(values.get(name))
            if version:
                return version.lstrip("^~")
    return None


def _version_from_user_agent(user_agent: str, prefix: str) -> str | None:
    if user_agent.startswith(prefix):
        return user_agent[len(prefix) :].split(" ", 1)[0].strip() or None
    return None


def _version_from_chat_user_agent(user_agent: str) -> str | None:
    match = re.search(r"ai-sdk/openai-compatible/(\S+)/codebuff", user_agent)
    return match.group(1) if match else None


def _extract_openai_compatible_fallback(source: str) -> str | None:
    match = re.search(r":\s*['\"]([^'\"]+)['\"]", source)
    return match.group(1) if match else None


def _extract_const_string(source: str, name: str) -> str | None:
    match = re.search(rf"\b{name}\s*=\s*['\"]([^'\"]+)['\"]", source)
    return match.group(1) if match else None


def _chrome_version_from_user_agent(user_agent: str) -> str | None:
    match = re.search(r"Chrome/([0-9.]+)", user_agent)
    return match.group(1) if match else None


def _browser_user_agent(os_name: str, chrome_version: str) -> str:
    normalized = os_name.strip().lower()
    if normalized in {"macos", "darwin", "mac"}:
        platform = "Macintosh; Intel Mac OS X 10_15_7"
    elif normalized in {"linux"}:
        platform = "X11; Linux x86_64"
    else:
        platform = "Windows NT 10.0; Win64; x64"
    return (
        f"Mozilla/5.0 ({platform}) AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{chrome_version} Safari/537.36"
    )


def _extract_openai_compatible_chat_keys(source: str) -> set[str]:
    block = _extract_object_block(source, "args:")
    if not block:
        return set()
    keys = set()
    depth = 0
    token_start = 0
    index = 0
    while index < len(block):
        char = block[index]
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            token_start = index + 1
        elif char == ":" and depth == 0:
            token = block[token_start:index].strip()
            match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)$", token)
            if match:
                keys.add(match.group(1))
            token_start = index + 1
        index += 1
    return keys


def _extract_object_block(source: str, marker: str) -> str:
    marker_index = source.find(marker)
    if marker_index < 0:
        return ""
    start = source.find("{", marker_index)
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    return ""
