import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from freebuff2api.upstream_fingerprint import (
    DEFAULT_CODEBUFF_JSON_USER_AGENT,
    DEFAULT_FREEBUFF_CLI_USER_AGENT,
    DEFAULT_UPSTREAM_CHAT_KEYS,
    UpstreamFingerprint,
    _extract_openai_compatible_chat_keys,
    load_upstream_fingerprint_config,
    write_upstream_fingerprint_config,
)


class UpstreamFingerprintTests(unittest.TestCase):
    def test_empty_or_invalid_config_uses_code_defaults(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fingerprint.json"
            path.write_text("", encoding="utf-8")

            empty = load_upstream_fingerprint_config(path)
            path.write_text("{not json", encoding="utf-8")
            invalid = load_upstream_fingerprint_config(path)

        self.assertEqual(empty.codebuff_json_user_agent, DEFAULT_CODEBUFF_JSON_USER_AGENT)
        self.assertEqual(invalid.freebuff_cli_user_agent, DEFAULT_FREEBUFF_CLI_USER_AGENT)

    def test_config_round_trip_keeps_cached_official_values(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fingerprint.json"
            fingerprint = UpstreamFingerprint(
                codebuff_json_user_agent="Bun/9.9.9",
                freebuff_cli_user_agent="Freebuff-CLI/1.2.3",
                upstream_chat_keys=("model", "messages", "verbosity"),
                source="test",
                synced_at="2026-06-24T00:00:00Z",
            )

            write_upstream_fingerprint_config(fingerprint, path)
            loaded = load_upstream_fingerprint_config(path)

        self.assertEqual(loaded.codebuff_json_user_agent, "Bun/9.9.9")
        self.assertEqual(loaded.freebuff_cli_user_agent, "Freebuff-CLI/1.2.3")
        self.assertEqual(loaded.upstream_chat_keys, ("messages", "model", "verbosity"))
        self.assertEqual(loaded.source, "test")

    def test_extracts_chat_request_keys_from_official_source_shape(self) -> None:
        source = """
        return {
          args: {
            model: this.modelId,
            max_tokens: maxOutputTokens,
            reasoning_effort: compatibleOptions.reasoningEffort,
            verbosity: compatibleOptions.textVerbosity,
            messages: convertToOpenAICompatibleChatMessages(prompt),
            tools: openaiTools,
          },
        }
        """

        keys = _extract_openai_compatible_chat_keys(source)

        self.assertLessEqual(
            {"model", "max_tokens", "reasoning_effort", "verbosity", "messages", "tools"},
            keys,
        )

    def test_fetch_official_fingerprint_from_mocked_github(self) -> None:
        from freebuff2api import upstream_fingerprint

        files = {
            "/package.json": json.dumps(
                {
                    "packageManager": "bun@1.3.15",
                    "overrides": {"@ai-sdk/provider-utils": "3.0.21"},
                }
            ),
            "/.bun-version": "1.3.15\n",
            "/freebuff/cli/release/package.json": json.dumps({"version": "0.0.114"}),
            "/packages/llm-providers/src/openai-compatible/version.ts": (
                "export const VERSION: string = '0.0.0-test'\n"
            ),
            "/packages/llm-providers/src/openai-compatible/chat/openai-compatible-chat-language-model.ts": (
                "return { args: { model: this.modelId, verbosity: value, messages: [] } }"
            ),
            "/cli/src/hooks/use-gravity-ad.ts": "const AD_CHROME_VERSION = '125.0.0.0'",
        }

        class MockAsyncClient:
            def __init__(self, **kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def get(self, url: str, headers=None):
                path = "/" + url.split("https://example.test/", 1)[1]
                return httpx.Response(
                    200,
                    text=files[path],
                    request=httpx.Request("GET", url),
                )

        original = upstream_fingerprint.httpx.AsyncClient
        upstream_fingerprint.httpx.AsyncClient = MockAsyncClient
        try:
            fetched = asyncio.run(
                upstream_fingerprint.fetch_official_upstream_fingerprint(
                    UpstreamFingerprint(),
                    raw_base_url="https://example.test",
                    os_name="windows",
                )
            )
        finally:
            upstream_fingerprint.httpx.AsyncClient = original

        self.assertEqual(fetched.codebuff_json_user_agent, "Bun/1.3.15")
        self.assertEqual(fetched.freebuff_cli_user_agent, "Freebuff-CLI/0.0.114")
        self.assertIn("ai-sdk/provider-utils/3.0.21", fetched.chat_completions_user_agent)
        self.assertIn("Chrome/125.0.0.0", fetched.har_browser_user_agent)
        self.assertIn("verbosity", fetched.upstream_chat_keys)
        self.assertLessEqual(set(DEFAULT_UPSTREAM_CHAT_KEYS), set(fetched.upstream_chat_keys))


if __name__ == "__main__":
    unittest.main()
