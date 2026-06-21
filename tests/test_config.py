import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from freebuff2api.config import DEFAULT_ADMIN_KEY, Settings, load_settings, write_env_values


class ConfigTests(unittest.TestCase):
    def test_proxy_url_is_ignored_when_proxy_is_disabled(self) -> None:
        settings = Settings(
            codebuff_token="token",
            local_api_key=None,
            proxy_enabled=False,
            proxy_url="http://127.0.0.1:7890",
        )

        self.assertIsNone(settings.upstream_proxy_url)

    def test_proxy_url_is_used_when_proxy_is_enabled(self) -> None:
        settings = Settings(
            codebuff_token="token",
            local_api_key=None,
            proxy_enabled=True,
            proxy_url=" http://127.0.0.1:7890 ",
        )

        self.assertEqual(settings.upstream_proxy_url, "http://127.0.0.1:7890")

    def test_load_settings_reads_proxy_toggle_and_url(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FREEBUFF_PROXY_ENABLED": "true",
                "FREEBUFF_PROXY_URL": "socks5://127.0.0.1:1080",
            },
        ):
            settings = load_settings()

        self.assertTrue(settings.proxy_enabled)
        self.assertEqual(settings.upstream_proxy_url, "socks5://127.0.0.1:1080")

    def test_freebuff_api_base_url_overrides_legacy_codebuff_base_url(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FREEBUFF_API_BASE_URL": " https://freebuff-api.example.test/ ",
                "CODEBUFF_BASE_URL": "https://legacy.example.test",
            },
        ):
            settings = load_settings()

        self.assertEqual(settings.codebuff_api_url, "https://freebuff-api.example.test")

    def test_legacy_codebuff_base_url_is_still_supported(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "CODEBUFF_BASE_URL": "https://legacy.example.test/",
            },
            clear=True,
        ):
            settings = load_settings()

        self.assertEqual(settings.codebuff_api_url, "https://legacy.example.test")

    def test_codebuff_tokens_splits_comma_separated_tokens(self) -> None:
        settings = Settings(
            codebuff_token="token-a, token-b,,token-c ",
            local_api_key=None,
        )

        self.assertEqual(
            settings.codebuff_tokens,
            ("token-a", "token-b", "token-c"),
        )

    def test_load_settings_reads_admin_key_and_log_limit(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FREEBUFF_ADMIN_KEY": "admin-secret",
                "FREEBUFF_ADMIN_LOG_LINES": "250",
            },
            clear=True,
        ):
            settings = load_settings()

        self.assertEqual(settings.admin_key, "admin-secret")
        self.assertEqual(settings.admin_log_lines, 250)

    def test_load_settings_defaults_admin_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            settings = load_settings()

        self.assertEqual(settings.admin_key, DEFAULT_ADMIN_KEY)

    def test_load_settings_does_not_expose_custom_browser_user_agent(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FREEBUFF_BROWSER_UA": "custom-client/1.0",
            },
            clear=True,
        ):
            settings = load_settings()

        self.assertFalse(hasattr(settings, "browser_user_agent"))

    def test_write_env_values_updates_known_keys_and_preserves_comments(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "# local config\nFREEBUFF_TOKEN=old\nFREEBUFF_TIMEOUT=60\n",
                encoding="utf-8",
            )

            write_env_values(
                {"FREEBUFF_TOKEN": "new-a,new-b", "FREEBUFF_ADMIN_KEY": "admin"},
                env_path,
            )

            self.assertEqual(
                env_path.read_text(encoding="utf-8"),
                "# local config\nFREEBUFF_TOKEN=new-a,new-b\nFREEBUFF_TIMEOUT=60\nFREEBUFF_ADMIN_KEY=admin\n",
            )


if __name__ == "__main__":
    unittest.main()
