import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from tool.web.main import _proxy_url, app


class TokenWebTests(unittest.TestCase):
    def test_public_page_has_no_backend_env_write_action(self) -> None:
        response = TestClient(app).get("/")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("/api/write_env", response.text)
        self.assertNotIn("writeEnvBtn", response.text)

    def test_write_env_endpoint_is_not_exposed(self) -> None:
        response = TestClient(app).post("/api/write_env", json={"token": "secret"})

        self.assertEqual(response.status_code, 404)

    def test_favicon_does_not_log_public_page_404_noise(self) -> None:
        response = TestClient(app).get("/favicon.ico")

        self.assertEqual(response.status_code, 204)

    def test_start_rejects_unknown_mode_before_upstream_call(self) -> None:
        response = TestClient(app).post("/api/start", json={"mode": "other"})

        self.assertEqual(response.status_code, 400)

    def test_web_tool_uses_proxy_only_when_enabled(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FREEBUFF_PROXY_ENABLED": "false",
                "FREEBUFF_PROXY_URL": "http://127.0.0.1:7890",
            },
        ):
            self.assertIsNone(_proxy_url())

        with patch.dict(
            "os.environ",
            {
                "FREEBUFF_PROXY_ENABLED": "true",
                "FREEBUFF_PROXY_URL": "http://127.0.0.1:7890",
            },
        ):
            self.assertEqual(_proxy_url(), "http://127.0.0.1:7890")


if __name__ == "__main__":
    unittest.main()
