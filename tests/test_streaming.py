import asyncio
import json
import unittest
from types import SimpleNamespace

from freebuff2api.app import _start_freebuff_run_chain, _stream_openai_chunks
from freebuff2api.codebuff import CodebuffError, FreebuffRun
from freebuff2api.config import Settings
from freebuff2api.models import resolve_model


class FakeClient:
    def __init__(self) -> None:
        self.recorded = False
        self.finished = False
        self.calls = []

    async def chat_events(self, payload):
        yield (
            'data: {"id":"chunk-1","object":"chat.completion.chunk",'
            '"created":1,"model":"deepseek/deepseek-v4-flash",'
            '"choices":[{"index":0,"delta":{"content":null,'
            '"reasoning_content":"hello"},"finish_reason":null}]}'
        )
        yield "data: [DONE]"

    async def record_run_step(self, *args, **kwargs) -> None:
        self.recorded = True
        self.calls.append(("step", args, kwargs))
        await asyncio.sleep(0)

    async def finish_run(self, *args, **kwargs) -> None:
        self.finished = True
        self.calls.append(("finish", args, kwargs))
        await asyncio.sleep(0)

    async def start_run(self, agent_id, ancestor_run_ids=None):
        run_id = f"run-{len([call for call in self.calls if call[0] == 'start']) + 1}"
        self.calls.append(("start", agent_id, ancestor_run_ids or [], run_id))
        await asyncio.sleep(0)
        return run_id


class FailingStreamClient(FakeClient):
    async def chat_events(self, payload):
        raise CodebuffError("Codebuff chat failed: 403 hierarchy", 502)
        yield


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_forwards_content_before_finalize(self) -> None:
        client = FakeClient()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    codebuff=client,
                    settings=Settings(
                        codebuff_token="token",
                        local_api_key=None,
                        debug=False,
                    ),
                )
            )
        )

        chunks = []
        run = FreebuffRun(
            run_id="run-1",
            agent_id="base2-free-deepseek-flash",
            started_at="2026-05-23T00:00:00.000Z",
        )
        async for chunk in _stream_openai_chunks(request, {}, run):
            chunks.append(chunk.decode("utf-8"))

        first_payload = json.loads(chunks[0].removeprefix("data: ").strip())

        delta = first_payload["choices"][0]["delta"]
        self.assertNotIn("content", delta)
        self.assertEqual(delta["reasoning_content"], "hello")
        self.assertEqual(chunks[1], "data: [DONE]\n\n")

        await asyncio.sleep(0.05)
        self.assertTrue(client.recorded)
        self.assertTrue(client.finished)

    async def test_run_chain_matches_freebuff_parent_child_shape(self) -> None:
        client = FakeClient()

        run = await _start_freebuff_run_chain(client, "base2-free-kimi")

        self.assertEqual(run.run_id, "run-1")
        self.assertEqual(run.child_run_id, "run-2")
        self.assertEqual(client.calls[0], ("start", "base2-free-kimi", [], "run-1"))
        self.assertEqual(
            client.calls[1],
            ("start", "context-pruner", ["run-1"], "run-2"),
        )
        self.assertEqual(client.calls[2][0], "step")
        self.assertEqual(client.calls[2][1], ("run-2",))
        self.assertEqual(client.calls[2][2]["step_number"], 1)
        self.assertEqual(client.calls[2][2]["child_run_ids"], [])
        self.assertEqual(client.calls[2][2]["message_id"], None)
        self.assertEqual(client.calls[3], ("finish", ("run-2",), {"total_steps": 2}))
        self.assertEqual(
            client.calls[4],
            (
                "step",
                ("run-1",),
                {
                    "step_number": 1,
                    "child_run_ids": ["run-2"],
                    "message_id": None,
                    "start_time": run.started_at,
                },
            ),
        )

    async def test_gemini_thinker_run_chain_uses_child_as_payload_run(self) -> None:
        client = FakeClient()

        run = await _start_freebuff_run_chain(
            client,
            resolve_model("google/gemini-3.1-pro-preview"),
        )

        self.assertEqual(run.run_id, "run-1")
        self.assertEqual(run.chat_run_id, "run-2")
        self.assertEqual(run.payload_run_id, "run-2")
        self.assertEqual(client.calls[0], ("start", "base2-free-kimi", [], "run-1"))
        self.assertEqual(
            client.calls[1],
            ("start", "thinker-with-files-gemini", ["run-1"], "run-2"),
        )

    async def test_gemini_flash_lite_run_chain_uses_session_root_parent(self) -> None:
        client = FakeClient()

        run = await _start_freebuff_run_chain(
            client,
            resolve_model("google/gemini-2.5-flash-lite"),
        )

        self.assertEqual(run.run_id, "run-1")
        self.assertEqual(run.chat_run_id, "run-2")
        self.assertEqual(run.payload_run_id, "run-2")
        self.assertEqual(
            client.calls[0],
            ("start", "base2-free-deepseek-flash", [], "run-1"),
        )
        self.assertEqual(
            client.calls[1],
            ("start", "file-picker", ["run-1"], "run-2"),
        )

    async def test_streaming_codebuff_error_is_returned_as_sse_error(self) -> None:
        client = FailingStreamClient()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    codebuff=client,
                    settings=Settings(
                        codebuff_token="token",
                        local_api_key=None,
                        debug=False,
                    ),
                )
            )
        )

        chunks = []
        run = FreebuffRun(
            run_id="run-1",
            agent_id="base2-free-deepseek-flash",
            started_at="2026-05-23T00:00:00.000Z",
        )
        with self.assertLogs("freebuff2api.app", level="WARNING"):
            async for chunk in _stream_openai_chunks(request, {}, run):
                chunks.append(chunk.decode("utf-8"))

        error_payload = json.loads(chunks[0].removeprefix("data: ").strip())
        self.assertEqual(error_payload["error"]["code"], "codebuff_error")
        self.assertEqual(chunks[1], "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()
