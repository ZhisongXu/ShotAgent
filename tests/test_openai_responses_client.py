import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from PIL import Image

from video_retouch.clients import (
    OpenAICompatibleVisionClient,
    OpenAIResponsesVisionClient,
)


class _ResponsesHandler(BaseHTTPRequestHandler):
    request_payload: dict[str, object] = {}
    request_path = ""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        type(self).request_payload = json.loads(self.rfile.read(length))
        type(self).request_path = self.path
        result = json.dumps(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"summary":"ok"}',
                            }
                        ],
                    }
                ]
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(result)))
        self.end_headers()
        self.wfile.write(result)

    def log_message(self, format, *args):
        del format, args


class _RateLimitHandler(BaseHTTPRequestHandler):
    request_count = 0

    def do_POST(self):
        type(self).request_count += 1
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if type(self).request_count == 1:
            result = json.dumps(
                {
                    "error": {
                        "code": 429,
                        "details": [
                            {
                                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                                "retryDelay": "3s",
                            }
                        ],
                    }
                }
            ).encode("utf-8")
            self.send_response(429)
        else:
            result = json.dumps(
                {"choices": [{"message": {"content": '{"summary":"ok"}'}}]}
            ).encode("utf-8")
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(result)))
        self.end_headers()
        self.wfile.write(result)

    def log_message(self, format, *args):
        del format, args


class _InvalidJsonHandler(BaseHTTPRequestHandler):
    request_count = 0
    request_payloads: list[dict[str, object]] = []

    def do_POST(self):
        type(self).request_count += 1
        length = int(self.headers.get("Content-Length", "0"))
        type(self).request_payloads.append(json.loads(self.rfile.read(length)))
        content = "{invalid" if type(self).request_count == 1 else '{"fixed":true}'
        result = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(result)))
        self.end_headers()
        self.wfile.write(result)

    def log_message(self, format, *args):
        del format, args


class OpenAIResponsesVisionClientTest(unittest.TestCase):
    def test_compatible_client_requests_json_and_repairs_invalid_json(self) -> None:
        _InvalidJsonHandler.request_count = 0
        _InvalidJsonHandler.request_payloads = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _InvalidJsonHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        old_key = os.environ.get("COMPATIBLE_TEST_KEY")
        os.environ["COMPATIBLE_TEST_KEY"] = "test-only"
        try:
            client = OpenAICompatibleVisionClient(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                model_id="vision-model",
                api_key_env="COMPATIBLE_TEST_KEY",
                json_mode=True,
                max_json_retries=1,
            )
            result = client.generate_json([], "return JSON")

            self.assertEqual(result, {"fixed": True})
            self.assertEqual(_InvalidJsonHandler.request_count, 2)
            first, second = _InvalidJsonHandler.request_payloads
            self.assertEqual(first["response_format"], {"type": "json_object"})
            self.assertEqual(second["messages"][-2]["role"], "assistant")
            self.assertIn("not valid JSON", second["messages"][-1]["content"])
        finally:
            if old_key is None:
                os.environ.pop("COMPATIBLE_TEST_KEY", None)
            else:
                os.environ["COMPATIBLE_TEST_KEY"] = old_key
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_compatible_client_retries_rate_limit_using_server_delay(self) -> None:
        _RateLimitHandler.request_count = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), _RateLimitHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        old_key = os.environ.get("COMPATIBLE_TEST_KEY")
        os.environ["COMPATIBLE_TEST_KEY"] = "test-only"
        try:
            client = OpenAICompatibleVisionClient(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                model_id="vision-model",
                api_key_env="COMPATIBLE_TEST_KEY",
                max_retries=1,
                retry_base_seconds=0.01,
                retry_max_seconds=10.0,
            )
            with patch("video_retouch.clients.time.sleep") as sleep:
                result = client.generate_json([], "return JSON")

            self.assertEqual(result, {"summary": "ok"})
            self.assertEqual(_RateLimitHandler.request_count, 2)
            sleep.assert_called_once_with(3.0)
        finally:
            if old_key is None:
                os.environ.pop("COMPATIBLE_TEST_KEY", None)
            else:
                os.environ["COMPATIBLE_TEST_KEY"] = old_key
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_sends_native_responses_multimodal_contract(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ResponsesHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        old_key = os.environ.get("RESPONSES_TEST_KEY")
        os.environ["RESPONSES_TEST_KEY"] = "test-only"
        try:
            client = OpenAIResponsesVisionClient(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                model_id="gpt-5.6-sol",
                api_key_env="RESPONSES_TEST_KEY",
                reasoning_effort="high",
                image_detail="high",
            )

            result = client.generate_json(
                [("frame_id=7", Image.new("RGB", (16, 12), (20, 40, 60)))],
                "return JSON",
            )

            self.assertEqual(result, {"summary": "ok"})
            self.assertEqual(_ResponsesHandler.request_path, "/v1/responses")
            request = _ResponsesHandler.request_payload
            self.assertEqual(request["model"], "gpt-5.6-sol")
            self.assertEqual(request["reasoning"], {"effort": "high"})
            content = request["input"][0]["content"]
            self.assertEqual(content[0]["type"], "input_text")
            image_item = next(
                item for item in content if item["type"] == "input_image"
            )
            self.assertEqual(image_item["detail"], "high")
            self.assertTrue(image_item["image_url"].startswith("data:image/jpeg"))
        finally:
            if old_key is None:
                os.environ.pop("RESPONSES_TEST_KEY", None)
            else:
                os.environ["RESPONSES_TEST_KEY"] = old_key
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
