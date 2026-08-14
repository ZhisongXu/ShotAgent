import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from video_retouch.clients import OpenAIResponsesVisionClient


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


class OpenAIResponsesVisionClientTest(unittest.TestCase):
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
