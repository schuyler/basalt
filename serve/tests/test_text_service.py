"""The server with a text-prompt engine: usage and message_start carry the engine's token count, and a prompt over the
context is still a 400 before the stream starts."""
import http.client
import json
import unittest
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.prompt import CTX_SLACK, PromptText
from serve.service import Service
from serve.util.text import IM_END, ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
CTX = 4096


class TextEngine(MockEngine):
    """Counts the text like an engine would (one token per byte here) and answers with the script."""
    takes_text = True

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):
        if isinstance(ids, PromptText):
            self.seen_text = ids.text
            n = len(ByteTokenizer().encode(ids.text, parse_special=True))   # what an engine would count
            room = CTX - CTX_SLACK - n
            if ids.clip:
                if room < 1:
                    raise ValueError(ids.explain(f"context_no_room prompt={n} context={CTX}: prompt ({n} tokens) leaves no room to answer in the context ({CTX})"))
                max_new = min(max_new, room)
            elif max_new > room:
                raise ValueError(ids.explain(f"context_overflow prompt={n} max_new={max_new} context={CTX}: prompt ({n} tokens) + max_new ({max_new}) exceeds the context ({CTX})"))
            ids.set_counted(n, max_new)
            yield None
            ids = ids.ids()
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class TextService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = TextEngine(tok, "ok", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, r.read().decode()
        finally:
            c.close()

    def chat(self, content, **kw):
        return {"model": "basalt", "messages": [{"role": "user", "content": content}], **kw}

    def test_usage_carries_the_engines_count_and_the_text_is_what_the_template_rendered(self):
        status, body = self.post("/v1/chat/completions", self.chat("hi", max_tokens=8))
        self.assertEqual(status, 200, body)
        usage = json.loads(body)["usage"]
        want = len(ByteTokenizer().encode(self.engine.seen_text, parse_special=True))
        self.assertEqual(usage["prompt_tokens"], want)
        self.assertIn("<|im_start|>user\nhi<|im_end|>", self.engine.seen_text)

    def test_anthropic_message_start_has_the_counted_input_tokens(self):
        status, body = self.post("/v1/messages", {"model": "basalt", "max_tokens": 8, "stream": True,
                                                  "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200, body)
        start = [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:")][0]
        self.assertEqual(start["type"], "message_start")
        self.assertEqual(start["message"]["usage"]["input_tokens"],
                         len(ByteTokenizer().encode(self.engine.seen_text, parse_special=True)))

    def test_responses_usage_carries_the_engines_count(self):
        for stream in (False, True):
            status, body = self.post("/v1/responses", {"model": "basalt", "input": "hi " * 50, "max_output_tokens": 8,
                                                       "stream": stream})
            self.assertEqual(status, 200, body)
            if stream:
                events = [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:")]
                response = [e for e in events if e.get("type") == "response.completed"][0]["response"]
            else:
                response = json.loads(body)
            usage = response["usage"]
            self.assertEqual(usage["input_tokens"], len(ByteTokenizer().encode(self.engine.seen_text, parse_special=True)))
            self.assertEqual(usage["total_tokens"], usage["input_tokens"] + usage["output_tokens"])

    def test_a_prompt_over_the_context_is_a_400_before_streaming(self):
        for stream in (False, True):
            status, body = self.post("/v1/chat/completions", self.chat("y" * (CTX + 10), max_tokens=8, stream=stream))
            self.assertEqual(status, 400, body)
            self.assertIn("exceeds the context window", json.loads(body)["error"]["message"])

    def test_an_unset_cap_is_clipped_to_the_room(self):
        status, body = self.post("/v1/chat/completions", self.chat("hi"))
        self.assertEqual(status, 200, body)


class ReplyEngine(TextEngine):
    """Scripts a long thinking, the close, then an answer ending in ANSWER_END. A text pass is counted and settled as
    the engine would; an id-list pass that does not fit the context is refused, as the engine's GEN refuses it."""
    THINK = "Thinking it through, one step at a time. " * 40
    ANSWER_END = "ANSWER-END"
    ANSWER = "ANSWER-START " + "the answer goes on. " * 25 + ANSWER_END

    def __init__(self, tok):
        super().__init__(tok, "x", max_context=CTX)
        end = tok.encode(IM_END, parse_special=True)
        self.full = tok.encode(self.THINK + "\n</think>\n\n" + self.ANSWER, parse_special=True) + end
        self.answer = tok.encode(self.ANSWER, parse_special=True) + end

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):
        if isinstance(ids, PromptText):
            n = len(ids.ids())
            max_new = ids.settle(n, int(max_new), CTX)
            yield None
            out = self.full
        else:
            if len(ids) + max_new + CTX_SLACK > CTX:
                raise ValueError(f"context_overflow prompt={len(ids)} max_new={max_new} context={CTX}")
            out = self.answer
        for t in out[:max_new]:
            if cancel.is_set():
                return
            yield t


class ReplyCap(unittest.TestCase):
    """A thinking budget limits only the thinking: the reply's cap is max_tokens, or the room the context leaves."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(ReplyEngine(tok), tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def ask(self, content="Say hi.", **extra):
        body = {"model": "basalt", "messages": [{"role": "user", "content": content}], **extra}
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, json.loads(r.read())
        finally:
            c.close()

    def assert_complete(self, **extra):
        status, body = self.ask(**extra)
        self.assertEqual(status, 200, body)
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertIn(ReplyEngine.ANSWER_END, choice["message"]["content"])

    def test_a_cap_below_the_room_with_a_budget_is_the_replys_cap(self):
        self.assert_complete(max_tokens=3000, reasoning_budget_tokens=100)

    def test_an_unset_cap_with_a_budget_is_the_room(self):
        self.assert_complete(reasoning_budget_tokens=100)

    def test_a_cap_below_the_room_without_a_budget_is_the_replys_cap(self):
        self.assert_complete(max_tokens=3000)

    def test_an_unset_cap_without_a_budget_is_the_room(self):
        self.assert_complete()

    def test_a_cap_above_the_context_with_a_budget_is_a_400(self):
        status, body = self.ask(max_tokens=CTX + 1000, reasoning_budget_tokens=100)
        self.assertEqual(status, 400, body)
        self.assertIn("exceeds the context window", body["error"]["message"])

    def test_a_cap_above_the_context_without_a_budget_is_a_400(self):
        status, body = self.ask(max_tokens=CTX + 1000)
        self.assertEqual(status, 400, body)
        self.assertIn("exceeds the context window", body["error"]["message"])

    def test_a_room_smaller_than_the_budget_ends_the_reply_at_the_room(self):
        status, body = self.ask("Say hi. " + "filler " * 490, reasoning_budget_tokens=1000)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(body["usage"]["completion_tokens"], CTX - CTX_SLACK - body["usage"]["prompt_tokens"])


if __name__ == "__main__":
    unittest.main()
