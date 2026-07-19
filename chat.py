"""
Interactive chat client for a running vLLM server.

Usage:
    python chat.py [--url URL] [--model MODEL] [--system PROMPT]
                   [--temperature T] [--max-tokens N] [--tools]

Defaults:
    url         = http://localhost:8000/v1
    model       = auto-detected from server
    temperature = 0.7
    max-tokens  = 512
    tools       = off (enable with --tools; the server must be started with
                  a tool parser, e.g. serve.py --tool-call-parser gemma4
                  --enable-auto-tool-choice)

Commands during chat:
    /clear           - clear conversation history
    /system          - print current system prompt
    /system <text>   - set a new system prompt
    /quit            - exit (also Ctrl+C or Ctrl+D)
"""

import argparse
import html.parser
import json
import re
import sys
import urllib.error
import urllib.request


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/v1")
    p.add_argument("--model", default=None,
                   help="Model name (auto-detected if omitted)")
    p.add_argument("--system", default="You are a helpful assistant.",
                   help="System prompt")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--tools", action="store_true",
                   help="Offer built-in tools (web_fetch) to the model. The "
                        "server must be started with a tool parser, e.g. "
                        "serve.py --tool-call-parser gemma4 --enable-auto-tool-choice")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Built-in tools

class _TextExtractor(html.parser.HTMLParser):
    """Collect visible text from an HTML page, skipping script/style."""

    _SKIP = {"script", "style", "noscript", "template"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self.parts.append(data)


def web_fetch(url: str, max_chars: int = 8000) -> str:
    if not url.startswith(("http://", "https://")):
        return "Error: only http(s) URLs are supported."
    # A browser-like UA: many sites reject obvious script UAs outright.
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) "
                      "Gecko/20100101 Firefox/140.0",
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            content_type = r.headers.get("Content-Type", "")
            body = r.read(1_000_000).decode("utf-8", "replace")
    except Exception as e:
        return f"Error fetching {url}: {e}"
    if "html" in content_type:
        extractor = _TextExtractor()
        extractor.feed(body)
        body = " ".join(extractor.parts)
        body = re.sub(r"\s+", " ", body)
    if len(body) > max_chars:
        body = body[:max_chars] + f"\n[truncated at {max_chars} chars]"
    return body


TOOL_SPECS = [{
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Fetch a web page over HTTP(S) and return its visible "
                       "text content (HTML is stripped to plain text). Some "
                       "consumer sites block automated access; prefer "
                       "API-friendly endpoints when they exist (e.g. for "
                       "weather use wttr.in/<city>, open-meteo.com, or "
                       "api.weather.gov instead of commercial weather sites).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
            },
            "required": ["url"],
        },
    },
}]

TOOL_IMPLS = {"web_fetch": web_fetch}

MAX_TOOL_ROUNDS = 5


def get_models(base_url):
    req = urllib.request.Request(f"{base_url}/models")
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read())
    return [m["id"] for m in data["data"]]


def chat_completion(base_url, model, messages, temperature, max_tokens,
                    tools=None):
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        body["tools"] = tools
    payload = json.dumps(body).encode()

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full_text = ""
    tool_calls = {}  # index -> {"id", "type", "function": {"name", "arguments"}}
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw_line in r:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                delta = json.loads(chunk)["choices"][0]["delta"]
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
            token = delta.get("content", "")
            if token:
                print(token, end="", flush=True)
                full_text += token
            for tc in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(tc.get("index", 0), {
                    "id": None, "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]

    print()
    return full_text, [tool_calls[i] for i in sorted(tool_calls)]


def run_tool_call(call):
    name = call["function"]["name"]
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return f"Error: unknown tool '{name}'"
    try:
        arguments = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError as e:
        return f"Error: could not parse tool arguments: {e}"
    try:
        return impl(**arguments)
    except TypeError as e:
        return f"Error: bad tool arguments: {e}"


def main():
    args = parse_args()

    # Detect model name from server
    model = args.model
    if model is None:
        try:
            models = get_models(args.url)
            model = models[0]
        except Exception as e:
            print(f"Could not reach server at {args.url}: {e}", file=sys.stderr)
            print("Is serve.py running?", file=sys.stderr)
            sys.exit(1)

    print(f"Connected to {args.url}")
    print(f"Model: {model}")
    print("Type /clear to reset history, /system to view/set the system prompt, "
          "/quit to exit.\n")

    system_prompt = args.system
    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye.")
            break
        elif user_input == "/clear":
            history = []
            print("[History cleared]\n")
            continue
        elif user_input == "/system" or user_input.startswith("/system "):
            new_prompt = user_input[len("/system"):].strip()
            if new_prompt:
                system_prompt = new_prompt
                print(f"[System prompt updated: {system_prompt}]\n")
            else:
                print(f"[System: {system_prompt}]\n")
            continue

        history.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                messages = [{"role": "system", "content": system_prompt}] + history
                reply, tool_calls = chat_completion(
                    args.url, model, messages, args.temperature,
                    args.max_tokens, tools=TOOL_SPECS if args.tools else None,
                )
                if not tool_calls:
                    history.append({"role": "assistant", "content": reply})
                    break
                history.append({"role": "assistant", "content": reply or None,
                                "tool_calls": tool_calls})
                for call in tool_calls:
                    name = call["function"]["name"]
                    print(f"[tool] {name}({call['function']['arguments']})")
                    result = run_tool_call(call)
                    history.append({"role": "tool",
                                    "tool_call_id": call["id"],
                                    "name": name,
                                    "content": result})
                print("Assistant: ", end="", flush=True)
            else:
                print(f"[Stopped after {MAX_TOOL_ROUNDS} tool rounds]")
        except urllib.error.URLError as e:
            print(f"\n[Request failed: {e}]")
        except Exception as e:
            print(f"\n[Error: {e}]")

        print()


if __name__ == "__main__":
    main()
