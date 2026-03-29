"""
Interactive chat client for a running vLLM server.

Usage:
    python chat.py [--url URL] [--model MODEL] [--system PROMPT]
                   [--temperature T] [--max-tokens N]

Defaults:
    url         = http://localhost:8000/v1
    model       = auto-detected from server
    temperature = 0.7
    max-tokens  = 512

Commands during chat:
    /clear    - clear conversation history
    /system   - print current system prompt
    /quit     - exit (also Ctrl+C or Ctrl+D)
"""

import argparse
import json
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
    return p.parse_args()


def get_models(base_url):
    req = urllib.request.Request(f"{base_url}/models")
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read())
    return [m["id"] for m in data["data"]]


def chat_completion(base_url, model, messages, temperature, max_tokens):
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full_text = ""
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
                token = delta.get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_text += token
            except (KeyError, json.JSONDecodeError):
                pass

    print()
    return full_text


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
    print(f"Type /clear to reset history, /quit to exit.\n")

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
        elif user_input == "/system":
            print(f"[System: {args.system}]\n")
            continue

        history.append({"role": "user", "content": user_input})

        messages = [{"role": "system", "content": args.system}] + history

        print("Assistant: ", end="", flush=True)
        try:
            reply = chat_completion(
                args.url, model, messages, args.temperature, args.max_tokens
            )
            history.append({"role": "assistant", "content": reply})
        except urllib.error.URLError as e:
            print(f"\n[Request failed: {e}]")
        except Exception as e:
            print(f"\n[Error: {e}]")

        print()


if __name__ == "__main__":
    main()
