

# client.py
def chat(prompt: str, model: str = "llama3.2:3b"):
    """
    Client to consume the FastAPI endpoint with streaming.
    """
    from urllib.parse import urlencode
    import requests

    params = urlencode({"prompt": prompt, "model": model})
    url = f"http://localhost:8000/chat?{params}"
    print(f"Requesting: {url}\n")
    with requests.get(url, stream=True) as response:
        for chunk in response.iter_content(chunk_size=None):
            print(chunk.decode("utf-8"), end='', flush=True)

if __name__ == "__main__":
    chat("Salom, ahvoling qalay?", "llama3.2:3b")

