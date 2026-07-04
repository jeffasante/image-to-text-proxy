import http.server
import json
import urllib.request
import urllib.error
import os
import sys
import hashlib
import subprocess

# Configuration
PORT = 8099
GLM_API_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
OPENCODE_ZEN_URL = "https://opencode.ai/zen/v1/chat/completions"

# Known OpenCode Zen free models (as reported by `opencode models`)
OPENCODE_MODELS = {
    "big-pickle",
    "mimo-v2.5-free",
    "north-mini-code-free",
    "nemotron-3-ultra-free",
    "deepseek-v4-flash-free",
    "claude-sonnet-5",
    # also accept with prefix in case Zed sends it
    "opencode/big-pickle",
    "opencode/mimo-v2.5-free",
    "opencode/north-mini-code-free",
    "opencode/nemotron-3-ultra-free",
    "opencode/deepseek-v4-flash-free",
    "opencode/claude-sonnet-5",
}

# Key cache path
KEY_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".env")

def load_cached_key():
    if os.path.exists(KEY_CACHE_FILE):
        try:
            with open(KEY_CACHE_FILE, "r") as f:
                for line in f:
                    if line.strip().startswith("NVIDIA_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        if key.startswith(('"', "'")) and key.endswith(('"', "'")):
                            key = key[1:-1]
                        if key:
                            print(f"[*] Loaded cached Nvidia API key from {KEY_CACHE_FILE}")
                            return key
        except Exception as e:
            print(f"[-] Error loading cached Nvidia key from .env: {e}")
def load_opencode_key():
    path = os.path.expanduser("~/.local/share/opencode/auth.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                # Try getting the standard key, then the go key fallback
                key = data.get("opencode", {}).get("key") or data.get("opencode-go", {}).get("key")
                if key:
                    return key
        except Exception as e:
            print(f"[-] Error loading OpenCode key: {e}")
    return None

OPENCODE_API_KEY = load_opencode_key()
if OPENCODE_API_KEY:
    print("[*] Loaded OpenCode Zen API key from auth.json")

def save_cached_key(key):
    try:
        lines = []
        key_found = False
        if os.path.exists(KEY_CACHE_FILE):
            with open(KEY_CACHE_FILE, "r") as f:
                lines = f.readlines()
        
        for i, line in enumerate(lines):
            if line.strip().startswith("NVIDIA_API_KEY="):
                lines[i] = f"NVIDIA_API_KEY={key}\n"
                key_found = True
                break
                
        if not key_found:
            lines.append(f"NVIDIA_API_KEY={key}\n")
            
        with open(KEY_CACHE_FILE, "w") as f:
            f.writelines(lines)
        print(f"[+] Saved Nvidia API key to {KEY_CACHE_FILE}")
    except Exception as e:
        print(f"[-] Error saving Nvidia key to .env: {e}")

# Global cached key in memory
NVIDIA_API_KEY = load_cached_key()

# Persistent image description cache path
IMAGE_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".image_cache.json")

def load_image_cache():
    if os.path.exists(IMAGE_CACHE_FILE):
        try:
            with open(IMAGE_CACHE_FILE, "r") as f:
                cache = json.load(f)
                print(f"[*] Loaded {len(cache)} cached image descriptions from {IMAGE_CACHE_FILE}")
                return cache
        except Exception as e:
            print(f"[-] Error loading image cache: {e}")
    return {}

def save_image_cache():
    try:
        with open(IMAGE_CACHE_FILE, "w") as f:
            json.dump(IMAGE_CACHE, f)
    except Exception as e:
        print(f"[-] Error saving image cache: {e}")

# Global image cache loaded from disk
IMAGE_CACHE = load_image_cache()

def get_image_description(base64_image_data, mime_type="image/jpeg"):
    """
    Calls the vision model to describe the image.
    If an Nvidia API key is cached, it uses the cloud-hosted nvidia/nemotron or gpt-oss model.
    Otherwise, it falls back to a local vision model server (e.g. Ollama).
    """
    # Strip headers from base64 if present (e.g. "data:image/jpeg;base64,")
    base64_raw = base64_image_data
    if "," in base64_image_data:
        base64_raw = base64_image_data.split(",")[1]
        
    # Clean whitespace, newlines, and backslashes from raw base64 data to normalize it
    base64_raw = base64_raw.strip().replace("\n", "").replace("\r", "").replace(" ", "").replace("\\", "")
    
    # Compute hash of raw base64 image data to check if we already described it
    img_hash = hashlib.md5(base64_raw.encode('utf-8')).hexdigest()
    print(f"[*] Image hash: {img_hash} (Length: {len(base64_raw)})")
    
    if img_hash in IMAGE_CACHE:
        print(f"[*] Found cached image description (hash: {img_hash})")
        return IMAGE_CACHE[img_hash]
        
    global NVIDIA_API_KEY
    if not NVIDIA_API_KEY:
        NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
        
    if NVIDIA_API_KEY:
        # Use cloud Nvidia NIM model (nvidia/nemotron-3-nano-omni-30b-a3b-reasoning)
        vision_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        vision_model = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        print(f"[*] Calling Nvidia vision model ({vision_model}) via cloud API...")
        
        payload = {
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what is in this image or screenshot in detail, highlighting any UI elements, text, or code shown."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_raw}"
                            }
                        }
                    ]
                }
            ],
            "stream": False
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {NVIDIA_API_KEY}"
        }
    else:
        # Fall back to local vision model (e.g. Ollama)
        vision_url = os.environ.get("VISION_API_URL", "http://localhost:11434/api/chat")
        vision_model = os.environ.get("VISION_MODEL", "minicpm-v")
        print(f"[*] Calling local vision model ({vision_model}) at {vision_url}...")
        
        payload = {
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": "Describe what is in this image or screenshot in detail, highlighting any UI elements, text, or code shown.",
                    "images": [base64_raw]
                }
            ],
            "stream": False
        }
        headers = {"Content-Type": "application/json"}

    try:
        req = urllib.request.Request(
            vision_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
            if NVIDIA_API_KEY:
                # OpenAI compatible response format
                description = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                # Local runner response format
                description = result.get("message", {}).get("content", "")
                
            if description:
                print("[+] Vision model description retrieved successfully.")
                IMAGE_CACHE[img_hash] = description
                save_image_cache()
                return description
            else:
                print("[-] Empty response from vision model.")
                return "[Error: Empty response from vision model]"
            
    except Exception as e:
        print(f"[-] Error calling vision model: {e}")
        # If the Nvidia API failed, clear the cached key (maybe it expired or was invalid)
        if NVIDIA_API_KEY and "401" in str(e):
            print("[-] Clearing invalid cached Nvidia key.")
            NVIDIA_API_KEY = None
            if os.path.exists(KEY_CACHE_FILE):
                try:
                    os.remove(KEY_CACHE_FILE)
                except:
                    pass
        return f"[Error generating image description: {e}]"

class VisionProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to log to stdout
        sys.stdout.write("%s - - [%s] %s\n" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          format%args))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_POST(self):
        normalized_path = self.path.replace("//", "/")
        if normalized_path == "/v1/chat/completions" or normalized_path == "/chat/completions":
            self.handle_chat_completions()
        else:
            self.send_error(404, "Not Found")

    def handle_chat_completions(self):
        content_length = int(self.headers.get("Content-Length", 0))
        req_body = self.rfile.read(content_length).decode("utf-8")
        
        try:
            payload = json.loads(req_body)
        except Exception as e:
            self.send_error(400, f"Invalid JSON: {e}")
            return

        # 1. Inspect and preprocess messages
        processed_messages = []
        messages = payload.get("messages", [])
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            # Format assistant messages as plain strings (required by GLM-5.2)
            if role == "assistant" and isinstance(content, list):
                # Flatten the list of text blocks into a single string
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = "\n".join(text_parts)
                msg["content"] = content

            # Handle any messages (user, system, etc.) with image blocks
            elif isinstance(content, list):
                new_content_blocks = []
                for block in content:
                    if not isinstance(block, dict):
                        new_content_blocks.append(block)
                        continue
                        
                    block_type = block.get("type")
                    if block_type == "text":
                        new_content_blocks.append(block)
                    elif block_type == "image_url":
                        image_url_obj = block.get("image_url", {})
                        url_data = image_url_obj.get("url", "")
                        
                        if url_data.startswith("data:image/"):
                            # It is a base64 image payload from Zed
                            description = get_image_description(url_data)
                            new_content_blocks.append({
                                "type": "text",
                                "text": f"\n[User uploaded an image. Vision Model Description:\n{description}\n]"
                            })
                        else:
                            # External URL (could try fetching it or just describe it as URL)
                            new_content_blocks.append({
                                "type": "text",
                                "text": f"\n[User uploaded an image link: {url_data}]"
                            })
                msg["content"] = new_content_blocks
            
            processed_messages.append(msg)
            
        payload["messages"] = processed_messages
        
        # 2. Forward request dynamically
        auth_header = self.headers.get("Authorization", "")
        model_name = payload.get("model", "").lower()
        
        global NVIDIA_API_KEY
        
        # Determine route
        # Strip opencode/ prefix for model comparison
        model_bare = model_name.split("/")[-1] if "/" in model_name else model_name
        full_model = payload.get("model", "")

        is_opencode_route = model_name in OPENCODE_MODELS or model_bare in OPENCODE_MODELS
        is_deepseek_route = "deepseek" in model_name and not is_opencode_route
        is_glm_route = "glm-5.2" in model_name
        is_nvidia_route = not is_opencode_route and not is_deepseek_route and not is_glm_route

        if is_opencode_route:
            # OpenCode Zen models require curl to bypass Cloudflare bot detection
            # Always strip provider prefix — the API only wants the bare model name
            bare = payload.get("model", "").split("/")[-1]
            payload["model"] = bare
            payload.pop("prompt_cache_key", None)
            opencode_key = OPENCODE_API_KEY or (auth_header.split(" ")[1] if auth_header.startswith("Bearer ") else "")
            print(f"[*] Routing request to OpenCode Zen via curl ({bare})...")
            try:
                curl_result = subprocess.run(
                    [
                        "curl", "-s", "-X", "POST", OPENCODE_ZEN_URL,
                        "-H", "Content-Type: application/json",
                        "-H", f"Authorization: Bearer {opencode_key}",
                        "--data-raw", json.dumps(payload),
                        "--max-time", "120",
                    ],
                    capture_output=True,
                    timeout=125
                )
                response_body = curl_result.stdout
                if not response_body:
                    response_body = json.dumps({"error": {"message": f"OpenCode empty response: {curl_result.stderr.decode()}", "type": "proxy_error"}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(response_body)
            except subprocess.TimeoutExpired:
                self.send_response(504)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": "OpenCode request timed out", "type": "proxy_error"}}).encode())
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"[-] Client disconnected (OpenCode route): {e}")
            except Exception as e:
                print(f"[-] OpenCode Proxy Error: {e}")
                try:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": {"message": str(e), "type": "proxy_error"}}).encode())
                except Exception:
                    pass
            return
        elif is_nvidia_route:
            target_url = "https://integrate.api.nvidia.com/v1/chat/completions"
            print(f"[*] Routing request to Nvidia API ({model_name})...")
            # Strip prompt_cache_key which is unsupported by Nvidia API
            payload.pop("prompt_cache_key", None)
            
            # Cache the Nvidia API key automatically when it passes through
            if auth_header and auth_header.startswith("Bearer "):
                key = auth_header.split(" ")[1]
                if key != NVIDIA_API_KEY:
                    NVIDIA_API_KEY = key
                    save_cached_key(key)
        elif is_deepseek_route:
            target_url = "https://api.deepseek.com/chat/completions"
            print(f"[*] Routing request to DeepSeek API ({model_name})...")
        else:
            target_url = GLM_API_URL
            print(f"[*] Routing request to Z.AI GLM API ({model_name})...")
            
        req = urllib.request.Request(
            target_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": auth_header
            },
            method="POST"
        )
        
        try:
            response = urllib.request.urlopen(req, timeout=120)
            
            # Copy response headers from upstream
            self.send_response(response.getcode())
            for header, val in response.getheaders():
                # Avoid duplicate access-control headers
                if header.lower() not in ["access-control-allow-origin", "transfer-encoding"]:
                    self.send_header(header, val)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            # Stream the response chunk-by-chunk back to Zed
            while True:
                chunk = response.readline()
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                
        except urllib.error.HTTPError as e:
            err_content = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(err_content)
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"[-] Client disconnected prematurely (Broken Pipe / Connection Reset): {e}")
        except Exception as e:
            print(f"[-] Proxy Error: {e}")
            try:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                err_msg = json.dumps({"error": {"message": f"Proxy Error: {e}", "type": "proxy_error"}})
                self.wfile.write(err_msg.encode("utf-8"))
            except Exception as write_err:
                print(f"[-] Failed to send 500 error response: {write_err}")

def run_server():
    server_address = ("", PORT)
    httpd = http.server.HTTPServer(server_address, VisionProxyHandler)
    print(f"[*] Vision Proxy Server running on port {PORT}...")
    print(f"[*] Target GLM API: {GLM_API_URL}")
    if NVIDIA_API_KEY:
        print(f"[*] Local Vision API: integrate.api.nvidia.com (Model: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) [Cached key loaded]")
    else:
        print(f"[*] Local Vision API: http://localhost:11434/api/chat (Model: minicpm-v) [Ollama fallback]")
    print("[*] To use in Zed, configure your api_url to: http://localhost:8099/v1")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
