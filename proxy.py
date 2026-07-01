import http.server
import json
import urllib.request
import urllib.error
import os
import sys
import hashlib

# Configuration
PORT = 8099
GLM_API_URL = "https://api.z.ai/api/paas/v4/chat/completions"

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
    return None

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

# In-memory image description cache to prevent re-processing the same images in chat history
IMAGE_CACHE = {}

def get_image_description(base64_image_data, mime_type="image/jpeg"):
    """
    Calls the vision model to describe the image.
    If an Nvidia API key is cached, it uses the cloud-hosted nvidia/nemotron or gpt-oss model.
    Otherwise, it falls back to a local vision model server (e.g. Ollama).
    """
    # Compute hash of base64 image data to check if we already described it
    img_hash = hashlib.md5(base64_image_data.encode('utf-8')).hexdigest()
    if img_hash in IMAGE_CACHE:
        print(f"[*] Found cached image description (hash: {img_hash})")
        return IMAGE_CACHE[img_hash]

    # Strip headers from base64 if present (e.g. "data:image/jpeg;base64,")
    base64_raw = base64_image_data
    if "," in base64_image_data:
        base64_raw = base64_image_data.split(",")[1]
        
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

            # Handle user messages with image blocks
            elif role == "user" and isinstance(content, list):
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
        
        # Check if the model requested is an Nvidia NIM model (contains nvidia, gpt-oss, nemotron, minimax, kimi, qwen, etc.)
        is_nvidia_route = False
        for keyword in ["nvidia", "gpt-oss", "nemotron", "minimax", "gemma-4", "kimi-k", "qwen3"]:
            if keyword in model_name:
                is_nvidia_route = True
                break
                
        if is_nvidia_route:
            target_url = "https://integrate.api.nvidia.com/v1/chat/completions"
            print(f"[*] Routing request to Nvidia API ({model_name})...")
            
            # Cache the Nvidia API key automatically when it passes through
            if auth_header and auth_header.startswith("Bearer "):
                key = auth_header.split(" ")[1]
                if key != NVIDIA_API_KEY:
                    NVIDIA_API_KEY = key
                    save_cached_key(key)
        elif "deepseek" in model_name:
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
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            err_msg = json.dumps({"error": {"message": f"Proxy Error: {e}", "type": "proxy_error"}})
            self.wfile.write(err_msg.encode("utf-8"))

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
