import http.server
import json
import urllib.request
import urllib.error
import os
import sys

# Configuration
PORT = 8099
GLM_API_URL = "https://api.z.ai/api/paas/v4/chat/completions"

# Vision Model Configuration
# By default, we support a local vision model server running a model like minicpm-v, llava, or qwen2.5-vl.
VISION_API_URL = os.environ.get("VISION_API_URL", "http://localhost:11434/api/chat")
VISION_MODEL = os.environ.get("VISION_MODEL", "minicpm-v")  # e.g., minicpm-v, llava, qwen2.5-vl

def get_image_description(base64_image_data, mime_type="image/jpeg"):
    """
    Calls the local vision model to describe the image.
    """
    print(f"[*] Calling vision model ({VISION_MODEL}) to describe image...")
    
    # Strip headers from base64 if present (e.g. "data:image/jpeg;base64,")
    if "," in base64_image_data:
        base64_image_data = base64_image_data.split(",")[1]
        
    try:
        # Request payload for the local vision API
        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Describe what is in this image or screenshot in detail, highlighting any UI elements, text, or code shown.",
                    "images": [base64_image_data]
                }
            ],
            "stream": False
        }
        
        req = urllib.request.Request(
            VISION_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            description = result.get("message", {}).get("content", "")
            print("[+] Vision model description retrieved successfully.")
            return description
            
    except Exception as e:
        print(f"[-] Error calling vision model: {e}")
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
        
        # 2. Forward request to Z.AI / GLM-5.2 or DeepSeek
        auth_header = self.headers.get("Authorization", "")
        
        # Route dynamically depending on the model
        model_name = payload.get("model", "").lower()
        if "deepseek" in model_name:
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
    print(f"[*] Local Vision API: {VISION_API_URL} (Model: {VISION_MODEL})")
    print("[*] To use in Zed, configure your api_url to: http://localhost:8000/v1")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
