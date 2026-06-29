import asyncio
import json
import requests
import websockets

def test_stream():
    # 1. Create a job
    url = "http://127.0.0.1:10800/api/v1/analyze"
    # A 1x1 transparent PNG base64
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    payload = {
        "image_base64": base64_image,
        "mode": "fast"
    }
    resp = requests.post(url, json=payload)
    print("Analyze response status:", resp.status_code)
    job_data = resp.json()
    print("Analyze response data:", job_data)
    job_id = job_data["job_id"]
    
    # 2. Connect to WebSocket stream
    async def listen_ws():
        # Inside the container network, we can connect to localhost:10800
        ws_url = f"ws://localhost:10800/api/v1/jobs/{job_id}/stream"
        print(f"Connecting to WebSocket: {ws_url}")
        async with websockets.connect(ws_url) as websocket:
            print("WebSocket connected successfully!")
            try:
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    print("Received event:", data)
                    if data.get("status") in ["completed", "failed"]:
                        print("Stream finished with status:", data.get("status"))
                        break
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server.")
                
    asyncio.run(listen_ws())

if __name__ == "__main__":
    test_stream()
