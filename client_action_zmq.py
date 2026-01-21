import zmq, cv2, base64, json

class GGUFActionClient:
    def __init__(self, server_ip="192.168.1.5", port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{server_ip}:{port}")
        print(f"[CLIENT] Connected to server tcp://{server_ip}:{port}")

    def send_frame(self, frame):
        _, jpeg = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(jpeg).decode("utf-8")
        self.socket.send_json({"image": b64})
        reply = self.socket.recv_json()
        return reply.get("action", None)
