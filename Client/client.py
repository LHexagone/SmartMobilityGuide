import cv2
import socket
import sys
import struct
import time

PC_IP = "YOUR_PC_IP_ADDRESS"
PORT = 5005
TARGET_FPS = 25

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print(f"Connecting to PC at {PC_IP}:{PORT}...")
try:
    client_socket.connect((PC_IP, PORT))
    print("Connected successfully!")
except Exception as e:
    print(f"Failed to connect: {e}")
    sys.exit()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


frame_delay = 1.0 / TARGET_FPS

try:
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        encoded, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        data = buffer.tobytes()
        size = len(data)

        try:
            client_socket.sendall(struct.pack("!I", size))
            client_socket.sendall(data)
        except (ConnectionResetError, BrokenPipeError):
            print("\nPC disconnected from the network stream.")
            break

        elapsed = time.time() - start_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

except KeyboardInterrupt:
    print("\nStreaming stopped by user.")
finally:
    cap.release()
    client_socket.close()
    print("Resources released. Goodbye.")
