import socket
import numpy as np
import struct
import threading
import queue
from model import *
from tts import LocalTTS

IP = "0.0.0.0"
PORT = 5005

frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

tts = LocalTTS()


def recv_all(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def frame_receiver(conn):
    while not stop_event.is_set():
        size_header = recv_all(conn, 4)
        if not size_header:
            break

        frame_size = struct.unpack("!I", size_header)[0]
        frame_data = recv_all(conn, frame_size)
        if not frame_data:
            break

        np_arr = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            # Drop older, unprocessed frames if the queue is full
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)

    stop_event.set()


def background_computation(frame):
    d = describe_image(encode_image_to_base64(frame))
    print(d)
    print("-"*7)
    tts.speak(d)
    return frame


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((IP, PORT))
server_socket.listen(1)

print(f"TCP Receiver listening on port {PORT}...")
conn, addr = server_socket.accept()
print(f"Accepted connection from Pi: {addr}")

receiver_thread = threading.Thread(target=frame_receiver, args=(conn,), daemon=True)
receiver_thread.start()

try:
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        processed_frame = background_computation(frame)

        cv2.imshow("Original Feed", frame)
        cv2.imshow("Processed Output", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nReceiver stopped by user.")
finally:
    stop_event.set()
    receiver_thread.join(timeout=1.0)
    cv2.destroyAllWindows()
    conn.close()
    server_socket.close()
