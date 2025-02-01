import socket
import cv2
import numpy as np
from datetime import datetime

# Server configuration
SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000

# Create a TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_IP, SERVER_PORT))
server_socket.listen(1)

print(f"Listening for connections on {SERVER_IP}:{SERVER_PORT}...")

# Accept a connection
client_socket, client_address = server_socket.accept()
print(f"Connected to {client_address}")

# List to store frames
frames = []

# Buffer size for receiving frame data
BUFFER_SIZE = 65536  # Adjust based on expected frame size

while True:
    try:
        # Read the frame data directly
        frame_data = client_socket.recv(BUFFER_SIZE)
        if not frame_data:
            print("No frame data received. Exiting...")
            break

        # Decode the JPEG frame
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Check if the frame is valid
        if frame is None:
            print("Received invalid frame. Skipping...")
            continue

        # Store the frame in the list
        frames.append(frame)
        print(f"Frame {len(frames)} received and stored.")

    except Exception as e:
        print(f"Error: {e}")
        break

# Cleanup network connection
client_socket.close()
server_socket.close()

# Check if any frames were received
if not frames:
    print("No frames received. Exiting...")
    exit()

# Generate a timestamp for the video filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_video = f"output_video_{timestamp}.mp4"

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
fps = 30  # Adjust FPS as needed
height, width, _ = frames[0].shape
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Check if the video writer was initialized successfully
if not video_writer.isOpened():
    print("Error: Could not open video writer.")
    exit()

# Write all frames to the video
for frame in frames:
    video_writer.write(frame)

# Release the video writer
video_writer.release()

print(f"Video saved as {output_video}")