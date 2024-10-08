import socket
import threading
from task_manager import TaskManager
from communication import ClientHandler

HOST = '127.0.0.1'
PORT = 12345
MAX_CLIENTS = 5

class Server:
    def __init__(self):
        self.task_manager = TaskManager('tasks.json')

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((HOST, PORT))
            server_socket.listen(MAX_CLIENTS)
            print("Server is running, waiting for connections...")

            while True:
                client_socket, addr = server_socket.accept()
                print(f"Connected by {addr}")
                handler = ClientHandler(client_socket, self.task_manager)
                threading.Thread(target=handler.handle).start()

if __name__ == "__main__":
    server = Server()
    server.start()
