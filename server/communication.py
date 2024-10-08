import socket
import threading
import numpy as np
from server.communication import send_results
from server.mmFedAvg import calculate_average
from task_manager import TaskManager

import numpy as np
from communication import send_results, receive_data

class ClientHandler:
    def __init__(self, client_socket, task_manager):
        self.client_socket = client_socket
        self.task_manager = task_manager

    def handle(self):
        with self.client_socket:
            hello_message = self.client_socket.recv(1024).decode()
            # print("Received from client:", hello_message)
            modalities = self.client_socket.recv(1024).decode().split()
            # print("Client modalities:", modalities)

            # 发送需要的模态信息
            required_modalities = self.get_required_modalities(modalities)
            self.client_socket.sendall("Required modalities: ".encode() + required_modalities.encode())

            # 第二次交互：接收训练结果
            training_result = receive_data(self.client_socket)  # 假设这是一个1D数组
            print("Received training result:", training_result)

            # 发送收到确认
            self.client_socket.sendall("Training result received.".encode())

            # 计算并发送数组平均值
            average_result = self.calculate_average(training_result)
            send_results(self.client_socket, average_result)

            # 第三次交互：完成确认
            self.client_socket.sendall("Completion confirmation.".encode())
            print("Connection ended.")

    def get_required_modalities(self, modalities):
        # 返回服务器需要的模态信息（示例逻辑）
        required_modalities = []  # 你需要根据任务管理来生成这个列表
        for task_id in self.task_manager.tasks.keys():
            required_modalities += self.task_manager.get_required_modalities(task_id)
        return ' '.join(set(required_modalities))

    def calculate_average(self, training_result):
        # 假设 training_result 是一维数组
        result_array = np.frombuffer(training_result, dtype=np.float32)
        average = np.mean(result_array)
        return average


def send_results(client_socket, results):
    client_socket.sendall(results.tobytes())
