import numpy as np

class ClientManager:
    _instance = None
    _client_id_counter = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClientManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.client_id = ClientManager._client_id_counter
        ClientManager._client_id_counter += 1
        self.modalities_results = None  # 用于存储按模态信息的二维数组
        self.task_results = None  # 用于存储按任务需求模态组合作为索引的二维数组
        self.modalities = []  # 模态信息的索引
        self.tasks = []  # 任务模态组合作为索引

    def set_modalities(self, modalities):
        self.modalities = modalities
        self.modalities_results = np.zeros((len(modalities), 0))  # 初始化二维数组

    def set_tasks(self, tasks):
        self.tasks = tasks
        self.task_results = np.zeros((len(tasks), 0))  # 初始化二维数组

    def update_modalities_results(self, client_id, result):
        if self.modalities_results is None:
            raise ValueError("Modalities results array is not initialized.")
        # 假设 result 是一维数组，将其添加到二维数组中
        self.modalities_results = np.append(self.modalities_results, result.reshape(-1, 1), axis=1)

    def update_task_results(self, task_index, result):
        if self.task_results is None:
            raise ValueError("Task results array is not initialized.")
        # 假设 result 是一维数组，将其添加到二维数组中
        self.task_results = np.append(self.task_results, result.reshape(-1, 1), axis=1)

    def get_client_id(self):
        return self.client_id

    def get_modalities_results(self):
        return self.modalities_results

    def get_task_results(self):
        return self.task_results
