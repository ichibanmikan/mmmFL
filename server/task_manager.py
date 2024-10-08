import json

class TaskManager:
    def __init__(self, filename):
        self.tasks = self.load_tasks(filename)

    def load_tasks(self, filename):
        with open(filename, 'r') as file:
            return json.load(file)

    def get_required_modalities(self, task_id):
        return self.tasks[task_id]["required_modalities"]

    def task_exists(self, task_id):
        return task_id in self.tasks
