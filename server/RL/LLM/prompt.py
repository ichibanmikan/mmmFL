
class Prompt_reward:
    def __init__(self):
        self.Context = """

# Context
We address a multi-task multimodal federated learning (FL) problem with the objective of minimizing the completion time for all tasks. The challenge arises from the heterogeneity in client resources and data distribution across participating clients. Specifically, clients have limited computational and communication capabilities, and the data distribution is non-uniform, potentially leading to missing modalities or labels. This results in imbalanced distributions of data, modalities, and client capabilities. Furthermore, different FL tasks have varying resource requirements and convergence times.

To tackle these challenges, we propose a framework for task and bandwidth allocation within each multi-task multimodal FL client. Task assignment involves selecting suitable clients for specific tasks based on their computational resources and data quality. For instance, a client with abundant computational resources and high-quality data for a long-running task would be prioritized for that task. Conversely, clients with limited resources or poor data quality may be assigned fewer training rounds or excluded from certain tasks. Bandwidth allocation aims to minimize the standard deviation of per-round time consumption across clients by allocating more bandwidth to slower tasks and less to faster ones, ensuring that transmission time does not become a bottleneck.

The original objective of "minimizing the total completion time for all multimodal FL tasks" is decomposed into two sub-objectives: (1) achieving convergence with the minimum number of rounds, and (2) minimizing the per-round computation and communication time. To ensure fairness and prevent overfitting, we impose a maximum participation time constraint for each client throughout the FL process. This constraint limits the cumulative time spent on computation and communication, preventing any single client from over-consuming resources or overfitting to a specific task.

Each round consists of three phases: (1) the server sends the global model to the clients, (2) clients perform local training, and (3) the server receives the updated models from the clients. For simplicity, we assume that the server's upload and download bandwidths are identical, allowing us to compute the transmission time based on a single bandwidth allocation ratio. The training time is determined by the local computation time of each client.

**After task assign for a round, the bandwidth is reallocated among the participating clients to ensure that the sum of their allocated bandwidth ratios equals one. Let K' denote the number of clients participating in a given round. This value can vary, with a maximum cap C. If the number of clients selected by the reinforcement learning system exceeds C, C clients are randomly chosen, and K'=C. Otherwise, K' equals the number of clients selected by the system. K' can be 0. To discourage rounds with too few participants, a penalty is imposed when K' is small. The total number of clients M and the number of tasks N remain constant throughout the process, and all models for the same task are of the same size. If the size of the array is M, then the index of the array means the number of the client. Similarly, the array index of size N is the task sequence number.**

Task assignment and bandwidth allocation are implemented using a hierarchical reinforcement learning (HRL) approach, which is a single-agent process. The server hosts an agent with two policies: a high-level policy for task assignment and a low-level policy for bandwidth allocation. In each round, the agent receives local information from each client and decides whether the client should participate in a specific task (represented by an integer from 1 to N) or not (represented by 0). This decision-making process ensures efficient resource utilization and fair participation across all clients.

Task Assignment Policy Inputs for Processing Client i:

1. Epoch Duration Vector (Shape: (N,)):
Estimated training time per task for client i in the current epoch, calculated using exponential weighted averaging based on historical data.

2. Training Loss Vector (Shape: (N,)):
Loss values for each task of client i. A high loss indicates poor data quality, while an excessively low loss suggests overfitting risks.

3. Accuracy Gap Vector (Shape: (N,)):
Distance of each task's accuracy from the convergence target.

4. Remaining Time (Shape: (1,)):
Client i's residual participation time in the current round. If selected, bandwidth allocation will be triggered.

Bandwidth Allocation Policy Inputs (Conditional on Client i's Selection):

1. Predicted Training Time (Shape: (1,)):
Estimated training time for client i in the current epoch, determined by the high-level assignment policy.

2. Remaining Time (Shape: (1,)):
Client i's residual participation time (repeated input for bandwidth calculation consistency).

3. Model Size (Shape: (1,)):
Size of the model parameters (in MB) associated with the tasks client i is expected to participate in.

**Notes:**

1. All vectors/matrices follow NumPy-style shape notation.
2. Inputs are logically grouped into two distinct modules (task assignment and bandwidth allocation).


        """
        self.Action = """ 
    
# Action
I hope you can, based on the above background, create a function that calculates the sub-rewards for the task assignment policy and the reward for bandwidth allocation for the hierarchical reinforcement learning described.

The following observational variables are provided for the current training round:

**Observational Set:**

1. Round training duration (shape: M)
   Facilitates assessment of the rationality of bandwidth allocation decisions. Greater than 0 if participated, otherwise 0.

2. Per-client transmission & training times (shape: M * 2)
   Quantifies the impact of client-task selection and bandwidth allocation on temporal efficiency. (a, b) and (a > 0 & b > 0) if participated, otherwise (0, 0).

3. Task accuracy increments (Δ) vs. previous round (shape: N)
   Computed as current accuracy - prior accuracy; indicates task assignment effectiveness on model improvement.
   Note: Reward scaling required (e.g., Δ0→1 merits higher reward than Δ90→91).

4. Current-round accuracy (shape: N)
   Provides reference for evaluating accuracy growth significance.

5. Predefined accuracy targets (shape: N)
   Enables convergence efficiency analysis by measuring progress toward system-level objectives.

6. Remaining training time budget (shape: M)
   Triggers penalty mechanisms if negative values occur.

7. Active client participation table (shape: M)
   Boolean type. True if participated, otherwise False.

**Action Decisions:**

1. Task assignment table (shape: M)
   High-level policy output determining client-task assignments.
   Implementation Note: Zero-assignment entries incur penalties when participation falls below minimum thresholds.

2. Bandwidth allocation table (shape: M)
   Low-level policy output governing resource distribution across clients.

I would like you to help me generate a function that calculates a set of 8 sub-rewards for the task assignment policy and a reward for bandwidth allocation. This set of sub-rewards includes factors such as the improvement in a client's accuracy, its impact on system stability, its effect on the training duration per round, some potential benefits, and other factors you select based on the input observations and action decisions.

**Notes:**
1. You must input all of them into the function you generate (even if you don't need them), and your function can select all or part of these inputs to calculate the sub-rewards.
2. A small subset of clients may be unsuitable for training certain tasks, resulting in excessively long training times for those tasks. Such clients and tasks should be identified and their participation in the task minimized.
3. A small subset of clients may exhibit limited computational capabilities, leading to prolonged training durations for every task. These clients should be avoided as much as possible.
4. A small subset of clients may experience missing modality data or labels. Decisions regarding their inclusion in a given task should be based on metrics such as loss performance or their reputation on that task (e.g., whether their participation consistently leads to a decline in test accuracy).
5. The data distribution among clients may be imbalanced, with each client potentially containing data corresponding to only one or a few label classes. Efforts should be made to equalize the frequency of participation of these clients.
6. The Active client participation table (Observational Set[7]) may be entirely False. Even if it is not entirely False, the sum of Per-client transmission & training times (Observational Set[2]) may still be 0, as some clients may be unable to participate in this training round due to insufficient remaining time or other reasons. Pay attention to boundary condition checks for the function inputs.
7. Considering 6., make sure to avoid NaN values in the function calculations. Also, ensure that each generated sub-reward does not contain NaN!
        """
        self.Purpose = """ 

# Purpose
Through the reward calculation function you provide for the two policies, I should be able to obtain a set of reasonable rewards in each round, effectively achieving the goals of minimizing the number of rounds required for convergence while simultaneously reducing the computational and communication time per round. And I can get the highest accuracy within a limited number of communication rounds.

        """
        
        self.Expectation = """

# Expectation
Please think step by step and generate content in the following JSON format (replace the content inside the () with your answer). Please do not use Python's triple quotes in the JSON string in your answers. Use double quotes "" instead.
{
  "Understand": (Your understanding of this task),  
  "Analyze": (Step-by-step analysis of which inputs can reflect potential positive and negative rewards),  
  "Functions": (A Python function in the form of `def reward_function(from Observational Set[1] to Observational Set[7], Action Decisions[1], Action Decisions[2]): ... return [reward_array, with shape (M, 9) -1 < for each r in reward_array[0-7] < 1 represent the sub-rewards for task assignment policy and reward_array[8] represents the reward for bandwidth allocation policy for client i, respectively]. Please do not use Python's triple quotes, as it will cause JSON errors. Use a single line of double quotes "" to wrap the function you generate.)
}
    """
    
    def get_context(self):
        return self.Context + self.Action + self.Purpose + self.Expectation
            
class Prompt_regenerate:
    def __init__(self, func, error_mess):
        self.content_1 = """
The code you generated:

        """
        
        self.content_2 = """

produced an error with the following message:
        
        """
        
        self.content_3 = """
Please think step by step and generate content in the following JSON format (replace the content inside the () with your answer). Please do not use Python's triple quotes in the JSON string in your answers. Use double quotes "" instead.
{
  "Understand": (Your understanding of this task),  
  "Analyze": (Step-by-step analysis of which inputs can reflect potential positive and negative rewards),  
  "Functions": (A Python function in the form of `def reward_function(from Observational Set[1] to Observational Set[7], Action Decisions[1], Action Decisions[2]): ... return [reward_array, with shape (M, 9) -1 < for each r in reward_array[0-7] < 1 represent the sub-rewards for task assignment policy and reward_array[8] represents the reward for bandwidth allocation policy for client i, respectively]. Please do not use Python's triple quotes, as it will cause JSON errors. Use a single line of double quotes "" to wrap the function you generate.)
}
        """
        self.func = func
        self.error_mess = error_mess
    def get_context(self):
        return self.content_1 + self.func + self.content_2 + self.error_mess + self.content_3
     
     

class Prompt_Summary:
   def __init__(self):
      self.Action_1 = """ 
    
# Action
Here is a series of reward calculation functions that you generated earlier:

"""
      self.Action_2 = """ 
Please summarize these functions and create a comprehensive function. The input and output should not be changed. Your scope of thinking should be limited to the functions that you have previously generated.      
"""
      self.Purpose = """ 

# Purpose
The comprehensive reward calculation function generated in this round takes into account all the features of the above functions. It can achieve better results than any of the individual functions above.

        """
        
      self.Expectation = """

# Expectation
Please think step by step and generate content in the following JSON format (replace the content inside the () with your answer). Please do not use Python's triple quotes in the JSON string in your answers. Use double quotes "" instead.
{
  "Understand": (Your understanding of this task),  
  "Analyze": (Step-by-step analysis of which inputs can reflect potential positive and negative rewards),  
  "Functions": (A Python function in the form of `def reward_function(from Observational Set[1] to Observational Set[7], Action Decisions[1], Action Decisions[2]): ... return [reward_array, with shape (M, 9) -1 < for each r in reward_array[0-7] < 1 represent the sub-rewards for task assignment policy and reward_array[8] represents the reward for bandwidth allocation policy for client i, respectively]. Please do not use Python's triple quotes, as it will cause JSON errors. Use a single line of double quotes "" to wrap the function you generate.)
}
    """
    
   def get_context(self, functions):
      str = self.Action_1
      for idx, func in enumerate(functions):
            str += f'{idx}. '
            str += func
            str += '\n'
      return str + self.Action_2 + self.Purpose + self.Expectation
   



