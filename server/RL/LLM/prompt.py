
class Prompt:
    def __init__(self):
        self.Context = """

# Context
We address a multi-task multimodal federated learning (FL) problem with the objective of minimizing the completion time for all tasks. The challenge arises from the heterogeneity in client resources and data distribution across participating clients. Specifically, clients have limited computational and communication capabilities, and the data distribution is non-uniform, potentially leading to missing modalities or labels. This results in imbalanced distributions of data, modalities, and client capabilities. Furthermore, different FL tasks have varying resource requirements and convergence times.

To tackle these challenges, we propose a framework for task and bandwidth allocation within each multi-task multimodal FL client. Task allocation involves selecting suitable clients for specific tasks based on their computational resources and data quality. For instance, a client with abundant computational resources and high-quality data for a long-running task would be prioritized for that task. Conversely, clients with limited resources or poor data quality may be assigned fewer training rounds or excluded from certain tasks. Bandwidth allocation aims to minimize the standard deviation of per-round time consumption across clients by allocating more bandwidth to slower tasks and less to faster ones, ensuring that transmission time does not become a bottleneck.

The original objective of "minimizing the total completion time for all multimodal FL tasks" is decomposed into two sub-objectives: (1) achieving convergence with the minimum number of rounds, and (2) minimizing the per-round computation and communication time. To ensure fairness and prevent overfitting, we impose a maximum participation time constraint for each client throughout the FL process. This constraint limits the cumulative time spent on computation and communication, preventing any single client from over-consuming resources or overfitting to a specific task.

Each round consists of three phases: (1) the server sends the global model to the clients, (2) clients perform local training, and (3) the server receives the updated models from the clients. For simplicity, we assume that the server's upload and download bandwidths are identical, allowing us to compute the transmission time based on a single bandwidth allocation ratio. The training time is determined by the local computation time of each client.

After task allocation for a round, the bandwidth is reallocated among the participating clients to ensure that the sum of their allocated bandwidth ratios equals one. Let K' denote the number of clients participating in a given round. This value can vary, with a maximum cap C. If the number of clients selected by the reinforcement learning system exceeds C, C clients are randomly chosen, and K'=C. Otherwise, K' equals the number of clients selected by the system. To discourage rounds with too few participants, a penalty is imposed when K' is small. The total number of clients M and the number of tasks N remain constant throughout the process, and all models for the same task are of the same size.

Task scheduling and bandwidth allocation are implemented using a hierarchical reinforcement learning (HRL) approach, which is a single-agent process. The server hosts an agent with two policies: a high-level policy for task allocation and a low-level policy for bandwidth allocation. In each round, the agent receives local information from each client and decides whether the client should participate in a specific task (represented by an integer from 1 to N) or not (represented by 0). This decision-making process ensures efficient resource utilization and fair participation across all clients.

Task Allocation Policy Inputs for Processing Client i:

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
Estimated training time for client i in the current epoch, determined by the high-level scheduling policy.

2. Remaining Time (Shape: (1,)):
Client i's residual participation time (repeated input for bandwidth calculation consistency).

3. Model Size (Shape: (1,)):
Size of the model parameters (in MB) associated with the tasks client i is expected to participate in.

**Notes:**

1. Variables are formatted as italicized terms.
2. All vectors/matrices follow NumPy-style shape notation.
3. Inputs are logically grouped into two distinct modules (task allocation and bandwidth allocation).

        """
        self.Action = """ 
    
# Action
I hope you can, based on the above background, create a reward calculation function for the hierarchical reinforcement learning described.

The following observational variables are provided for the current training round:

**Observational Set:**

1. Round training duration (shape: K')
Facilitates assessment of the rationality of bandwidth allocation decisions.

2. Per-client transmission & training times (shape: K' * 2)
   Quantifies the impact of client-task selection and bandwidth allocation on temporal efficiency.

3. Task accuracy increments (Δ) vs. previous round (shape: N)
   Computed as current accuracy - prior accuracy; indicates task allocation effectiveness on model improvement.
   Note: Reward scaling required (e.g., Δ0→1 merits higher reward than Δ90→91).

4. Prior-round accuracy baselines (shape: N)
   Provides reference for evaluating accuracy growth significance.

5. Predefined accuracy targets (shape: N)
   Enables convergence efficiency analysis by measuring progress toward system-level objectives.

6. Remaining training time budget (shape: M)
   Triggers penalty mechanisms if negative values occur.

7. Active client participation table (shape: M)
   Identifies clients engaged in current training.

**Action Space:**

1. Task scheduling table (shape: M)
   High-level policy output determining client-task assignments.
   Implementation Note: Zero-scheduling entries incur penalties when participation falls below minimum thresholds.

2. Bandwidth allocation table (shape: M)
   Low-level policy output governing resource distribution across clients.

This structured representation preserves causal relationships between observational variables and policy actions while maintaining formal academic terminology. The hierarchical formatting enhances readability and emphasizes the dual-layer policy architecture.

I would like you to help me generate a function that calculates the rewards for two policies based on the input observations and action decisions.        
        
        """
        self.Purpose = """ 

# Purpose
Through the reward calculation function you provide for the two policies, I should be able to obtain a reasonable reward in each round, effectively achieving the goals of minimizing the number of rounds required for convergence while simultaneously reducing the computational and communication time per round.
        """
        
        self.Expectation = """
# Expectation
Please think step by step and generate content in the following JSON format (replace the content inside the () with your answer).
{
  "Understand": "(Your understanding of this task)",  
  "Analyze": "(Step-by-step analysis of which inputs can reflect potential positive and negative rewards)",  
  "Functions": "(A Python function in the form of `def get_reward(selected_array_from_observations, action_decisions): ... return [reward_array, with shape (M, 2) reward_array[i][0] and reward_array[i][1] represent the task allocation reward and bandwidth allocation reward for client i, respectively])"
}

    """