
class Prompt_reward:
    def __init__(self):
        self.Context = """

# Context
We address a multi-task federated learning (FL) problem with the objective of minimizing the completion time for all tasks. The challenge arises from the heterogeneity in client resources and data distribution across participating clients. Specifically, clients have limited computational and communication capabilities, and the data distribution is non-uniform, potentially leading to missing modalities or labels. This results in imbalanced distributions of data, modalities, and client capabilities. Furthermore, different FL tasks have varying resource requirements and convergence times.

To tackle these challenges, we propose a framework for task and bandwidth allocation within each multi-task FL client. Task assignment involves selecting suitable clients for specific tasks based on their computational resources and data quality. For instance, a client with abundant computational resources and high-quality data for a long-running task would be prioritized for that task. Conversely, clients with limited resources or poor data quality may be assigned fewer training rounds or excluded from certain tasks. Bandwidth allocation aims to minimize the standard deviation of per-round time consumption across clients by allocating more bandwidth to slower tasks and less to faster ones, ensuring that transmission time does not become a bottleneck.

The original objective of "minimizing the total completion time for all FL tasks" is decomposed into two sub-objectives: (1) achieving convergence with the minimum number of rounds, and (2) minimizing the per-round computation and communication time. To ensure fairness and prevent overfitting, we impose a maximum participation time constraint for each client throughout the FL process. This constraint limits the cumulative time spent on computation and communication, preventing any single client from over-consuming resources or overfitting to a specific task.

Each training round consists of three stages: (1) the server broadcasts the global model to the selected clients, (2) the clients perform local training on their assigned tasks, and (3) the clients upload their updated model parameters to the server. Following common practice in federated learning and consistent with our system model, we focus exclusively on the uplink communication cost, since the base station typically has sufficiently high transmission power and stable energy supply, making the downlink latency negligible in comparison. Thus, the communication time of each client is determined solely by its uplink transmission rate under the allocated bandwidth fraction, while the training time depends on its local computation latency. **Since we are using synchronous FL, the latency of each round is determined by the client with the maximum latency.**

**After task assign for a round, the bandwidth is reallocated among the participating clients to ensure that the sum of their allocated bandwidth ratios equals one. Let K' denote the number of clients participating in a given round. This value can vary, with a maximum cap C. If the number of clients selected by the reinforcement learning system exceeds C, C clients are randomly chosen, and K'=C. Otherwise, K' equals the number of clients selected by the system. K' can be 0. To discourage rounds with too few participants, a penalty is imposed when K' is small. The total number of clients M and the number of tasks N remain constant throughout the process, and all models for the same task are of the same size. If the size of the array is M, then the index of the array means the number of the client. Similarly, the array index of size N is the task sequence number.**

Task assignment and bandwidth allocation are implemented using a hierarchical reinforcement learning (HRL) approach, which is a single-agent process. The server hosts an agent with two policies: a high-level policy for task assignment and a low-level policy for bandwidth allocation. In each round, the agent receives local information from each client and decides whether the client should participate in a specific task (represented by an integer from 1 to N) or not (represented by 0). This decision-making process ensures efficient resource utilization and fair participation across all clients.

Task Assignment Policy Inputs for Processing Client i:

1. Training Loss Vector (Shape: (N,)):
Loss values for each task of client i. A high loss indicates poor data quality, while an excessively low loss suggests overfitting risks.

2. Model Staleness Gap (Shape: (N,)):
The difference between the current round and the last round in which client i participated in task j. A large gap indicates outdated local models, leading to potential gradient drift and unstable convergence.

3. Remaining energy ratio (Shape: (1,)):
The ratio of remaining energy of client i relative to its total energy budget. A low remaining energy level requires cautious task assignment, while a completely depleted energy state makes the client ineligible for participation.

Bandwidth Allocation Policy Inputs (Conditional on Client i's Selection):

1. Model Size (Shape: (1,)):
Size of the model parameters (in MB) associated with the tasks client i is expected to participate in.

2. Remaining energy ratio (Shape: (1,)):
The ratio of remaining energy of client i relative to its total energy budget. A low remaining energy level requires cautious task assignment, while a completely depleted energy state makes the client ineligible for participation.

3. The channel gain (Shape: (1,)):
The channel gain of client i. A larger channel gain results in lower latency, while a smaller channel gain leads to higher latency. Therefore, bandwidth allocation needs to take this factor into account.

**Notes:**

1. All vectors/matrices follow NumPy-style shape notation.
2. Inputs are logically grouped into two distinct modules (task assignment and bandwidth allocation).


        """
        self.Action = """ 
    
# Action
You are a coder. I hope you can, based on the above background, create a function for the described hierarchical reinforcement learning. The inputs to this function include the current round’s FL training performance, the actions taken by each participant (including bandwidth allocation and task assignment), and the final computed global reward. The purpose of this function is to fairly distribute the global reward to each client based on their performance in the round. Even if the global reward is low, the function should still reward well-performing clients generously; conversely, it must identify clients that truly impact the system’s performance negatively and penalize them accordingly. The goal is to enable stable convergence of the HRL system.

## Input

The following variables are provided for the current training round:

**server:**
1. server is an instance of a class that contains a variable history_data, which is initially initialized as an empty dictionary (history_data = {}). You can store any useful information in this dictionary, such as each client's historical data, participation rounds, and the current communication round of federated learning.

**Observational Set:**

Observational Set[0]: Round training duration (shape: M)
   Facilitates assessment of the rationality of bandwidth allocation decisions. Greater than 0 if participated, otherwise 0.

Observational Set[1]: Per-client transmission & training times (shape: M * 2)
   Quantifies the impact of client-task selection and bandwidth allocation on temporal efficiency. (transmission time a, training time b) and (a > 0 & b > 0) if participated, otherwise (0, 0). The time for a round equals transmission time a + training time b.

Observational Set[2]: Task accuracy increments (Δ) vs. previous round (shape: N)
   Computed as current accuracy - prior accuracy; indicates task assignment effectiveness on model improvement.
   Note: Reward scaling required (e.g., Δ0→1 merits higher reward than Δ90→91).

Observational Set[3]: Current-round accuracy (shape: N)
   Provides reference for evaluating accuracy growth significance.

Observational Set[4]: Predefined accuracy targets (shape: N)
   Enables convergence efficiency analysis by measuring progress toward system-level objectives.

Observational Set[5]: Active client participation table (shape: M)
   Boolean type. True if participated, otherwise False.

Observational Set[6]: Energy Consumption Ratio (shape: M)
   The proportion of energy consumed by each client in the current round relative to its total available energy.

Observational Set[7]: Remaining Energy Ratio (shape: M)
   The proportion of energy remaining after the current round relative to each client’s total available energy. If this ratio becomes negative while the client is still scheduled to participate in the round (i.e., the corresponding entry in Observational Set [5] is True), an additional penalty should be applied.


**Action Decisions:**

Action Decisions[0]: Task assignment table (shape: M)
   High-level policy output determining client-task assignments.
   Implementation Note: Zero-assignment entries incur penalties when participation falls below minimum thresholds.

Action Decisions[1]: Bandwidth allocation table (shape: M)
   Low-level policy output governing resource distribution across clients.

**Global Reward:**

Global Reward: The global reward in this round. (shape: 1)
   This reward is calculated as w0 * np.sum(Observational Set[2]) - w1 * np.max(Observational Set[0]) - w2 * soft_penalty - w3 * hard_penalty, representing the FL reward value of the round. Here, w0–w3 are constant hyperparameters; soft_penalty denotes the ratio between the total energy consumed by all clients in this round and the total remaining energy of all clients; hard_penalty is a hard penalty applied to clients that still participate in training despite having negative remaining energy, or whose energy consumption in this round exceeds their remaining energy.

   
## Output

Reward Array(M, 2):
For the reward distribution of each client, Reward Array[i][0] represents the reward given by the task assignment agent to client i for the assigned task in this round, and Reward Array[i][1] represents the reward given by the bandwidth allocation agent to client i for the allocated bandwidth ratio in this round.

It must be ensured that:
   **np.sum(Reward Array) = Global Reward**


## Implementation
I hope you can generate a function that calculates rewards for both the task assignment policy and the bandwidth allocation policy, and the code length must exceed 200 lines.

The reward set for the task assignment policy should include factors such as improvement in client accuracy, its impact on system stability, its influence on round training duration, potential benefits, and other factors that you select based on the input observation data and action decisions.

For the bandwidth allocation rewards, you may consider factors such as the standard deviation of Observational Set[0]. For example, if the standard deviation is too large, this may indicate that clients with strong computing capabilities are allocated larger bandwidth, while clients with weak computing capabilities simultaneously obtain small bandwidth, which slows down the entire round. In this situation, both types of clients need to be penalized.

**The function you generate must be sufficiently long (for example, more than 200 lines), to ensure that both the bandwidth allocation and task assignment rewards must consider at least 8 factors. In your function, you must first generate 8 sub-rewards for each client’s bandwidth allocation and task assignment, respectively. Then sum the 8 sub-rewards into the corresponding client’s bandwidth allocation reward or task assignment reward.**

**Please make sure to carefully check the shape of each element in the observation set to avoid any broadcasting errors.**

## Notes
1. You must input all of them into the function you generate (even if you don't need them), and your function can select all or part of these inputs to calculate the sub-rewards. The number of lines in the generated code must be greater than 100.
2. A small subset of clients may be unsuitable for training certain tasks, resulting in excessively long training times or extremely high energy consumption for those tasks. Such clients and tasks should be identified and their participation in the task minimized.
3. A small subset of clients may exhibit limited computational capabilities, leading to prolonged training durations for every task. These clients should be avoided as much as possible.
4. A small subset of clients may experience missing modality data or labels. Decisions regarding their inclusion in a given task should be based on metrics such as loss performance or their reputation on that task (e.g., whether their participation consistently leads to a decline in test accuracy).
5. The data distribution among clients may be imbalanced, with each client potentially containing data corresponding to only one or a few label classes. Efforts should be made to equalize the frequency of participation of these clients.
6. The Active client participation table (Observational Set[5]) may be entirely False. Even if it is not entirely False, the sum of Per-client transmission & training times (Observational Set[1]) may still be 0, as some clients may be unable to participate in this training round due to insufficient remaining energy or other reasons. Pay attention to boundary condition checks for the function inputs.
7. During training, there may be slight fluctuations in accuracy. A minor decrease in accuracy does not necessarily indicate that the data from the participating clients is unreliable. You can use server.history_data{} to store each client's historical data, participation rounds, and the current communication round of federated learning. This server.history_data is an empty dictionary.
8. Considering 6., make sure to avoid NaN values in the function calculations. Also, ensure that each generated sub-reward does not contain NaN! You can add "np.nan_to_num" to the function return.
9. Ensure that the rewards generated by your function in each round fall within a reasonable range(e.g., -5 to 5), meaning that the absolute value of the rewards should not be too large.
10. **Most importantly, my reinforcement learning system is most afraid of a scenario where no nodes participate in training during a round. Please make every effort to avoid this situation: if no clients participate in a task during a round, then impose the MAXIMUM penalty in the ENTIRE system—greater than penalties for nodes that encounter issues due to data, training time, or other factors—on at least 80% clients who have performed the best and have ample remaining participation energy (for example, the remaining energy exceeds a certain ratio of the total energy). **It is necessary to ensure that there is no round in which no client participates in training. If the high-level task assignment policy consistently decides not to involve a well-performing node in many rounds (i.e., action is 0), then the high-level policy should be penalized. However, if a node **performs(reputation) poorly** , assigning it an action of 0 is a reasonable approach and does not require punishment.
11. Considering 10., each set of sub-rewards (sub-rewards 0 - 7) corresponding to the high-level task assignment policy should be initialized to 0. However, if the high-level task assignment policy decides to **exclude a reputable client(e.g., top 80%) from participating in the current training round**, or decides that **a client should participate in a task that has already reached its convergence goal (Current-round accuracy >= Predefined accuracy targets)**, then each set of sub-rewards corresponding to the high-level task assignment policy should receive the maximum penalty to ensure the punishment is maximized.
12. My environment ensures that each client can effectively participate in at least two tasks. I hope to achieve balanced growth of my tasks, where clients contribute their data fairly to each task. I have a scheme in mind: when a certain task i has not been updated for too long, it is necessary to incentivize the high-level task assignment policy to allocate more decisions to this task through rewards; if a task has not been trained for hundreds of rounds, it may indicate that the task is highly volatile and prone to penalties. In such cases, the penalties can be relaxed or even transformed into rewards for participation until the task has been trained by many clients for a certain number of rounds, after which the original reward and penalty mechanism can be restored. I hope you can refine this scheme or propose a new one and incorporate it into the function generation.
13. In line with point 12, I hope to achieve balanced growth of my tasks. **Store in the history the number of times each client has participated in each task. If, for a given client, the participation counts across all of their high-reputation tasks become imbalanced (for example, the variance of those counts exceeds a specified threshold), impose a heavier penalty—initially setting task assignment sub_rewards[0:8] = -1 and then progressively increase the penalty as the imbalance grows (i.e., as the variance rises), until the variance falls back below the threshold.**
14. **Be sure to generate rewards for the bandwidth allocation policy decisions. When generating rewards for the bandwidth allocation strategy, you must comprehensively consider both task participation and participation time.**
15. For the reward decision regarding bandwidth allocation strategies, you must utilize the Action Decisions[1] (Bandwidth allocation table) to ensure that a unique reward is generated for each bandwidth allocation result. In the Bandwidth allocation table, values that cause excessive time consumption in the current round (i.e., bandwidth allocations that are too large or too small) should correspond to lower rewards, while allocations closer to the middle should yield higher rewards. The reward baseline can be calculated based on the standard deviation of the time taken by each client in the current round. Note that in Observation Set [1], the transmission time for each client is the actual measured time, determined by the model size, the bandwidth allocation value and the channel gain. The round's transmission time is calculated as ** transmission time, and the process of computing the transmission time does not require multiplying by the bandwidth allocation ratio**.
16. **The above 15 rules are only intended for handling extreme edge cases in the system. They should not constitute the entire reward function. Meanwhile please do not use training accuracy as the sole evaluation criterion; the rewards for each client in every training round should comprehensively consider multiple factors, and the rewards should vary between clients accordingly.**
17. Observational Set[0][i] = Observational Set[1][i][0] + Observational Set[1][i][1]
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
  "Functions": (A Python function in the form of `def reward_function(server, from Observational Set[0] to Observational Set[6], Action Decisions[0], Action Decisions[1], Global Reward): ... return reward_array (reward_array: A numpy array, reward_array, with shape (M, 2), is used such that every element is ensured to be within an appropriate range (for example, between -5 and 5) and not be a NaN. Specifically, for client i, reward_array[0] represents the reward for the task assignment policy, while reward_array[1] represents the reward for the bandwidth allocation policy.)`. Please do not use Python's triple quotes, as it will cause JSON errors. Use a single line of double quotes "" to wrap the function you generate.)
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
  "Functions": (A Python function in the form of `def reward_function(server, from Observational Set[0] to Observational Set[6], Action Decisions[0], Action Decisions[1], Global Reward): ... return reward_array (reward_array: A numpy array, reward_array, with shape (M, 2), is used such that every element is ensured to be within an appropriate range (for example, between -5 and 5) and not be a NaN. Specifically, for client i, reward_array[0] represents the reward for the task assignment policy, while reward_array[1] represents the reward for the bandwidth allocation policy.)`. Please do not use Python's triple quotes, as it will cause JSON errors. Use a single line of double quotes "" to wrap the function you generate.)
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

Please think step by step and generate content in the following JSON format (replace the content inside the () with your answer). Please do not use Python's triple quotes in the JSON string in your answers. Use double quotes "" instead.
{
  "Understand": (Your understanding of this task),  
  "Analyze": (Step-by-step analysis of which inputs can reflect potential positive and negative rewards),  
  "Functions": (A Python function in the form of `def reward_function(server, from Observational Set[0] to Observational Set[6], Action Decisions[0], Action Decisions[1], Global Reward): ... return reward_array (reward_array: A numpy array, reward_array, with shape (M, 2), is used such that every element is ensured to be within an appropriate range (for example, between -5 and 5) and not be a NaN. Specifically, for client i, reward_array[0] represents the reward for the task assignment policy, while reward_array[1] represents the reward for the bandwidth allocation policy.)`. Please do not use Python's triple quotes, as it will cause JSON errors. Use a single line of double quotes "" to wrap the function you generate.)
}
    """
    
   def get_context(self, functions):
      str = self.Action_1
      pr = Prompt_reward
      for idx, func in enumerate(functions):
            str += f'{idx}. '
            str += func
            str += '\n'
      return pr.Context + pr.Action + str + self.Action_2 + self.Purpose + self.Expectation
   



