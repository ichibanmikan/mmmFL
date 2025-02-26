import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches

count = 0
times = []

def plot(time_table, 
         round, 
         output_dir = os.path.join(os.path.dirname(__file__), 'client_graph')
         ):
    os.makedirs(output_dir, exist_ok=True)
    global count, times
    count += 1
    times.append(time_table)
    if count % 100 == 0:
        times = np.array(times).reshape(100, 60)
        with open(os.path.join(output_dir, 'times.csv'), 'a') as f:
            np.savetxt(f, times, delimiter=",", fmt="%.6f")
        times = []
    valid_clients = []
    for i in range(time_table.shape[0]):
        t0 = time_table[i, 0]
        t1 = time_table[i, 1]
        total_time = 2 * t0 + t1
        
        if total_time > 0:
            client = {
                "name": f"Client {i}",
                "start": 0,
                "s1": t0,
                "s2": t0 + t1,
                "end": total_time
            }
            valid_clients.append(client)
    
    if not valid_clients:
        print("No Clients")
        return

    plt.rcParams['font.family'] = 'Times New Roman'
    colors = ['#87AAB6', '#2F5763', '#87AAB6']

    num_clients = len(valid_clients)
    fig_height = max(2, num_clients * 0.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    for client in valid_clients:
        ax.barh(client['name'], client['s1'], left=client['start'], color=colors[0])
        ax.barh(client['name'], client['s2']-client['s1'], left=client['s1'], color=colors[1])
        ax.barh(client['name'], client['end']-client['s2'], left=client['s2'], color=colors[2])

    ax.set_xlabel('Time', fontsize=14)
    ax.set_title(f'Client Time Distribution in Round {round}', fontsize=16)

    max_time = max(client['end'] for client in valid_clients)
    
    if max_time > 100 and max_time <= 500:
        ax.set_xticks(np.arange(0, max_time + 50, 50))
    elif max_time > 500:
        ax.set_xticks(np.arange(0, max_time + 50, 100))
    elif max_time > 1000:
        ax.set_xticks(np.arange(0, max_time + 50, 250))
    else:
        ax.set_xticks(np.arange(0, max_time + 5, 5))

    ax.tick_params(axis='both', labelsize=12)
    legend_handles = [
        patches.Patch(facecolor=colors[0], label='Transmission time'),
        patches.Patch(facecolor=colors[1], label='Training time')
    ]
    ax.legend(handles=legend_handles, 
             loc='upper right',
             bbox_to_anchor=(1.25, 1),
             frameon=True,
             framealpha=1,
             edgecolor='black')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'round_{round}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Picture is saved at: {output_path}")

# # for example
# if __name__ == "__main__":
#     example_data = np.array([
#         [12.5, 10],   # Client 1
#         [12.5, 20],   # Client 2
#         [7.5, 25],    # Client 3
#         [12.5, 10]    # Client 4
#     ])
    
#     plot(example_data, 0)