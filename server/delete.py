from pathlib import Path

path_root = Path(__file__).resolve().parent

log_file = path_root / 'server.log'
if log_file.exists():
    log_file.unlink()
    print(f"Deleted log file: {log_file}")
else:
    print(f"No log file found at: {log_file}")

context_file = path_root / 'context.pkl'
if context_file.exists():
    context_file.unlink()
    print(f"Deleted log file: {context_file}")
else:
    print(f"No log file found at: {context_file}")

RL_data = path_root / 'RL/data'
RL_agent = path_root / 'RL/RLModel'
models = path_root / 'global_models/models'
csvs = path_root / 'Experiment/client_graph'
for folder in [RL_data, RL_agent, models, csvs]:
    if folder.exists() and folder.is_dir():
        for f in folder.rglob("*"):
            if f.is_file():
                print(f"Deleting file: {f}")
                f.unlink()


