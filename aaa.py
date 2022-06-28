import time

import wandb
import os

os.environ["WANDB_API_KEY"] = "sas"
os.environ["WANDB_MODE"] = "offline"

config = {
    "dataset": "CIFAR10",
    "machine": "offline cluster",
    "model": "CNN",
    "learning_rate": 0.01,
    "batch_size": 128,
}

wandb.init(project="offline-demo")

for i in range(100):
    time.sleep(1)
    wandb.log({"accuracy": i})
    print(i)
