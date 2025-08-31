import re
import matplotlib.pyplot as plt
from pathlib import Path

steps = []
train_loss = []
val_loss = []
filepath = Path("temp/k100_gelu_gpt_train.txt")

with open(filepath, "r", encoding="utf-8") as f:
    for line in f:
        match = re.match(r"step (\d+): train ([\d\.]+), val ([\d\.]+)", line)
        if match:
            steps.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))

plt.plot(steps, train_loss, label="Train Loss")
plt.plot(steps, val_loss, label="Val Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("GPT Training Loss")
plt.yscale("log")
plt.legend()
plt.show()