import json
import matplotlib.pyplot as plt

# Load the JSON log file
log_file_path = "/home/wayne/Desktop/PR_final/segformer/work_dirs/segformer_mit-b5.log.json"
with open(log_file_path, 'r') as file:
    log_data = [json.loads(line) for line in file if "mode" in line and json.loads(line)["mode"] == "train"]

# Extract iterations, accuracy, and loss
iterations = [entry["iter"] for entry in log_data]
accuracy = [entry["decode.acc_seg"] for entry in log_data]
loss = [entry["loss"] for entry in log_data]

# Plot accuracy and loss
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(iterations, accuracy, label="Accuracy", color="blue")
plt.xlabel("Iterations")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy over Iterations")
plt.grid(True)
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(iterations, loss, label="Loss", color="red")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("/home/wayne/Desktop/PR_final/result/result.png")
