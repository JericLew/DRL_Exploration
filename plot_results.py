import csv
import argparse
import matplotlib.pyplot as plt

# Arguement parser
parser = argparse.ArgumentParser(description="Script to plot loss and returns graph")
parser.add_argument("csv_path", type=str, help="path to csv file")
args = parser.parse_args()

# Initialize empty lists to store data
iterations = []
avg_actor_loss = []
avg_critic_loss = []
avg_episodic_return = []

# Open and read the CSV file
with open(args.csv_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        iterations.append(int(row['Iteration']))
        avg_actor_loss.append(float(row['Average Actor Loss']))
        avg_critic_loss.append(float(row['Average Critic Loss']))
        avg_episodic_return.append(float(row['Average Episodic Return']))

# Create a grid of subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot Average Actor Loss
axes[0].plot(iterations, avg_actor_loss, marker='o', linestyle='-', label='Average Actor Loss', color='blue')
axes[0].set_ylabel('Average Actor Loss')
axes[0].grid(True)
axes[0].legend()

# Plot Average Critic Loss
axes[1].plot(iterations, avg_critic_loss, marker='o', linestyle='-', label='Average Critic Loss', color='green')
axes[1].set_ylabel('Average Critic Loss')
axes[1].grid(True)
axes[1].legend()

# Plot Average Episodic Return
axes[2].plot(iterations, avg_episodic_return, marker='o', linestyle='-', label='Average Episodic Return', color='red')
axes[2].set_xlabel('Iterations')
axes[2].set_ylabel('Average Episodic Return')
axes[2].grid(True)
axes[2].legend()

# Adjust layout
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('loss-graph.png')

# Show the plot
plt.show()