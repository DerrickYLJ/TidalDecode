import matplotlib.pyplot as plt
import numpy as np

# Paths to the log files
log_files = {
    'Full': 'results/ppl/full/full_log.txt',
    # 'Correction L9-2048': 'results/ppl/2048_9/index_9_log.txt',
    # 'Correction L13-2048': 'results/ppl/2048_13/index_13_log.txt',
    # 'Correction L15-2048': 'results/ppl/2048_15/index_15_log.txt',
    # 'Quest-2048': 'results/ppl/2048_quest/quest_log.txt',
    'Correction L9-4096': 'results/ppl/4096_9/log.txt',
    'Correction L13-4096': 'results/ppl/4096_13/log.txt',
    'Correction L15-4096': 'results/ppl/4096_15/log.txt',
    'Quest-4096': 'results/ppl/4096_quest/log.txt',
}

# Function to read the NLL values from the log file and compute perplexity step by step
def read_nll_values_and_compute_perplexity(file_path):
    cumulative_nll = 0.0  # Cumulative sum of NLL values
    perplexities = []
    
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            try:
                nll = float(line.strip())
                cumulative_nll += nll  # Accumulate the NLL values
                
                # Compute the average NLL so far
                avg_nll = cumulative_nll / (idx + 1)
                
                # Compute perplexity as exp of the average NLL
                ppl = np.exp(avg_nll)
                perplexities.append(ppl)
            except ValueError:
                pass  # Ignore lines that cannot be parsed as float
    
    return perplexities

# Reading NLL and computing perplexities for each approach
perplexities = {label: read_nll_values_and_compute_perplexity(file) for label, file in log_files.items()}

# Plotting perplexities for all approaches on the same graph with limited y-axis range
plt.figure(figsize=(10, 6))

for label, perp in perplexities.items():
    plt.plot(perp, label=label)

    print(f"Final perplexity for {label}: {perp[-1]}")


plt.xlabel("Input Length")
plt.ylabel("Perplexity (the lower the better)")
plt.title("Perplexity with Context Length")
plt.ylim(7, 9.5)  # Limit the y-axis range between 6.5 and 9.5
plt.legend()

# Save the plot
plt.savefig("results/ppl/perplexity_4096.png")

plt.show()
