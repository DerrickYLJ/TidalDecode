import numpy as np
import faiss
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


# Function to perform search on a given index
def search_index(index, xq, k):
    start_time = time.time()
    distances, indices = index.search(xq, k)
    search_time = time.time() - start_time
    return search_time

def warm_up(num_warmups, nq_list, nb_list, k_list):
    for _ in range(num_warmups):
        for nq in nq_list:
            for nb in nb_list:
                for k in k_list:
                    # Create random vectors
                    xb = np.random.random((nb, 128)).astype('float32')
                    xb[:, 0] += np.arange(nb) / 1000.

                    # Create and populate index
                    index_cpu = faiss.IndexFlatL2(128)
                    index_cpu.add(xb)

                    # Generate query vectors
                    xq = np.random.random((nq, 128)).astype('float32')
                    xq[:, 0] += np.arange(nq) / 1000.

                    # Warm-up search on CPU
                    index_cpu.search(xq, k)

                    # Transfer index to GPU
                    res = faiss.StandardGpuResources()
                    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

                    # Warm-up search on GPU
                    index_gpu.search(xq, k)

# Function to measure and record search times
def measure_latency(num_itr, nq_list, nb_list, k_list, num_kv_heads):
    results = []
    res = [faiss.StandardGpuResources() for _ in range(num_kv_heads)]

    for nq in nq_list:
        for nb in nb_list:
            for k in k_list:
                # Create random vectors
                xb = np.random.random((nb, 128)).astype('float32')
                xb[:, 0] += np.arange(nb) / 1000.
                search_times_cpu = []
                search_times_gpu = []

                # Measure build and search time on CPU
                index_cpus = [faiss.IndexFlatL2(128) for _ in range(num_kv_heads)]
                for index_cpu in index_cpus:
                    index_cpu.add(xb)

                for i in range(num_itr):
                    new_time = None
                    for j in range(num_kv_heads):
                        np.random.seed(i)
                        xq = np.random.random((nq, 128)).astype('float32')
                        xq[:, 0] += np.arange(nq) / 1000.
                        new_time = search_index(index_cpus[j], xq, k) if new_time is None else new_time+ search_index(index_cpus[j], xq, k)
                    search_times_cpu.append(new_time)

                # Measure build and search time on GPU
                
                index_gpus = [faiss.index_cpu_to_gpu(r, 0, index_cpu) for index_cpu, r in zip(index_cpus, res)]

                for i in range(num_itr):
                    new_time = None
                    for j in range(num_kv_heads):
                        np.random.seed(i)
                        xq = np.random.random((nq, 128)).astype('float32')
                        xq[:, 0] += np.arange(nq) / 1000.
                        new_time = search_index(index_gpus[j], xq, k) if new_time is None else new_time+ search_index(index_gpus[j], xq, k)
                    search_times_gpu.append(new_time)

                results.append((nq, nb, k, [i / num_itr for i in search_times_cpu], [i / num_itr for i in search_times_gpu]))

    return results

def measure_latency_parallel(num_itr, nq_list, nb_list, k_list, num_kv_heads):
    results = []
    res = [faiss.StandardGpuResources() for _ in range(num_kv_heads)]
    for nq in nq_list:
        for nb in nb_list:
            for k in k_list:
                # Create random vectors
                xb = np.random.random((nb, 128)).astype('float32')
                xb[:, 0] += np.arange(nb) / 1000.
                search_times_cpu = []
                search_times_gpu = []

                # Measure build and search time on CPU
                index_cpus = [faiss.IndexFlatL2(128) for _ in range(num_kv_heads)]
                for index_cpu in index_cpus:
                    index_cpu.add(xb)

                # Prepare query vectors
                xq_list = [np.random.random((nq, 128)).astype('float32') for _ in range(num_kv_heads)]
                for xq in xq_list:
                    xq[:, 0] += np.arange(nq) / 1000.

                # Perform parallel search on CPU
                for i in range(num_itr):
                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(search_index, index_cpu, xq, k) for index_cpu, xq in zip(index_cpus, xq_list)]
                        tmp = [future.result() for future in futures]
                        new_time = max(tmp)
                    search_times_cpu.append(new_time)

                # Measure build and search time on GPU
                index_gpus = [faiss.index_cpu_to_gpu(r, 0, index_cpu) for index_cpu, r in zip(index_cpus, res)]

                # Perform parallel search on GPU
                for i in range(num_itr):
                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(search_index, index_gpu, xq, k) for index_gpu, xq in zip(index_gpus, xq_list)]
                        tmp = [future.result() for future in futures]
                        new_time = max(tmp)
                        print(tmp)
                    search_times_gpu.append(new_time)
                results.append((nq, nb, k, [i / num_itr for i in search_times_cpu], [i / num_itr for i in search_times_gpu]))

    return results


# Plot results
def plot_results(results, nb_list):
    # Individual nb plots
    for nb in nb_list:
        filtered_results = [result for result in results if result[1] == nb]
        x_labels = [f'({nq}, {k})' for nq, _, k, _, _ in filtered_results]
        fig, ax = plt.subplots(figsize=(15, 7))

        index = np.arange(len(filtered_results))
        bar_width = 0.35

        for i, (nq, _, k, search_times_cpu, search_times_gpu) in enumerate(filtered_results):
            # CPU first iteration
            ax.bar(index[i] + bar_width/2, search_times_cpu[0], bar_width, label='CPU - 1st Iteration' if i == 0 else "", color='blue')
            # CPU rest iterations
            ax.bar(index[i] + bar_width/2, np.sum(search_times_cpu[1:]), bar_width, bottom=search_times_cpu[0], label='CPU - All Rest' if i == 0 else "", color='lightblue')

            # GPU first iteration
            ax.bar(index[i] - bar_width/2, search_times_gpu[0], bar_width, label='GPU - 1st Iteration' if i == 0 else "", color='red')
            # GPU rest iterations
            ax.bar(index[i] - bar_width/2, np.sum(search_times_gpu[1:]), bar_width, bottom=search_times_gpu[0], label='GPU - All Rest' if i == 0 else "", color='lightcoral')

        ax.set_xticks(index)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Combination of batched_size and top-k')
        ax.set_ylabel('Latency (seconds)')
        ax.set_title(f'Search Latency for Different Configurations with database={nb} ')
        ax.legend()
        plt.xticks(rotation=90)
        plt.tight_layout()
        fig.savefig(f'data/latency_plot/index_retrieval/search_latency_nb_{nb}.png')
    
    # Complete GPU Summary
    fig, ax = plt.subplots(figsize=(15, 7))
    x_labels = [f'({nq}, {nb//1000}K, {k})' for nq, nb, k, _, _ in results]
    index = np.arange(len(results))

    for i, (nq, nb, k, search_times_cpu, search_times_gpu) in enumerate(results):
        # GPU first iteration
        ax.bar(index[i], search_times_gpu[0], bar_width, label='GPU - 1st Iteration' if i == 0 else "", color='red')
        # GPU rest iterations
        ax.bar(index[i], np.sum(search_times_gpu[1:]), bar_width, bottom=search_times_gpu[0], label='GPU - All Rest' if i == 0 else "", color='lightcoral')

    ax.set_xticks(index)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Combination of batch_size, database_size, and top-k')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title(f'Complete GPU Summary ')
    ax.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig.savefig(f'data/latency_plot/index_retrieval/complete_gpu_summary.png')

    # Complete CPU Summary
    fig, ax = plt.subplots(figsize=(15, 7))
    x_labels = [f'({nq}, {nb//1000}K, {k})' for nq, nb, k, _, _ in results]
    index = np.arange(len(results))

    for i, (nq, nb, k, search_times_cpu, search_times_gpu) in enumerate(results):
        # CPU first iteration
        ax.bar(index[i], search_times_cpu[0], bar_width, label='CPU - 1st Iteration' if i == 0 else "", color='blue')
        # CPU rest iterations
        ax.bar(index[i], np.sum(search_times_cpu[1:]), bar_width, bottom=search_times_cpu[0], label='CPU - All Rest' if i == 0 else "", color='lightblue')

    ax.set_xticks(index)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Combination of batch_size, database_size, and top-k')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title(f'Complete CPU Summary ')
    ax.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig.savefig(f'data/latency_plot/index_retrieval/complete_cpu_summary.png')


if __name__ == "__main__":
    # Define parameters
    num_itr = 10
    nq_list = [1, 2, 8, 16, 32]
    nb_list = [2000, 4000, 8000, 16000, 32000]
    k_list = [200, 400, 800]
    num_kv_heads = 8

    # Warm-up
    warm_up(2, nq_list, nb_list, k_list)

    # Measure latency
    results = measure_latency(num_itr, nq_list, nb_list, k_list, num_kv_heads)
    # Plot the results for each nb and complete summaries
    plot_results(results, nb_list)