import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_dashboard(csv_file='benchmark_results.csv'):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    # Set theme
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']

    df = pd.read_csv(csv_file)
    
    # Calculate speedup relative to CPU Serial
    baseline_time = df.loc[df['Method'] == 'CPU Serial', 'AvgTimeMs'].values[0]
    df['Speedup'] = baseline_time / df['AvgTimeMs']

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=100)
    
    # 1. Execution Time Comparison
    colors = sns.color_palette("viridis", len(df))
    sns.barplot(x='AvgTimeMs', y='Method', data=df, ax=axes[0], palette=colors)
    axes[0].set_title('Execution Time Comparison (Lower is Better)', fontsize=16, fontweight='bold', pad=20)
    axes[0].set_xlabel('Average Time (ms)', fontsize=12)
    axes[0].set_ylabel('', fontsize=12)
    
    # Add labels to bars
    for i, v in enumerate(df['AvgTimeMs']):
        axes[0].text(v + 0.1, i, f"{v:.3f} ms", color='black', va='center', fontweight='bold')

    # 2. Throughput Comparison
    sns.barplot(x='ThroughputGbs', y='Method', data=df, ax=axes[1], palette=colors)
    axes[1].set_title('Throughput Performance (Higher is Better)', fontsize=16, fontweight='bold', pad=20)
    axes[1].set_xlabel('Throughput (GB/s)', fontsize=12)
    axes[1].set_ylabel('', fontsize=12)

    # Add labels to bars
    for i, v in enumerate(df['ThroughputGbs']):
        axes[1].text(v + 0.1, i, f"{v:.2f} GB/s", color='black', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('performance_dashboard.png')
    print("Dashboard saved as performance_dashboard.png")

if __name__ == "__main__":
    generate_dashboard()
