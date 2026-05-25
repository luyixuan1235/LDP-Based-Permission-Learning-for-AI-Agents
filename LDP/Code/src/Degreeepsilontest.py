import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os
import math
import numpy as np

def calculate_degree_gain(attack_degrees, perturbed_degree, targets, user_num, num_of_attackers):
    gain = 0.0
    for i in targets:
        attacked_gain = attack_degrees[i] / (user_num + num_of_attackers)
        original_normalized = perturbed_degree[i] / (user_num + num_of_attackers)
        diff = attacked_gain - original_normalized
        gain += abs(diff)
    return gain

def calculate_untargeted_global_gain(before_degrees, after_degrees, total_users):
    def average_degree_centrality(degrees):
        N = total_users
        if N <= 1:
            return 0.0
        return sum(d / (N - 1) for d in degrees) / N    
    
    avg_before = average_degree_centrality(before_degrees) 
    avg_after = average_degree_centrality(after_degrees)
    return abs(avg_after - avg_before)

def read_degrees(file_path):
    with open(file_path, 'r') as file:
        return [int(line.strip()) for line in file if line.strip()]

def get_epsilon_sorted_files(directory, dataset, beta, gamma):
    file_pattern = os.path.join(directory, str(dataset), f'*_{beta}_{gamma}_.txt')
    files = glob.glob(file_pattern)
    return sorted(files, key=lambda x: float(os.path.basename(x).split('_')[0]))

def read_attack_degrees(attack_type, dataset, beta, gamma):
    directory = f'../output/degree/attackedtargeted/{attack_type}'
    file_paths = get_epsilon_sorted_files(directory, dataset, beta, gamma)
    print(f"Looking for files in: {directory}")
    print(f"Found {len(file_paths)} files for {attack_type}")
    if len(file_paths) == 0:
        print(f"  No files found matching pattern: *_{beta}_{gamma}_.txt")
    else:
        print(f"  Files: {[os.path.basename(f) for f in file_paths]}")
    return [read_degrees(file_path) for file_path in file_paths]

def read_untargeted_attack_degrees(dataset, beta, gamma):
    directory = f'../output/degree/untargeted_all_nodes/{dataset}'
    before_pattern = os.path.join(directory, f'before_attack_*_{beta}_{gamma}.txt')
    after_pattern = os.path.join(directory, f'after_attack_*_{beta}_{gamma}.txt')
    before_files = sorted(glob.glob(before_pattern), key=lambda x: float(os.path.basename(x).split('_')[2]))
    after_files = sorted(glob.glob(after_pattern), key=lambda x: float(os.path.basename(x).split('_')[2]))
    
    print(f"Looking for untargeted files in: {directory}")
    print(f"Found {len(before_files)} before files and {len(after_files)} after files")
    
    before_degrees_list = [read_degrees(file_path) for file_path in before_files]
    after_degrees_list = [read_degrees(file_path) for file_path in after_files]
    return before_degrees_list, after_degrees_list

def read_perturbed_degrees(dataset):
    directory = f'../output/degree/perturbed/{dataset}'
    file_pattern = os.path.join(directory, '*.txt')
    files = glob.glob(file_pattern)

    epsilon_files = []
    for file in files:
        filename = os.path.basename(file)
        try:
            epsilon = float(filename.replace('.txt', ''))
            epsilon_files.append((epsilon, file))
        except ValueError:
            continue

    epsilon_files.sort(key=lambda x: x[0], reverse=True)

    print(f"Found {len(epsilon_files)} perturbed degree files:")
    for epsilon, file in epsilon_files:
        print(f"  Epsilon {epsilon}: {os.path.basename(file)}")

    perturbed_degrees = []
    for _, file_path in epsilon_files:
        if os.path.exists(file_path):
            degrees = read_degrees(file_path)
            perturbed_degrees.append(degrees)
        else:
            print(f"Warning: File not found: {file_path}")

    return perturbed_degrees

def calculate_degree_gains(attack_degrees_list, perturbed_degrees_list, targetNodes, user_num, num_of_attackers):
    gains = []
    for attack_degrees, perturbed_degrees in zip(attack_degrees_list, perturbed_degrees_list):
        gain = calculate_degree_gain(attack_degrees, perturbed_degrees, targetNodes, user_num, num_of_attackers)
        gains.append(gain)
    return gains

def calculate_untargeted_global_gains(before_degrees_list, after_degrees_list, total_users):
    # calculate the global gain of untargeted attack
    gains = []
    for before_degrees, after_degrees in zip(before_degrees_list, after_degrees_list):
        gain = calculate_untargeted_global_gain(before_degrees, after_degrees, total_users)
        gains.append(gain)
    return gains

def plot_targeted_attacks(epsilons, random_value_gains, random_node_gains, maximum_gain_gains):
    """绘制三种目标攻击的增益变化"""
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(500 * px, 350 * px))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    ax.plot(epsilons, random_value_gains, marker='s', linestyle='--', label="RVA", color='#1f77b4')
    ax.plot(epsilons, random_node_gains, marker='o', linestyle='-', label="RNA", color='#ff7f0e')
    ax.plot(epsilons, maximum_gain_gains, marker='^', linestyle='-.', label="MGA", color='#2ca02c')

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel(r'Privacy Budget $\epsilon$', fontsize=12)
    ax.set_ylabel(r'Targeted Gain $\left| G \right|$', fontsize=12)
    ax.set_yscale('symlog', linthresh=0.001, linscale=1)
    
    # no untargeted attacked gains(it is global gains)
    all_gains = random_value_gains + random_node_gains + maximum_gain_gains
    min_gain = min(all_gains) * 0.8
    max_gain = max(all_gains) * 1.2
    ax.set_ylim(min_gain, max_gain)
    
    ax.grid(True, alpha=0.7)
    ax.legend(loc='best')
    plt.title('Targeted Attacks Gain Comparison')

    for spine in ['left', 'bottom', 'right', 'top']:
        ax.spines[spine].set_color('#c8c8c8')

    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.savefig('targeted_attacks_gain.svg', dpi=300)
    plt.show()

def plot_untargeted_attack(epsilons, untargeted_gains):
    """单独绘制无目标攻击的全局增益变化"""
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(500 * px, 350 * px))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    # untargeted attacked
    ax.plot(epsilons, untargeted_gains, marker='X', linestyle='-', label="UA", color='#d62728', linewidth=2.5)
    
    # give the avg of untargeted_gains
    avg_gain = np.mean(untargeted_gains)
    ax.axhline(y=avg_gain, color='#9467bd', linestyle='--', label=f'Avg Gain: {avg_gain:.4f}')
    
    
    for i, gain in enumerate(untargeted_gains):
        ax.annotate(f'{gain:.4f}', 
                    (epsilons[i], gain),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel(r'Privacy Budget $\epsilon$', fontsize=12)
    ax.set_ylabel(r'Global Gain $\Delta C$', fontsize=12)
    
    # untargeted_gains set
    min_gain = min(untargeted_gains) * 0.9
    max_gain = max(untargeted_gains) * 1.1
    ax.set_ylim(min_gain, max_gain)
    
    ax.grid(True, alpha=0.7)
    ax.legend(loc='best')
    plt.title('Untargeted Attack Global Gain')

    for spine in ['left', 'bottom', 'right', 'top']:
        ax.spines[spine].set_color('#c8c8c8')

    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.savefig('untargeted_attack_global_gain.svg', dpi=300)
    plt.show()

def extract_int(pattern, text, default=0):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else default

def find_parameter_file(dataset, beta, gamma):
    config_dir = f'../output/degree/config/{dataset}'
    if not os.path.exists(config_dir):
        print(f"Config directory does not exist: {config_dir}")
        return None

    files = os.listdir(config_dir)
    possible_formats = [
        f"parameters_{beta}.txt",
        f"parameters_{beta}_{gamma}.txt",
        f"parameters_{beta:.1f}_{gamma:.1f}.txt",
        f"parameters_{beta:.2f}_{gamma:.2f}.txt",
        f"parameters_{beta:.3f}_{gamma:.3f}.txt",
    ]

    for format_name in possible_formats:
        file_path = os.path.join(config_dir, format_name)
        if os.path.exists(file_path):
            print(f"Found parameter file: {format_name}")
            return file_path

    for file in files:
        if file.startswith("parameters_") and file.endswith(".txt"):
            print(f"Found potential parameter file: {file}")
            return os.path.join(config_dir, file)

    print("No parameter file found!")
    return None

def read_parameters(dataset, beta, gamma):
    param_file = find_parameter_file(dataset, beta, gamma)
    if param_file is None:
        print("Parameter file not found!")
        return None, None, None, None

    with open(param_file, 'r') as file:
        data_parameter = file.read()

    totalNum = extract_int(r'totalNum:\s*(\d+)', data_parameter)
    attackerNum = extract_int(r'attackerNum:\s*(\d+)', data_parameter)
    targetNum = extract_int(r'targetNum:\s*(\d+)', data_parameter)
    targetNodes = [int(node) for node in re.findall(r'targetNode \d+:\s*(\d+)', data_parameter)]

    return totalNum, attackerNum, targetNum, targetNodes

def calculate_and_plot_degree_gain(dataset, beta, gamma, targetNodes, user_num, num_of_attackers):
    print(f"Starting calculation for dataset {dataset}")
    
    perturbed_degrees_list = read_perturbed_degrees(dataset)[::-1]
    print(f"Loaded {len(perturbed_degrees_list)} perturbed degree files")
    
    if len(perturbed_degrees_list) == 0:
        print("ERROR: No perturbed degree files found!")
        return
    
    random_node_degrees = read_attack_degrees('randomNodeAttack', dataset, beta, gamma)
    random_value_degrees = read_attack_degrees('randomValueAttack', dataset, beta, gamma)
    maximum_gain_degrees = read_attack_degrees('maximumGainAttack', dataset, beta, gamma)

    # 统一用 randomNodeAttack 的文件生成 epsilons
    attack_files = get_epsilon_sorted_files(
        f'../output/degree/attackedtargeted/randomNodeAttack', dataset, beta, gamma)
    epsilons = [float(os.path.basename(f).split('_')[0]) for f in attack_files]
    
    print(f"Found {len(epsilons)} epsilon values: {epsilons}")

    if len(epsilons) == 0:
        print("ERROR: No epsilon files found!")
        return

    # user_num = 4039
    total_users = user_num
    print(f"totaluser is: {total_users}")
    random_node_gains = calculate_degree_gains(random_node_degrees, perturbed_degrees_list, targetNodes, user_num, num_of_attackers)[:len(epsilons)]
    random_value_gains = calculate_degree_gains(random_value_degrees, perturbed_degrees_list, targetNodes, user_num, num_of_attackers)[:len(epsilons)]
    maximum_gain_gains = calculate_degree_gains(maximum_gain_degrees, perturbed_degrees_list, targetNodes, user_num, num_of_attackers)[:len(epsilons)]

    before_degrees_list, after_degrees_list = read_untargeted_attack_degrees(dataset, beta, gamma)
    untargeted_gains = calculate_untargeted_global_gains(before_degrees_list, after_degrees_list, total_users)[:len(epsilons)]

    print(f"Calculated gains:")
    print(f"  Random node gains: {random_node_gains}")
    print(f"  Random value gains: {random_value_gains}")
    print(f"  Maximum gain gains: {maximum_gain_gains}")
    print(f"  Untargeted gains: {untargeted_gains}")

    # use two graphs to compare
    plot_targeted_attacks(epsilons, random_value_gains, random_node_gains, maximum_gain_gains)
    plot_untargeted_attack(epsilons, untargeted_gains)

if __name__ == '__main__':
    dataset = 1
    beta = 0.05
    gamma = 0.05
    
    totalNum, attackerNum, targetNum, targetNodes = read_parameters(dataset, beta, gamma)
    
    if totalNum is None:
        print("Failed to read parameters.")
        exit(1)
    
    print(f'Dataset: {dataset}')
    print(f'Beta: {beta}')
    print(f'Gamma: {gamma}')
    print(f'AttackerNum: {attackerNum}')
    print(f'TargetNum: {targetNum}')
    print(f'TargetNodes: {targetNodes}')
    
    calculate_and_plot_degree_gain(dataset, beta, gamma, targetNodes, totalNum, attackerNum)