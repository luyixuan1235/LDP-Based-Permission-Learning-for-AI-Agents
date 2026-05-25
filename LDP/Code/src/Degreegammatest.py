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

def get_gamma_sorted_files(directory, dataset, epsilon, beta):
    """获取按gamma排序的文件列表"""
    file_pattern = os.path.join(directory, str(dataset), f'{epsilon}_{beta}_*_.txt')
    files = glob.glob(file_pattern)
    return sorted(files, key=lambda x: float(os.path.basename(x).split('_')[2]))

def read_attack_degrees_by_gamma(attack_type, dataset, epsilon, beta):
    """根据gamma值读取攻击度数数据"""
    directory = f'../output/degree/attackedtargeted/{attack_type}'
    file_paths = get_gamma_sorted_files(directory, dataset, epsilon, beta)
    print(f"Looking for files in: {directory}")
    print(f"Found {len(file_paths)} files for {attack_type}")
    if len(file_paths) == 0:
        print(f"  No files found matching pattern: {epsilon}_{beta}_*_.txt")
    else:
        print(f"  Files: {[os.path.basename(f) for f in file_paths]}")
    return [read_degrees(file_path) for file_path in file_paths]

def read_untargeted_attack_degrees_by_gamma(dataset, epsilon, beta):
    """根据gamma值读取无目标攻击度数数据"""
    directory = f'../output/degree/untargeted_all_nodes/{dataset}'
    before_pattern = os.path.join(directory, f'before_attack_{epsilon}_{beta}_*.txt')
    after_pattern = os.path.join(directory, f'after_attack_{epsilon}_{beta}_*.txt')
    before_files = sorted(glob.glob(before_pattern), key=lambda x: float(os.path.basename(x).split('_')[3].replace('.txt', '')))
    after_files = sorted(glob.glob(after_pattern), key=lambda x: float(os.path.basename(x).split('_')[3].replace('.txt', '')))
    
    print(f"Looking for untargeted files in: {directory}")
    print(f"Found {len(before_files)} before files and {len(after_files)} after files")
    
    before_degrees_list = [read_degrees(file_path) for file_path in before_files]
    after_degrees_list = [read_degrees(file_path) for file_path in after_files]
    return before_degrees_list, after_degrees_list

def read_perturbed_degrees_by_gamma(dataset, epsilon):
    """读取指定epsilon的扰动度数数据"""
    directory = f'../output/degree/perturbed/{dataset}'
    file_path = os.path.join(directory, f'{epsilon}.txt')
    
    if os.path.exists(file_path):
        print(f"Found perturbed degree file: {os.path.basename(file_path)}")
        return read_degrees(file_path)
    else:
        print(f"Warning: Perturbed degree file not found: {file_path}")
        return None

def calculate_degree_gains(attack_degrees_list, perturbed_degrees, targetNodes_list, user_num, num_of_attackers_list):
    """计算度数增益，每个gamma值使用对应的目标节点"""
    gains = []
    for i, attack_degrees in enumerate(attack_degrees_list):
        if i < len(targetNodes_list) and i < len(num_of_attackers_list):
            gain = calculate_degree_gain(attack_degrees, perturbed_degrees, targetNodes_list[i], user_num, num_of_attackers_list[i])
            gains.append(gain)
        else:
            print(f"Warning: Missing target nodes or attacker info for gamma index {i}")
            gains.append(0.0)
    return gains

def calculate_untargeted_global_gains(before_degrees_list, after_degrees_list, total_users):
    gains = []
    for before_degrees, after_degrees in zip(before_degrees_list, after_degrees_list):
        gain = calculate_untargeted_global_gain(before_degrees, after_degrees, total_users)
        gains.append(gain)
    return gains

def plot_targeted_attacks_vs_gamma(gammas, random_value_gains, random_node_gains, maximum_gain_gains, epsilon, beta):
    """绘制三种目标攻击随gamma变化的增益"""
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(500 * px, 350 * px))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    ax.plot(gammas, random_value_gains, marker='s', linestyle='--', label="RVA", color='#1f77b4')
    ax.plot(gammas, random_node_gains, marker='o', linestyle='-', label="RNA", color='#ff7f0e')
    ax.plot(gammas, maximum_gain_gains, marker='^', linestyle='-.', label="MGA", color='#2ca02c')

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel(r'Gamma $\gamma$', fontsize=12)
    ax.set_ylabel(r'Targeted Gain $\left| G \right|$', fontsize=12)
    
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=0.001, linscale=1)
    
    all_gains = random_value_gains + random_node_gains + maximum_gain_gains
    min_gain = min(all_gains) * 0.8
    max_gain = max(all_gains) * 1.2
    ax.set_ylim(min_gain, max_gain)
    
    ax.grid(True, alpha=0.7)
    ax.legend(loc='best')
    plt.title(f'Targeted Attacks Gain vs Gamma (ε={epsilon}, β={beta})')

    for spine in ['left', 'bottom', 'right', 'top']:
        ax.spines[spine].set_color('#c8c8c8')

    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.savefig(f'targeted_attacks_gain_vs_gamma_eps{epsilon}_beta{beta}.svg', dpi=300)
    plt.show()

def plot_untargeted_attack_vs_gamma(gammas, untargeted_gains, epsilon, beta):
    """绘制无目标攻击随gamma变化的全局增益"""
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(500 * px, 350 * px))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    ax.plot(gammas, untargeted_gains, marker='X', linestyle='-', label="UA", color='#d62728', linewidth=2.5)
    
    avg_gain = np.mean(untargeted_gains)
    ax.axhline(y=avg_gain, color='#9467bd', linestyle='--', label=f'Avg Gain: {avg_gain:.4f}')
    
    for i, gain in enumerate(untargeted_gains):
        ax.annotate(f'{gain:.4f}', 
                    (gammas[i], gain),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel(r'Gamma $\gamma$', fontsize=12)
    ax.set_ylabel(r'Global Gain $\Delta C$', fontsize=12)
    
    ax.set_xscale('log')
    
    min_gain = min(untargeted_gains) * 0.9
    max_gain = max(untargeted_gains) * 1.1
    ax.set_ylim(min_gain, max_gain)
    
    ax.grid(True, alpha=0.7)
    ax.legend(loc='best')
    plt.title(f'Untargeted Attack Global Gain vs Gamma (ε={epsilon}, β={beta})')

    for spine in ['left', 'bottom', 'right', 'top']:
        ax.spines[spine].set_color('#c8c8c8')

    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.savefig(f'untargeted_attack_global_gain_vs_gamma_eps{epsilon}_beta{beta}.svg', dpi=300)
    plt.show()

def extract_int(pattern, text, default=0):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else default

def read_parameters_for_gamma(dataset, beta, gamma):
    """为特定的gamma值读取参数文件"""
    config_dir = f'../output/degree/config/{dataset}'
    if not os.path.exists(config_dir):
        print(f"Config directory does not exist: {config_dir}")
        return None, None, None, None

    possible_formats = [
        f"parameters_{beta}_{gamma}.txt",
        f"parameters_{beta:.1f}_{gamma:.1f}.txt",
        f"parameters_{beta:.2f}_{gamma:.2f}.txt",
        f"parameters_{beta:.3f}_{gamma:.3f}.txt",
    ]

    for format_name in possible_formats:
        file_path = os.path.join(config_dir, format_name)
        if os.path.exists(file_path):
            print(f"Found parameter file for gamma {gamma}: {format_name}")
            with open(file_path, 'r') as file:
                data_parameter = file.read()

            totalNum = extract_int(r'totalNum:\s*(\d+)', data_parameter)
            attackerNum = extract_int(r'attackerNum:\s*(\d+)', data_parameter)
            targetNum = extract_int(r'targetNum:\s*(\d+)', data_parameter)
            targetNodes = [int(node) for node in re.findall(r'targetNode \d+:\s*(\d+)', data_parameter)]

            return totalNum, attackerNum, targetNum, targetNodes

    print(f"Warning: No parameter file found for gamma {gamma}")
    return None, None, None, None

def get_all_gamma_values(dataset, epsilon, beta):
    directory = f'../output/degree/attackedtargeted/randomNodeAttack/{dataset}'
    file_pattern = os.path.join(directory, f'{epsilon}_{beta}_*_.txt')
    files = glob.glob(file_pattern)
    gammas = []
    for file in files:
        basename = os.path.basename(file)
        gamma = float(basename.split('_')[2])
        gammas.append(gamma)
    return sorted(gammas)

def read_all_parameters_by_gamma(dataset, beta, epsilon):
    gammas = get_all_gamma_values(dataset, epsilon, beta)
    parameters_list = []
    
    print(f"Reading parameters for {len(gammas)} gamma values: {gammas}")
    
    for gamma in gammas:
        totalNum, attackerNum, targetNum, targetNodes = read_parameters_for_gamma(dataset, beta, gamma)
        if totalNum is not None:
            parameters_list.append({
                'gamma': gamma,
                'totalNum': totalNum,
                'attackerNum': attackerNum,
                'targetNum': targetNum,
                'targetNodes': targetNodes
            })
        else:
            print(f"Failed to read parameters for gamma {gamma}")
    
    return parameters_list

def calculate_and_plot_degree_gain_vs_gamma(dataset, epsilon, beta):
    """固定epsilon和beta，观察gamma变化对攻击增益的影响"""
    print(f"Starting calculation for dataset {dataset} with fixed epsilon={epsilon}, beta={beta}")
    
    # 读取所有gamma值对应的参数
    parameters_list = read_all_parameters_by_gamma(dataset, beta, epsilon)
    if not parameters_list:
        print("ERROR: No parameter files found!")
        return
    
    perturbed_degrees = read_perturbed_degrees_by_gamma(dataset, epsilon)
    if perturbed_degrees is None:
        print("ERROR: No perturbed degree file found!")
        return
    
    gammas = [params['gamma'] for params in parameters_list]
    targetNodes_list = [params['targetNodes'] for params in parameters_list]
    attackerNum_list = [params['attackerNum'] for params in parameters_list]
    totalNum = parameters_list[0]['totalNum']  
    
    print(f"Found {len(gammas)} gamma values: {gammas}")
    print(f"Total users: {totalNum}")
    
    random_node_degrees = read_attack_degrees_by_gamma('randomNodeAttack', dataset, epsilon, beta)
    random_value_degrees = read_attack_degrees_by_gamma('randomValueAttack', dataset, epsilon, beta)
    maximum_gain_degrees = read_attack_degrees_by_gamma('maximumGainAttack', dataset, epsilon, beta)

    if len(gammas) == 0:
        print("ERROR: No gamma files found!")
        return
    
    random_node_gains = calculate_degree_gains(random_node_degrees, perturbed_degrees, targetNodes_list, totalNum, attackerNum_list)
    random_value_gains = calculate_degree_gains(random_value_degrees, perturbed_degrees, targetNodes_list, totalNum, attackerNum_list)
    maximum_gain_gains = calculate_degree_gains(maximum_gain_degrees, perturbed_degrees, targetNodes_list, totalNum, attackerNum_list)

    before_degrees_list, after_degrees_list = read_untargeted_attack_degrees_by_gamma(dataset, epsilon, beta)
    untargeted_gains = calculate_untargeted_global_gains(before_degrees_list, after_degrees_list, totalNum)

    print(f"Calculated gains:")
    print(f"  Random node gains: {random_node_gains}")
    print(f"  Random value gains: {random_value_gains}")
    print(f"  Maximum gain gains: {maximum_gain_gains}")
    print(f"  Untargeted gains: {untargeted_gains}")

    # 绘制两个分离的图表
    plot_targeted_attacks_vs_gamma(gammas, random_value_gains, random_node_gains, maximum_gain_gains, epsilon, beta)
    plot_untargeted_attack_vs_gamma(gammas, untargeted_gains, epsilon, beta)

if __name__ == '__main__':
    dataset = 1
    epsilon = 4.0  # 固定epsilon
    beta = 0.05    # 固定beta
    
    print(f'Dataset: {dataset}')
    print(f'Fixed Epsilon: {epsilon}')
    print(f'Fixed Beta: {beta}')
    
    calculate_and_plot_degree_gain_vs_gamma(dataset, epsilon, beta)
