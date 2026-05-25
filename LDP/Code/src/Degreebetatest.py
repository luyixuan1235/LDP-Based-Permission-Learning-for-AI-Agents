import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os
import math
import numpy as np
from scipy import stats

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

def get_beta_sorted_files(directory, dataset, epsilon, gamma):

    file_pattern = os.path.join(directory, str(dataset), f'{epsilon}_*_{gamma}_.txt')
    files = glob.glob(file_pattern)
    return sorted(files, key=lambda x: float(os.path.basename(x).split('_')[1]))

def read_attack_degrees_for_betas(attack_type, dataset, epsilon, gamma):
    directory = f'../output/degree/attackedtargeted/{attack_type}'
    file_paths = get_beta_sorted_files(directory, dataset, epsilon, gamma)
    print(f"Looking for files in: {directory}")
    print(f"Found {len(file_paths)} files for {attack_type}")
    if len(file_paths) == 0:
        print(f"  No files found matching pattern: {epsilon}_*_{gamma}_.txt")
    else:
        print(f"  Files: {[os.path.basename(f) for f in file_paths]}")
    return [read_degrees(file_path) for file_path in file_paths]

def read_untargeted_attack_degrees_for_betas(dataset, epsilon, gamma):
    directory = f'../output/degree/untargeted_all_nodes/{dataset}'
    before_pattern = os.path.join(directory, f'before_attack_{epsilon}_*_{gamma}.txt')
    after_pattern = os.path.join(directory, f'after_attack_{epsilon}_*_{gamma}.txt')
    before_files = sorted(glob.glob(before_pattern), key=lambda x: float(os.path.basename(x).split('_')[3]))
    after_files = sorted(glob.glob(after_pattern), key=lambda x: float(os.path.basename(x).split('_')[3]))
    
    print(f"Looking for untargeted files in: {directory}")
    print(f"Found {len(before_files)} before files and {len(after_files)} after files")
    
    before_degrees_list = [read_degrees(file_path) for file_path in before_files]
    after_degrees_list = [read_degrees(file_path) for file_path in after_files]
    return before_degrees_list, after_degrees_list

def read_perturbed_degrees(dataset, epsilon):
    directory = f'../output/degree/perturbed/{dataset}'
    file_path = os.path.join(directory, f'{epsilon}.txt')
    
    if os.path.exists(file_path):
        print(f"Found perturbed degree file: {epsilon}.txt")
        return read_degrees(file_path)
    else:
        print(f"Warning: Perturbed degree file not found: {file_path}")
        return None

def calculate_degree_gains_by_beta(attack_degrees_list, perturbed_degrees, targetNodes_list, user_num, attackerNum_list):
    gains = []
    for i, attack_degrees in enumerate(attack_degrees_list):
        if i < len(targetNodes_list) and i < len(attackerNum_list):
            gain = calculate_degree_gain(attack_degrees, perturbed_degrees, targetNodes_list[i], user_num, attackerNum_list[i])
            gains.append(gain)
        else:
            print(f"Warning: Missing target nodes or attacker info for beta index {i}")
            gains.append(0.0)
    return gains

def calculate_untargeted_global_gains(before_degrees_list, after_degrees_list, total_users):
    gains = []
    for before_degrees, after_degrees in zip(before_degrees_list, after_degrees_list):
        gain = calculate_untargeted_global_gain(before_degrees, after_degrees, total_users)
        gains.append(gain)
    return gains

def plot_targeted_attacks_vs_beta(betas, random_value_gains, random_node_gains, maximum_gain_gains):
    # the three attacked graphs
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(500 * px, 350 * px))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    ax.plot(betas, random_value_gains, marker='s', linestyle='--', label="RVA", color='#1f77b4')
    ax.plot(betas, random_node_gains, marker='o', linestyle='-', label="RNA", color='#ff7f0e')
    ax.plot(betas, maximum_gain_gains, marker='^', linestyle='-.', label="MGA", color='#2ca02c')

    for i, beta in enumerate(betas):
        ax.annotate(f'{random_value_gains[i]:.3f}', 
                    (beta, random_value_gains[i]),
                    textcoords="offset points", 
                    xytext=(5,5), 
                    ha='left',
                    fontsize=8)
        ax.annotate(f'{random_node_gains[i]:.3f}', 
                    (beta, random_node_gains[i]),
                    textcoords="offset points", 
                    xytext=(5,5), 
                    ha='left',
                    fontsize=8)
        ax.annotate(f'{maximum_gain_gains[i]:.3f}', 
                    (beta, maximum_gain_gains[i]),
                    textcoords="offset points", 
                    xytext=(5,5), 
                    ha='left',
                    fontsize=8)

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel(r'Beta ($\beta$)', fontsize=12)
    ax.set_ylabel(r'Targeted Gain $\left| G \right|$', fontsize=12)
    ax.set_xscale('log')  
    ax.set_yscale('log')  
    
    ax.grid(True, alpha=0.7)
    ax.legend(loc='best')
    plt.title(r'Targeted Attacks Gain vs Beta ($\epsilon=4.0$)')

    for spine in ['left', 'bottom', 'right', 'top']:
        ax.spines[spine].set_color('#c8c8c8')

    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.savefig('targeted_attacks_gain_vs_beta.svg', dpi=300)
    plt.show()

def plot_untargeted_attack_vs_beta(betas, untargeted_gains):
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(500 * px, 350 * px))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    ax.plot(betas, untargeted_gains, marker='X', linestyle='-', label="UA", color='#d62728', linewidth=2.5)
    
    log_betas = np.log10(betas)
    log_gains = np.log10(untargeted_gains)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_betas, log_gains)
    trend_line = 10**(slope * log_betas + intercept)
    ax.plot(betas, trend_line, linestyle='--', color='#9467bd', alpha=0.7, 
            label=f'Trend (R²={r_value**2:.3f})')
    
    avg_gain = np.mean(untargeted_gains)
    ax.axhline(y=avg_gain, color='#17becf', linestyle=':', alpha=0.7, 
               label=f'Avg: {avg_gain:.4f}')
  
    for i, gain in enumerate(untargeted_gains):
        ax.annotate(f'{gain:.4f}', 
                    (betas[i], gain),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel(r'Beta ($\beta$)', fontsize=12)
    ax.set_ylabel(r'Global Gain $\Delta C$', fontsize=12)
    ax.set_xscale('log')  
    ax.set_yscale('log')  
    
    ax.grid(True, alpha=0.7)
    ax.legend(loc='best')
    plt.title(r'Untargeted Attack Global Gain vs Beta ($\epsilon=4.0$)')

    for spine in ['left', 'bottom', 'right', 'top']:
        ax.spines[spine].set_color('#c8c8c8')

    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.savefig('untargeted_attack_global_gain_vs_beta.svg', dpi=300)
    plt.show()

def extract_int(pattern, text, default=0):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else default

def read_parameters_for_beta(dataset, beta, gamma):
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
            print(f"Found parameter file for beta {beta}: {format_name}")
            with open(file_path, 'r') as file:
                data_parameter = file.read()

            totalNum = extract_int(r'totalNum:\s*(\d+)', data_parameter)
            attackerNum = extract_int(r'attackerNum:\s*(\d+)', data_parameter)
            targetNum = extract_int(r'targetNum:\s*(\d+)', data_parameter)
            targetNodes = [int(node) for node in re.findall(r'targetNode \d+:\s*(\d+)', data_parameter)]

            return totalNum, attackerNum, targetNum, targetNodes

    print(f"Warning: No parameter file found for beta {beta}")
    return None, None, None, None

def get_all_beta_values(dataset, epsilon, gamma):
    """从文件名中提取所有beta值"""
    directory = f'../output/degree/attackedtargeted/randomNodeAttack/{dataset}'
    file_pattern = os.path.join(directory, f'{epsilon}_*_{gamma}_.txt')
    files = glob.glob(file_pattern)
    betas = []
    for file in files:
        basename = os.path.basename(file)
        beta = float(basename.split('_')[1])
        betas.append(beta)
    return sorted(betas)

def read_all_parameters_by_beta(dataset, epsilon, gamma):
    """读取所有beta值对应的参数"""
    betas = get_all_beta_values(dataset, epsilon, gamma)
    parameters_list = []
    
    print(f"Reading parameters for {len(betas)} beta values: {betas}")
    
    for beta in betas:
        totalNum, attackerNum, targetNum, targetNodes = read_parameters_for_beta(dataset, beta, gamma)
        if totalNum is not None:
            parameters_list.append({
                'beta': beta,
                'totalNum': totalNum,
                'attackerNum': attackerNum,
                'targetNum': targetNum,
                'targetNodes': targetNodes
            })
        else:
            print(f"Failed to read parameters for beta {beta}")
    
    return parameters_list

def calculate_and_plot_beta_effects(dataset, epsilon, gamma):
    #defalut gamma and epsilon,find the relationship between beta and gain
    print(f"Starting beta analysis for dataset {dataset}, epsilon={epsilon}, gamma={gamma}")
    
    parameters_list = read_all_parameters_by_beta(dataset, epsilon, gamma)
    if not parameters_list:
        print("ERROR: No parameter files found!")
        return
    
    perturbed_degrees = read_perturbed_degrees(dataset, epsilon)
    if perturbed_degrees is None:
        print("ERROR: No perturbed degree file found!")
        return
    
    # 提取信息
    betas = [params['beta'] for params in parameters_list]
    targetNodes_list = [params['targetNodes'] for params in parameters_list]
    attackerNum_list = [params['attackerNum'] for params in parameters_list]
    totalNum = parameters_list[0]['totalNum']  
    
    print(f"Found {len(betas)} beta values: {betas}")
    print(f"Total users: {totalNum}")
    
    random_node_degrees_list = read_attack_degrees_for_betas('randomNodeAttack', dataset, epsilon, gamma)
    random_value_degrees_list = read_attack_degrees_for_betas('randomValueAttack', dataset, epsilon, gamma)
    maximum_gain_degrees_list = read_attack_degrees_for_betas('maximumGainAttack', dataset, epsilon, gamma)

    if len(betas) == 0:
        print("ERROR: No beta files found!")
        return
    
    random_node_gains = calculate_degree_gains_by_beta(random_node_degrees_list, perturbed_degrees, targetNodes_list, totalNum, attackerNum_list)
    random_value_gains = calculate_degree_gains_by_beta(random_value_degrees_list, perturbed_degrees, targetNodes_list, totalNum, attackerNum_list)
    maximum_gain_gains = calculate_degree_gains_by_beta(maximum_gain_degrees_list, perturbed_degrees, targetNodes_list, totalNum, attackerNum_list)

    before_degrees_list, after_degrees_list = read_untargeted_attack_degrees_for_betas(dataset, epsilon, gamma)
    untargeted_gains = calculate_untargeted_global_gains(before_degrees_list, after_degrees_list, totalNum)

    min_length = min(len(random_node_gains), len(random_value_gains), len(maximum_gain_gains), len(untargeted_gains), len(betas))
    
    betas = betas[:min_length]
    random_node_gains = random_node_gains[:min_length]
    random_value_gains = random_value_gains[:min_length]
    maximum_gain_gains = maximum_gain_gains[:min_length]
    untargeted_gains = untargeted_gains[:min_length]

    print(f"Calculated gains for {min_length} beta values:")
    for i, beta in enumerate(betas):
        print(f"  Beta {beta}: RVA={random_value_gains[i]:.4f}, RNA={random_node_gains[i]:.4f}, MGA={maximum_gain_gains[i]:.4f}, UA={untargeted_gains[i]:.4f}")

    plot_targeted_attacks_vs_beta(betas, random_value_gains, random_node_gains, maximum_gain_gains)
    plot_untargeted_attack_vs_beta(betas, untargeted_gains)

if __name__ == '__main__':
    dataset = 1
    epsilon = 4.0  
    gamma = 0.05  
    
    print(f'Dataset: {dataset}')
    print(f'Fixed Epsilon: {epsilon}')
    print(f'Fixed Gamma: {gamma}')
    
    calculate_and_plot_beta_effects(dataset, epsilon, gamma)
