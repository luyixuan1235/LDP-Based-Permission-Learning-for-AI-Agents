import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os
import math
import numpy as np

def calculate_global_degree_centrality(degrees, total_nodes):
    """
    计算全局平均度中心性
    :param degrees: 节点的度数列表
    :param total_nodes: 总节点数
    :return: 全局平均度中心性
    """
    if total_nodes <= 1:
        return 0.0
    
    # 计算每个节点的度中心性 c_i = d_i / (N-1)
    c_values = [d / (total_nodes - 1) for d in degrees]
    
    # 全局平均度中心性 C = 1/N * Σc_i
    return sum(c_values) / len(c_values)

def calculate_degree_gain(attack_degrees, perturbed_degree, user_num, num_of_attackers):
    gain = 0.0
    for idx in range(len(attack_degrees)):
        attacked_gain = attack_degrees[idx] / (user_num + num_of_attackers)
        original_normalized = perturbed_degree[idx] / (user_num)
        diff = attacked_gain - original_normalized
        gain += abs(diff)
    return gain

def read_degrees(file_path):
    with open(file_path, 'r') as file:
        return [int(line.strip()) for line in file if line.strip()]

def get_epsilon_sorted_files(directory, dataset, beta, gamma):
    file_pattern = os.path.join(directory, str(dataset), f'*_{beta}_{gamma}.txt')
    files = glob.glob(file_pattern)
    return sorted(files, key=lambda x: float(os.path.basename(x).split('_')[0]))

def read_attack_degrees(attack_type, dataset, beta, gamma):
    directory = f'../output/degree/attackedtargeted/{attack_type}'
    file_paths = get_epsilon_sorted_files(directory, dataset, beta, gamma)
    print(f"Looking for files in: {directory}")
    print(f"Found {len(file_paths)} files for {attack_type}")
    if len(file_paths) == 0:
        print(f"  No files found matching pattern: *_{beta}_{gamma}.txt")
    else:
        print(f"  Files: {[os.path.basename(f) for f in file_paths]}")
    return [read_degrees(file_path) for file_path in file_paths]

def read_perturbed_degrees(dataset):
    directory = '../output/degree/perturb'
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
        degrees = read_degrees(file_path)
        perturbed_degrees.append(degrees)

    return perturbed_degrees

def calculate_degree_gains(attack_degrees_list, perturbed_degrees_list, user_num, num_of_attackers):
    gains = []
    for attack_degrees, perturbed_degrees in zip(attack_degrees_list, perturbed_degrees_list):
        gain = calculate_degree_gain(attack_degrees, perturbed_degrees, user_num, num_of_attackers)
        gains.append(gain)
    return gains

def plot_degree_gains(epsilons, random_value_gains, random_node_gains, untargeted_gains, maximum_gain_gains, untargeted_theory_gains):
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(400 * px, 300 * px))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    ax.plot(epsilons, random_value_gains, marker='s', linestyle='--', label="RVA")
    ax.plot(epsilons, random_node_gains, marker='o', linestyle='-', label="RNA")
    ax.plot(epsilons, untargeted_gains, marker='x', linestyle=':', label="UA (experiment)")  # 无目标攻击实验值
    ax.plot(epsilons, untargeted_theory_gains, marker='+', linestyle='-.', label="UA (theoretical)")  # 无目标攻击理论值
    ax.plot(epsilons, maximum_gain_gains, marker='^', linestyle='-.', label="MGA")

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel(r'$\epsilon$', fontsize=12)
    ax.set_ylabel(r'$\left| G \right|$', fontsize=12)
    ax.set_yscale('symlog', linthresh=0.001, linscale=1)
    ax.set_ylim(0.06, 20)
    ax.grid(True, alpha=0.7)
    ax.legend()

    for spine in ['left', 'bottom', 'right', 'top']:
        ax.spines[spine].set_color('#c8c8c8')

    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    plt.savefig('degree_gain_epsilon_plot.svg', dpi=300)
    plt.show()

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

    return None

def read_parameters(dataset, beta, gamma):
    param_file = find_parameter_file(dataset, beta, gamma)
    if param_file is None:
        print("Parameter file not found!")
        return None, None, None, None

    with open(param_file, 'r') as file:
        data_parameter = file.read()

    totalNum = int(re.search(r'totalNum:\s*(\d+)', data_parameter).group(1))
    attackerNum = int(re.search(r'attackerNum:\s*(\d+)', data_parameter).group(1))
    targetNum = int(re.search(r'targetNum:\s*(\d+)', data_parameter).group(1))
    targetNodes = [int(node) for node in re.findall(r'targetNode \d+:\s*(\d+)', data_parameter)]

    return totalNum, attackerNum, targetNum, targetNodes

def theoretical_gain(avg_degree, num_attackers, num_targets, total_nodes):
    N = total_nodes
    m = num_attackers
    r = num_targets
    d_bar = avg_degree
    gain = (m * r) / (N - 1) * (min(r, d_bar) / r - d_bar / (N - 1))
    return abs(gain)

def calculate_untargeted_gain(dataset, beta, gamma, epsilon):
    """
    计算无目标攻击的实际gain值：攻击后 - 攻击前 全局平均度中心性
    :param dataset: 数据集ID
    :param beta: β值
    :param gamma: γ值
    :param epsilon: ε值
    :return: gain值（攻击后 - 攻击前）
    """
    base_dir = f'../output/degree/untargeted_all_nodes/{dataset}'
    
    # 读取攻击前度数
    before_file = os.path.join(base_dir, f'before_attack_{epsilon}_{beta}_{gamma}.txt')
    if not os.path.exists(before_file):
        print(f"File not found: {before_file}")
        return 0.0
    
    # 读取攻击后度数
    after_file = os.path.join(base_dir, f'after_attack_{epsilon}_{beta}_{gamma}.txt')
    if not os.path.exists(after_file):
        print(f"File not found: {after_file}")
        return 0.0
    
    # 读取参数文件
    param_file = find_parameter_file(dataset, beta, gamma)
    if not param_file:
        print("Parameter file not found for untargeted gain calculation")
        return 0.0
    
    with open(param_file, 'r') as f:
        param_data = f.read()
    
    totalNum = int(re.search(r'totalNum:\s*(\d+)', param_data).group(1))
    attackerNum = int(re.search(r'attackerNum:\s*(\d+)', param_data).group(1))
    realUserNum = totalNum - attackerNum
    
    degrees_before = read_degrees(before_file)
    degrees_after = read_degrees(after_file)
    
    # 计算攻击前的全局平均度中心性（只考虑真实用户）
    # C_before = (1/n) * Σ(d_i/(n-1)) for i in real users
    C_before = calculate_global_degree_centrality(degrees_before, realUserNum)
    
    # 计算攻击后的全局平均度中心性（考虑所有节点）
    # C_after = (1/N) * Σ(d_i/(N-1)) for i in all nodes
    C_after = calculate_global_degree_centrality(degrees_after, totalNum)
    
    # 计算gain = C_after - C_before
    gain = C_after - C_before
    
    print(f"Untargeted attack gain for ε={epsilon}: {gain:.6f} "
          f"(C_before={C_before:.6f}, C_after={C_after:.6f})")
    
    return gain

def untargeted_theoretical_max_gain(dataset, beta, gamma, epsilon):
    """
    计算无目标攻击的理论最大gain值：攻击后 - 攻击前 全局平均度中心性
    :param dataset: 数据集ID
    :param beta: β值
    :param gamma: γ值
    :param epsilon: ε值
    :return: 理论最大gain值
    """
    # 读取参数文件
    param_file = find_parameter_file(dataset, beta, gamma)
    if not param_file:
        print("Parameter file not found for theoretical max gain")
        return 0.0
    
    with open(param_file, 'r') as f:
        param_data = f.read()
    
    totalNum = int(re.search(r'totalNum:\s*(\d+)', param_data).group(1))
    attackerNum = int(re.search(r'attackerNum:\s*(\d+)', param_data).group(1))
    realUserNum = totalNum - attackerNum
    
    real_degree_file = f'../output/degree/real/RealDegree_{dataset}.txt'
    if os.path.exists(real_degree_file):
        real_degrees = read_degrees(real_degree_file)
        real_user_degrees = real_degrees[:realUserNum]
        D = sum(real_user_degrees)
    else:
        print(f"Real degree file not found: {real_degree_file}")
        return 0.0
    
    # 攻击前的全局平均度中心性
    # C_before = (1/n) * Σ(d_i/(n-1)) = (1/n) * D/(n-1) = D/(n*(n-1))
    C_before = D / (realUserNum * (realUserNum - 1)) if realUserNum > 1 else 0.0
    
    # 策略1: 假节点报告最小度（孤立节点）
    # 攻击后总度数仍为D（假节点不增加任何连接）
    # C_after_min = D / (N * (N-1))
    C_after_min = D / (totalNum * (totalNum - 1)) if totalNum > 1 else 0.0
    gain_min = C_after_min - C_before
    
    # 策略2: 假节点报告最大度（连接所有其他节点）
    # 攻击后总度数 = D + m * (N-1) （每个假节点连接其他N-1个节点）
    D_max = D + attackerNum * (totalNum - 1)
    # C_after_max = D_max / (N * (N-1))
    C_after_max = D_max / (totalNum * (totalNum - 1)) if totalNum > 1 else 0.0
    gain_max = C_after_max - C_before
    
    # 选择绝对值较大的策略
    if abs(gain_max) > abs(gain_min):
        max_gain = gain_max
        strategy = "max"
    else:
        max_gain = gain_min
        strategy = "min"
    
    print(f"Theoretical max gain for ε={epsilon}: {max_gain:.6f} "
          f"(strategy={strategy}, min_gain={gain_min:.6f}, max_gain={gain_max:.6f})")
    
    return max_gain

def calculate_and_plot_degree_gain(dataset, beta, gamma, targetNodes, user_num, num_of_attackers, target_num):
    perturbed_degrees = read_perturbed_degrees(dataset)[::-1]
    print(f"Read {len(perturbed_degrees)} perturbed degree files")

    random_node_degrees = read_attack_degrees('randomNodeAttack', dataset, beta, gamma)
    random_value_degrees = read_attack_degrees('randomValueAttack', dataset, beta, gamma)
    
    attack_files = get_epsilon_sorted_files(f'../output/degree/attackedtargeted/randomNodeAttack', dataset, beta, gamma)
    epsilons = [float(os.path.basename(f).split('_')[0]) for f in attack_files]
 
    untargeted_gains = [calculate_untargeted_gain(dataset, beta, gamma, epsilon) for epsilon in epsilons]
    
    untargeted_theory_gains = [untargeted_theoretical_max_gain(dataset, beta, gamma, epsilon) for epsilon in epsilons]
    
    random_node_gains = calculate_degree_gains(random_node_degrees, perturbed_degrees, user_num, num_of_attackers)
    random_value_gains = calculate_degree_gains(random_value_degrees, perturbed_degrees, user_num, num_of_attackers)
    
    avg_degree = 46  
    maximum_gain_gains = [theoretical_gain(avg_degree, num_of_attackers, target_num, user_num + num_of_attackers) for _ in epsilons]

    plot_degree_gains(epsilons, random_value_gains, random_node_gains, untargeted_gains, maximum_gain_gains, untargeted_theory_gains)
    print('untargeted gains (experimental):', untargeted_gains)
    print('untargeted gains (theoretical):', untargeted_theory_gains)

if __name__ == '__main__':
    dataset = 1
    beta = 0.05
    gamma = 0.05

    totalNum, attackerNum, targetNum, targetNodes = read_parameters(dataset, beta, gamma)
    if totalNum is None:
        print("Failed to read parameters.")
        exit(1)

    print(f"Dataset: {dataset}")
    print(f"Beta: {beta}")
    print(f"Gamma: {gamma}")
    print(f"AttackerNum: {attackerNum}")
    print(f"TargetNum: {targetNum}")
    print(f"TargetNodes: {targetNodes}")

    calculate_and_plot_degree_gain(dataset, beta, gamma, targetNodes, totalNum, attackerNum, targetNum)
