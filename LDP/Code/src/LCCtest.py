import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os

def calculate_clustering_gain(attack_coeffs, perturbed_coeffs, targets, user_num, num_of_attackers):
   #calculate another three types attack's gain
    gain = 0.0
    for i in targets:
        if i < len(attack_coeffs) and i < len(perturbed_coeffs):
            attacked_coeff = attack_coeffs[i]
            original_coeff = perturbed_coeffs[i]
            diff = attacked_coeff - original_coeff
            gain += abs(diff)
    return gain

def calculate_untargeted_global_gain(before_coeffs, after_coeffs, total_users):
    def average_clustering_coefficient(coeffs):
        if not coeffs:
            return 0.0
        return sum(coeffs) / len(coeffs)  # find avg Clustering Coefficient of global
    
    avg_before = average_clustering_coefficient(before_coeffs)
    avg_after = average_clustering_coefficient(after_coeffs)
    return abs(avg_after - avg_before)

def read_clustering_coefficients(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file if line.strip()]

def get_epsilon_sorted_files(directory, dataset, beta, gamma):
    full_dir = os.path.join(directory, str(dataset))
    file_pattern = os.path.join(full_dir, f'*_{beta}_{gamma}*.txt')  
    files = glob.glob(file_pattern)
    def extract_epsilon(filename):
        try:
            return float(os.path.basename(filename).split('_')[0])
        except (IndexError, ValueError):
            return float('inf')  
    return sorted(files, key=extract_epsilon)

def read_attack_clustering_coeffs(attack_type, dataset, beta, gamma):
    directory = f'../output/attack/{attack_type}'
    file_paths = get_epsilon_sorted_files(directory, dataset, beta, gamma)
    print(f"Looking for clustering files in: {directory}/{dataset}")
    print(f"Found {len(file_paths)} files for {attack_type}")
    if len(file_paths) == 0:
        print(f"  No files found matching pattern: *_{beta}_{gamma}*.txt")
        if os.path.exists(os.path.join(directory, str(dataset))):
            print(f"  Actual files: {os.listdir(os.path.join(directory, str(dataset)))}")
    else:
        print(f"  Files: {[os.path.basename(f) for f in file_paths]}")
    return [read_clustering_coefficients(file_path) for file_path in file_paths]

def read_untargeted_attack_clustering_coeffs(dataset, beta, gamma):

    directory = f'../output/attack/untargetedAttack'

    file_paths = get_epsilon_sorted_files(directory, dataset, beta, gamma)

    print(f"Looking for untargeted files in: {directory}/{dataset}")
    print(f"Found {len(file_paths)} files for untargetedAttack")
    if len(file_paths) == 0:
        print(f"  No files found matching pattern: *_{beta}_{gamma}*.txt")
        if os.path.exists(os.path.join(directory, str(dataset))):
            print(f"  Actual files: {os.listdir(os.path.join(directory, str(dataset)))}")
        return [], []

    after_coeffs_list = [read_clustering_coefficients(fp) for fp in file_paths]

    perturbed_coeffs_list = read_perturbed_clustering_coeffs(dataset)[::-1]  # make sure that the sorting is same as RVA

    min_len = min(len(after_coeffs_list), len(perturbed_coeffs_list))
    before_coeffs_list = perturbed_coeffs_list[:min_len]
    after_coeffs_list  = after_coeffs_list[:min_len]

    print(f"Aligned untargeted data: {min_len} points")
    return before_coeffs_list, after_coeffs_list

def read_perturbed_clustering_coeffs(dataset):
    directory = f'../output/perturbed/{dataset}'
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

    print(f"Found {len(epsilon_files)} perturbed clustering coefficient files:")
    for epsilon, file in epsilon_files:
        print(f"  Epsilon {epsilon}: {os.path.basename(file)}")

    perturbed_coeffs = []
    for _, file_path in epsilon_files:
        if os.path.exists(file_path):
            coeffs = read_clustering_coefficients(file_path)
            perturbed_coeffs.append(coeffs)
        else:
            print(f"Warning: File not found: {file_path}")

    return perturbed_coeffs

def calculate_clustering_gains(attack_coeffs_list, perturbed_coeffs_list, targetNodes, user_num, num_of_attackers):
    gains = []
    min_len = min(len(attack_coeffs_list), len(perturbed_coeffs_list))
    for i in range(min_len):
        gain = calculate_clustering_gain(attack_coeffs_list[i], perturbed_coeffs_list[i], targetNodes, user_num, num_of_attackers)
        gains.append(gain)
    return gains

def calculate_untargeted_global_gains(before_coeffs_list, after_coeffs_list, total_users):
    #caluculate the global gains
    gains = []
    min_len = min(len(before_coeffs_list), len(after_coeffs_list))
    for i in range(min_len):
        gain = calculate_untargeted_global_gain(before_coeffs_list[i], after_coeffs_list[i], total_users)
        gains.append(gain)
    return gains

def plot_clustering_gains(epsilons, random_value_gains, random_node_gains, maximum_gain_gains):
    min_len = min(len(epsilons), len(random_value_gains), len(random_node_gains), len(maximum_gain_gains))
    if min_len == 0:
        print("ERROR: No data to plot!")
        return

    epsilons = epsilons[:min_len]
    random_value_gains = random_value_gains[:min_len]
    random_node_gains = random_node_gains[:min_len]
    maximum_gain_gains = maximum_gain_gains[:min_len]

    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(400 * px, 300 * px))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    ax.plot(epsilons, random_value_gains, marker='s', linestyle='--', label="RVA")
    ax.plot(epsilons, random_node_gains, marker='o', linestyle='-', label="RNA")
    ax.plot(epsilons, maximum_gain_gains, marker='^', linestyle='-.', label="MGA")

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel(r'$\epsilon$', fontsize=12)
    ax.set_ylabel(r'Targeted Clustering Gain $\left| G \right|$', fontsize=12)
    ax.set_yscale('symlog', linthresh=0.001, linscale=1)
    ax.grid(True, alpha=0.7)
    ax.legend()

    for spine in ['left', 'bottom', 'right', 'top']:
        ax.spines[spine].set_color('#c8c8c8')

    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    plt.savefig('targeted_clustering_gain_epsilon_plot.svg', dpi=300)
    plt.show()

def plot_untargeted_global_gains(epsilons, untargeted_gains):
    if len(untargeted_gains) == 0:
        print("WARNING: No UA data to plot!")
        return

    min_len = min(len(epsilons), len(untargeted_gains))
    epsilons = epsilons[:min_len]
    untargeted_gains = untargeted_gains[:min_len]

    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(400 * px, 300 * px))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    ax.plot(epsilons, untargeted_gains, marker='D', linestyle='-', color='purple', label="Untargeted Attack")

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel(r'$\epsilon$', fontsize=12)
    ax.set_ylabel(r'Global Clustering Gain $\Delta C$', fontsize=12)
    ax.set_yscale('symlog', linthresh=0.001, linscale=1)
    ax.grid(True, alpha=0.7)
    ax.legend()

    for spine in ['left', 'bottom', 'right', 'top']:
        ax.spines[spine].set_color('#c8c8c8')

    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    plt.savefig('untargeted_global_clustering_gain_epsilon_plot.svg', dpi=300)
    plt.show()

def extract_int(pattern, text, default=0):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else default

def find_parameter_file(dataset, beta, gamma):
    config_dir = f'../output/config/{dataset}'
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

def calculate_and_plot_clustering_gain(dataset, beta, gamma, targetNodes, user_num, num_of_attackers):
    print(f"Starting clustering coefficient calculation for dataset {dataset}")
    
    perturbed_coeffs_list = read_perturbed_clustering_coeffs(dataset)[::-1]
    print(f"Loaded {len(perturbed_coeffs_list)} perturbed clustering coefficient files")
    
    if len(perturbed_coeffs_list) == 0:
        print("ERROR: No perturbed clustering coefficient files found!")
        return
    
    random_node_coeffs = read_attack_clustering_coeffs('randomNodeAttack', dataset, beta, gamma)
    random_value_coeffs = read_attack_clustering_coeffs('randomValueAttack', dataset, beta, gamma)
    maximum_gain_coeffs = read_attack_clustering_coeffs('maximumGainAttack', dataset, beta, gamma)
    
    before_coeffs_list, after_coeffs_list = read_untargeted_attack_clustering_coeffs(dataset, beta, gamma)

    attack_files = get_epsilon_sorted_files(
        f'../output/attack/randomNodeAttack', dataset, beta, gamma)
    epsilons = []
    for f in attack_files:
        try:
            epsilons.append(float(os.path.basename(f).split('_')[0]))
        except (IndexError, ValueError):
            continue
    
    print(f"Found {len(epsilons)} epsilon values: {epsilons}")

    if len(epsilons) == 0:
        print("ERROR: No epsilon files found!")
        return

    print(f"Total users: {user_num}")
    
    random_node_gains = calculate_clustering_gains(
        random_node_coeffs, perturbed_coeffs_list, targetNodes, user_num, num_of_attackers)[:len(epsilons)]
    random_value_gains = calculate_clustering_gains(
        random_value_coeffs, perturbed_coeffs_list, targetNodes, user_num, num_of_attackers)[:len(epsilons)]
    maximum_gain_gains = calculate_clustering_gains(
        maximum_gain_coeffs, perturbed_coeffs_list, targetNodes, user_num, num_of_attackers)[:len(epsilons)]
    untargeted_gains = calculate_untargeted_global_gains(
        before_coeffs_list, after_coeffs_list, user_num)[:len(epsilons)]

    print(f"Calculated clustering coefficient gains:")
    print(f"  Random node gains (len={len(random_node_gains)}): {random_node_gains}")
    print(f"  Random value gains (len={len(random_value_gains)}): {random_value_gains}")
    print(f"  Untargeted gains (len={len(untargeted_gains)}): {untargeted_gains}")
    print(f"  Maximum gain gains (len={len(maximum_gain_gains)}): {maximum_gain_gains}")

    if len(untargeted_gains) == 0:
        print("WARNING: No untargeted data found, skipping UA in plot.")

    # make graphs of the two attack
    plot_clustering_gains(epsilons, random_value_gains, random_node_gains, maximum_gain_gains)
    plot_untargeted_global_gains(epsilons, untargeted_gains)

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
    
    calculate_and_plot_clustering_gain(dataset, beta, gamma, targetNodes, totalNum, attackerNum)
