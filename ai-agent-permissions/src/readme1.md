# RQ1 与 MoE Gating Leakage Attack 复现实验说明

本文档说明如何复现当前阶段的两个核心实验：

1. **RQ1 性能验证**：比较 AgentPerms(IC+CF)、No Personalization、Clustering、MoE-only、Federated MoE 的准确率与高置信覆盖率。
2. **MoE Gating Leakage Attack**：验证中心化 `MoE-only` 的门控路由是否会在黑盒输出轨迹中泄露用户所属 expert 与用户偏好。

当前攻击实验只针对 **MoE-only**，不是 FedMoE，也不引入 LDP。LDP 防御实验应在攻击存在性被验证后再做。

---

## 1. 环境与运行位置

所有命令都从 `ai-agent-permissions/src/` 目录运行：

```bash
cd ai-agent-permissions/src
```

主要依赖：

- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `matplotlib`

如果缺少依赖，先安装：

```bash
pip install -r ../requirements.txt
```

---

## 2. RQ1：性能验证

### 2.1 实验目的

RQ1 关注：

> MoE 是否能提升权限预测准确率与高置信覆盖率？

当前比较 5 个方法：

1. `AgentPerms (IC+CF)`：原始论文方法，预测文件为 `../results/ic_cf_predictions.json`
2. `No Personalization`：无个性化单模型
3. `Clustering`：聚类个性化模型
4. `Federated MoE`：联邦个性化 MoE
5. `MoE-only`：中心化 MoE，上界/ablation，不是隐私保护主方法

### 2.2 生成各方法预测结果

如果已有结果文件，可以跳过对应步骤。

```bash
python permission_no_personalization.py
python permission_clustering.py
python permission_moe.py
python permission_federated_moe.py
```

`AgentPerms (IC+CF)` 使用原始论文预测结果：

```text
../results/ic_cf_predictions.json
```

如果需要重新生成 IC+CF，需要先有 CF 分数：

```bash
python permission_cf_only.py
python permission_ic_cf.py
```

注意：`permission_ic_cf.py` 会调用 OpenAI-compatible API，例如 DeepSeek，需要正确配置 `.env`。

### 2.3 生成 RQ1 曲线

```bash
python rq1_visualization.py
```

输出文件：

```text
../results/rq1_accuracy_vs_threshold.png
../results/rq1_coverage_vs_threshold.png
../results/rq1_threshold_curves.json
```

### 2.4 RQ1 指标公式

对每个预测样本，有：

```text
confidence = max(p_allow, 1 - p_allow)
pred = 1[p_allow >= 0.5]
```

给定置信度阈值 `tau`：

```text
Accuracy(tau) = # correct predictions with confidence >= tau / # predictions with confidence >= tau
```

```text
Coverage(tau) = # predictions with confidence >= tau / # all predictions
```

### 2.5 当前结果如何解释

当前 RQ1 图中：

- `MoE-only` 整体准确率高于原始 `AgentPerms(IC+CF)`，说明 MoE 个性化结构有性能潜力。
- `Federated MoE` 覆盖率较好，但准确率仍低于 `MoE-only`，说明联邦训练会带来性能损失。
- `MoE-only` 是中心化上界，不是最终隐私保护方案。

---

## 3. MoE Gating Leakage Attack

### 3.1 实验目的

攻击实验当前只验证一个新攻击面：

> 攻击者是否能通过一系列 crafted queries 的黑盒输出轨迹，推断用户所属 expert 或用户偏好？

该实验不引入 LDP，不做防御。目的是先证明 vanilla MoE 存在 gating leakage。

### 3.2 威胁模型

攻击者是 **black-box API adversary**。

攻击者可以观察：

```text
p_allow
confidence
pred
margin = |p_allow - 0.5|
```

攻击者不能看到：

```text
gate_weights
top_expert
用户训练历史
真实用户偏好标签
```

`gate_weights` 和用户偏好标签只在实验评估时使用，作为 ground truth，不提供给攻击者。

### 3.3 运行攻击实验

先确保已经有 MoE-only 预测文件：

```text
../results/moe_rq1_predictions.json
```

如果没有，先运行：

```bash
python permission_moe.py
```

然后运行攻击实验：

```bash
python attack_moe_gating_leakage_asr.py
```

输出文件：

```text
../results/attack_moe_gating_leakage_asr.json
../results/attack_moe_expert_asr_vs_queries.png
../results/attack_moe_preference_asr_vs_queries.png
../results/attack_moe_preference_amplification_vs_queries.png
```

---

## 4. 攻击指标定义

### 4.1 用户 dominant expert

MoE 的 routing 是 query-level 的，因此先定义用户级 dominant expert：

```text
E_i* = argmax_k mean_q g_{i,q,k}
```

含义：

- `i` 是用户
- `q` 是该用户的查询样本
- `g_{i,q,k}` 是该样本路由到 expert `k` 的 gate weight
- `E_i*` 是用户整体上最常被路由到的 expert

这是 ground truth，只用于评估。

### 4.2 Expert-ASR

攻击者输入前 `m` 个 crafted queries 的黑盒输出轨迹，预测用户 dominant expert：

```text
hat_E_i = A_E(x_i)
```

其中 `x_i` 包含该用户在 `m` 个查询上的：

```text
[p_allow, confidence, pred, margin]
```

攻击成功率：

```text
Expert-ASR(m) = # {i : hat_E_i = E_i*} / # all users
```

随机基线：

```text
1 / K
```

当前 `K=4`，所以随机基线是 `0.25`。

### 4.3 用户偏好标签

用户偏好来自真实训练历史，不伪造标签。

先计算每个用户的历史允许率：

```text
allow_rate_i = # allow decisions / # all training decisions
```

再按三分位划分：

```text
privacy-sensitive: allow_rate_i <= q33
balanced:          q33 < allow_rate_i < q67
utility-driven:    allow_rate_i >= q67
```

### 4.4 Preference-ASR

攻击者预测用户偏好：

```text
hat_Y_i = A_Y(x_i)
```

攻击成功率：

```text
Preference-ASR(m) = # {i : hat_Y_i = Y_i*} / # all users
```

随机基线：

```text
1 / 3
```

因为偏好分成三类。

### 4.5 Routing amplification

为了证明 routing 是否额外放大偏好泄露，比较两个攻击器：

1. `Direct-output attack`：只使用黑盒输出轨迹
2. `Routing-augmented attack`：使用黑盒输出轨迹 + 攻击者推断出的 expert posterior

放大效应定义为：

```text
Amplification(m) = Preference-ASR_routing(m) - Preference-ASR_direct(m)
```

如果该值大于 0，说明 inferred routing 对偏好推断有额外帮助。

---

## 5. 当前攻击结果说明

当前 `attack_moe_gating_leakage_asr.py` 的结果大致为：

### 5.1 Expert 推断

随着 crafted queries 数量增加，Expert-ASR 从约 `0.58` 增加到约 `0.79`。

这说明：

> 攻击者只观察 MoE 的黑盒输出轨迹，就能远高于随机基线地恢复用户 dominant expert。

这是当前最强的 gating leakage 证据。

### 5.2 Preference 推断

Direct-output attack 和 Routing-augmented attack 都明显高于 `1/3` 随机基线。

这说明：

> MoE 输出行为本身已经泄露用户偏好。

### 5.3 Routing amplification

当前 Routing amplification 较弱，大多在 `0% ~ 2%` 左右。

这说明：

> 现阶段可以强证明 expert routing 可恢复，但 routing 对 preference inference 的额外放大还不够强，需要后续优化攻击设计或改进偏好标签定义。

---

## 6. 当前阶段结论

当前阶段只得出以下结论：

1. `MoE-only` 的 expert routing 可以被黑盒输出轨迹恢复。
2. 用户偏好可以通过 MoE 输出行为被推断，成功率高于随机基线。
3. Routing amplification 当前存在但较弱。
4. 这些结果说明 vanilla MoE 存在新的 gating leakage attack surface。
5. 之后才应引入防御，例如 noise gating、expert smoothing、entropy regularization、Selective LDP。

---

## 7. 文件索引

### RQ1

```text
../results/rq1_accuracy_vs_threshold.png
../results/rq1_coverage_vs_threshold.png
../results/rq1_threshold_curves.json
```

### MoE-only

```text
../results/moe_rq1_predictions.json
../results/moe_rq1_metrics.json
```

### Gating Leakage Attack

```text
../results/attack_moe_gating_leakage_asr.json
../results/attack_moe_expert_asr_vs_queries.png
../results/attack_moe_preference_asr_vs_queries.png
../results/attack_moe_preference_amplification_vs_queries.png
```
