# REAL README: RQ1 MoE 实验运行指南

这个文件记录当前仓库里新增的 RQ1 实验怎么跑、结果放在哪里、图怎么生成。

当前 RQ1 目标是验证：

> MoE-only 是否能提升 AI agent 权限预测的准确率和高置信覆盖率。

本阶段暂时不引入 LDP / 联邦学习，只比较四个方法：

1. `AgentPerms (IC+CF)`：原始论文方法，LLM + collaborative filtering。
2. `No Personalization`：单模型，所有用户共享一个模型。
3. `Clustering`：先按用户历史偏好聚类，每个 cluster 训练一个模型。
4. `MoE-only`：多个 experts + gate，根据用户画像和当前权限请求动态路由。

---

## 1. 目录位置

所有命令建议在：

```powershell
ai-agent-permissions\src
```

下运行。

也就是：

```powershell
cd "C:\Users\luyixuan\Desktop\大三下\计算机网络安全\Privacy of ai-agent-permission\ai-agent-permissions\src"
```

不要在 `ai-agent-permissions` 根目录直接运行这些脚本，因为脚本里使用了相对路径：

```text
../data
../queries.json
../results
```

---

## 2. 安装依赖

如果还没安装依赖，先回到 `ai-agent-permissions` 目录：

```powershell
cd ..
pip install -r requirements.txt
```

新增 RQ1 实验需要：

```text
torch
matplotlib
scikit-learn
numpy
pandas
```

如果 `torch` 安装失败，可以单独安装 CPU 版本：

```powershell
pip install torch
```

然后再回到 `src`：

```powershell
cd src
```

---

## 3. 输入数据

新实验主要读取这些文件：

```text
data\processed_dataset.json
data\data_types.csv
queries.json
```

其中：

- `processed_dataset.json`：181 个过滤后的用户，包含每个用户的 `training` / `testing` 权限决策。
- `data_types.csv`：raw data type 到 generic data type 的映射。
- `queries.json`：65 个 AI agent 任务场景，包含 domain、tool、data type 等信息。

实验只使用二分类标签：

```text
Yes, always share -> 1
No, never share   -> 0
```

其他 ask-first 类型标签会被过滤。

---

## 4. 运行顺序

### Step 1: 单模型 No Personalization

```powershell
python permission_no_personalization.py
```

输出：

```text
results\no_personalization_predictions.json
results\no_personalization_metrics.json
```

这个模型不使用用户画像，也不使用用户 ID。它只看当前权限请求：

```text
query_id + receiver + generic_data_type + domain
```

作用：作为没有个性化能力的全局 baseline。

---

### Step 2: 聚类个性化 Clustering

```powershell
python permission_clustering.py
```

输出：

```text
results\clustering_personalization_predictions.json
results\clustering_personalization_metrics.json
```

这个模型先根据用户训练历史构造用户画像，然后用 KMeans 聚类：

```text
默认 cluster 数量 = 4
```

每个 cluster 单独训练一个 MLP。

作用：作为粗粒度个性化 baseline。

---

### Step 3: MoE-only

```powershell
python permission_moe.py
```

输出：

```text
results\moe_rq1_predictions.json
results\moe_rq1_metrics.json
```

MoE 架构：

```text
ContextEncoder(query_id, receiver, generic_data_type, domain)
        -> context vector z

concat(z, user_profile)
        -> gate network
        -> expert weights

experts(z)
        -> expert predictions

weighted sum
        -> final p_allow
```

默认配置：

```text
n_experts = 4
top_k = 2
epochs = 60
load_balance_coef = 0.01
```

作用：验证 MoE 是否比单模型和聚类个性化更有效。

---

### Step 4: 生成两张 RQ1 折线图

```powershell
python rq1_visualization.py
```

它会读取：

```text
results\ic_cf_predictions.json
results\no_personalization_predictions.json
results\clustering_personalization_predictions.json
results\moe_rq1_predictions.json
```

然后输出：

```text
results\rq1_threshold_curves.json
results\rq1_accuracy_vs_threshold.png
results\rq1_coverage_vs_threshold.png
```

两张图分别是：

```text
Accuracy vs. confidence threshold τ
Coverage vs. confidence threshold τ
```

---

## 5. 四个方法的结果来源

| 方法 | 是否需要新跑 | 预测文件 |
|---|---|---|
| AgentPerms (IC+CF) | 不一定，默认复用旧结果 | `results\ic_cf_predictions.json` |
| No Personalization | 需要跑 | `results\no_personalization_predictions.json` |
| Clustering | 需要跑 | `results\clustering_personalization_predictions.json` |
| MoE-only | 需要跑 | `results\moe_rq1_predictions.json` |

注意：

当前 `AgentPerms (IC+CF)` 的旧结果可以直接用于画图，但旧代码里可能存在一个问题：

```text
cf_scores.csv 的 user_id 是整数，
permission_ic_cf.py 查找的是 P039 这种用户 ID。
```

因此旧 `IC+CF` 结果可能没有真正注入 CF 推荐。正式写论文前，最好修复这个 ID 映射问题后重跑 IC+CF。

---

## 6. 指标解释

每个模型都会输出：

```text
accuracy
precision
recall
f1
fpr
fnr
auc
high_conf_accuracy
coverage
```

其中最重要的是：

### Accuracy

全体测试样本上的准确率：

```text
预测正确数量 / 测试样本总数
```

### Confidence

模型对最终预测标签的置信度。

对于我们自己的三个神经模型：

```text
p_allow = 模型认为应该 allow 的概率
confidence = max(p_allow, 1 - p_allow)
```

例如：

```text
p_allow = 0.90 -> 预测 allow，confidence = 0.90
p_allow = 0.10 -> 预测 deny， confidence = 0.90
p_allow = 0.55 -> 预测 allow，confidence = 0.55
```

### Confidence threshold τ

阈值 `τ` 表示：

```text
只保留 confidence >= τ 的预测
```

`τ` 越大，模型越谨慎。

### Coverage / 高置信覆盖率

```text
coverage(τ) = confidence >= τ 的样本数量 / 全部测试样本数量
```

它表示在阈值 `τ` 下，模型能自动处理多少比例的权限请求。

随着 `τ` 增大，coverage 应该单调下降或保持不变。

### Accuracy vs. τ

可视化里的 accuracy 不是全体 accuracy，而是：

```text
在 confidence >= τ 的样本子集上的 accuracy
```

通常：

```text
τ 越大 -> accuracy 越高，但 coverage 越低
```

这体现的是自动化权限决策中的 tradeoff：

```text
更可靠的自动决策 <-> 更少的自动覆盖范围
```

---

## 7. 推荐完整运行命令

在 PowerShell 中：

```powershell
cd "C:\Users\luyixuan\Desktop\大三下\计算机网络安全\Privacy of ai-agent-permission\ai-agent-permissions\src"

python permission_no_personalization.py
python permission_clustering.py
python permission_moe.py
python rq1_visualization.py
```

运行结束后看：

```text
ai-agent-permissions\results\rq1_accuracy_vs_threshold.png
ai-agent-permissions\results\rq1_coverage_vs_threshold.png
```

---

8. 如果只想看最终数值

直接打开：

```text
results\no_personalization_metrics.json
results\clustering_personalization_metrics.json
results\moe_rq1_metrics.json
results\ic_cf_metrics.json
```

重点看：

```text
accuracy
high_conf_accuracy
coverage
auc
```

当前 RQ1 论文叙事可以围绕：

```text
MoE 在较低/中等 confidence threshold 下准确率最高；
MoE 在较高 threshold 下仍保持较高 coverage；
MoE 不需要 LLM API，也不需要上传完整用户历史，是后续 Federated + LDP 的基础。
```

---

9. 常见问题

Q1: 报错找不到 `../data/processed_dataset.json`

说明运行目录错了。

请进入：

```powershell
ai-agent-permissions\src
```

再运行脚本。

Q2: `torch` 找不到

安装：

```powershell
pip install torch
```

### Q3: 可视化脚本跳过某个 baseline

如果看到：

```text
[skip] missing: ...
```

说明对应预测文件还没生成。

先运行对应脚本。

### Q4: 旧结果会不会被覆盖？

会。以下文件会被覆盖：

```text
results\no_personalization_predictions.json
results\no_personalization_metrics.json
results\clustering_personalization_predictions.json
results\clustering_personalization_metrics.json
results\moe_rq1_predictions.json
results\moe_rq1_metrics.json
results\rq1_threshold_curves.json
results\rq1_accuracy_vs_threshold.png
results\rq1_coverage_vs_threshold.png
```

如果要保留旧结果，先备份：

```powershell
cd ..\results
mkdir backup_rq1
copy no_personalization_* backup_rq1\
copy clustering_personalization_* backup_rq1\
copy moe_rq1_* backup_rq1\
copy rq1_* backup_rq1\
```

