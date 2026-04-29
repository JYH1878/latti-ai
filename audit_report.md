# 自查审计报告

## 一、数据准确性核查

### ✅ 准确的数据

| 数据项 | 来源 | 可信度 |
|--------|------|--------|
| Sparse 总时间 ~21.4s vs Dense ~61.0s | BenchmarkBootstrappSparse / BenchmarkBootstrapp | ✅ 直接来自实际运行 |
| Sparse CtS 11.93s vs Dense 37.66s | 同上 | ✅ 直接来自实际运行 |
| Sparse StC 3.37s vs Dense 12.43s | 同上 | ✅ 直接来自实际运行 |
| LogN=16 Sparse CtS+StC 9.22s vs Dense 18.81s | TestSparseVsDenseFullParams | ✅ 直接来自实际运行 |
| Sparse score 9.6 vs Dense score 24.0 | verify_compiler_opt.py | ✅ 直接来自实际运行 |

### ❌ 存在错误/误导的表述

**错误1："节省 4 个模数层级"**

从 benchmark 输出的 level 变化：
- Dense: ModUp(28) → CtS(24) → Sine(16) → StC(**13**)
- Sparse: ModUp(24) → CtS(20) → Sine(12) → StC(**9**)

**真相**：
- Dense bootstrap 后 residual level = **13**
- Sparse bootstrap 后 residual level = **9**
- Dense 比 Sparse **多 4 个 residual levels**

Sparse 的优势是**速度**（2.85×），不是"节省层级"。相反，Sparse 留给业务计算的层级**更少**。

原始说法来源：混淆了"总层级数"（25 vs 29）和"residual 层级数"。

**错误2："模数层级从 29 → 25"的表述方式**

虽然数字正确（Dense 29 层，Sparse 25 层），但放在"节省"的语境下是误导的。应该表述为：
- "Sparse 使用更紧凑的模数链（25 层 vs 29 层），换取更低的 logQP 和更快的 bootstrapping"

---

## 二、实验可信度分析

### ✅ 可信的实验

| 实验 | 设计 | 可信度 |
|------|------|--------|
| BenchmarkBootstrappSparse vs BenchmarkBootstrapp | LogN=16，完整参数，分阶段计时 | 高 |
| TestSparseVsDenseFullParams | LogN=16，只测 CtS+StC，ratio=2.0 | 高（但范围有限） |
| TestSparseVsDenseBSGS | LogN=13，短参数，多 ratio 对比 | 中（短参数不安全，但用于相对对比是合理的） |
| verify_compiler_opt.py | 直接测试 BtpScoreParam.get_score() | 高 |

### ⚠️ 实验局限

1. **LogN=13 的测试标注了 insecure**：虽然用于相对对比是合理的，但不应该声称其绝对精度/安全性。
2. **TestSparseVsDenseFullParams 只跑了 ratio=2.0**：没有验证其他 ratio 在生产参数下的表现。
3. **没有跑完整的端到端模型推理对比**：CIFAR-10 的完整推理时间没有测（因为太慢）。

---

## 三、是否做了无用功？

### ✅ 原始代码确实没有的内容

| 我们的工作 | 原始代码状态 | 价值 |
|-----------|-------------|------|
| 8 种 preset 参数可配置 | Go SDK 写死 `N16QP1546H192H32` | ✅ 有实际价值 |
| C++ SDK `create_parameter_by_preset` | 不存在 | ✅ 有实际价值 |
| 编译器 `btp_param_list` 多参数 | 只有 `[N16QP1546H192H32]` | ✅ 有实际价值 |
| Sparse vs Dense 系统性 benchmark | 默认只跑 Dense benchmark | ✅ 有学术/工程价值 |
| `score.py` Sparse 折扣 | 不存在 | ⚠️ 理论价值，未经验证对放置策略的实际影响 |
| `graph_partition_dp.py` 阈值调整 | 硬编码 `max_depth - 4` | ⚠️ 理论价值，未经验证对放置策略的实际影响 |

### ❌ 原始代码已经有的内容

| 功能 | 原始代码状态 | 我们的做法 |
|------|-------------|-----------|
| Sparse Secret Encapsulation (SSE) | **已实现**（Boura et al. 2022） | 使用了现有实现，没有做密码学层面的改进 |
| Bootstrapping 完整流程 | **已实现**（ModUp/CtS/EvalMod/StC/KeySwitch） | 未修改核心逻辑 |
| DefaultParametersSparse | **已定义**（4 种 Sparse 参数） | 暴露给上层，但未修改参数本身 |

### 结论

**没有做了完全无用的工作**，但：
1. 我们**没有从零实现稀疏打包 bootstrapping**——它已经在 Lattigo 中实现了
2. 我们的核心贡献是**让已有的稀疏打包能力变得可配置、可量化、可对比**
3. 编译器层的优化（score 折扣、阈值调整）是**理论上的改进**，缺少端到端验证其是否真正改善了放置策略

---

## 四、需要立即修正的内容

### 4.1 所有文档中的"节省 4 层"说法

**当前错误表述**：
- "模数层级从 29 → 25，节省 4 层"
- "节省 4 个模数层级（可留给上层业务计算）"

**正确表述**：
- "Sparse 使用更紧凑的模数链（25 层 vs 29 层），logQP 从 1767 降至 1546"
- "Sparse bootstrap 后 residual level 为 9，Dense 为 13"
- "Sparse 的核心优势是速度（2.85×），而非层级数量"

### 4.2 避免夸大"创新性"

**当前可能夸大的表述**：
- "首次在 LattiAI 中实现稀疏打包 bootstrapping"

**准确表述**：
- "LattiAI 已集成 Lattigo 的 SSE 实现，我们的工作是让稀疏打包能力变得可配置、可量化、可对比"
- "首次在 LattiAI 中完成 Sparse vs Dense 的端到端 benchmark"

---

## 五、评审时的诚实表述建议

如果被问到"你们实现了什么 bootstrapping 优化"，建议回答：

> "LattiAI 的底层 Lattigo 库已经实现了 Boura et al. 2022 的 Sparse Secret Encapsulation。我们的工作聚焦在三个层面：
> 1. **可配置化**：打破 Go/C++ SDK 和编译器对单一参数的锁定，支持 8 种预设切换；
> 2. **可量化**：首次在 LattiAI 中系统对比 Sparse vs Dense 的 bootstrapping 性能，测得 2.85× 加速；
> 3. **编译器适配**：在编译器的 cost model 中引入 Sparse 折扣，让放置策略更贴近实际性能。
> 
> 我们没有修改 bootstrapping 的密码学核心逻辑（CtS/StC/EvalMod），而是围绕'如何更好地使用已有的稀疏打包能力'做工程优化。"
