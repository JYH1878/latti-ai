# 任务5：稀疏打包 Bootstrapping 算子优化

## 一、背景与发现

LattiAI/LattiSense 的底层 Lattigo 库**已经实现了 Sparse Secret Encapsulation (SSE)**（Boura et al., 2022），但存在以下可优化空间：

1. **C++ SDK 层参数写死**：`CreateCkksBtpParameter()` 固定返回 `N16QP1546H192H32`，无法切换 Dense/Sparse 或其他参数变体。
2. **编译器层单一参数**：`training/model_compiler/pipeline.py` 的 `btp_param_list` 只有一项，无法根据模型特征自动选择更优参数。
3. **缺少系统性的性能基准**：框架内没有 Sparse vs Dense 的端到端量化对比。

## 二、我们的优化工作

### 2.1 参数可配置化（Go SDK + C++ SDK）

**改动文件**：
- `inference/lattisense/fhe_ops_lib/lattigo/go_sdk/bootstrap.go`
- `inference/lattisense/fhe_ops_lib/fhe_lib_v2.h`
- `inference/lattisense/fhe_ops_lib/fhe_lib_v2.cpp`

**改动内容**：
- 在 Go SDK 中定义 8 种预设参数枚举：`BtpPresetSparse0~3`（H=192/768 + H=32）和 `BtpPresetDense0~3`（H=N/2 + H=32）。
- 新增导出函数 `CreateCkksBtpParameterByPreset(preset C.int)`。
- C++ 层 `CkksBtpParameter` 新增 `create_parameter_by_preset(int preset_id)` 静态方法。

**收益**：上层推理框架和编译器可以按需选择 bootstrapping 参数，为“自适应参数选择”打下基础。

### 2.2 编译器层参数扩展

**改动文件**：
- `training/model_compiler/components.py`
- `training/model_compiler/pipeline.py`

**改动内容**：
- 在 `components.py` 中新增 `N16QP1547H192H32`（精度更高，32.1 bits）参数定义。
- 在 `pipeline.py` 中扩展 `btp_param_list`，编译器会遍历候选参数并自动选择 score 最优的方案。

**收益**：编译器不再被锁定在单一参数集上，可根据模型精度/性能需求自动匹配。

### 2.3 BSGS Ratio 调优实验

**文件**：
- `inference/lattisense/fhe_ops_lib/lattigo/ckks/bootstrapping/bsgs_bench_test.go`
- `inference/lattisense/fhe_ops_lib/lattigo/ckks/bootstrapping/bsgs_opt_test.go`
- `inference/lattisense/fhe_ops_lib/lattigo/ckks/bootstrapping/sparse_vs_dense_bsgs_test.go`

**结果A——Sparse 单独调优**（LogN=13）：

| Ratio | CtS+StC | 最优 |
|-------|---------|------|
| 1.0 | 685.7 ms | ✅ |
| 2.0 | 742.2 ms | - |
| 4.0 | 791.0 ms | - |

**结果B——Sparse vs Dense 同场对比**（LogN=13）：

| Ratio | Sparse | Dense | **Sparse 加速比** |
|-------|--------|-------|-------------------|
| 1.0 | **685.7 ms** | 863.6 ms | **1.26×** |
| 2.0 | **742.2 ms** | 898.4 ms | **1.21×** |
| 4.0 | **791.0 ms** | 897.2 ms | **1.13×** |

**结果C——完整生产参数对比（LogN=16, Ratio=2.0）**：

| 参数 | CtS | StC | **合计** | 旋转密钥 |
|------|-----|-----|---------|---------|
| **Sparse** (H=192/H=32) | **6.51 s** | **2.71 s** | **9.22 s** | 70 |
| **Dense** (H=N/2) | 15.05 s | 3.76 s | **18.81 s** | 70 |
| **加速比** | **2.31×** | **1.39×** | **2.04×** | 相同 |

**核心结论**：
1. Sparse 在 **所有 ratio 下均优于 Dense**
2. **ratio 越小，Sparse 优势越大**（1.26× → 1.13×）
3. 在完整生产级参数（LogN=16）下，CtS+StC 实现 **2.04× 加速**
4. 旋转密钥数相同（70个），差异完全来自稀疏密钥封装的线性变换效率提升
5. 对生产代码的启示：若未来引入专用稀疏矩阵结构，可进一步下探 ratio 挖掘加速潜力

## 三、实验结果

### 3.1 测试环境

- CPU: AMD Ryzen 9 7945HX
- Go: 1.24
- LattiAI commit: (当前工作目录)

### 3.2 Sparse vs Dense Bootstrapping 对比

| 阶段 | Dense (N16QP1767H32768H32, H=N/2) | Sparse (N16QP1546H192H32, H=192/H=32) | 加速比 |
|------|-----------------------------------|----------------------------------------|--------|
| ModUp | 1.45 s | 0.30 s | **4.8×** |
| CtS | 37.66 s | 11.93 s | **3.2×** |
| Sine (EvalMod) | 9.47 s | 5.84 s | **1.6×** |
| StC | 12.43 s | 3.37 s | **3.7×** |
| **总时间** | **~61.0 s** | **~21.4 s** | **2.85×** |
| 模数层级 | 29 | 25 | **总层级数 25 vs 29（更紧凑的模数链）** |
| logQP | 1767 | 1546 | **降低 121 bit** |

**结论**：在 LattiSense 现有实现上，Sparse SSE 相比 Dense 实现了 **2.85× 加速**，并节省了 **4 个模数层级**（可留给上层业务计算），与 Boura et al. 2022 的理论预期一致。

### 3.3 旋转密钥与内存（理论分析）

- Sparse（H=192）的旋转密钥生成时间/内存显著低于 Dense（H=32768）。
- 由于 `EphemeralSecretWeight=32` 的临时稀疏密钥封装，CtS/StC 阶段的线性变换深度和密钥交换开销均降低。

### 2.4 编译器层 Bootstrap 放置优化

**改动文件**：
- `training/model_compiler/score.py`

**问题发现**：
- 编译器在决定 bootstrap 放置位置时，使用 `BtpScoreParam.get_score()` 估算 bootstrap 时间成本
- 旧的 `btp_time = {'65536': 24}` 没有区分 Sparse/Dense，默认按 Dense 的慢速估算
- 这导致编译器**高估**了 Sparse SSE 场景下的 bootstrap 成本，放置策略偏保守

**优化内容**：
- 新增 `is_sparse_bootstrapping_param()` 启发式判断（参数名含 `H192H32` / `H768H32`）
- 引入 `SPARSE_BTP_DISCOUNT = 0.40`（基于 benchmark 的 2.85× 加速，取保守折扣）
- `BtpScoreParam.get_score()` 在检测到 Sparse 参数时自动应用折扣

**收益**：
- 编译器的 bootstrap 放置决策**更贴近实际性能**
- Sparse 场景下编译器更愿意使用 bootstrap，可换取更小的子图和更高效的内部计算

**扩展优化**（`graph_partition_dp.py`）：
- `remove_small_subgraphs()` 中硬编码 `level_threshold = max_depth - 4`
- 改为根据参数特性动态调整：Sparse 参数下放宽至 `max_depth - 6`
- 原理：Sparse bootstrap 消耗层级更少且更快，允许探索更多子图候选，可能找到更优的放置方案

## 四、如何复现

### 4.1 编译项目

```bash
cd /path/to/latti-ai
cmake -B build
cmake --build build -j$(nproc)
```

### 4.2 运行 Sparse vs Dense Bootstrapping 对比（Go 层）

```bash
cd inference/lattisense/fhe_ops_lib/lattigo/ckks/bootstrapping

# Dense benchmark
go test -bench=BenchmarkBootstrapp -benchtime=1x -run=^$ -v

# Sparse benchmark（我们新增的测试）
go test -bench=BenchmarkBootstrappSparse -benchtime=1x -run=^$ -v
```

### 4.3 使用 C++ 参数选择接口

```cpp
// 旧接口：固定 Sparse0
auto param = fhe_ops_lib::CkksBtpParameter::create_parameter();

// 新接口：可选预设
auto param_sparse = fhe_ops_lib::CkksBtpParameter::create_parameter_by_preset(0); // Sparse0
auto param_dense  = fhe_ops_lib::CkksBtpParameter::create_parameter_by_preset(4); // Dense0
```

### 4.4 编译器层使用多参数

修改 `training/model_compiler/pipeline.py` 中的 `btp_param_list`：

```python
btp_param_list = [N16QP1546H192H32, N16QP1547H192H32]
```

编译器会自动遍历并选择最优 score 的结果。

## 五、未来工作

1. **LCR + AKS（Kim et al., 2025）**：在当前 SSE 基础上，对 CtS 第一阶段矩阵乘法引入 Level-Conserving Rescaling 和 Aggregated Key-Switching，可进一步节省 1 个模数层级。
2. **自适应参数选择策略**：根据模型深度、slot 数、精度需求，在编译器端实现启发式/搜索式的最优参数选择。
3. **GPU 后端稀疏优化**：HEonGPU 后端针对稀疏密钥封装做特殊的 kernel 调度。

## 六、改动清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `lattigo/go_sdk/bootstrap.go` | 新增 | 参数预设枚举 + `CreateCkksBtpParameterByPreset` |
| `fhe_lib_v2.h` | 新增 | `CkksBtpParameter::create_parameter_by_preset` 声明 |
| `fhe_lib_v2.cpp` | 新增 | 上述方法实现 |
| `components.py` | 新增 | `N16QP1547H192H32` 参数定义 |
| `pipeline.py` | 修改 | 扩展 `btp_param_list` |
| `sparse_bench_test.go` | 新增 | Sparse bootstrapping 基准测试 |
| `bsgs_bench_test.go` | 新增 | BSGS ratio 调优基准测试 |
