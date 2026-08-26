# HeatSolver `main` 分支代码说明

本文对应 `main` 分支提交 `7dbc818`。当前代码实现的是基于 PyTorch 的三维稳态热传导反演，支持直接优化热参数和使用多源特征网络生成热参数。两种模式都由同一个 PDE 求解器计算温度场，网络不直接预测三维温度。

## 1. 整体流程

```text
Data/*.csv 和 thermal_properties.json
    |
    +-- data_loading.py：读取地表温度、热流和多源地球物理数据
    +-- lithosphere_model.py：建立规则网格、层位掩膜和参考 K0/c/A
    |
    v
参数来源
    +-- direct_parameter.py：直接优化参数
    +-- prior_network.py：多源特征 -> 2-D CNN -> 参数
    |
    v
parameterization.py
    低维系数 -> T_bot、K0、A 三维物理场
    |
    v
heat_solver.py
    稳态热传导 PDE -> 三维温度 T -> 地表热流 q_pred
    |
    v
inversion.py
    热流损失 + 统计矩损失 + 参数正则化 -> 反向传播与优化
    |
    +-- evaluation.py：重建并导出温度、热导率和热流场
    +-- inversion_uq.py：数据划分、随机种子和配置组成的经验集合
    +-- shap.py：prior_net 输入特征的温度归因分析
```

## 2. 主要入口

安装依赖并运行测试：

```powershell
uv sync
uv run python -B -m pytest -q -p no:cacheprovider
```

直接反演，默认优化 `3 x 3` DCT 底部温度，固定热导率和生热率：

```powershell
uv run python inversion.py --mode direct
```

使用多源特征网络同时生成底温、热导率和生热率参数：

```powershell
uv run python inversion.py --mode prior_net --optimizer adamw --lr 0.005 --steps 100 --k-mode dct --a-mode dct --lam-q-std 0.4 --alpha-scale 1200 --k-prop-scale 0.3 --a-prop-scale 0.5 --t-ref 1250
```

评估已有结果：

```powershell
uv run python evaluation.py outputs/<run_dir>
```

运行经验不确定性集合：

```powershell
uv run python inversion_uq.py --mode prior_net
```

对 `prior_net` checkpoint 做特征归因：

```powershell
uv run python shap.py outputs/<run_dir>
```

## 3. 数据与参考模型

`data_loading.py` 将散点或规则数据统一映射到计算网格。反演必需的主要文件是：

- `grid_1km.csv`：基础 ID、X/Y 和经纬度坐标。
- `grid_1km_layer.csv`：沉积层、地壳和 Moho 几何信息。
- `grid_1km_stratum_filled.csv`：补齐后的沉积地层界面。
- `grid_1km_thermal.csv`：地表温度。
- `thermal_properties.json`：各层参考热导率、生热率和温度系数。
- `sample_heatflow_processed.csv`：用于目标函数的地表热流观测。

`prior_net` 还会读取 DEM、重力、磁异常、CSRM 地震速度和 MT 电阻率。地震与 MT 数据先按深度区间平均，再用最近点映射成二维通道；所有通道按通道均值和标准差归一化。

`lithosphere_model.py` 根据地层和 Moho 几何建立五个热学单元：

```text
sediment / upper_crust / middle_crust / lower_crust / mantle
```

每个单元提供参考 `K0`、`c`、`A` 和平滑层位掩膜，供参数化和 PDE 求解器共同使用。

## 4. 参数化

`parameterization.py` 对三类反演量使用统一的 `fixed`、`dct` 和 `full` 模式：

- `T_bot(x,y)`：二维底部温度场，以 `T_ref` 为中心叠加扰动。
- `K0(x,y,z)`：五个热学单元的二维对数乘子，通过层位掩膜扩展到三维。
- `A(x,y,z)`：与 `K0` 相同的分层对数乘子结构。

`fixed` 使用参考场，`dct` 只优化低频 DCT 系数，`full` 为网格上的自由二维场。缩放后的参数经过 `tanh` 限幅，再解码为实际物理场。

两种参数来源的区别只有系数如何产生：

- `DirectParameterModel` 直接把底温、`K0` 和 `A` 系数注册为可训练参数。
- `PriorParameterModel` 将所有二维特征早期融合到一个 CNN，输出一个确定性的系数向量。

当前 `prior_net` 不是概率先验，也没有独立的二维/三维特征分支或协方差输出。

## 5. 热传导求解器

`heat_solver.py` 求解

$$
-\nabla\cdot\left(k(x,T)\nabla T\right)=A(x),
\qquad
k(x,T)=\frac{K_0(x)}{\max(1+c(x)T,0.1)}.
$$

边界条件为顶部和底部 Dirichlet 温度、四周零法向热流。离散时使用界面热导率的调和平均；温度相关热导率通过 Picard 迭代更新，线性系统使用 CG，并由 PyAMG 提供预条件。

求解器封装为自定义 `torch.autograd.Function`。前向阶段调用 NumPy/SciPy 稀疏求解，反向阶段通过隐式伴随方程计算目标函数对 `T_bot`、`K0` 和 `A` 的梯度，因此不需要保存整个迭代过程的自动微分图。

## 6. 反演目标与优化

`inversion.py` 的主循环依次执行：

1. 模型生成参数系数。
2. 参数化模块解码 `T_bot`、`K0` 和 `A`。
3. PDE 求解器计算三维温度和地表热流。
4. 在有观测的网格点计算热流残差。
5. 反向传播并用 L-BFGS、Adam 或 AdamW 更新参数。

目标函数由四部分组成：

$$
L=L_q+L_{\mathrm{moment}}+L_{\mathrm{coeff}},
$$

其中 `L_q` 是地表热流均方误差，`L_moment` 可约束预测热流的均值和标准差，`L_coeff` 分别惩罚底温、`K0` 和 `A` 系数。代码还支持提前停止和分组梯度诊断。

每次运行默认写入 `outputs/<timestamp>/`：

- `checkpoint.pt`：模型状态和重建配置。
- `summary.json`：运行参数、损失权重和模型配置。
- `train_log.json`：逐步损失、RMSE、参数范围和求解器统计。
- `loss_curve.png`：目标函数曲线。
- `panels.png`：观测热流、预测热流、残差和底温。

## 7. 评估、UQ 与归因

`evaluation.py` 从 checkpoint 重建网格、参数模型和 PDE 状态，导出温度、热导率、三向热流、地表字段、深度切片和统计摘要。它可以与 `grid_1km_temperature_0_200km.csv` 比较，但该文件是外部参考模型，不是独立地下真值。

`inversion_uq.py` 通过观测训练/留出划分、随机种子、初值和参数配置组成经验集合，汇总均值、样本标准差、分位数和校准曲线。它描述的是实验配置下的集合离散程度，不是贝叶斯后验采样。

`shap.py` 只适用于 `prior_net`。它对不同深度的温度异常进行 Gradient-SHAP 风格的通道归因，并输出通道/数据源重要性、beeswarm、依赖图和空间分布图。

## 8. 文件职责

| 文件 | 职责 |
|---|---|
| `inversion.py` | 命令行入口、目标函数、优化和 checkpoint 输出 |
| `heat_solver.py` | 三维稳态热传导正演和自定义伴随反向传播 |
| `parameterization.py` | `fixed/dct/full` 参数模式及物理场解码 |
| `direct_parameter.py` | 直接可训练参数模型 |
| `prior_network.py` | 多源二维特征 CNN 和系数输出 |
| `data_loading.py` | 网格、热流、地表温度和多源特征读取 |
| `lithosphere_model.py` | 层位几何、参考热物性和层位掩膜 |
| `evaluation.py` | checkpoint 重建、字段导出和参考温度比较 |
| `inversion_uq.py` | 经验集合运行与聚合 |
| `shap.py` | `prior_net` 输入特征归因 |

## 9. 当前边界

- 温度必须由 PDE 求解器计算，网络只约束边界和热物性参数。
- `prior_net` 对一个区域输出一个确定性参数向量，不是跨样本预训练的概率模型。
- UQ 是数据划分、随机种子和配置扰动形成的经验集合。
- 地表热流拟合良好不等于地下 `T_bot/K0/A` 唯一可恢复。
- 参考温度 CSV 只能用于模型间比较，不能作为独立真值。
- 当前结果主要用于方法验证，不能直接描述为已验证的 1 km 盆地级地下温度产品。
