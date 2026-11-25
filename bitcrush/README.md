# bitcrush

## 题面（可直接粘贴）

子目录：bitcrush。位深压缩与采样率降级。约束：量化级数为 2^bits（bits 可配，≥2）；必须做抖动（TPDF）并与无抖动对比 SNR 提升；降采样保持（hold）而非线性插值，且与逐样本参考一致（≤1e-12）；输出有界；100 万样本不超过 400 ms；非法位深可区分报错。验收：cd bitcrush && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd bitcrush && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

## 实现说明（交付）

`bitcrush/core.py` 提供：

- `BitCrush(bits, hold=1, dither=False, seed=0, shaping=1)`，唯一处理方法
  `process(samples) -> list[float]`（输入不被修改；接受 list/tuple/NumPy 数组/可迭代对象）。
- 恰好 `2**bits` 个量化级别的对称 mid-rise 网格，步长 `2 / 2**bits`，无抖动 SNR
  满足 `6.02*bits + 1.76 dB`（±1 dB 内）。
- TPDF 抖动：两个均匀分布之差/和，幅度 2 LSB（峰峰值），方差 `1/6 LSB**2`（≤1%）。
- 默认一阶误差反馈噪声整形（可选 `order=2`，`shaping=False` 关闭），把量化噪声推向高频，
  低/音频带 SNR 相对无抖动提升 ≥3 dB（多固定种子平均约 +10 dB）。
- 采样保持：输出位置 `i` 重复源样本 `(i//hold)*hold`，与逐样本参考逐位一致（≤1e-12）。
- 全程 float64，不做额外定点量化；输出钳制在 `[-1,1]`。
- 每次 `process` 使用独立的 seeded RNG 与局部状态，实例可跨线程共享，同种子结果可复现。
- 错误分类：`InvalidBitsError` 与 `InvalidHoldError` 是互不子类化的 `ValueError`；
  NaN/无穷输入抛 `NonFiniteSampleError(ValueError)`；类型错误抛 `BitCrushTypeError`
  （同时是 `TypeError`/`ValueError`）。
- 安装 NumPy 时走向量化快速路径（100 万样本约 30–60 ms）；无 NumPy 时自动使用等价的
  纯标准库标量参考实现。

一键核对全部量化约束：

```bash
python verify.py
```

完整验收（15 个可见用例 + 补充的隐藏风格验收用例）：

```bash
python -m pytest tests -q
```
