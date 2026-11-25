# waveshape

## 题面（可直接粘贴）

子目录：waveshape。波形整形与过采样。约束：整形函数必须奇对称且单调；4 倍过采样后混叠下降 ≥12 dB；驱动参数单调提升谐波；反奇对称（f(-x)=-f(x)，≤1e-12）；输出有界（≤1.0）；100 万样本不超过 600 ms；非法驱动可区分报错。验收：cd waveshape && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd waveshape && python -m pytest tests -q
```

初始 15 failed（`test_core.py`），现全部通过；另有 `tests/test_spec.py` 覆盖完整约束与边界。

## 实现概要

`waveshape/core.py` 实现 `WaveShape(drive, oversample=4)`：

- 整形函数 `f(x) = tanh(drive·x)`：逐位奇对称、严格单调、值域 `[-1, 1]`。
- 过采样链：零插值 → Kaiser 窗 sinc 低通（多相实现，增益折入相位）→ 高采样率
  `tanh` → 同一低通 → 抽取。4 倍时混叠相对 1 倍路径显著下降。
- 性能：100 万样本约 150–200 ms（预算 600 ms）。
- 错误分类（均为 `WaveShapeError`，同时是 `ValueError`/`TypeError`）：
  `InvalidDriveError`、`InvalidOversampleError`、`InvalidSampleError`、
  `NonFiniteInputError`（细分 `NaNInputError` / `InfInputError`）。
- 并发安全：实例只读，`process` 无共享可变状态。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
