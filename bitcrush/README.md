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
