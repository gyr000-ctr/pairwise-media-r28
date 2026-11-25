# spectrogram

## 题面（可直接粘贴）

子目录：spectrogram。STFT 谱图与逆变换。约束：STFT 与朴素 DFT 一致（≤1e-9）；ISTFT 在 COLA 窗下完美重建（≤1e-6）；帧数按 hop 与窗长计算正确（含尾部补齐）；幅相分离可逆；100 万样本不超过 900 ms；非法 hop 可区分报错。验收：cd spectrogram && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd spectrogram && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
