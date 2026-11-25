# beat

## 题面（可直接粘贴）

子目录：beat。节拍跟踪。约束：先用谱通量做 onset，再用自相关估计 tempo（BPM 与真值差 ≤2%）；节拍相位对齐到强 onset（偏差 ≤20 ms）；倍频歧义必须选定基频；100 万样本不超过 900 ms；非法 BPM 范围可区分报错。验收：cd beat && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd beat && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
