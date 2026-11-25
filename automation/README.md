# automation

## 题面（可直接粘贴）

子目录：automation。自动化曲线求值。约束：分段三次（Hermite）插值，端点斜率按邻点差分；曲线连续且一阶连续（≤1e-6）；节点时间必须严格递增；越界按端点值钳制；求值单调性由节点值保序决定；10 万次求值不超过 150 ms；并发安全。验收：cd automation && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd automation && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
