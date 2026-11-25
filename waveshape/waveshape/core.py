"""waveshape 核心 API（初始未实现）。"""


class WaveShape:
    def __init__(self, drive, oversample=4):
        """构造"""
        raise NotImplementedError

    def process(self, samples):
        """处理"""
        raise NotImplementedError

    def shape(self, x):
        """整形函数"""
        raise NotImplementedError
