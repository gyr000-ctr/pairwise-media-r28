"""beat 核心 API（初始未实现）。"""


class Beat:
    def __init__(self, rate, bpm_range):
        """构造"""
        raise NotImplementedError

    def tempo(self, samples):
        """估计 BPM"""
        raise NotImplementedError

    def beats(self, samples):
        """节拍位置"""
        raise NotImplementedError
