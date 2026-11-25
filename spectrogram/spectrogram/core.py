"""spectrogram 核心 API（初始未实现）。"""


class Spectrogram:
    def __init__(self, frame, hop):
        """构造"""
        raise NotImplementedError

    def stft(self, samples):
        """正变换"""
        raise NotImplementedError

    def istft(self, spec):
        """逆变换"""
        raise NotImplementedError
