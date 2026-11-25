"""automation 核心 API（初始未实现）。"""


class Automation:
    def __init__(self, points):
        """构造"""
        raise NotImplementedError

    def value_at(self, t):
        """求值"""
        raise NotImplementedError

    def slope_at(self, t):
        """斜率"""
        raise NotImplementedError
