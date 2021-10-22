"""
@Date: 2021/10/21 下午3:28
@Author: Chen Zhang
@Brief: 学习速率规划
"""


class LRScheduleManager:

    def __init__(self):
        self.schedules = {
            0: schedule_0
        }

    def print_schedules(self):
        """打印现有的schedule"""
        print('ID', '    ', 'Name')
        for key, val in self.schedules.items():
            print(key, '    ', val)

    def get_schedule(self, id: int):
        return self.schedules[id]


def schedule_0(epoch, lr):
    """
    常数分段式学习速率：
        1 <= epoch <= 10: lr = 0.01;
        11 <= epoch <= 50: lr = 0.001;
        51 <= epoch <= 100: lr = 0.0001
    """
    if epoch < 10:
        return 0.01
    elif epoch < 50:
        return 0.001
    else:
        return 0.0001


if __name__ == '__main__':
    s = LRScheduleManager()
    s.print_schedules()
