
class Metric:
    def __init__(self, name='Metric', *args, **kwargs):
        self.name = name
    def calc(self, network_output, target):
        raise NotImplementedError
