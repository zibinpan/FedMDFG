
import fedplat as fp
class Recall(fp.Metric):
    def __init__(self):
        super().__init__(name='recall')
    def calc(self, network_output, target):
        true_positive = ((target * network_output) > .1).int().sum(axis=-1)
        return (true_positive / (target.sum(axis=-1) + 1e-13)).sum().item()
