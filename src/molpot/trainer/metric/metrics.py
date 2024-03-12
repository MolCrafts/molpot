import torch

from molpot.statistic.tracker import Tracker

__all__ = ["Accuracy", "TopKAccuracy", "MAE", "Identity", "StepSpeed"]

class Metric:
    
    def __init__(self, name):
        self.name = name

    def __call__(self, output, data):
        raise NotImplementedError
    
class Identity(Metric):

    def __init__(self, key):
        super().__init__('identity')
        self.key = key

    def __call__(self, output, data):
        return output[self.key]

class Accuracy(Metric):
    def __init__(self, result_key, target_key):
        super().__init__('accuracy')
        self.result_key = result_key
        self.target_key = target_key

    def __call__(self, output, data):

        result = output[self.result_key]
        target = data[self.target_key]
        with torch.no_grad():
            pred = torch.argmax(result, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()
        return correct / len(target)
    
class TopKAccuracy(Metric):
    def __init__(self, result_key, target_key, k=3):
        super().__init__('top_k_accuracy')
        self.result_key = result_key
        self.target_key = target_key
        self.k = k

    def __call__(self, output, data):

        result = output[self.result_key]
        target = data[self.target_key]
        with torch.no_grad():
            pred = torch.topk(result, self.k, dim=1)[1]
            assert pred.shape[0] == len(target)
            correct = 0
            for i in range(self.k):
                correct += torch.sum(pred[:, i] == target).item()
        return correct / len(target)
    
class MAE(Metric):
    def __init__(self, result_key, target_key, reduction="mean"):
        super().__init__('mae')
        self.result_key = result_key
        self.target_key = target_key
        self.kernel = torch.nn.L1Loss(reduction=reduction)

    def __call__(self, output, data):
            
        result = output[self.result_key]
        target = data[self.target_key]
        with torch.no_grad():
            mae = self.kernel(result, target)
            return mae
            
def track(metric):
    tracker = Tracker()
    def update(step, output, data):
        tracker(metric(step, output, data))
        return (tracker.mean, tracker.stddev)
    return update

class StepSpeed(Metric):

    def __init__(self):
        super().__init__("step_speed")

    def __call__(self, outputs, inputs):
        elaspse_time = outputs["this_report_time"] - outputs["last_report_time"]
        return outputs["elaspse_time"] / elaspse_time 