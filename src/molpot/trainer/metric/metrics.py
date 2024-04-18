import torch

from molpot.statistic.tracker import Tracker

__all__ = ["Accuracy", "TopKAccuracy", "MAE", "Identity", "StepSpeed"]

class Metric:
    
    def __init__(self, name):
        self.name = name

    def __call__(self, outputs):
        raise NotImplementedError
    
    @property
    def keys(self):
        return []
    
class Identity(Metric):

    def __init__(self, key):
        super().__init__('identity')
        self.key = key

    def __call__(self, outputs):
        return outputs[self.key]
    
    @property
    def keys(self):
        return [self.key]

class Accuracy(Metric):
    def __init__(self, result_key, target_key):
        super().__init__('accuracy')
        self.result_key = result_key
        self.target_key = target_key

    def __call__(self, outputs):

        result = outputs[self.result_key]
        target = outputs[self.target_key]
        with torch.no_grad():
            pred = torch.argmax(result, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()
        return correct / len(target)
    
    @property
    def keys(self):
        return [self.result_key, self.target_key]
    
class TopKAccuracy(Metric):
    def __init__(self, result_key, target_key, k=3):
        super().__init__('top_k_accuracy')
        self.result_key = result_key
        self.target_key = target_key
        self.k = k

    def __call__(self, outputs):

        result = outputs[self.result_key]
        target = outputs[self.target_key]
        with torch.no_grad():
            pred = torch.topk(result, self.k, dim=1)[1]
            assert pred.shape[0] == len(target)
            correct = 0
            for i in range(self.k):
                correct += torch.sum(pred[:, i] == target).item()
        return correct / len(target)
    
    @property
    def keys(self):
        return [self.result_key, self.target_key]
    
class MAE(Metric):
    def __init__(self, result_key, target_key, reduction="mean"):
        super().__init__('mae')
        self.result_key = result_key
        self.target_key = target_key
        self.kernel = torch.nn.L1Loss(reduction=reduction)

    def __call__(self, outputs):
            
        result = outputs[self.result_key]
        target = outputs[self.target_key]
        with torch.no_grad():
            mae = self.kernel(result, target)
            return mae
        
    @property
    def keys(self):
        return [self.result_key, self.target_key]
            
def track(metric):
    tracker = Tracker()
    def update(step, outputs):
        tracker(metric(step, outputs))
        return (tracker.mean, tracker.stddev)
    return update

class StepSpeed(Metric):

    def __init__(self):
        super().__init__("step_speed")

    def __call__(self, outputs):
        elaspse_time = outputs["this_report_time"] - outputs["last_report_time"]
        return outputs["elaspse_time"] / elaspse_time 
    
    @property
    def keys(self):
        return ["this_report_time", "last_report_time", "elaspse_time"]