import torch

__all__ = ["Accuracy", "TopKAccuracy"]

class Metric:
    
    def __init__(self, name):
        self.name = name

    def __call__(self, step, output, data):
        raise NotImplementedError

class Accuracy(Metric):
    def __init__(self, result_key, target_key):
        super().__init__("accuracy")
        self.result_key = result_key
        self.target_key = target_key

    def __call__(self, nstep, output, data):

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
        super().__init__("top_k_acc")
        self.result_key = result_key
        self.target_key = target_key
        self.k = k

    def __call__(self, step, output, data):

        result = output[self.result_key]
        target = data[self.target_key]
        with torch.no_grad():
            pred = torch.topk(result, self.k, dim=1)[1]
            assert pred.shape[0] == len(target)
            correct = 0
            for i in range(self.k):
                correct += torch.sum(pred[:, i] == target).item()
        return correct / len(target)