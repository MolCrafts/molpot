import logging
import logging.config
from pathlib import Path
import torch

class LogAdapter:
    def __init__(self, metrics, handlers):
        self.metrics = metrics
        self.handlers = handlers

    def init(self):
        for handler in self.handlers:
            handler.init(self.metrics)

    def eval_metrics(self, nstep, output, data):
        return {header: metric(nstep, output, data) for header, metric in self.metrics.items()}

    def __call__(self, nstep, output, data):
        result = self.eval_metrics(nstep, output, data)
        for handler in self.handlers:
            handler(result)

class ConsoleHandler:
    def __init__(self):
        pass

    def init(self, metrics):
        print(f" | ".join([f"{key:<10s}" for key in metrics.keys()]))

    def __call__(self, results):
        formatted_result = []
        for result in results.values():
            if isinstance(result, float):
                formatted_result.append(f"{result:<10.4f}")
            elif isinstance(result, torch.Tensor):
                formatted_result.append(f"{result.item():<10.4f}")
            else:
                formatted_result.append(f"{str(result):<10s}")
        msg = f" | ".join(formatted_result)
        print(msg)