import logging
import logging.config
from pathlib import Path
import torch

class LogAdapter:
    def __init__(self, name, metrics, handlers, save_dir):
        self.name = name
        self.metrics = metrics
        self.handlers = handlers
        self.save_dir = save_dir

    def init(self):
        for handler in self.handlers:
            handler.init(self.name, self.metrics, self.save_dir)

    def eval_metrics(self, output, data):
        return {header: metric(output, data) for header, metric in self.metrics.items()}

    def __call__(self, nstep, nepoch, output, data):
        result = self.eval_metrics(output, data)
        for handler in self.handlers:
            handler(nstep, nepoch, result)

class ConsoleHandler:
    def __init__(self):
        pass

    def init(self, name, metrics, save_dir):
        print("step | epoch | ", f" | ".join([f"{key:<10s}" for key in metrics.keys()]))

    def __call__(self, nstep, nepoch, results):
        formatted_result = [str(nstep), str(nepoch)]
        for result in results.values():
            if isinstance(result, float):
                formatted_result.append(f"{result:<10.4f}")
            elif isinstance(result, torch.Tensor):
                formatted_result.append(f"{result.item():<10.4f}")
            else:
                formatted_result.append(f"{str(result):<10s}")
        msg = f" | ".join(formatted_result)
        print(msg)

class TensorBoardHandler:
    def __init__(self):
        pass
        
    def init(self, name, metrics, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(save_dir, comment=name)

    def __call__(self, nstep, nepoch, results):
        for key, value in results.items():
            self.writer.add_scalar(key, value, nstep)

    def __del__(self):
        if hasattr(self, "writer"):
            self.writer.close()