from .metric.metric import accuracy, top_k_acc
from .trainer import *
from .optimizers import get_optimizer, get_lr_scheduler
from .loss import *

def get_metric(config):
    if config['type'] == 'accuracy':
        return accuracy
    elif config['type'] == 'top_k_acc':
        return top_k_acc
    else:
        raise NotImplementedError