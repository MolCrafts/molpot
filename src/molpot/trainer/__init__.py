from .loss import nll_loss
from .metric import accuracy, top_k_acc
from .trainer import Trainer
from .optimizers import get_optimizer, get_lr_scheduler

def get_criterion(config):
    if config['type'] == 'nll_loss':
        return nll_loss
    else:
        raise NotImplementedError
    
def get_metric(config):
    if config['type'] == 'accuracy':
        return accuracy
    else:
        raise NotImplementedError