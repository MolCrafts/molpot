from tensordict.tensordict import TensorDict, LazyStackedTensorDict

class Frames(LazyStackedTensorDict):
    pass

class Frame(TensorDict):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'predicts' not in self:
            self['predicts'] = {}
        if 'labels' not in self:
            self['labels'] = {}