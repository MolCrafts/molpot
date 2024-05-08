import logging
from copy import deepcopy

class Fix:
    """Base class for fix.

    Fix can be registered in :class:`cpu.trainer.Trainer`. Each fix can implement 6 methods
    (:meth:`before_train`, :meth:`after_train`, :meth:`before_epoch`, :meth:`after_epoch`,
    :meth:`before_iter`, :meth:`after_iter`). The way they are called is demonstrated
    in the following snippet:

    .. code-block:: python

        fix.before_train()
        for epoch in range(start_epoch, max_epochs):
            fix.before_epoch()
            for iter in range(epoch_len):
                fix.before_iter()
                train_one_iter()
                fix.after_iter()
            fix.after_epoch()
        fix.after_train()

    In the fix method, users can access ``self.trainer`` to access more
    properties about the context (e.g., model, optimizer, current epoch).

    Each fix has a priority, which is an integer from 1 to 10.
    The smaller the number, the higher the priority. Fix are executed
    in order of priority from high to low. If two fixes have the same priority,
    they are executed in the order they are registered.
    """

    # A weak reference to the trainer object. Set by the trainer when the fix is registered.
    trainer: "Trainer" = None
    priority: int = 5

    def __init__(self, every_n_steps:int, every_n_epochs:int=0) -> None:

        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.logger = logging.getLogger(__name__)
        self._name = self.__class__.__name__
        assert self.every_n_steps >= 0 and self.every_n_epochs >= 0, "interval must be positive."

    def do_before_train(self) -> None:
        """Called before the first epoch."""
        self.before_train()

    def do_after_train(self) -> None:
        """Called after the last epoch."""
        self.after_train()

    def do_before_epoch(self) -> None:
        """Called before each epoch."""
        if self.is_every_n_epochs() or self.is_last_epoch():
            self.before_epoch()

    def do_after_epoch(self) -> None:
        """Called after each epoch."""
        if self.is_every_n_epochs() or self.is_last_epoch():
            self.before_epoch()

    def do_before_iter(self) -> None:
        """Called before each iteration."""
        if self.is_every_n_steps() or self.is_last_iter():
            self.before_iter()

    def do_after_iter(self) -> None:
        """Called after each iteration."""
        if self.is_every_n_steps() or self.is_last_iter():
            self.after_iter()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_iter(self):
        pass

    def after_iter(self):
        pass

    @property
    def checkpointable(self) -> bool:
        """A fix is checkpointable when it implements :meth:`state_dict` method.
        Its state will be saved into checkpoint.
        """
        return callable(getattr(self, "state_dict", None))

    @property
    def name(self) -> str:
        """The class name of the fix."""
        return self._name
    
    @name.setter
    def name(self, name:str):
        self._name = name

    def log(self, *args, **kwargs) -> None:
        self.trainer.log(*args, **kwargs)

    # belows are some helper functions that are often used in fix
    def is_every_n_epochs(self) -> bool:
        return (self.trainer.elasped_epochs) % self.every_n_epochs == 0 if self.every_n_epochs > 0 else False

    def is_every_n_steps(self) -> bool:
        return (self.trainer.elasped_steps) % self.every_n_steps == 0 if self.every_n_steps > 0 else False

    def is_last_epoch(self) -> bool:
        return self.trainer.elasped_epochs == self.trainer.train_epochs

    def is_last_iter(self) -> bool:
        return self.trainer.elasped_steps == self.trainer.train_steps
    
    def copy(self):
        trainer = self.trainer
        self.trainer = None
        new_fix = deepcopy(self)
        self.trainer = trainer
        return new_fix
    

class FixManager(list):

    def add_fix(self, fix:Fix):
        assert isinstance(fix, Fix)
        assert fix.trainer is not None
        inserted = False
        for i in range(len(self) - 1, -1, -1):
            if fix.priority >= self[i].priority:
                self.insert(i + 1, fix)
                inserted = True
                break
        if not inserted:
            self.insert(0, fix)

    def append(self, fix:Fix):
        self.add_fix(fix)

    