import abc
class BaseStrategy(abc.ABC):
    """Defines interface for custom training strategies."""
    def __init__(self, trainer):
        self.trainer = trainer  # has .model, .criterion, data loaders, etc.

    @abc.abstractmethod
    def train_one_epoch(self, epoch:int): ...

    @abc.abstractmethod
    def validate(self, epoch:int): ...
