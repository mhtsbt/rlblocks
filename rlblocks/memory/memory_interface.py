from abc import ABCMeta, abstractmethod


class MemoryBase(metaclass=ABCMeta):

    @abstractmethod
    def sample(self, size):
        pass

    @abstractmethod
    def store(self, transition):
        pass
