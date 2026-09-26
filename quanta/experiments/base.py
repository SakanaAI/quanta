from abc import ABC, abstractmethod


class Experiment(ABC):
    @abstractmethod
    def run(self):
        pass
