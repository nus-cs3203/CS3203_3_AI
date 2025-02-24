from abc import ABC, abstractmethod

class InsightGenerator(ABC):
    @abstractmethod
    def extract_insights(self, post):
        pass
