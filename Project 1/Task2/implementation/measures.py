from abc import ABC, abstractmethod
from enum import Enum, auto

from sklearn.metrics import recall_score, precision_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score

# interface of Measure
class IMeasure(ABC):
    @staticmethod
    @abstractmethod
    def get(true, pred):
        return

    @staticmethod
    @abstractmethod
    def get_name():
        return

    @staticmethod
    @abstractmethod
    def get_type():
        return

class Measure:
    # finite number of specified measures
    class Type(Enum):
        recall = auto()
        precision = auto()
        F1_score = auto()
        balanced_accuracy = auto()
        AUC_ROC = auto()
        AUC_PR = auto()

    # get Measure object based on passed Measure.Type
    @staticmethod
    def from_type(p_type: "Measure.Type"):
        assert isinstance(p_type, Measure.Type)
        
        match p_type:
            case Measure.Type.recall:
                return Measure.Recall()
            case Measure.Type.precision:
                return Measure.Precision()
            case Measure.Type.F1_score:
                return Measure.F1Score()
            case Measure.Type.balanced_accuracy:
                return Measure.BalancedAccuracy()
            case Measure.Type.AUC_ROC:
                return Measure.AUC_ROC()
            case Measure.Type.AUC_PR:
                return Measure.AUC_PR()
            case _:
                raise Exception(f"Unhandled type '{p_type}' passed")

    class Recall(IMeasure):
        @staticmethod
        def get(true, pred):
            return recall_score(true, pred)

        @staticmethod
        def get_name():
            return "recall"
    
        @staticmethod
        def get_type():
            return Measure.Type.recall
    
    class Precision(IMeasure):
        @staticmethod
        def get(true, pred):
            return precision_score(true, pred)

        @staticmethod
        def get_name():
            return "precision"
    
        @staticmethod
        def get_type():
            return Measure.Type.precision
    
    class F1Score(IMeasure):
        @staticmethod
        def get(true, pred):
            return f1_score(true, pred)

        @staticmethod
        def get_name():
            return "F1-score"
    
        @staticmethod
        def get_type():
            return Measure.Type.F1_score
    
    class BalancedAccuracy(IMeasure):
        @staticmethod
        def get(true, pred):
            return balanced_accuracy_score(true, pred)

        @staticmethod
        def get_name():
            return "balanced accuracy"
    
        @staticmethod
        def get_type():
            return Measure.Type.balanced_accuracy
    
    class AUC_ROC(IMeasure):
        @staticmethod
        def get(true, scores):
            return roc_auc_score(true, scores)

        @staticmethod
        def get_name():
            return "auc roc"
    
        @staticmethod
        def get_type():
            return Measure.Type.AUC_ROC
    
    class AUC_PR(IMeasure):
        @staticmethod
        def get(true, scores):
            return average_precision_score(true, scores)

        @staticmethod
        def get_name():
            return "auc pr"
    
        @staticmethod
        def get_type():
            return Measure.Type.AUC_PR