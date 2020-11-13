import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel
import os

class ToxicCommentClassifier:

    LABELS_FILE_NAME = 'labels.txt'
    MAIN_DIRECTORY = 'ToxicComment'
    MODEL_DIRECTORY = 'roberta_model_distilled'
    
    def __init__(self):
        self.labels_values = self.load_labels_values()
        self.model = self.load_model()

    def load_model(self):
        model = MultiLabelClassificationModel('roberta', 
                                                os.path.join(self.MAIN_DIRECTORY, self.MODEL_DIRECTORY),
                                                use_cuda=False)
        return model
    
    def load_labels_values(self):
        labels_values = None
        with open(os.path.join(self.MAIN_DIRECTORY, self.LABELS_FILE_NAME), 'r') as f:
            labels_values = [x.strip() for x in f.readlines()]  # Get all label values as a list
        return labels_values

    def predict(self, input_sentences=[]):
        preds, outputs = self.model.predict(input_sentences)
        result = {}
        for i in range(len(input_sentences)):
            sentence_output = {}
            for j in range(len(self.labels_values)):
                sentence_output[self.labels_values[j]] = preds[i][j]
            result["sentence "+str(i+1)] = sentence_output
        return result