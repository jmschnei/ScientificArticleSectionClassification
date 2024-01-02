from mydata import *

def classify_sections_bert(sect):

    # We have to get the text of the section
    t = sect.text

    # TODO Classification using trained BERT model
    # labs = BertForTextClassification.classify(t)
    labs = None

    # We have to store the labels
    sect.labels = labs
    return sect


def classify_sections_scibert(sect):
    # TODO method to be implemented
    pass


def classify_sections_roberta(sect):
    # TODO method to be implemented
    pass

