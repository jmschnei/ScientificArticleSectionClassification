from mydata import *

def classify_sections_bert(document):
    sections = document.getSections()
    for sect in sections:
        # We have to get the text of the section
        t = sect.text
        # TODO Classification using trained BERT model
        # labs = BertForTextClassification.classify(t)
        labs = None
        # We have to store the labels
        sect.labels = labs

    document.setSections(sections)
    return document
