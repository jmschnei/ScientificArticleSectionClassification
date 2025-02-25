from mydata import *
from models.bert import *
from models.ollama import *
from models.rules import *
import re
 
def classify_text(document,model):
    if model=='rules':
        document = classify_sections_rules(document)
    elif model=='bert':
        document = classify_sections_bert(document)
    elif model=='ollama':
        document = classify_sections_ollama(document)
    else:
        pass
    return document


if __name__ == '__main__':
    st = classify_text('I am trying to get a new line. And there I will be able to do something with the text.','bert')
    print(st)
