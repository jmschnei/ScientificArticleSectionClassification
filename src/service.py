from mydata import *
from models.transformer_based import *
import re
 
def classify_text(d,model):
    t = MySection(d)
    if model=='bert':
        t = classify_sections_bert(t)
    elif model=='scibert':
        t = classify_sections_scibert(t)
    elif model=='roberta':
        t = classify_sections_roberta(t)
    else:
        pass
    return t


if __name__ == '__main__':
    st = classify_text('I am trying to get a new line. And there I will be able to do something with the text.','bert')
    print(st)
