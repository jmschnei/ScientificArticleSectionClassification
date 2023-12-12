from mydata import *
import re

def classify_text(d,model):
    t = MySection(d)
    if model=='bert':
        pass
    elif model=='scibert':
        pass
    elif model=='roberta':
        pass
    else:
        pass
    return t



if __name__ == '__main__':
    st = classify_text('I am trying to get a new line. And there I will be able to do something with the text.','bert')
    print(st)
