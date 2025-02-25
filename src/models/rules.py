from mydata import *


rules = [
    {"id": 1, "label": "introduction", "texts": ["introduction"]},
    {"id": 2, "label": "related", "texts": ["related work"]},
    {"id": 3, "label": "approach", "texts": ["approach"]},
    {"id": 4, "label": "evaluation", "texts": ["evaluation"]},
    {"id": 5, "label": "experiments", "texts": ["experiments"]},
    {"id": 6, "label": "discussion", "texts": ["discussion"]},
    {"id": 7, "label": "conclusions", "texts": ["conclusions"]},
]

def classify_sections_rules(document):
    sections = document.getSections()

    for sect in sections:
        # We have to get the text of the section
        title = sect.title
        text = sect.text

        labs = []
        # Classification using Rules
        for rule in rules:
            if title in rule['texts']:
                labs.append(rule['label'])

        # We have to store the labels
        sect.labels = labs
    #document.setSections(sections)
    return document