import json


class ScilakeDocument:
    """A class to store a Scilake Document (RDF like)"""
    def __init__(self, sections=[]):
        self.sections = sections

    def addSection(self, section):
        self.sections.append(section)

    def setSection(self, sections):
        self.sections = sections

    def getSections(self):
        return self.sections

    def decode_txt(self, data):
        lines = data.splitlines()
        title = lines[0]
        text = '\n'.join(lines[1:])
        sect = MySection(title, text)
        self.addSection(sect)

    def decode_json(self,data):
        json_d = json.loads(data)
        sections = json_d['sections']
        for section in sections:
            sect = MySection(section['title'],section['text'])
            self.addSection(sect)

    def decode_rdf(self,data):
        # TODO
        pass

    def _to_json(self):
        data = {}
        json_sections = []
        for sect in self.sections:
            json_sect = sect._to_json()
            json_sections.append(json_sect)
        data['sections'] = json_sections
        json_data = json.dumps(data)
        return json_data

    def _to_rdf(self):
        data = {}
        data['text'] = self.text
        data['labels'] = self.labels
        json_data = json.dumps(data)
        return json_data

    '''
    def simplified_to_xml(self):
        labels = ','.join(labels)
        data = '<span label="'+str(labels)+'">'+self.text+'</span>'
        return data
    '''

class MySection:
    """A class to store a scientific article section and its classified information"""

    def __init__(self, title, text):
        self.title = title
        self.text = text
        self.labels = []

    def addLabel(self, label):
        self.labels.append(label)

    def labels(self):
        return ','.join(self.labels)

    def _to_str_json(self):
        data = {}
        if self.title:
            data['title'] = self.title
        data['text'] = self.text
        data['labels'] = self.labels
        json_data = json.dumps(data)
        return json_data

    def _to_json(self):
        data = {}
        if self.title:
            data['title'] = self.title
        data['text'] = self.text
        data['labels'] = self.labels
        return data

    def _to_xml(self):
        labels = ','.join(labels)
        if self.title:
            data = '<sect><title>'+str(self.title)+'</title><span label="'+str(labels)+'">'+self.text+'</span></sect>'
        else:
            data = '<span label="'+str(labels)+'">'+self.text+'</span>'
        return data

    def _to_rdf(self):
        labels = ','.join(labels)
        data = '<span label="'+str(labels)+'">'+self.text+'</span>'
        return data
