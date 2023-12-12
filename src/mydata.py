import json

class MySection:
    """A class to store a scientific article section and its classified information"""

    def __init__(self, text):
        self.text = text
        self.labels = []

    def labels(self):
        return ','.join(self.labels)

    def simplified_to_json(self):
        data = {}
        data['text'] = self.text
        data['labels'] = self.labels
        json_data = json.dumps(data)
        return json_data

    def simplified_to_xml(self):
        labels = ','.join(labels)
        data = '<span label="'+str(labels)+'">'+self.text+'</span>'
        return data
