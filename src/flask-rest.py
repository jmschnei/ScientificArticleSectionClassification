#!/usr/bin/python3
from flask import Flask, request, jsonify

from flask_cors import CORS
import os
import shutil
#from werkzeug.utils import secure_filename
import zipfile

from flask_redoc import Redoc
import service
from mydata import *

import json

app = Flask(__name__)
app.secret_key = "super secret key"
CORS(app)

#API_URL = '/openapi.yaml'
#redoc = Redoc(
#    app,
#    'static/openapi.yml'
#)

models = [
    {"id": 1, "name": "rules", "description": "Several rules try to classify the sections, based primarily on titles content."},
    {"id": 2, "name": "bert", "description": "The approach uses BERT to classify sections based on PUBMED section types."},
    {"id": 3, "name": "ollama", "description": "The approach uses ollama to classify the sections content."},
]
 
@app.get("/sasc/models")
def get_models():
    return jsonify(models)

@app.get("/sasc/")
def sanity_check():
    return "The Scientific Article Section Classification (SASC) REST Controller is working properly.\n"

@app.route('/sasc/classify_text', methods=['POST'])
def analyze():
    if request.method == 'POST':
        cType = request.headers["Content-Type"]
        accept = request.headers["Accept"]

        model = request.args.get('model')
        model_names = [m['name'] for m in models]
        if not model in model_names:
            return 'ERROR: the model name "'+str(model)+'" is not supported. Check /sasc/models for suitable models'

        data=request.stream.read().decode("utf-8")

        if accept == 'text/plain':
            return 'ERROR: the Accept header '+accept+' is not implemented yet!'
        elif accept == 'application/json':
            pass
        elif accept == 'application/rdf':
            pass
        else:
            return 'ERROR: the Accept header '+accept+' is not supported!'
        document = None
        document = ScilakeDocument([])
        if cType == 'text/plain':
            document.decode_txt(data)
        elif cType == 'application/json':
            document.decode_json(data)
        elif cType == 'application/rdf':
            document.decode_rdf(data)
        else:
            return 'ERROR: the contentType header '+cType+' is not supported!'
        classified = service.classify_text(document,model)
        #if accept == 'text/plain':
        #    return classified._to_txt()
        #elif accept == 'application/json':
        if accept == 'application/json':
            return classified._to_json()
        elif accept == 'application/rdf':
            return classified._to_rdf()
        else:
            print('WARNING: the accept Header ('+accept+') is not supported. We are using "text/plain"')
            return classified.labels()
    else:
        return 'ERROR, only POST method is allowed.'

if __name__ == '__main__':
    port = int(os.environ.get('PORT',8093))
    app.run(host='localhost', port=port, debug=True)
