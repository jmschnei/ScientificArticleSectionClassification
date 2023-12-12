#!/usr/bin/python3
from flask import Flask, request, jsonify

from flask_cors import CORS
import os
import shutil
#from werkzeug.utils import secure_filename
import zipfile

from flask_redoc import Redoc
import service

import json

app = Flask(__name__)
app.secret_key = "super secret key"
CORS(app)

API_URL = '/openapi.yaml'
redoc = Redoc(
    app,
    'static/openapi.yml'
)

models = [
    {"id": 1, "name": "bert", "description": ""},
    {"id": 2, "name": "scibert", "description": ""},
]

@app.get("/models")
def get_countries():
    return jsonify(models)

@app.get("/")
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
            return 'ERROR: the model name '+model+' is not supported. Check /models endpoint for suitable models'

        data=request.stream.read().decode("utf-8")

        if accept == 'text/plain':
            pass
        elif accept == 'application/json':
            pass
        else:
            return 'ERROR: the Accept header '+accept+' is not supported!'
        if cType == 'text/plain':
            d = data
        elif cType == 'application/json':
            d = json.loads(data)
        else:
            return 'ERROR: the contentType header '+cType+' is not supported!'

        classified = service.classify_text(d,model)
        if accept == 'text/plain':
            return classified.labels()
        elif accept == 'application/json':
            return classified.toJSON()
        else:
            print('WARNING: the accept Header ('+accept+') is not supported. We are using "text/plain"')
            return classified.labels()
    else:
        return 'ERROR, only POST method is allowed.'

if __name__ == '__main__':
    port = int(os.environ.get('PORT',8080))
    app.run(host='localhost', port=port, debug=True)
