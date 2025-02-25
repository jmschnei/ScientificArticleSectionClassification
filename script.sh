#!/bin/bash

# Start the first process
ollama serve &

ollama pull llama3.1:8b

# Start the second process
FLASK_APP=flask-rest.py flask run --host=0.0.0.0 --port=8093 &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?