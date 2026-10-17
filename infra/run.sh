#!/bin/bash
# Lambda handler. The Lambda Web Adapter layer (AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap) runs this script,
# waits until GET /api/health answers on port 8080, then turns each invoke into an HTTP request to uvicorn.
# Same shape as awslabs/aws-lambda-web-adapter examples/fastapi-response-streaming-zip/app/run.sh.
PATH=$PATH:$LAMBDA_TASK_ROOT/bin \
    PYTHONPATH=$LAMBDA_TASK_ROOT:$PYTHONPATH:/opt/python:$LAMBDA_RUNTIME_DIR \
    exec python -m uvicorn --port=8080 app:app
