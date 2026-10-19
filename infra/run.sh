#!/bin/bash
# Lambda handler. The Lambda Web Adapter layer (AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap) runs this script,
# waits until GET /api/health answers on port 8080, then turns each invoke into an HTTP request to uvicorn.
# Same shape as awslabs/aws-lambda-web-adapter examples/fastapi-response-streaming-zip/app/run.sh.
# --no-proxy-headers: uvicorn would otherwise trust the adapter's (127.0.0.1) X-Forwarded-For and log the address
# it names as the client, so the access log would keep viewer IPs (docs/aws.md, "Limits").
PATH=$PATH:$LAMBDA_TASK_ROOT/bin \
    PYTHONPATH=$LAMBDA_TASK_ROOT:$PYTHONPATH:/opt/python:$LAMBDA_RUNTIME_DIR \
    exec python -m uvicorn --port=8080 --no-proxy-headers drishyam.web:app
