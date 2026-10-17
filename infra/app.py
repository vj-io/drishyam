"""CDK app for the AWS copy of Drishyam. Run it through scripts/aws.sh, which builds build/lambda first.

Context: reserved_concurrency (default 5, or "none" when the account's Lambda quota leaves no room for a
reservation) and commit (the git SHA the package was built from, shown by /api/health).
"""

import os
from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_lambda as lambda_
from drishyam_stack import DrishyamStack

PACKAGE = Path(__file__).resolve().parent.parent / "build" / "lambda"
if not (PACKAGE / "run.sh").exists():
    raise SystemExit(f"{PACKAGE} is missing: run scripts/build_lambda.sh first")

app = cdk.App()
reserved = app.node.try_get_context("reserved_concurrency") or "5"
DrishyamStack(
    app, "Drishyam", code=lambda_.Code.from_asset(str(PACKAGE)),
    reserved_concurrency=None if reserved == "none" else int(reserved),
    commit=app.node.try_get_context("commit") or "",
    # CloudFront's WAF scope and the Lambda Web Adapter ARN below are us-east-1; the account comes from the profile.
    env=cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region="us-east-1"))
app.synth()
