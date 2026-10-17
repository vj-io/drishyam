#!/usr/bin/env bash
# Drishyam on AWS (docs/aws.md): one command per operation. Account 242441136319 through the
# CLI profile drishyam (override with AWS_PROFILE), region us-east-1 (CloudFront's WAF scope lives there).
#
#   scripts/aws.sh deploy    build the Lambda package, bootstrap CDK if needed, deploy, print the URLs
#   scripts/aws.sh smoke     scripts/smoke.sh against the deployed URL, plus the raw Function URL check;
#                            appends the results to runs/aws-smoke.log (SMOKE_LOG overrides)
#   scripts/aws.sh stop      reserved concurrency 0: every request is refused, nothing runs or bills
#   scripts/aws.sh start     undo stop (reserved concurrency back to 5, or no reservation with RESERVED=none)
#   scripts/aws.sh destroy   delete the stack: function, distribution, WAF, table, secret, logs, alarms, budget
#
# RESERVED=none deploys without a reserved concurrency, for an account whose Lambda quota is too small for one.
# ONNX=models/drishyam_dacl10k.onnx (optionally ONNX_FP16=models/drishyam_dacl10k.fp16.onnx) deploys with that
# segmenter instead of the classical engine; the file must be in models/, which build_lambda.sh packages.
set -euo pipefail
cd "$(dirname "$0")/.."
export AWS_PROFILE=${AWS_PROFILE:-drishyam} AWS_REGION=us-east-1 CDK_DISABLE_CLI_TELEMETRY=true
ACCOUNT=242441136319 STACK=Drishyam FN=drishyam-web RESERVED=${RESERVED:-5}
CDK="npx --yes aws-cdk@2.1145.0"

out() {
  aws cloudformation describe-stacks --stack-name "$STACK" --output text \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue"
}

case ${1:-} in
  deploy)
    # The stack only works with the app-side changes in docs/aws.md: without them KILL never reaches the
    # job, each environment signs with its own key and every browser POST gets 403.
    grep -q DRISHYAM_KILL_TABLE drishyam/web.py && grep -q x-amz-content-sha256 public/js/api.js || {
      echo "app side not merged: drishyam/web.py must read DRISHYAM_KILL_TABLE and public/js/api.js must send x-amz-content-sha256 (docs/aws.md)" >&2
      exit 1
    }
    acct=$(aws sts get-caller-identity --query Account --output text)
    [ "$acct" = "$ACCOUNT" ] || { echo "profile $AWS_PROFILE is account $acct, not $ACCOUNT" >&2; exit 1; }
    limit=$(aws lambda get-account-settings --query AccountLimit.ConcurrentExecutions --output text)
    if [ "$RESERVED" != none ] && [ "$limit" -lt $((RESERVED + 100)) ]; then
      cat >&2 <<EOF
This account's Lambda concurrency quota is $limit. Reserving $RESERVED needs at least $((RESERVED + 100)),
because Lambda keeps 100 unreserved. Either ask for the default quota of 1000 and deploy once it is granted:
  aws service-quotas request-service-quota-increase --service-code lambda --quota-code L-B99A9384 --desired-value 1000 --profile $AWS_PROFILE --region us-east-1
or deploy with no reservation, so the account quota of $limit is the cap instead:
  RESERVED=none scripts/aws.sh deploy
EOF
      exit 1
    fi
    scripts/build_lambda.sh
    if [ ! -x .venv-infra/bin/python ]; then
      uv venv --quiet --python 3.12 .venv-infra
      uv pip install --quiet --python .venv-infra/bin/python -r infra/requirements.txt
    fi
    commit=$(git rev-parse HEAD)$(git diff --quiet HEAD || echo -dirty)
    (cd infra && $CDK bootstrap "aws://$ACCOUNT/$AWS_REGION" &&
      $CDK deploy "$STACK" -c reserved_concurrency="$RESERVED" -c commit="$commit" \
        ${ONNX:+-c "onnx=$ONNX"} ${ONNX_FP16:+-c "onnx_fp16=$ONNX_FP16"})  # asks before IAM changes
    echo "site:   $(out Url)"
    echo "alerts: aws sns subscribe --topic-arn $(out AlertTopicArn) --protocol email --notification-endpoint YOUR_EMAIL --profile $AWS_PROFILE --region us-east-1"
    ;;
  smoke)
    mkdir -p runs
    FUNCTION_URL=$(out FunctionUrl) SMOKE_LOG=${SMOKE_LOG:-runs/aws-smoke.log} \
      CODE_SHA256=$(aws lambda get-function-configuration --function-name "$FN" --query CodeSha256 --output text) \
      scripts/smoke.sh "$(out Url)"
    ;;
  stop) aws lambda put-function-concurrency --function-name "$FN" --reserved-concurrent-executions 0 ;;
  start)
    if [ "$RESERVED" = none ]; then aws lambda delete-function-concurrency --function-name "$FN"
    else aws lambda put-function-concurrency --function-name "$FN" --reserved-concurrent-executions "$RESERVED"; fi
    ;;
  destroy)
    aws cloudformation delete-stack --stack-name "$STACK"
    aws cloudformation wait stack-delete-complete --stack-name "$STACK"
    echo "deleted $STACK (the CDKToolkit bootstrap stack stays; delete it in CloudFormation if nothing else uses it)"
    ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac
