"""Synthesises the AWS stack (infra/) and checks what an auditor would ask about: who may invoke the
function, what the function may do, how requests are throttled and what stops the spend.

The project environment has no CDK, so this file skips there. Run it with (python -m, so pytest runs
under the interpreter that sees the CDK overlay; CI runs the same command):
    uv run --with-requirements infra/requirements.txt python -m pytest -q tests/test_infra.py
"""

import json
import sys
from pathlib import Path

import pytest

cdk = pytest.importorskip("aws_cdk")
from aws_cdk import assertions
from aws_cdk import aws_lambda as lambda_

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))
from drishyam_stack import DrishyamStack

ACCOUNT = "123456789012"
ROOT = Path(__file__).resolve().parents[1]


def synth(tmp_path, **kw) -> dict:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "run.sh").write_text("#!/bin/bash\n")
    app = cdk.App(outdir=str(tmp_path / "cdk.out"))
    stack = DrishyamStack(app, "Test", code=lambda_.Code.from_asset(str(tmp_path / "code")),
                          env=cdk.Environment(account=ACCOUNT, region="us-east-1"), **kw)
    return assertions.Template.from_stack(stack).to_json()


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> dict:
    return synth(tmp_path_factory.mktemp("infra"), commit="abc123")


def of_type(t: dict, kind: str) -> list[dict]:
    return [r["Properties"] for r in t["Resources"].values() if r["Type"] == kind]


def one(t: dict, kind: str) -> dict:
    (props,) = of_type(t, kind)
    return props


def web_fn(t: dict) -> dict:  # the app; the stack's other functions are CDK's deploy-time custom resources
    (props,) = [p for p in of_type(t, "AWS::Lambda::Function") if p.get("FunctionName") == "drishyam-web"]
    return props


def oac(t: dict, origin_type: str) -> dict:
    (cfg,) = [p["OriginAccessControlConfig"] for p in of_type(t, "AWS::CloudFront::OriginAccessControl")
              if p["OriginAccessControlConfig"]["OriginAccessControlOriginType"] == origin_type]
    return cfg


def test_function_is_arm64_streaming_behind_iam(template):
    fn = web_fn(template)
    assert fn["Architectures"] == ["arm64"] and fn["Runtime"] == "python3.12" and fn["Handler"] == "run.sh"
    assert fn["Layers"] == ["arn:aws:lambda:us-east-1:753240598075:layer:LambdaAdapterLayerArm64:30"]
    assert fn["ReservedConcurrentExecutions"] == 5 and fn["Timeout"] == 60
    assert fn["TracingConfig"] == {"Mode": "Active"}
    env = fn["Environment"]["Variables"]
    assert env["AWS_LAMBDA_EXEC_WRAPPER"] == "/opt/bootstrap" and env["AWS_LWA_INVOKE_MODE"] == "response_stream"
    assert env["DRISHYAM_COMMIT"] == "abc123" and "DRISHYAM_SERVE_STATIC" not in env  # static files come from S3
    assert set(env) >= {"DRISHYAM_KILL_TABLE", "DRISHYAM_SIGNING_KEY_SECRET_ARN"}
    assert "DRISHYAM_ONNX" not in env and "DRISHYAM_ONNX_FP16" not in env  # classical-v1 unless asked
    url = one(template, "AWS::Lambda::Url")
    assert url["AuthType"] == "AWS_IAM" and url["InvokeMode"] == "RESPONSE_STREAM"


def test_only_this_distribution_may_invoke_and_both_dual_auth_actions_are_granted(template):
    perms = of_type(template, "AWS::Lambda::Permission")
    assert sorted(p["Action"] for p in perms) == ["lambda:InvokeFunction", "lambda:InvokeFunctionUrl"]
    dist_id = next(k for k, r in template["Resources"].items() if r["Type"] == "AWS::CloudFront::Distribution")
    for p in perms:
        assert p["Principal"] == "cloudfront.amazonaws.com"
        assert {"Ref": dist_id} in p["SourceArn"]["Fn::Join"][1]  # this distribution, not any distribution
    assert (oac(template, "lambda")["SigningBehavior"], oac(template, "lambda")["SigningProtocol"]) == ("always", "sigv4")
    assert all("OriginAccessControlId" in o
               for o in one(template, "AWS::CloudFront::Distribution")["DistributionConfig"]["Origins"])


def test_role_is_least_privilege(template):
    role_id = web_fn(template)["Role"]["Fn::GetAtt"][0]
    role = template["Resources"][role_id]["Properties"]
    assert "ManagedPolicyArns" not in role and "Policies" not in role
    statements = [s for p in of_type(template, "AWS::IAM::Policy") if p["Roles"] == [{"Ref": role_id}]
                  for s in p["PolicyDocument"]["Statement"]]
    grants = {}
    for s in statements:
        assert s["Effect"] == "Allow"
        for action in [s["Action"]] if isinstance(s["Action"], str) else s["Action"]:
            grants[action] = s["Resource"]
    assert set(grants) == {"logs:CreateLogStream", "logs:PutLogEvents", "dynamodb:GetItem", "dynamodb:PutItem",
                           "secretsmanager:GetSecretValue", "xray:PutTraceSegments", "xray:PutTelemetryRecords"}
    assert [a for a, r in grants.items() if r == "*"] == ["xray:PutTraceSegments", "xray:PutTelemetryRecords"]
    assert grants["logs:PutLogEvents"]["Fn::GetAtt"][1] == "Arn"  # the function's own log group only


def test_kill_table_and_secret(template):
    table = one(template, "AWS::DynamoDB::Table")
    assert table["BillingMode"] == "PAY_PER_REQUEST"
    assert table["KeySchema"] == [{"AttributeName": "k", "KeyType": "HASH"}]
    assert table["TimeToLiveSpecification"] == {"AttributeName": "exp", "Enabled": True}
    gen = one(template, "AWS::SecretsManager::Secret")["GenerateSecretString"]
    assert gen["PasswordLength"] >= 32 and gen["ExcludePunctuation"] is True  # web.py needs >= 32 chars
    assert [g["RetentionInDays"] for g in of_type(template, "AWS::Logs::LogGroup")] == [14, 14]


def test_waf_rate_limits_per_ip(template):
    acl = one(template, "AWS::WAFv2::WebACL")
    assert acl["Scope"] == "CLOUDFRONT" and acl["DefaultAction"] == {"Allow": {}}
    rules = {r["Name"]: r for r in acl["Rules"]}
    jobs = rules["job-starts"]["Statement"]["RateBasedStatement"]
    # WAF's smallest limit is 10; over 120 s that is 5 job starts a minute, the reserved concurrency.
    assert (jobs["Limit"], jobs["EvaluationWindowSec"], jobs["AggregateKeyType"]) == (10, 120, "IP")
    scope = json.dumps(jobs["ScopeDownStatement"])
    assert '"/api/jobs"' in scope and '"POST"' in scope and "URL_DECODE" in scope
    kr = rules["kill-and-review"]["Statement"]["RateBasedStatement"]
    assert (kr["Limit"], kr["EvaluationWindowSec"]) == (30, 60)
    assert '"/api/kill"' in json.dumps(kr) and '"/api/review"' in json.dumps(kr)
    everything = rules["all-requests"]["Statement"]["RateBasedStatement"]
    assert (everything["Limit"], everything["EvaluationWindowSec"]) == (300, 60) and "ScopeDownStatement" not in everything
    assert all(r["Action"] == {"Block": {}} for r in acl["Rules"])
    assert one(template, "AWS::CloudFront::Distribution")["DistributionConfig"]["WebACLId"]["Fn::GetAtt"][0] == "Waf"


def test_api_behaviour_is_uncached_and_forwards_the_body_hash(template):
    cfg = one(template, "AWS::CloudFront::Distribution")["DistributionConfig"]
    (api,) = cfg["CacheBehaviors"]
    assert api["PathPattern"] == "/api/*" and "POST" in api["AllowedMethods"] and api["Compress"] is False
    assert api["CachePolicyId"] == "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"  # Managed-CachingDisabled
    assert api["OriginRequestPolicyId"] == "b689b0a8-53d0-40ab-baf2-68738e2966ac"  # AllViewerExceptHostHeader
    (fn_origin,) = [o for o in cfg["Origins"] if "CustomOriginConfig" in o]
    assert api["TargetOriginId"] == fn_origin["Id"] and fn_origin["CustomOriginConfig"]["OriginReadTimeout"] == 60


def test_static_site_is_a_private_bucket_with_the_local_headers(template):
    resources = template["Resources"]
    dist_id = next(k for k, r in resources.items() if r["Type"] == "AWS::CloudFront::Distribution")
    cfg = resources[dist_id]["Properties"]["DistributionConfig"]
    (site_origin,) = [o for o in cfg["Origins"] if "S3OriginConfig" in o]
    default = cfg["DefaultCacheBehavior"]
    assert default["TargetOriginId"] == site_origin["Id"] and cfg["DefaultRootObject"] == "index.html"
    assert oac(template, "s3")["SigningBehavior"] == "always"
    assert set(one(template, "AWS::S3::Bucket")["PublicAccessBlockConfiguration"].values()) == {True}
    (read,) = [s for s in one(template, "AWS::S3::BucketPolicy")["PolicyDocument"]["Statement"]
               if s["Effect"] == "Allow" and "s3:GetObject" in s["Action"]]
    assert read["Principal"] == {"Service": "cloudfront.amazonaws.com"}
    assert {"Ref": dist_id} in read["Condition"]["StringEquals"]["AWS:SourceArn"]["Fn::Join"][1]

    policy_id = next(k for k, r in resources.items() if r["Type"] == "AWS::CloudFront::ResponseHeadersPolicy")
    assert default["ResponseHeadersPolicyId"] == {"Ref": policy_id}
    policy = resources[policy_id]["Properties"]["ResponseHeadersPolicyConfig"]
    sec, custom = policy["SecurityHeadersConfig"], policy["CustomHeadersConfig"]["Items"]
    hsts = sec["StrictTransportSecurity"]
    served = {"Content-Security-Policy": sec["ContentSecurityPolicy"]["ContentSecurityPolicy"],
              "X-Frame-Options": sec["FrameOptions"]["FrameOption"],
              "Referrer-Policy": sec["ReferrerPolicy"]["ReferrerPolicy"],
              "Strict-Transport-Security": f"max-age={hsts['AccessControlMaxAgeSec']}"
              + "; includeSubDomains" * hsts["IncludeSubdomains"] + "; preload" * hsts["Preload"],
              **({"X-Content-Type-Options": "nosniff"} if "ContentTypeOptions" in sec else {}),
              **{h["Header"]: h["Value"] for h in custom}}
    local = {h["key"]: h["value"] for h in json.loads((ROOT / "infra" / "headers.json").read_text())}
    assert served == local  # the same headers as local runs (web.py serves public/ with this file)
    assert all(v["Override"] for v in [*sec.values(), *custom])

    deploy = one(template, "Custom::CDKBucketDeployment")
    assert deploy["DistributionId"] == {"Ref": dist_id} and deploy["DistributionPaths"] == ["/*"]


def test_alarms_budget_and_topic_without_subscribers(template):
    assert of_type(template, "AWS::SNS::Subscription") == []
    topic_id = next(k for k, r in template["Resources"].items() if r["Type"] == "AWS::SNS::Topic")
    alarms = {a["MetricName"]: a for a in of_type(template, "AWS::CloudWatch::Alarm")}
    assert set(alarms) == {"Errors", "Throttles", "Duration", "AppErrors"}
    app_errors = one(template, "AWS::Logs::MetricFilter")
    assert app_errors["FilterPattern"] == '?"ERROR drishyam" ?"Exception in ASGI application"'
    assert app_errors["MetricTransformations"][0]["MetricName"] == "AppErrors"
    assert alarms["Duration"]["ExtendedStatistic"] == "p95" and alarms["Duration"]["Threshold"] == 45_000
    assert all(a["AlarmActions"] == [{"Ref": topic_id}] for a in alarms.values())
    budget = one(template, "AWS::Budgets::Budget")
    data = budget["Budget"]
    assert data["BudgetLimit"] == {"Amount": 20, "Unit": "USD"} and data["TimeUnit"] == "MONTHLY"
    assert data["CostTypes"]["IncludeCredit"] is False
    notes = budget["NotificationsWithSubscribers"]
    assert sorted(n["Notification"]["Threshold"] for n in notes) == [50, 100]
    assert all(n["Subscribers"] == [{"SubscriptionType": "SNS", "Address": {"Ref": topic_id}}] for n in notes)
    publishers = {s["Principal"]["Service"] for s in one(template, "AWS::SNS::TopicPolicy")["PolicyDocument"]["Statement"]}
    assert publishers == {"cloudwatch.amazonaws.com", "budgets.amazonaws.com"}
    body = one(template, "AWS::CloudWatch::Dashboard")["DashboardBody"]
    assert "{Drishyam,Status}" in json.dumps(body) and "JobSeconds" in json.dumps(body)


def test_reservation_can_be_left_out_and_no_em_dashes(template, tmp_path):
    assert "ReservedConcurrentExecutions" not in web_fn(synth(tmp_path, reserved_concurrency=None))
    assert "—" not in json.dumps(template, ensure_ascii=False)  # repo rule for AWS names and descriptions


def test_onnx_context_sets_the_segmenter_env(tmp_path):
    def env(sub, **kw):
        (tmp_path / sub).mkdir()
        return web_fn(synth(tmp_path / sub, **kw))["Environment"]["Variables"]

    fp32 = env("a", onnx="models/drishyam_dacl10k.onnx")
    assert fp32["DRISHYAM_ONNX"] == "/var/task/models/drishyam_dacl10k.onnx" and "DRISHYAM_ONNX_FP16" not in fp32
    both = env("b", onnx="models/m.onnx", onnx_fp16="models/m.fp16.onnx")
    assert (both["DRISHYAM_ONNX"], both["DRISHYAM_ONNX_FP16"]) == ("/var/task/models/m.onnx", "/var/task/models/m.fp16.onnx")
