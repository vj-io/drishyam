"""The web demo on AWS, as one stack (docs/aws.md).

The FastAPI app (drishyam/web.py) runs on Lambda arm64 (Python 3.12 zip plus the Lambda Web Adapter layer),
streams NDJSON through a Function URL (RESPONSE_STREAM, AWS_IAM auth) and is reached only through
CloudFront, which signs every origin request with Origin Access Control. CloudFront serves the static
site (public/) from a private S3 bucket, also through OAC, so only /api/* reaches the function. A WAF web
ACL rate-limits requests per IP. KILL flags live in a DynamoDB table and the work-order signing key in
Secrets Manager, so both work across Lambda instances. Logs, a dashboard, alarms and a monthly budget
complete it.
"""

from __future__ import annotations

import json
import mmap
import re
from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Size, Stack
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_cloudfront as cf
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3deploy
from aws_cdk import aws_secretsmanager as sm
from aws_cdk import aws_sns as sns
from aws_cdk import aws_wafv2 as waf
from constructs import Construct

# Lambda Web Adapter 1.1.0 for arm64, published by AWS account 753240598075 in every region. Version 30 is
# the one the awslabs/aws-lambda-web-adapter README lists; `aws lambda get-layer-version-by-arn` on it in
# us-east-1 reports Description "Layer for AWS Lambda Adapter arm64: 1.1.0", created 2026-09-18.
LWA_LAYER = "arn:aws:lambda:{region}:753240598075:layer:LambdaAdapterLayerArm64:30"
FUNCTION_NAME = "drishyam-web"
ROOT = Path(__file__).resolve().parents[1]
# POST /api/jobs per IP: 10 per 120 s. WAF's smallest limit is 10, so the 120 s window is what brings
# it to 5 starts a minute. The live deployment has no reserved concurrency (RESERVED=none): the account
# quota of 10 is the cap (docs/aws.md, "Limits").
JOBS_LIMIT, JOBS_WINDOW_S = 10, 120
KILL_REVIEW_PER_MINUTE = 30  # /api/kill and /api/review together, per IP per 60 s
REQUESTS_PER_MINUTE = 300  # every request per IP per 60 s; a page load is about 30
# What a blocked caller reads: each rule answers 429 with {"error": "rate limited: <this> per IP; wait and retry"}.
RATE_LIMITS = {"job-starts": f"{JOBS_LIMIT} job starts per {JOBS_WINDOW_S // 60} min",
               "kill-and-review": f"{KILL_REVIEW_PER_MINUTE} kill and review calls per min",
               "all-requests": f"{REQUESTS_PER_MINUTE} requests per min"}
BUDGET_USD = 20
# video/build.py --standin renders the film with stand-in data (a relabelled trained-segmenter result) and writes this
# into the MP4's metadata and file name; such a film must never reach the site.
STANDIN_MARK = b"STAND-IN DATA, NOT FOR PUBLICATION"


def refuse_standin(public: Path) -> None:
    """Stop the deployment if any MP4 under public/ was rendered with stand-in data, whatever it is called now."""
    for f in public.rglob("*.mp4"):
        with f.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            if "standin" in f.name or m.find(STANDIN_MARK) >= 0:
                raise SystemExit(f"{f} is a stand-in film (video/build.py --standin), not for publication")


def _path_is(path: str) -> waf.CfnWebACL.StatementProperty:
    return waf.CfnWebACL.StatementProperty(byte_match_statement=waf.CfnWebACL.ByteMatchStatementProperty(
        field_to_match=waf.CfnWebACL.FieldToMatchProperty(uri_path={}), positional_constraint="EXACTLY",
        search_string=path,  # after URL decoding, so /api/%6Aobs is counted like /api/jobs (Starlette decodes it)
        text_transformations=[waf.CfnWebACL.TextTransformationProperty(priority=0, type="URL_DECODE")]))


def _rate_rule(name: str, priority: int, limit: int, scope: waf.CfnWebACL.StatementProperty | None, window: int = 60):
    block = waf.CfnWebACL.BlockActionProperty(custom_response=waf.CfnWebACL.CustomResponseProperty(
        response_code=429, custom_response_body_key=name))  # the body is RATE_LIMITS[name], on the web ACL
    return waf.CfnWebACL.RuleProperty(
        name=name, priority=priority, action=waf.CfnWebACL.RuleActionProperty(block=block),
        statement=waf.CfnWebACL.StatementProperty(rate_based_statement=waf.CfnWebACL.RateBasedStatementProperty(
            limit=limit, evaluation_window_sec=window, aggregate_key_type="IP", scope_down_statement=scope)),
        visibility_config=waf.CfnWebACL.VisibilityConfigProperty(
            cloud_watch_metrics_enabled=True, metric_name=name, sampled_requests_enabled=True))


class DrishyamStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, code: lambda_.Code, reserved_concurrency: int | None = 5,
                 commit: str = "", onnx: str = "", onnx_fp16: str = "", **kwargs) -> None:
        super().__init__(scope, cid, description="Drishyam bridge inspector on Lambda arm64 behind CloudFront and WAF",
                         **kwargs)

        kills = ddb.Table(self, "KillTable", partition_key=ddb.Attribute(name="k", type=ddb.AttributeType.STRING),
                          billing_mode=ddb.BillingMode.PAY_PER_REQUEST, time_to_live_attribute="exp",
                          removal_policy=RemovalPolicy.DESTROY)  # 5-minute flags, nothing worth keeping
        key = sm.Secret(self, "SigningKey", description="HMAC key that signs Drishyam work orders",
                        generate_secret_string=sm.SecretStringGenerator(password_length=64, exclude_punctuation=True))
        log_group = logs.LogGroup(self, "Logs", retention=logs.RetentionDays.TWO_WEEKS,
                                  removal_policy=RemovalPolicy.DESTROY)

        # Least privilege: no managed policies. Logs on its own group, two item actions on the one table,
        # one read on the one secret; Tracing.ACTIVE adds the two X-Ray actions, which only accept "*".
        role = iam.Role(self, "Role", assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
                        description="Drishyam web function: own log group, kill table items, signing key")
        log_group.grant_write(role)
        kills.grant(role, "dynamodb:GetItem", "dynamodb:PutItem")
        role.add_to_policy(iam.PolicyStatement(actions=["secretsmanager:GetSecretValue"], resources=[key.secret_arn]))

        env = {
            "AWS_LAMBDA_EXEC_WRAPPER": "/opt/bootstrap",  # the adapter starts run.sh and proxies to port 8080
            "AWS_LWA_INVOKE_MODE": "response_stream",
            "AWS_LWA_READINESS_CHECK_PATH": "/api/health",
            "DRISHYAM_KILL_TABLE": kills.table_name,
            "DRISHYAM_SIGNING_KEY_SECRET_ARN": key.secret_arn,
            # cv2.setNumThreads at startup (web.py): 3008 MB is 1.7 vCPU of time (1,769 MB = 1 vCPU), so 2 threads
            "DRISHYAM_CV_THREADS": "2",
        }
        if commit:
            env["DRISHYAM_COMMIT"] = commit
        # The segmenter: an ONNX file inside the package (models/...), which Lambda unpacks to /var/task.
        for name, path in (("DRISHYAM_ONNX", onnx), ("DRISHYAM_ONNX_FP16", onnx_fp16)):
            if path:
                env[name] = f"/var/task/{path}"
        fn = lambda_.Function(
            self, "Web", function_name=FUNCTION_NAME, description="Drishyam FastAPI app behind the Lambda Web Adapter",
            runtime=lambda_.Runtime.PYTHON_3_12, architecture=lambda_.Architecture.ARM_64, handler="run.sh", code=code,
            layers=[lambda_.LayerVersion.from_layer_version_arn(self, "WebAdapter", LWA_LAYER.format(region=self.region))],
            memory_size=3008, timeout=Duration.seconds(60),  # 60 s: MAX_PACED_S and the 30 s seconds cap fit in it
            reserved_concurrent_executions=reserved_concurrency, role=role, log_group=log_group,
            tracing=lambda_.Tracing.ACTIVE, environment=env)
        url = fn.add_function_url(auth_type=lambda_.FunctionUrlAuthType.AWS_IAM,
                                  invoke_mode=lambda_.InvokeMode.RESPONSE_STREAM)

        acl = waf.CfnWebACL(
            self, "Waf", scope="CLOUDFRONT", default_action=waf.CfnWebACL.DefaultActionProperty(allow={}),
            description="Per IP rate limits on Drishyam job starts, kills, reviews and all requests",
            visibility_config=waf.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True, metric_name="drishyam", sampled_requests_enabled=True),
            custom_response_bodies={name: waf.CfnWebACL.CustomResponseBodyProperty(
                content_type="APPLICATION_JSON",
                content=json.dumps({"error": f"rate limited: {what} per IP; wait and retry"}))
                for name, what in RATE_LIMITS.items()},
            rules=[
                _rate_rule("job-starts", 0, JOBS_LIMIT, waf.CfnWebACL.StatementProperty(
                    and_statement=waf.CfnWebACL.AndStatementProperty(statements=[
                        _path_is("/api/jobs"),
                        waf.CfnWebACL.StatementProperty(byte_match_statement=waf.CfnWebACL.ByteMatchStatementProperty(
                            field_to_match=waf.CfnWebACL.FieldToMatchProperty(method={}),
                            positional_constraint="EXACTLY", search_string="POST",
                            text_transformations=[waf.CfnWebACL.TextTransformationProperty(priority=0, type="NONE")]))])),
                    window=JOBS_WINDOW_S),
                _rate_rule("kill-and-review", 1, KILL_REVIEW_PER_MINUTE, waf.CfnWebACL.StatementProperty(
                    or_statement=waf.CfnWebACL.OrStatementProperty(statements=[
                        _path_is("/api/kill"), _path_is("/api/review")]))),
                _rate_rule("all-requests", 2, REQUESTS_PER_MINUTE, None),
            ])

        # The static site: a private bucket read only by this distribution (OAC), with the response headers
        # of infra/headers.json, so CSP and HSTS are the same as in local runs.
        site = s3.Bucket(self, "Site", block_public_access=s3.BlockPublicAccess.BLOCK_ALL, enforce_ssl=True,
                         encryption=s3.BucketEncryption.S3_MANAGED, removal_policy=RemovalPolicy.DESTROY,
                         auto_delete_objects=True)
        headers = {h["key"]: h["value"] for h in json.loads((ROOT / "infra" / "headers.json").read_text())}
        hsts = headers.pop("Strict-Transport-Security")
        headers.pop("X-Content-Type-Options")  # its only value is nosniff
        page_headers = cf.ResponseHeadersPolicy(
            self, "PageHeaders", comment="Drishyam page headers, from infra/headers.json",
            security_headers_behavior=cf.ResponseSecurityHeadersBehavior(
                content_security_policy=cf.ResponseHeadersContentSecurityPolicy(
                    content_security_policy=headers.pop("Content-Security-Policy"), override=True),
                content_type_options=cf.ResponseHeadersContentTypeOptions(override=True),
                frame_options=cf.ResponseHeadersFrameOptions(
                    frame_option=cf.HeadersFrameOption[headers.pop("X-Frame-Options")], override=True),
                referrer_policy=cf.ResponseHeadersReferrerPolicy(referrer_policy=cf.HeadersReferrerPolicy[
                    headers.pop("Referrer-Policy").replace("-", "_").upper()], override=True),
                strict_transport_security=cf.ResponseHeadersStrictTransportSecurity(
                    access_control_max_age=Duration.seconds(int(re.search(r"max-age=(\d+)", hsts)[1])),
                    include_subdomains="includeSubDomains" in hsts, preload="preload" in hsts, override=True)),
            custom_headers_behavior=cf.ResponseCustomHeadersBehavior(custom_headers=[
                cf.ResponseCustomHeader(header=k, value=v, override=True) for k, v in headers.items()]))

        origin = origins.FunctionUrlOrigin.with_origin_access_control(url, read_timeout=Duration.seconds(60))
        dist = cf.Distribution(
            self, "Cdn", comment="Drishyam web app", web_acl_id=acl.attr_arn, price_class=cf.PriceClass.PRICE_CLASS_100,
            http_version=cf.HttpVersion.HTTP2_AND_3, default_root_object="index.html",
            enable_ipv6=False,  # WAF rate rules count per address; an IPv6 /64 would give a client millions of them
            default_behavior=cf.BehaviorOptions(  # the static site: cached; the deployment below invalidates it
                origin=origins.S3BucketOrigin.with_origin_access_control(site),
                viewer_protocol_policy=cf.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cf.CachePolicy.CACHING_OPTIMIZED, response_headers_policy=page_headers),
            additional_behaviors={"/api/*": cf.BehaviorOptions(  # never cached; query, body and headers pass through
                origin=origin, viewer_protocol_policy=cf.ViewerProtocolPolicy.HTTPS_ONLY,
                allowed_methods=cf.AllowedMethods.ALLOW_ALL, cache_policy=cf.CachePolicy.CACHING_DISABLED,
                origin_request_policy=cf.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER, compress=False)})
        # Uploads public/ on each deploy that changes it, deletes files no longer there and invalidates /*.
        # 1 GiB of memory and 2 GiB of /tmp: the asset holds the video (public/video/drishyam.mp4, git-ignored).
        refuse_standin(ROOT / "public")
        s3deploy.BucketDeployment(
            self, "SiteFiles", sources=[s3deploy.Source.asset(str(ROOT / "public"))], destination_bucket=site,
            distribution=dist, distribution_paths=["/*"], memory_limit=1024,
            ephemeral_storage_size=Size.mebibytes(2048),
            log_group=logs.LogGroup(self, "SiteFilesLogs", retention=logs.RetentionDays.TWO_WEEKS,
                                    removal_policy=RemovalPolicy.DESTROY))

        # The stack adds no subscriber to the alerts topic; the owner subscribes an email (see docs/aws.md).
        topic = sns.Topic(self, "Alerts", display_name="Drishyam alerts")
        for service, source in (("cloudwatch", f"arn:aws:cloudwatch:{self.region}:{self.account}:alarm:*"),
                                ("budgets", f"arn:aws:budgets::{self.account}:*")):
            topic.add_to_resource_policy(iam.PolicyStatement(
                principals=[iam.ServicePrincipal(f"{service}.amazonaws.com")], actions=["sns:Publish"],
                resources=[topic.topic_arn],
                conditions={"StringEquals": {"aws:SourceAccount": self.account}, "ArnLike": {"aws:SourceArn": source}}))

        five_min = Duration.minutes(5)
        # Behind the Web Adapter a FastAPI 500 or a failed job (an {"type": "error"} event in a 200 stream) is
        # a successful invocation, so the Errors alarm misses it. Both log at ERROR from the drishyam logger
        # (web.py's format "%(levelname)s %(name)s %(message)s"); uvicorn logs an exception that escapes the
        # app as "Exception in ASGI application".
        app_errors = logs.MetricFilter(
            self, "AppErrors", log_group=log_group, metric_namespace="Drishyam", metric_name="AppErrors",
            filter_pattern=logs.FilterPattern.any_term("ERROR drishyam", "Exception in ASGI application"))
        for name, metric, threshold, op, text in (
            ("Errors", fn.metric_errors(period=five_min, statistic="Sum"), 1,
             cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD, "At least one failed invocation in 5 minutes"),
            ("Throttles", fn.metric_throttles(period=five_min, statistic="Sum"), 1,
             cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD, "Requests refused at the concurrency cap"),
            ("DurationP95", fn.metric_duration(period=five_min, statistic="p95"), 45_000,
             cw.ComparisonOperator.GREATER_THAN_THRESHOLD, "p95 invocation time above 45 s (timeout is 60 s)"),
            ("AppErrors", app_errors.metric(period=five_min, statistic="Sum"), 1,
             cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD, "At least one app error or failed job in 5 minutes"),
        ):
            cw.Alarm(self, f"{name}Alarm", metric=metric, threshold=threshold, evaluation_periods=1,
                     comparison_operator=op, treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
                     alarm_description=text).add_alarm_action(cw_actions.SnsAction(topic))

        def emf_metric(name: str, stat: str, prefix: str = "", dim: str = "Status") -> cw.MathExpression:
            # One line per value of the EMF dimension (Status: completed, halted, awaiting_review; Decision:
            # approve, reject), named by it, without hard-coding them.
            return cw.MathExpression(expression=f"SEARCH('{{Drishyam,{dim}}} MetricName=\"{name}\"', '{stat}', 300)",
                                     label=prefix + f"${{PROP('Dim.{dim}')}}", period=five_min)

        dash = cw.Dashboard(self, "Dashboard", dashboard_name="Drishyam")
        dash.add_widgets(
            cw.GraphWidget(title="Jobs by status", left=[emf_metric("JobSeconds", "SampleCount")]),
            cw.GraphWidget(title="Job seconds (mean)", left=[emf_metric("JobSeconds", "Average")]),
            cw.GraphWidget(title="Pixels used (fraction of full resolution)", left=[emf_metric("PixelsFrac", "Average")]),
            cw.GraphWidget(title="Steps and findings per job", left=[emf_metric("Steps", "Average", "steps ")],
                           right=[emf_metric("Findings", "Average", "findings ")]))
        dash.add_widgets(
            cw.GraphWidget(title="Lambda invocations, errors, throttles",
                           left=[fn.metric_invocations(period=five_min, statistic="Sum"),
                                 fn.metric_errors(period=five_min, statistic="Sum"),
                                 fn.metric_throttles(period=five_min, statistic="Sum")]),
            cw.GraphWidget(title="Lambda duration (ms)",
                           left=[fn.metric_duration(period=five_min, statistic=s) for s in ("p50", "p95", "Maximum")]),
            cw.GraphWidget(title="Concurrent executions",
                           left=[fn.metric("ConcurrentExecutions", period=five_min, statistic="Maximum")]),
            cw.GraphWidget(title="Reviews by decision", left=[emf_metric("Reviews", "Sum", dim="Decision")]))

        budget = budgets.CfnBudget(
            self, "Budget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name="drishyam-monthly", budget_type="COST", time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(amount=BUDGET_USD, unit="USD"),
                cost_types=budgets.CfnBudget.CostTypesProperty(include_credit=False)),  # credits must not hide spend
            notifications_with_subscribers=[budgets.CfnBudget.NotificationWithSubscribersProperty(
                notification=budgets.CfnBudget.NotificationProperty(
                    notification_type="ACTUAL", comparison_operator="GREATER_THAN", threshold=pct,
                    threshold_type="PERCENTAGE"),
                subscribers=[budgets.CfnBudget.SubscriberProperty(subscription_type="SNS", address=topic.topic_arn)])
                for pct in (50, 100)])
        budget.node.add_dependency(topic)  # Budgets checks it may publish when the budget is created

        for name, value in (("Url", f"https://{dist.distribution_domain_name}"), ("FunctionUrl", url.url),
                            ("DistributionId", dist.distribution_id), ("FunctionName", fn.function_name),
                            ("AlertTopicArn", topic.topic_arn), ("KillTableName", kills.table_name),
                            ("LogGroupName", log_group.log_group_name)):
            CfnOutput(self, name, value=value)
