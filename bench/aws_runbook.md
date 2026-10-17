# AWS runbook: the COOL three-arm benchmark

Copy-paste commands for arms A (c8g.2xlarge, COOL 3.1 AMI), B (c8g.2xlarge, Ubuntu 24.04, stock wheel) and C (c7i.2xlarge, Ubuntu 24.04, stock wheel); what each arm isolates is in [README.md](README.md). **You run every command here; nothing in this repo calls AWS, and none of these commands has been run by the author.** They were written against the AWS CLI v2 reference. Every value you must supply is marked `<FILL: ...>`.

How it fits together: [`aws_user_data.sh`](aws_user_data.sh) runs at first boot (installs git, the AWS CLI and uv, clones the public repo, schedules a hard stop after 3 hours). [`aws_arm.sh`](aws_arm.sh) is sent through SSM to run `bench/run_all.sh <arm>` against a dataset and parity crops you upload once, then puts `results/<arm>/` in S3. `python -m bench.compare_arms` turns the three result folders into `docs/cool_benchmark.md` and `public/results/cool.json`.

## Cost per arm

Rates are from the cost ledger in [docs/PRD.md](../docs/PRD.md) (us-east-1, on demand). Each instance terminates itself 3 hours after boot, so the right-hand column is the most an arm can cost. The ledger budgets 3 hours per arm. A normal run should take about 10 to 20 minutes from launch to upload, and the instance terminates itself 15 minutes after that, so an arm should cost roughly $0.20. That is my estimate from a local dry run of the same scripts; it has not been measured on EC2.

| Arm | Instance | $/h | Most it can cost (3 h) |
| --- | --- | --- | --- |
| A | c8g.2xlarge, COOL 3.1 AMI | 0.319 + 0.02 COOL fee = 0.339 (the fee is waived during the 7-day trial) | ≈ $1.02 (the PRD shares arm A with development, so it has no ledger row of its own) |
| B | c8g.2xlarge, Ubuntu 24.04 | 0.319 | ≈ $0.96 (ledger: $1) |
| C | c7i.2xlarge, Ubuntu 24.04 | 0.357 | ≈ $1.07 (ledger: $1) |

On top of that you pay for a public IPv4 address ($0.005/h per instance), an 8 GB root volume for the same hours, and about 110 MB in S3. Each is a few cents at most.

## 0. Before you start

Open a terminal, type `bash`, and paste the blocks below **in order into that one shell**. They are bash: zsh's default settings choke on the `#` comments. If you have to open a new shell, paste this block again and re-set any ids you already have (`IA`, `IB`, `IC`, `CA`, `CB`, `CC`). You need the AWS CLI v2 configured for the account, and this repo checked out with `.venv` working (`make test` passes).

```bash
cd <FILL: path to your checkout of this repo>
export AWS_PROFILE=<FILL: CLI profile for the account>
export AWS_DEFAULT_REGION=us-east-1        # one region for everything; the rates above are us-east-1
EMAIL=<FILL: address for budget alerts>
BUCKET=<FILL: a new, globally unique S3 bucket name, e.g. drishyam-bench-4821>
KEY_NAME=<FILL: name of an existing EC2 key pair in this region; used only by the SSH fallback>
ACCT=$(aws sts get-caller-identity --query Account --output text)
REF=$(git ls-remote https://github.com/vj-io/drishyam refs/heads/main | cut -f1)   # every arm checks out this commit
echo "account $ACCT, commit $REF"
aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --query Quota.Value --output text
```

If you have no key pair in the region, create one first with `aws ec2 create-key-pair --key-name drishyam-bench --query KeyMaterial --output text > ~/drishyam-bench.pem && chmod 400 ~/drishyam-bench.pem`, then set `KEY_NAME=drishyam-bench`. The file goes in your home directory, outside the repo.

The last line prints your on-demand vCPU quota for standard instances. Running all three arms at once needs 24. If the quota is lower, request an increase, or run steps 5 to 9 for one arm at a time.

## 1. Spend guardrails first

Create a $100 monthly budget that emails you at $50 and at $100. `IncludeCredit: false` makes it count usage that credits pay for, so credits cannot hide it. Skip this block if you already have such a budget, and use its name in step 6.

```bash
cat > /tmp/drishyam-budget-alerts.json <<EOF
[{"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 50, "ThresholdType": "ABSOLUTE_VALUE"},
  "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]},
 {"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 100, "ThresholdType": "ABSOLUTE_VALUE"},
  "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]}]
EOF
aws budgets create-budget --account-id "$ACCT" --notifications-with-subscribers file:///tmp/drishyam-budget-alerts.json \
  --budget '{"BudgetName": "drishyam-bench", "BudgetType": "COST", "TimeUnit": "MONTHLY",
             "BudgetLimit": {"Amount": "100", "Unit": "USD"}, "CostTypes": {"IncludeCredit": false}}'
aws iam create-role --role-name drishyam-budget-action --assume-role-policy-document \
  '{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "budgets.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name drishyam-budget-action \
  --policy-arn arn:aws:iam::aws:policy/AWSBudgetsActions_RolePolicyForResourceAdministrationWithSSM
```

The role lets Budgets stop instances. The stop action itself is created in step 6, right after launch, because a Budgets stop action has to name the instance ids. Budgets updates only a few times a day, so the stop action is the slow backstop. The fast one is the 3-hour self-termination that every instance schedules at boot.

## 2. Subscribe to COOL (console only) and find the AMI ids

You must do this in the console; the CLI cannot subscribe. Open AWS Marketplace, find the COOL 3.1 listing for Graviton4, choose **View purchase options**, then **Subscribe**, and accept the terms. Check that the listing supports c8g.2xlarge. Subscribing starts the 7-day free trial on the $0.02/h fee, so subscribe on the day you run. If the account is on the AWS Free Plan and you cannot subscribe, upgrade to the Paid Plan; credits carry over, and step 1 is already in place. Then choose **Continue to Configuration**, pick the version and your region, and copy the AMI id it shows.

```bash
AMI_A=<FILL: the COOL AMI id from "Continue to Configuration">
# optional cross-check; the name pattern is a guess, the console value above is authoritative:
#   aws ec2 describe-images --owners aws-marketplace --filters "Name=name,Values=*COOL*" "Name=architecture,Values=arm64" \
#     --query 'reverse(sort_by(Images,&CreationDate))[:5].[ImageId,Name]' --output table
AMI_B=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
  --query Parameter.Value --output text)
AMI_C=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query Parameter.Value --output text)
echo "A $AMI_A  B $AMI_B  C $AMI_C"
```

Each arm's `macro.json` records the AMI it actually ran on.

## 3. Results bucket and an instance role limited to SSM

The instance role gets `AmazonSSMManagedInstanceCore` plus read and write on one S3 prefix, which the results travel through. It gets nothing else.

```bash
aws s3 mb "s3://$BUCKET"
cat > /tmp/drishyam-s3-policy.json <<EOF
{"Version": "2012-10-17", "Statement": [
 {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": "arn:aws:s3:::$BUCKET/drishyam-bench/*"},
 {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::$BUCKET",
  "Condition": {"StringLike": {"s3:prefix": "drishyam-bench/*"}}}]}
EOF
aws iam create-role --role-name drishyam-bench-ec2 --assume-role-policy-document \
  '{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name drishyam-bench-ec2 --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam put-role-policy --role-name drishyam-bench-ec2 --policy-name drishyam-bench-s3 \
  --policy-document file:///tmp/drishyam-s3-policy.json
aws iam create-instance-profile --instance-profile-name drishyam-bench-ec2
aws iam add-role-to-instance-profile --instance-profile-name drishyam-bench-ec2 --role-name drishyam-bench-ec2
sleep 15   # a brand-new instance profile takes a few seconds to become usable by EC2
```

## 4. Shared inputs, built once on your laptop

All three arms time the same 200 images and segment the same 24 parity crops. `bench.compare_arms` refuses arms that did not.

```bash
[ -f data/synth200/annotations.json ] || .venv/bin/python -m evals.synth dataset --out data/synth200 --n 200 --seed 7
.venv/bin/python -m bench.parity --save /tmp/drishyam-parity-inputs.npz
aws s3 sync --only-show-errors data/synth200 "s3://$BUCKET/drishyam-bench/inputs/synth200"
aws s3 cp /tmp/drishyam-parity-inputs.npz "s3://$BUCKET/drishyam-bench/inputs/parity.npz"
```

## 5. Launch the three arms

The instances use the default VPC and its default security group, which allows no inbound traffic; SSM needs none. If the region has no default VPC, add `--subnet-id <FILL: a public subnet>` to the call.

```bash
launch() {  # launch <arm> <instance type> <ami>: prints the instance id
  aws ec2 run-instances --image-id "$3" --instance-type "$2" --key-name "$KEY_NAME" \
    --iam-instance-profile Name=drishyam-bench-ec2 --instance-initiated-shutdown-behavior terminate \
    --metadata-options HttpTokens=required --user-data file://bench/aws_user_data.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=drishyam-bench-$1},{Key=project,Value=drishyam-bench}]" \
                         "ResourceType=volume,Tags=[{Key=project,Value=drishyam-bench}]" \
    --query 'Instances[0].InstanceId' --output text
}
IA=$(launch A c8g.2xlarge "$AMI_A"); IB=$(launch B c8g.2xlarge "$AMI_B"); IC=$(launch C c7i.2xlarge "$AMI_C")
echo "A $IA  B $IB  C $IC"
```

An empty id means that launch failed, and the error is printed above it. The usual causes are a Marketplace subscription that is not active yet (arm A), the vCPU quota, or a key pair from another region.

## 6. Attach the $100 stop action to the three instances

```bash
aws budgets create-budget-action --account-id "$ACCT" --budget-name drishyam-bench \
  --notification-type ACTUAL --action-type RUN_SSM_DOCUMENTS \
  --action-threshold ActionThresholdValue=100,ActionThresholdType=ABSOLUTE_VALUE \
  --definition "{\"SsmActionDefinition\": {\"ActionSubType\": \"STOP_EC2_INSTANCES\", \"Region\": \"$AWS_DEFAULT_REGION\",
                \"InstanceIds\": [\"$IA\", \"$IB\", \"$IC\"]}}" \
  --execution-role-arn "arn:aws:iam::$ACCT:role/drishyam-budget-action" --approval-model AUTOMATIC \
  --subscribers SubscriptionType=EMAIL,Address="$EMAIL"
```

## 7. Run `bench/run_all.sh` on each arm through SSM

First wait until all three instances have registered with SSM (one to three minutes after boot):

```bash
for i in $(seq 40); do   # up to 10 minutes
  n=$(aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$IA,$IB,$IC" \
      --query 'length(InstanceInformationList[?PingStatus==`Online`])' --output text)
  [ "$n" = 3 ] && break; sleep 15
done
aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$IA,$IB,$IC" \
  --query 'InstanceInformationList[].[InstanceId,PingStatus,PlatformName]' --output table
```

If an instance is still missing after 10 minutes, its AMI has no SSM agent. That is most likely arm A, since only the Ubuntu AMIs are known to ship the agent. Use the SSH fallback at the end for that arm and run this step for the others.

```bash
send() {  # send <instance id> <arm> <usd per hour> [cool python]: runs bench/aws_arm.sh there, prints the command id
  b64=$(base64 < bench/aws_arm.sh | tr -d '\n')
  aws ssm send-command --instance-ids "$1" --document-name AWS-RunShellScript --comment "drishyam bench $2" \
    --parameters "{\"executionTimeout\": [\"7200\"], \"commands\": [\"echo $b64 | base64 -d > /tmp/arm.sh && bash /tmp/arm.sh $2 $BUCKET $REF $3 $4\"]}" \
    --query Command.CommandId --output text
}
CA=$(send "$IA" A 0.339); CB=$(send "$IB" B 0.319); CC=$(send "$IC" C 0.357)
echo "A $CA  B $CB  C $CC"
```

The third argument is the arm's total $/h, including the COOL fee for A. It becomes `USD_PER_HOUR` and the `$/1k images` figure in `macro.json`, so change it if you are not in us-east-1.

Each arm runs the micro benchmark, the macro benchmark at 1, 2, 4 and 8 workers, and the parity save. On arm A, `aws_arm.sh` uses the first `python3` under `/opt/cool` that imports `cv2`. I have not seen the AMI's `/opt/cool` layout. If that search picks the wrong interpreter, or finds none, the output lists `/opt/cool`; re-send with the right interpreter, for example `CA=$(send "$IA" A 0.339 /opt/cool/<FILL: venv>/bin/python3)`.

## 8. Wait, then copy the results back

```bash
wait_cmd() {  # wait_cmd <command id> <instance id>: polls until the run ends, then prints its status and output tail
  until s=$(aws ssm get-command-invocation --command-id "$1" --instance-id "$2" --query Status --output text 2>/dev/null) &&
        [ "$s" != Pending ] && [ "$s" != InProgress ] && [ "$s" != Delayed ]; do sleep 30; done
  echo "== $2: $s"
  aws ssm get-command-invocation --command-id "$1" --instance-id "$2" \
    --query '[StandardOutputContent,StandardErrorContent]' --output text | tail -30
}
wait_cmd "$CA" "$IA"; wait_cmd "$CB" "$IB"; wait_cmd "$CC" "$IC"
for arm in A B C; do aws s3 cp --only-show-errors --recursive "s3://$BUCKET/drishyam-bench/results/$arm" "results/$arm"; done
ls results/A results/B results/C
```

Each folder should hold `micro.json`, `macro.json`, `macro_w2.json`, `macro_w4.json`, `macro_w8.json` and `parity.npz`. A status other than `Success` comes with the output that explains it. Fix the cause and re-send that arm with `send`. An instance terminates itself 15 minutes after a successful upload, and 3 hours after boot regardless.

## 9. Terminate, and check nothing is left running

```bash
aws ec2 terminate-instances --instance-ids "$IA" "$IB" "$IC" --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
aws ec2 wait instance-terminated --instance-ids "$IA" "$IB" "$IC"
aws ec2 describe-instances --filters Name=tag:project,Values=drishyam-bench \
  Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text
aws ec2 describe-volumes --filters Name=tag:project,Values=drishyam-bench --query 'Volumes[].VolumeId' --output text
```

The last two commands should print nothing.

## 10. Fold the results into the repo

```bash
.venv/bin/python -m bench.compare_arms results/A results/B results/C
git add results/A/*.json results/B/*.json results/C/*.json docs/cool_benchmark.md public/results/cool.json
```

The command writes the report and the website summary. It refuses arms that ran different OpenCV major versions, different cv2 builds within one arm, or different images or parity crops, and it names which one. The `parity.npz` files are about 10 MB each and hold the same crops three times; the `git add` above leaves them out. Add them if you want the raw masks in the repo; otherwise they stay in S3 until the cleanup below.

## Fallback: an arm whose AMI has no SSM agent

The same script runs over SSH. Fill in the key file and the login user from the listing's usage instructions (often `ubuntu` or `ec2-user`).

```bash
SG=$(aws ec2 create-security-group --group-name drishyam-bench-ssh --description "SSH from one address" --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$(curl -s https://checkip.amazonaws.com)/32"
aws ec2 modify-instance-attribute --instance-id "$IA" --groups "$SG"
IP=$(aws ec2 describe-instances --instance-ids "$IA" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
PEM=<FILL: path to the .pem for $KEY_NAME>; LOGIN=<FILL: login user>
scp -i "$PEM" bench/aws_arm.sh "$LOGIN@$IP:/tmp/arm.sh"
ssh -i "$PEM" "$LOGIN@$IP" "sudo nohup bash /tmp/arm.sh A $BUCKET $REF 0.339 > /tmp/arm.log 2>&1 < /dev/null &"
ssh -i "$PEM" "$LOGIN@$IP" tail -n 30 /tmp/arm.log   # repeat until it ends with "uploaded results/A ..."
```

Then continue with the S3 copy in step 8 and with step 9. Delete the security group after termination: `aws ec2 delete-security-group --group-id "$SG"`.

## Cleanup (optional, after the results are committed)

The budget, the roles and the instance profile cost nothing and can stay. The stop action names instances that no longer exist; remove it with `aws budgets describe-budget-actions-for-budget --account-id "$ACCT" --budget-name drishyam-bench` and then `aws budgets delete-budget-action --account-id "$ACCT" --budget-name drishyam-bench --action-id <FILL: ActionId>`. Removing the bucket **permanently deletes** the uploaded dataset and every arm's `parity.npz`:

```bash
aws s3 rm --recursive "s3://$BUCKET/drishyam-bench" && aws s3 rb "s3://$BUCKET"
```
