STACK_NAME=LambdaSalinasLexAgent
BUCKET_NAME=example-sam-deploy-bucket
REGION=us-east-1
BEDROCK_REGION=us-east-1
AGENT_ID=FAKE_AGENT_ID
AGENT_ALIAS_ID=FAKE_ALIAS_ID
FUNCTION_NAME=LambdaSalinasLexAgent
EVENT_FILE=events/test-event.json
ENV_FILE=env.json
SAM_BUILD_MODE=debug
FUNCTION_LOGICAL_ID=LambdaSalinasLexAgent

.PHONY: build clean config-deploy guided-deploy highlimit-run local-invoke logs online-invoke print-env purge run validate

build:
	@echo "🔨 Building Lambda with SAM (mode: $(SAM_BUILD_MODE))..."
	SAM_BUILD_MODE=$(SAM_BUILD_MODE) sam build --beta-features

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf .aws-sam target

config-deploy:
	@echo "📦 Deploying with config params..."
	sam deploy \
	  --stack-name $(STACK_NAME) \
	  --s3-bucket $(BUCKET_NAME) \
	  --capabilities CAPABILITY_IAM \
	  --region $(REGION) \
	  --parameter-overrides \
	    AgentIdParam=$(AGENT_ID) \
			AgentAliasIdParam=$(AGENT_ALIAS_ID) \
	    BedrockRegionParam=$(BEDROCK_REGION)

guided-deploy:
	@echo "🧭 Starting guided deploy..."
	sam deploy --guided

highlimit-run:
	@echo "🧪 Running build with high file limit..."
	bash -c 'ulimit -n 65535 && SAM_BUILD_MODE=debug sam build --beta-features && echo "✅ High limit applied inside subshell"'

local-invoke:
	@echo "🚀 Running Lambda locally..."
	sam local invoke $(FUNCTION_NAME) -e $(EVENT_FILE) --env-vars $(ENV_FILE)

logs:
	@echo "📜 Tailing Lambda logs..."
	aws logs tail /aws/lambda/$$(aws lambda list-functions --query "Functions[?starts_with(FunctionName, 'LambdaSalinasLex')].FunctionName" --output text --region $(REGION)) --follow --region $(REGION)

online-invoke:
	@echo "☁️ Invoking Lambda in the cloud..."
	aws lambda invoke --function-name $$(aws cloudformation describe-stack-resource --stack-name $(STACK_NAME) --logical-resource-id $(FUNCTION_LOGICAL_ID) --query 'StackResourceDetail.PhysicalResourceId' --output text --region $(REGION)) --region $(REGION) --payload fileb://$(EVENT_FILE) output.json && cat output.json | jq

print-env:
	@echo "🔍 Printing environment variables inside Amazon Linux container..."
	docker run --rm \
		-e AGENT_ID=$(AGENT_ID) \
		-e AGENT_ALIAS_ID=$(AGENT_ALIAS_ID) \
		-e BEDROCK_REGION=$(BEDROCK_REGION) \
		amazonlinux \
		/bin/sh -c "echo AGENT_ID=\$$AGENT_ID && echo AGENT_ALIAS_ID=\$$AGENT_ALIAS_ID && echo BEDROCK_REGION=\$$BEDROCK_REGION"

purge:
	@echo "🔥 Cleaning Rust + Zig caches..."
	cargo clean
	rm -rf target
	rm -rf ~/.cache/zig

run: clean build online-invoke config-deploy online-invoke

validate:
	@echo "✅ Validating SAM template..."
	sam validate --lint
