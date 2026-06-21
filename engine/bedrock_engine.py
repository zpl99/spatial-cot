import os
from botocore.exceptions import ClientError
from botocore.client import Config

import boto3


def bedrock_converse_engine(client, engine, msg, temperature, top_p):
    # Newer Claude models (e.g., Claude 4.x) reject specifying both
    # `temperature` and `top_p`. We pass only `temperature` (the primary
    # sampling control used in the paper), which is accepted by all models.
    response = client.converse(
        modelId=engine,
        messages=msg,
        inferenceConfig={"maxTokens": 2000, "temperature": temperature},
    )

    return response


class BedrockEngine():
    """Amazon Bedrock engine (e.g., Claude models via the Converse API).

    Authentication uses a Bedrock API key (bearer token) read from the
    environment variable ``AWS_BEARER_TOKEN_BEDROCK``. Set it in your shell or
    in a local ``.env`` file (never commit it):

        export AWS_BEARER_TOKEN_BEDROCK="<your-bedrock-api-key>"

    boto3 (>=1.39) automatically uses this bearer token for the
    ``bedrock-runtime`` service when the variable is present.
    """

    def __init__(self, llm_engine_name):
        if not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
            raise ValueError(
                "AWS_BEARER_TOKEN_BEDROCK not found in environment variables. "
                "Please set it in your shell or in a .env file."
            )
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            config=Config(retries={"total_max_attempts": 10}),
        )
        self.llm_engine_name = llm_engine_name

    def respond(self, user_input, temperature, top_p):
        conversation = [
            {"role": turn["role"], "content": [{"text": turn["content"]}]}
            for turn in user_input
        ]

        try:
            response = bedrock_converse_engine(
                self.client,
                self.llm_engine_name,
                conversation,
                temperature,
                top_p
            )
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{self.llm_engine_name}'. Reason: {e}")
            return "ERROR", 0, 0

        return response["output"]["message"]["content"][0]["text"], response["usage"]["inputTokens"], response["usage"]["outputTokens"]
