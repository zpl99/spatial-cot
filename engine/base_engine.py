from dotenv import load_dotenv
import os

load_dotenv()


class LLMEngine():
    def __init__(self, llm_engine_name, use_azure=False):
        self.llm_engine_name = llm_engine_name
        self.use_azure = use_azure
        self.engine = None
        self.is_embedding_model = False
        
        if "embedding" in llm_engine_name.lower():
            self.is_embedding_model = True
            from engine.openai_engine import OpenaiEmbeddingEngine
            self.engine = OpenaiEmbeddingEngine(llm_engine_name)
        elif use_azure:
            # Azure OpenAI
            from engine.openai_engine import AzureOpenaiEngine
            self.engine = AzureOpenaiEngine(llm_engine_name)
        elif llm_engine_name.lower().startswith("qwen") or llm_engine_name.lower().startswith("mistral"):
            # Qwen (vllm)
            from engine.qwen_engine import QwenEngine
            self.engine = QwenEngine(llm_engine_name)
        elif llm_engine_name.startswith("gpt") or llm_engine_name.startswith("o1"):
            # OpenAI API
            from engine.openai_engine import OpenaiEngine
            self.engine = OpenaiEngine(llm_engine_name)
        elif "claude" in llm_engine_name.lower() or "anthropic" in llm_engine_name.lower():
            # AWS Bedrock (Claude models via the Converse API).
            # Accepts plain ids (claude-...) and cross-region inference profile
            # ids (e.g., us.anthropic.claude-3-5-haiku-20241022-v1:0).
            from engine.bedrock_engine import BedrockEngine
            self.engine = BedrockEngine(llm_engine_name)
        else:
            raise ValueError(f"Unknown engine type for model: {llm_engine_name}")

    def respond(self, user_input, temperature=0.2, top_p=0.92, n=1):
        if self.is_embedding_model:
            raise ValueError("This is an embedding model. Use get_embedding() instead of respond()")
        
        if self.llm_engine_name.lower().startswith("qwen"):
            return self.engine.respond(user_input, n=n, temperature=temperature, top_p=top_p)
        elif self.use_azure:
            return self.engine.respond(user_input, temperature, top_p, n=n)
        elif "claude" in self.llm_engine_name.lower() or "anthropic" in self.llm_engine_name.lower():
            return self.engine.respond(user_input, temperature, top_p)
        else:
            return self.engine.respond(user_input, n)
    
    def get_embedding(self, text):
        if not self.is_embedding_model:
            raise ValueError("This is not an embedding model. Use respond() instead of get_embedding()")
        return self.engine.get_embedding(text)

if __name__ == "__main__":
    engine = LLMEngine("Qwen/Qwen3-30B-A3B-Instruct-2507")
    
    response, prompt_tokens, completion_tokens = engine.respond(
        "Give me a short introduction to large language model.", 
        temperature=0.8, 
        top_p=0.95
    )
    print(response)