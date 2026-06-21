from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, AzureOpenAI
import os
from dotenv import load_dotenv
import backoff

load_dotenv()

@backoff.on_exception(backoff.expo, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError))
def openai_chat_engine(client, engine, msg, temperature, top_p):
    if engine.startswith("gpt"):
        response = client.chat.completions.create(
            model=engine,
            messages=msg,
            temperature=temperature,
            max_tokens=2000,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0
        )
    else:
        response = client.chat.completions.create(
            model=engine,
            messages=msg,
        )

    return response

class OpenaiEngine():

    def __init__(self, llm_engine_name):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file")
        
        self.client = OpenAI(
            api_key=api_key,
            max_retries=3, 
            timeout=120.0
        )
        self.llm_engine_name = llm_engine_name

    def respond(self, user_input, n=1):

        if n > 1:
            results = []
            prompt_tokens = 0
            completion_tokens = 0
            for _ in range(n):
                request_params = {
                    "model": self.llm_engine_name,
                    "input": user_input,
                    "max_output_tokens": 10000,
                }
                if self.llm_engine_name.startswith("o1"):
                    request_params["reasoning"] = {"effort": "low"}
                
                response = self.client.responses.create(**request_params)
                text = self._extract_response_text(response)
                results.append(text)
                prompt_tokens += getattr(response.usage, 'input_tokens', 0) or getattr(response.usage, 'prompt_tokens', 0)
                completion_tokens += getattr(response.usage, 'output_tokens', 0) or getattr(response.usage, 'completion_tokens', 0)
            return results, prompt_tokens, completion_tokens
        else:
            response = self.client.responses.create(
                model=self.llm_engine_name,
                input=user_input
            )
            
            prompt_tokens = getattr(response.usage, 'input_tokens', None) or getattr(response.usage, 'prompt_tokens', 0)
            completion_tokens = getattr(response.usage, 'output_tokens', None) or getattr(response.usage, 'completion_tokens', 0)
            
            text = self._extract_response_text(response)
            return text, prompt_tokens, completion_tokens
    
    def _extract_response_text(self, response):

        if hasattr(response, 'output_text') and response.output_text:
            return response.output_text
        
        if hasattr(response, 'output') and response.output:
            parts = []
            for item in response.output:
                if hasattr(item, 'content'):
                    for content_part in item.content:
                        if hasattr(content_part, 'text'):
                            if hasattr(content_part.text, 'value'):
                                parts.append(content_part.text.value)
                            else:
                                parts.append(str(content_part.text))
            if parts:
                return ''.join(parts)
        
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        
        return ""



class AzureOpenaiEngine():

    def __init__(self, llm_engine_name):
        self.llm_engine_name = llm_engine_name
        
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        base_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ist-apim-aoai.azure-api.net/load-balancing")
        
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables. Please set it in .env file")
        
        if llm_engine_name == "gpt-4.1":
            azure_endpoint = f"{base_endpoint}/gpt-4.1"
        elif llm_engine_name == "gpt-4":
            azure_endpoint = f"{base_endpoint}/gpt-4"
        elif llm_engine_name == "gpt-4o":
            azure_endpoint = f"{base_endpoint}/gpt-4o"
        elif llm_engine_name == "gpt-5":
            azure_endpoint = f"{base_endpoint}/gpt-5"
        elif llm_engine_name == "gpt-5-mini":
            azure_endpoint = f"{base_endpoint}/gpt-5-mini"
        elif llm_engine_name == "gpt-o1":
            azure_endpoint = f"{base_endpoint}/gpt-o1"
            api_version = "2025-02-01-preview"
        elif llm_engine_name == "gpt-35-turbo":
            azure_endpoint = f"{base_endpoint}/gpt-35-turbo"
        else:
            raise ValueError("Unknown OpenAI engine {}".format(llm_engine_name))
        
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )

    def respond(self, user_input, temperature, top_p,n=1):
        if self.llm_engine_name in ["gpt-35-turbo", "gpt-4"]:
            response = self.client.chat.completions.create(
                model=self.llm_engine_name,
                messages=user_input,
                temperature=temperature,
                top_p=top_p,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                n=n)
        elif self.llm_engine_name in ["gpt-o1","gpt-5", "gpt-5-mini"]:
            response = self.client.chat.completions.create(
                model=self.llm_engine_name,
                messages=user_input,
                max_completion_tokens=10000,
                n=n)
        else:
            response = self.client.chat.completions.create(
                model=self.llm_engine_name,
                messages=user_input,
                temperature=temperature,
                top_p=top_p,
                max_tokens=10000,
                frequency_penalty=0,
                presence_penalty=0,
                n=n)

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        if n != 1:
            results = [choice.message.content for choice in response.choices]
        else:
            results = response.choices[0].message.content

        return results, prompt_tokens, completion_tokens


class OpenaiEmbeddingEngine():

    def __init__(self, model_name):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file")
        
        self.client = OpenAI(
            api_key=api_key,
            max_retries=3, 
            timeout=120.0
        )
        self.model_name = model_name
    
    @backoff.on_exception(backoff.expo, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError))
    def get_embedding(self, text):

        if isinstance(text, str):
            text = [text]
            return_single = True
        else:
            return_single = False
        
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        
        embeddings = [item.embedding for item in response.data]
        
        if return_single:
            return embeddings[0]
        else:
            return embeddings
