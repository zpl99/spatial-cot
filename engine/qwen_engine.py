from vllm import LLM, SamplingParams

class QwenEngine:
    def __init__(self, model_name):
        self.model_name = model_name
        self.llm = LLM(model=model_name, tensor_parallel_size=2, trust_remote_code=True)

    def respond(self, user_input, n=1, temperature=0.8, top_p=0.95):
        """
        Generate response using Qwen model via vllm.
        """
        # Ensure prompts is a list of strings
        if isinstance(user_input, str):
            prompts = [user_input]
        elif isinstance(user_input, list) and isinstance(user_input[0], str):
             prompts = user_input
        else:
            # Handle other cases if necessary, e.g. chat messages
            # For now, convert to string
            prompts = [str(user_input)]

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=5000, n=n)
        
        # vllm generate
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Process outputs
        # Assuming single prompt input for now as per base_engine usage pattern (usually one query)
        # If multiple prompts were passed, we might need to handle differently, 
        # but base_engine.respond typically returns (text, p_tokens, c_tokens) for one input.
        
        output = outputs[0]
        prompt_tokens = len(output.prompt_token_ids)
        
        generated_texts = [o.text for o in output.outputs]
        completion_tokens = sum(len(o.token_ids) for o in output.outputs)
        
        if n == 1:
            return generated_texts[0], prompt_tokens, completion_tokens
        else:
            return generated_texts, prompt_tokens, completion_tokens
