import yaml
import os
from jinja2 import Template

class PromptManager:
    def __init__(self, prompt_dir="prompts"):
        self.prompt_dir = prompt_dir

    def load_prompt(self, agent, prompt_name):
        path = os.path.join(self.prompt_dir, agent, f"{prompt_name}.yaml")
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data

    def render_prompt(self, agent, prompt_name, variables):
        data = self.load_prompt(agent, prompt_name)
        template = Template(data['template'])
        rendered = template.render(**variables)
        return rendered
