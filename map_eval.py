from engine.base_engine import LLMEngine
from collections import defaultdict
from prompt import prompt_manager
import re
from tqdm import tqdm
import argparse
import time
import json
import ct_rag

manager = prompt_manager.PromptManager("./prompt")
rag_system = ct_rag.SCT_GraphRAG.load("rag_knowledge_graph.pkl")


def _get_difficulty_label(item):
    """
    Extract difficulty label from item, supporting two formats:
    - "difficulty": {"difficulty": "L3", "reason": "..."}
    - "difficulty": "L3"
    Returns "UNK" if missing or invalid.
    """
    d = item.get("difficulty", None)
    if isinstance(d, dict):
        val = d.get("difficulty", None)
        return val if isinstance(val, str) and val in {"L1", "L2", "L3"} else "UNK"
    elif isinstance(d, str):
        return d if d in {"L1", "L2", "L3"} else "UNK"
    else:
        return "UNK"


def evaluate_mapeval_classification(dataset):
    """
    Evaluate prediction results and calculate accuracy metrics by category and difficulty.
    
    Args:
        dataset: List of prediction items with answer, prediction, classification, and difficulty
        
    Returns:
        Dictionary containing overall metrics and breakdowns by category/difficulty
    """
    total = 0
    correct = 0
    unanswerable = 0

    by_category = defaultdict(lambda: {"correct": 0, "total": 0, "unanswerable": 0})
    by_difficulty = defaultdict(lambda: {"correct": 0, "total": 0, "unanswerable": 0})
    by_cat_diff = defaultdict(lambda: {"correct": 0, "total": 0, "unanswerable": 0})

    for item in dataset:
        gt = item.get('answer', None)
        pred = item.get('prediction', None)
        category = item.get('classification', "UNK")
        diff = _get_difficulty_label(item)

        if pred is None or gt is None:
            continue

        total += 1
        by_category[category]['total'] += 1
        by_difficulty[diff]['total'] += 1
        by_cat_diff[(category, diff)]['total'] += 1

        if pred == 0:
            unanswerable += 1
            by_category[category]['unanswerable'] += 1
            by_difficulty[diff]['unanswerable'] += 1
            by_cat_diff[(category, diff)]['unanswerable'] += 1

        if pred == gt:
            correct += 1
            by_category[category]['correct'] += 1
            by_difficulty[diff]['correct'] += 1
            by_cat_diff[(category, diff)]['correct'] += 1

    def _to_rates(counter):
        return {
            k: {
                "accuracy": (v["correct"] / v["total"]) if v["total"] > 0 else 0.0,
                "unanswerable_rate": (v["unanswerable"] / v["total"]) if v["total"] > 0 else 0.0,
                "total": v["total"],
                "correct": v["correct"],
                "unanswerable": v["unanswerable"],
            }
            for k, v in counter.items()
        }

    overall_accuracy = correct / total if total > 0 else 0.0
    overall_unanswerable_rate = unanswerable / total if total > 0 else 0.0

    per_category = _to_rates(by_category)
    per_difficulty = _to_rates(by_difficulty)

    nested_cat_diff = {}
    for (cat, diff), vals in by_cat_diff.items():
        nested_cat_diff.setdefault(cat, {})
        v = vals
        nested_cat_diff[cat][diff] = {
            "accuracy": (v["correct"] / v["total"]) if v["total"] > 0 else 0.0,
            "unanswerable_rate": (v["unanswerable"] / v["total"]) if v["total"] > 0 else 0.0,
            "total": v["total"],
            "correct": v["correct"],
            "unanswerable": v["unanswerable"],
        }

    return {
        "overall": {
            "accuracy": overall_accuracy,
            "unanswerable_rate": overall_unanswerable_rate,
            "total_evaluated": total,
            "correct": correct,
            "unanswerable": unanswerable,
        },
        "by_category": per_category,
        "by_difficulty": per_difficulty,
        "by_category_difficulty": nested_cat_diff,
    }


class SpatialAgent:
    def __init__(self, llm_engine_name, context_cutoff=18000, max_retries=3, backoff=0.8):
        self.llm_engine = LLMEngine(llm_engine_name)
        self.context_cutoff = context_cutoff
        self.max_retries = max_retries
        self.backoff = backoff
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
    
    def _safe_respond(self, user_input, n=1, accumulate_tokens=True, **kwargs):
        """Safely respond with retry logic and token tracking."""
        delay = self.backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.llm_engine.respond(user_input, n=n, **kwargs)
                out, prompt_tokens, completion_tokens = result
                
                if accumulate_tokens:
                    self.last_prompt_tokens += prompt_tokens
                    self.last_completion_tokens += completion_tokens
                else:
                    self.last_prompt_tokens = prompt_tokens
                    self.last_completion_tokens = completion_tokens
                return out
            except Exception as e:
                print(e)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
        raise Exception("Too many retries")
    
    def get_answer_id(self, response):
        """Extract option ID from model response using multiple regex patterns."""
        patterns = [
            r'\*{0,2}Option\s*id[:：]?\*{0,2}\s*(\d+)',
            r'Option\s+id[:：]?\s*(?:\[|\*\*?)?(\d+)(?:\]|\*\*?)?',
            r'(?i)option\s+(\d+)(?!\s*id)',
        ]
        
        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            all_matches.extend(matches)
            
        if all_matches:
            return int(all_matches[0])
        else:
            print(response)
            print("No option id found in the response.")
            return 0
    
    def spatial_cot_solve_task(self, x, existing_path=None, seperated_stages=False):
        """Solve task using Chain-of-Thought with Core Concepts."""

        system_format = manager.render_prompt(
            agent="coreconcepts",
            prompt_name="core_concepts",
            variables={"question": x}
        )

        if existing_path:
            user_input = [{"role": "user",
                            "content": system_format + f'''Existing identified path {existing_path}. Think step by step and use transformation path to help you solve the problem. Then, select the correct answer option based on your reasoning. Output the selected option's id using the format: Option id: id and your reasoning process based on the transformation path using this format : Reasoning: your reasoning. Do not include any other explanation. If there is no answer, just output "Option id: 0".'''}]
        else:
            user_input = [{"role": "user",
                            "content": system_format +
                                        '''
                                        Identify the relevant core concepts and transformation steps needed to solve it. Think step by step and use transformation path to help you solve the problem. Then, select the correct answer option based on your reasoning.
                                        Only output your reasoning process, and the selected option's id using the format: Option id: id. If there is no answer, just output "Option id: 0". Explain your reasoning clearly before giving the final answer.'''}]

        result = self._safe_respond(user_input, n=1)
        id = self.get_answer_id(result)

        return result, id
    
    def spatial_cotp_solve_task(self, x, id):
        """Solve task using CoT with Core Concepts and RAG-based rules."""
        info = rag_system.generate_transformation_path_iteratively(x, mode='concept_transformations_knowledge', max_steps=5)
        
        system_format = manager.render_prompt(
            agent="coreconcepts",
            prompt_name="core_concepts_transformation_path",
            variables={"info": info, "question": x}
        )

        user_input = [{"role": "user",
                   "content": system_format + '''Based on the information provided. Select the correct answer option based on your reasoning. Do reasoning first step by step then output your choice.
                              For your choice. Please using the format: Option id: id. If there is no answer, just output "Option id: 0". Let's think step by step.'''}]

        result = self._safe_respond(user_input, n=1)
        id = self.get_answer_id(result)

        return result, id
    
    def cot_plain_solve_task(self, x):
        """Solve task using standard Chain-of-Thought."""
        user_input = [{"role": "user",
                       "content": x + '''Think step by step， then output your choice of the option's id, in the format of "Option id: id". If you think there is no answer, just output "Option id: 0" to indicate no answer. Explain your reasoning clearly before giving the final answer.'''}]

        result = self._safe_respond(user_input, n=1)
        id = self.get_answer_id(result)
        return result, id

    def io_plain_solve_task(self, x):
        """Solve task using direct input-output (no reasoning)."""
        user_input = [{"role": "user",
                       "content": x + '''Only output your option. If you think there is no answer, just output "Option id: 0" to indicate no answer. Here is an example of your output: Option id: 1.'''}]

        result = self._safe_respond(user_input, n=1)
        id = self.get_answer_id(result)
        return result, id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="spatial_cot")
    parser.add_argument("--model_name", type=str, default="gpt-4o")

    args = parser.parse_args()
    args.output_dir = f"map_eval_{args.model_name}_{args.strategy}.json"
    print(f"Results will be saved to {args.output_dir}")
    
    agent = SpatialAgent(args.model_name)
    
    all_result = {}
    with open("data/mapeval/mapeval_textual_difficulty.json") as f:
        mapeval_textual = json.load(f)
    
    for item in tqdm(mapeval_textual):
        prompt = (
            "You are a highly intelligent assistant. "
            "Based on the given context, answer the multiple-choice question by selecting the correct option.\n\n"
            "Context:\n" + item["context"] + "\n\n"
            "Question:\n" + item["question"] + "\n\n"
            "Options:\n"
        )
        for i, option in enumerate(item["options"], start=1):
            prompt += f"{i}. {option}\n"
        
        gt = item["answer"]
        id_for_item = item["id"]
        
        if args.strategy == "io":
            llm_output, id = agent.io_plain_solve_task(prompt)
        elif args.strategy == "cot":
            llm_output, id = agent.cot_plain_solve_task(prompt)
        elif args.strategy == "spatial_cot":
            llm_output, id = agent.spatial_cot_solve_task(prompt)
        elif args.strategy == "spatial_cotp":
            llm_output, id = agent.spatial_cotp_solve_task(prompt, id=id_for_item)
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")

        item["prompt"] = prompt
        item["prediction"] = id
        item["raw_output"] = llm_output
    
    result = evaluate_mapeval_classification(mapeval_textual)
    all_result.update({"benchmark_acc": result})
    print(all_result)
    
    if "/" in args.output_dir:
        args.output_dir = args.output_dir.replace("/", "-")
    
    with open(args.output_dir, "w") as f:
        json.dump(
            {
                "item_level_results": mapeval_textual,
                "summary": all_result
            },
            f,
            indent=4,
            ensure_ascii=False
        )


if __name__ == "__main__":
    main()
