import ast
from engine.base_engine import LLMEngine
from litellm import model_cost
import re
from tqdm import tqdm
import argparse
import os
from pathlib import Path
from poi_pipeline import (
    pipeline_prepare_candidates, parse_label, evaluate_rank
)
import pandas as pd
import json


class ExperimentDesignAgent:
    def __init__(self, llm_engine_name, context_cutoff=18000):
        self.llm_engine = LLMEngine(llm_engine_name)
        self.llm_cost = model_cost[llm_engine_name]
        self.context_cutoff = context_cutoff
        self.n_generate_sample = 5
        self.prompt_sample = "cct"

    def cot_plain_solve_task(self, prompt):
        # Accept either a plain string prompt or a dict with {"system": ..., "user": ...}
        if isinstance(prompt, dict) and "system" in prompt and "user" in prompt:
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        result = self.llm_engine.respond(messages, n=1)[0]
        return result

    def extract_result(self, x):
        user_input = [{"role": "user",
                       "content": f"""You are an information extraction assistant. From the given text, extract the final ranked list of POI names.

Rules:
1. If the text contains a "Final Ranking" or "Final List" section, extract POI names in that order.
2. If no such section exists, extract all unique POI names found inside angle brackets <...>, in the order they first appear.
3. Remove the angle brackets but keep everything inside them, including any parentheses.
4. Return only a plain Python list of strings, e.g. ["poi1", "poi2", "poi3"].
5. Do not include any explanations, metadata, or extra text—only output the list.
Ensure your output can be processed by json.loads(<output>) and return a list
DO NOT include text like "json", "the list is", etc. Only the python list like this: ["x", "y", "z"]
Here is the text to process:
{x}"""}]
        result = self.llm_engine.respond(user_input, n=1)[0]
        start = result.find("[")
        end = result.rfind("]") + 1
        result = result[start:end]
        return result

def robust_extract_predictions(agent, raw_text: str, max_retries: int = 3):
    """
    Extract Ranking list from LLM output using regular expressions.
    Expected output format:
    'Reasoning: ... \n\nRanking: ["poi1", "poi2", ...]'
    
    Returns: (pred_ranked_names: List[str], extracted_list_text: str)
    """
    
    # Method 1: Directly extract the list after "Ranking:" using regex
    # Match JSON list following "Ranking:"
    ranking_pattern = r'Ranking:\s*(\[.*?\])'
    match = re.search(ranking_pattern, raw_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        try:
            list_str = match.group(1)
            # Clean up potential formatting issues
            list_str = list_str.strip()
            parsed = json.loads(list_str)
            
            # Normalize: ensure it's a list of strings
            if isinstance(parsed, list):
                parsed = [str(x).strip() for x in parsed]
                return parsed, list_str
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for extracted list: {e}")
            print(f"Extracted string: {list_str}")
    
    # Method 2: If "Ranking:" not found, try to find any JSON list
    json_list_pattern = r'\[(?:"[^"]*"\s*,?\s*)+\]'
    matches = re.findall(json_list_pattern, raw_text, re.DOTALL)
    
    if matches:
        # Try to parse the last found list (usually the final answer)
        for list_candidate in reversed(matches):
            try:
                parsed = json.loads(list_candidate)
                if isinstance(parsed, list) and len(parsed) > 0:
                    parsed = [str(x).strip() for x in parsed]
                    return parsed, list_candidate
            except json.JSONDecodeError:
                continue
    try:
        start = raw_text.find("[")
        end = raw_text.rfind("]") + 1
        list_str = raw_text[start:end]

        parsed = ast.literal_eval(list_str)
        return parsed, list_str
    except:
        # Method 3: Fallback to LLM extraction (if all above methods fail)
        last_extracted = None
        last_err = None
        
        for _ in range(max_retries):
            try:
                extracted = agent.extract_result(raw_text)
                parsed = json.loads(extracted)
                
                if not isinstance(parsed, list):
                    raise ValueError("Parsed JSON is not a list")
                parsed = [str(x).strip() for x in parsed]
                
                return parsed, extracted
            except Exception as e:
                last_extracted = extracted if 'extracted' in locals() else None
                last_err = e
                print(f"LLM extraction retry failed: {e}")
                continue
    
    # Final fallback: return empty list
    print(f"[WARNING] Failed to extract predictions from text:\n{raw_text[:500]}...")
    raise RuntimeError(f"Failed to extract predictions after all attempts. Last error: {last_err}")
    # return ["none"]*20, "extraction_failed"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--strategy", type=str, default="spatial_cotp")
    parser.add_argument("--model_name", type=str, default="gpt-4o")

    parser.add_argument("--save_file", type=str, default="poi_eval_results_spatial_cot.json")
    args = parser.parse_args()
    args.save_file = f"poi_eval_results_{args.model_name}_{args.strategy}.json"
    poi_path = Path("data/poi_qa/ENG/POI_ENG.txt")
    val_data = pd.read_csv("data/poi_qa/ENG/traj_ENG_val.csv")
    agent = ExperimentDesignAgent(args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.save_file)

    # Accumulated metrics (overall)
    total_metrics = {"HR@5": 0, "NDCG@5": 0, "HR@10": 0, "NDCG@10": 0, "HR@20": 0, "NDCG@20": 0}

    # New: per-item record list
    per_item_records = []
    for index, row in tqdm(val_data.iterrows(), total=len(val_data), desc="Evaluating"):

        traj_prompt = row["prompt"]
        prep = pipeline_prepare_candidates(poi_path, traj_prompt, max_candidates=120,strategy=args.strategy)
        raw_pred_text = agent.cot_plain_solve_task(prep["llm_prompt"])


        pred_ranked_names, extracted_list_text = robust_extract_predictions(agent, raw_pred_text, max_retries=3)

        gold_names = parse_label(row['label'])

        metrics = evaluate_rank(pred_ranked_names, gold_names)
        print(metrics)

        for k, v in metrics.items():
            total_metrics[k] += v

        per_item_records.append({
            "index": int(index),

            "input": {
                "llm_prompt": prep["llm_prompt"],

            },
            "model": {
                "name": args.model_name,
                "strategy": args.strategy,
            },
            "prediction": {
                "raw_output": raw_pred_text,
                "extracted_list_text": extracted_list_text,
                "pred_ranked_names": pred_ranked_names
            },
            "ground_truth": gold_names,
            "metrics": metrics
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "num_samples": len(val_data),
                "total_metrics": total_metrics
            },
            "records": per_item_records
        }, f, ensure_ascii=False, indent=2)

    print(f"[Saved] per-item results & metrics -> {save_path}")
    print("Total metrics:", total_metrics)


