"""
Supports both MapEval and POI-QA datasets with various LLM strategies.

Usage:
    python run.py --task map_eval --model gpt-4o --strategy cot
    python run.py --task poiqa --model gpt-4o --strategy spatial_cotp
"""

import argparse
import subprocess
import sys


# Available configurations
SUPPORTED_TASKS = ["map_eval", "poiqa"]

SUPPORTED_MODELS = [
    "gpt-4o",
    "gpt-5",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",

]

SUPPORTED_STRATEGIES = {
    "map_eval": [
        "io",              # Direct input-output (no reasoning)
        "cot",             # Standard Chain-of-Thought (CoT)
        "spatial_cot",          # CoT with Core Concepts (Spatial CoT)
        "spatial_cotp",# CoT with Core Concepts + RAG rules (Spatial CoT+)
    ],
    "poiqa": [
        "io",              # Direct input-output (no reasoning)
        "cot",             # Standard Chain-of-Thought (CoT)
        "spatial_cot",          # CoT with Core Concepts (Spatial CoT)
        "spatial_cotp",# CoT with Core Concepts + RAG rules (Spatial CoT+)
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified runner for spatial reasoning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --task map_eval --model gpt-4o --strategy cot
  python run.py --task poiqa --model gpt-4o --strategy spatial_cotp
  python run.py --list  # Show all available options
        """
    )
    
    parser.add_argument(
        "--task", 
        type=str, 
        choices=SUPPORTED_TASKS,
        help="Task to run: map_eval or poiqa"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o",
        help=f"LLM model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="cot",
        help="Reasoning strategy to use (default: cot)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results",
        help="Output directory for results (default: ./results)"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all supported tasks, models, and strategies"
    )
    
    return parser.parse_args()


def show_options():
    """Display all available configuration options."""
    print("\n" + "="*60)
    print("SUPPORTED CONFIGURATIONS")
    print("="*60)
    
    print("\n📊 Tasks:")
    for task in SUPPORTED_TASKS:
        print(f"  - {task}")
    
    print("\n🤖 Models:")
    for model in SUPPORTED_MODELS:
        print(f"  - {model}")
    
    print("\n🧠 Strategies:")
    for task, strategies in SUPPORTED_STRATEGIES.items():
        print(f"\n  [{task}]")
        for s in strategies:
            print(f"    - {s}")
    
    print("\n" + "="*60 + "\n")


def run_map_eval(model: str, strategy: str, output_dir: str):
    """Run MapEval benchmark."""
    print(f"\nRunning MapEval with model={model}, strategy={strategy}")
    
    cmd = [
        sys.executable, "map_eval.py",
        "--model_name", model,
        "--strategy", strategy,
    ]
    
    subprocess.run(cmd, check=True)


def run_poiqa(model: str, strategy: str, output_dir: str):
    """Run POI-QA benchmark."""
    print(f"\nRunning POI-QA with model={model}, strategy={strategy}")
    
    cmd = [
        sys.executable, "poiqa.py",
        "--model_name", model,
        "--strategy", strategy,
        "--output_dir", output_dir,
    ]
    
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    
    if args.list:
        show_options()
        return
    
    # Validate task is provided
    if not args.task:
        print("Error: --task is required. Use --list to see available options.")
        sys.exit(1)
    
    # Validate strategy for the selected task
    valid_strategies = SUPPORTED_STRATEGIES.get(args.task, [])
    if args.strategy not in valid_strategies:
        print(f"Error: Strategy '{args.strategy}' is not valid for task '{args.task}'.")
        print(f"Valid strategies: {valid_strategies}")
        sys.exit(1)
    
    # Run the selected task
    if args.task == "map_eval":
        run_map_eval(args.model, args.strategy, args.output_dir)
    elif args.task == "poiqa":
        run_poiqa(args.model, args.strategy, args.output_dir)
    else:
        print(f"Error: Unknown task '{args.task}'")
        sys.exit(1)
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
