import argparse
import json
import os
import logging
import re
import time
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
import torch

from src.utils import load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_system_prompt(max_len: int) -> str:
    """
    Returns the system prompt string with the 'max_len' words placeholder replaced.
    """
    return f"""You are solving AIME (American Invitational Mathematics Examination) problems. 
Please finish your proof within {max_len//2} words!

Important: Always end your solution with the final answer in one of these two formats:

1. \\[
   \\boxed{{X}}.
   \\]

2. $n=\\boxed{{X}}$

where X is your integer answer between 0 and 999."""


def load_2024_dataset() -> List[dict]:
    """
    Load the dataset of problems specifically for 2024.
    By default, this loads from 'AI-MO/aimo-validation-aime' and filters for 2024.
    Adjust if you want to pull from your own AIME_2024 dataset or anything else.
    """
    dataset_original = load_dataset("AI-MO/aimo-validation-aime")
    # Filter out problems that are not from 2024
    dataset = dataset_original["train"].filter(lambda example: "2024" in example["url"])
    logging.debug(f"Filtered dataset size: {len(dataset)}.")
    assert (
        len(dataset) == 30
    ), f"Expected 30 problems from 2024, but found {len(dataset)}"
    return dataset


def extract_answer(response: str) -> Optional[int]:
    """
    Extract the numerical answer from the LLM's solution text.
    Looks for \boxed{} or other final-answer patterns. Returns the *last* match found.
    """
    if not response:
        return None

    response = " ".join(response.split())  # remove extra whitespace

    patterns = [
        r"\$n=\\boxed{(\d+)}\$",
        r"\\\[\\boxed{(\d+)}\\\]",
        r"\\\[\\boxed{(\d+)}\.\\\]",
        r"\\boxed{(\d+)}",
        r"\$\\boxed{(\d+)}\$",
        r"boxed{(\d+)}",
        r"\\boxed\s*{\s*(\d+)\s*}",
        r"\bboxed\s*{\s*(\d+)\s*}",
        r"final answer is[^\d]*(\d+)",
        r"answer is[^\d]*(\d+)",
        r"answer:[^\d]*(\d+)",
        r"= ?(\d+)$",
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, response, re.IGNORECASE))
        if matches:
            # Take the last match
            last_match = matches[-1]
            try:
                return int(last_match.group(1))
            except (ValueError, IndexError):
                continue

    numbers = re.findall(r"(\d+)", response)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass

    return None


def tidal_inference_single(
    model, tokenizer, prompt: str, max_tokens: int = 8192, stride: int = 256
) -> str:
    """
    Perform Tidal inference with intermediate "cache correction":
      (1) Generate 'stride' tokens in Tidal decode mode.
      (2) Evict Tidal decode cache.
      (3) Forward pass newly generated tokens to build a proper "full-attention" cache.
      (4) Repeat until we generate a total of 'max_tokens' tokens.
    """
    system_prompt_str = get_system_prompt(max_tokens)
    full_prompt_text = (
        f"USER: {system_prompt_str}\n" f"Problem:\n{prompt}\n\n" "ASSISTANT: "
    )
    context_ids = tokenizer(full_prompt_text, return_tensors="pt")["input_ids"]
    rounds = (max_tokens // stride) + (1 if max_tokens % stride != 0 else 0)
    total_generated_ids = []
    total_tokens_generated = 0

    for r in range(rounds):
        tokens_this_round = min(stride, max_tokens - total_tokens_generated)
        if tokens_this_round <= 0:
            break
        outputs = model.generate(
            input_ids=context_ids,
            max_new_tokens=tokens_this_round,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        new_ids = outputs[:, context_ids.shape[-1] :]
        context_ids = torch.cat([context_ids, new_ids], dim=-1)
        with torch.no_grad():
            _ = model(
                input_ids=context_ids,
            )
        total_tokens_generated += tokens_this_round
        total_generated_ids.append(new_ids)
        if total_tokens_generated >= max_tokens:
            break
    if total_generated_ids:
        all_new_ids = torch.cat(total_generated_ids, dim=-1)
        all_new_ids = all_new_ids[0]
        out_text = tokenizer.decode(all_new_ids, skip_special_tokens=True)
    else:
        out_text = ""

    gen_length = len(out_text.split())
    logger.info(
        f"Generated {all_new_ids.shape[0]} tokens for this problem (approx. {gen_length} words)."
    )

    return out_text.strip()


def get_llm_response(model, tokenizer, problem: str, max_tokens: int = 8192) -> str:
    """
    Wrap the single Tidal inference to match the original function’s style.
    Returns the single response string for pass@1.
    """
    response_text = tidal_inference_single(model, tokenizer, problem, max_tokens)
    return response_text


def make_n_attempts(
    problem: str, model, tokenizer, n: int, max_tokens: int = 8192
) -> List[Dict]:
    """
    Make n attempts to solve a problem (pass@n).
    Returns a list of attempt dicts, each with 'response' and 'predicted_answer'.
    """
    attempts = []
    for attempt_index in range(n):
        response = get_llm_response(model, tokenizer, problem, max_tokens=max_tokens)
        predicted_answer = extract_answer(response)
        attempts.append(
            {
                "attempt_number": attempt_index + 1,
                "response": response,
                "predicted_answer": predicted_answer,
            }
        )
    return attempts


def evaluate_pass_at_n(
    attempts: List[Dict], correct_answer: int
) -> Tuple[bool, Optional[int]]:
    """
    Determine if *any* of the attempts is correct. If so, return first correct attempt number.
    """
    for attempt in attempts:
        if attempt["predicted_answer"] == correct_answer:
            return True, attempt["attempt_number"]
    return False, None


def load_existing_results(filename: str) -> List[Dict]:
    """Load existing results from file if it exists, else return empty list."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []


def save_result(filename: str, result: Dict):
    """Append a single result to the JSON file."""
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)


def analyze_results(results: List[Dict], n: int):
    """
    Summarize overall pass@N performance.
    """
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total if total > 0 else 0

    print("\n=== Results Summary ===")
    print(f"Evaluation mode: pass@{n}")
    print(f"Total problems: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    # If any were correct, see which attempt they got correct
    successful_attempts = [
        r["first_correct_attempt"] for r in results if r["is_correct"]
    ]
    if successful_attempts:
        avg_attempts = sum(successful_attempts) / len(successful_attempts)
        print(f"\nFor correct solutions:")
        print(f"Average attempts needed: {avg_attempts:.2f}")
        print(f"Attempt distribution:")
        for i in range(1, n + 1):
            count = sum(1 for x in successful_attempts if x == i)
            print(f"  Attempt {i}: {count} problems")

    print("\n=== Incorrect Problems ===")
    for r in results:
        if not r["is_correct"]:
            print(f"Problem {r['index']}:")
            print(f"Expected: {r['correct_answer']}")
            print(
                "Predicted answers across attempts:",
                [attempt["predicted_answer"] for attempt in r["attempts"]],
            )
            print("---")


def main(args):
    """
    Main driver:
    1) Load the Tidal model/tokenizer
    2) Load the 2024 AIME dataset
    3) Evaluate pass@N
    4) Print final analysis
    """
    # Load Tidal model + tokenizer
    logger.info("Loading Tidal model and tokenizer...")
    model, tokenizer = load(
        args.model_name,
        attn_type=getattr(args, "attn_type", None),
        top_k=getattr(args, "top_k", 128),
        sparse_layer_start=getattr(args, "sparse_layer_start", 2),
        correction_layer=getattr(args, "correction_layer", 13),
    )

    # Prepare results file
    os.makedirs("results", exist_ok=True)
    n_attempts = args.n
    top_k = args.top_k if args.top_k else ""
    results_file = f"aime_{args.attn_type}_{top_k}_{args.model_name.replace('/', '_')}_pass_at_{n_attempts}.json"

    # Load the dataset (30 problems from 2024)
    dataset = load_2024_dataset()

    existing_results = load_existing_results(results_file)
    processed_indexes = {r["index"] for r in existing_results}

    # Evaluate each problem
    for item in tqdm(dataset, desc="Evaluating problems"):
        problem_id = int(item["id"])
        if problem_id in processed_indexes:
            # Already processed
            continue

        problem_text = item["problem"]
        correct_answer = int(item["answer"])

        # Make n attempts
        attempts = make_n_attempts(
            problem_text, model, tokenizer, n_attempts, max_tokens=args.max_gen_len
        )
        is_correct, first_correct = evaluate_pass_at_n(attempts, correct_answer)

        # Record result
        result_dict = {
            "index": problem_id,
            "problem": problem_text,
            "attempts": attempts,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "first_correct_attempt": first_correct,
        }
        save_result(results_file, result_dict)

    # Final analysis
    final_results = load_existing_results(results_file)
    analyze_results(final_results, n_attempts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Tidal model on AIME 2024 problems"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name or local path for Tidal decode.",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of attempts per problem (pass@n)."
    )

    # Additional Tidal arguments if needed
    parser.add_argument(
        "--attn_type", type=str, default=None, help="Attention type (e.g., tidal)."
    )
    parser.add_argument("--top_k", type=int, default=128, help="Top-k for Tidal decode")
    parser.add_argument(
        "--sparse_layer_start",
        type=int,
        default=2,
        help="Layer from which to start sparse attention.",
    )
    parser.add_argument(
        "--correction_layer",
        type=int,
        default=13,
        help="Layer used for corrections in Tidal decode.",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=8192,
        help="Maximum tokens to generate per problem (also replaces '4000' in system prompt).",
    )

    args = parser.parse_args()

    # If you allow 'None' to be passed as a string on CLI:
    if args.attn_type == "None":
        args.attn_type = None

    main(args)
