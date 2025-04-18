import argparse
import json
import os
import logging
import re
from typing import List, Dict, Optional
from datasets import load_dataset
from tqdm import tqdm
from src.utils import load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# (3, \\frac{\\pi}{2})
# \\left( 3, \\frac{\\pi}{2} \\right)
def get_math_prompt(problem_text: str) -> str:
    """
    Return a prompt string in the style recommended by the MATH-500 instructions.
    """
    return f"""Solve the following math problem efficiently and clearly. 
The last line of your response should be of the following format:
'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct'

Think step by step before answering.

{problem_text}
"""


def extract_boxed_answer(response: str) -> Optional[str]:
    """
    Attempt to capture the content within $...$ and \boxed{...} – allowing either
    one slash or two slashes, plus optional whitespace.

    Examples we handle:
    - Therefore, the final answer is: $\\boxed{(3, \\frac{\\pi}{2})}$. I hope it is correct
    - 'Therefore, the final answer is: $ \boxed{(3, \frac{\pi}{2})}$. I hope it is correct'
    """
    if not response:
        return None
    
    # Remove extra newlines/tabs, keep single spacing
    response = " ".join(response.split())

    # This pattern says:
    #   \$            # a literal '$'
    #   \s*           # optional whitespace
    #   \\?boxed      # either \boxed or \\boxed
    #   \s*           # optional whitespace
    #   {([^}]+)}     # capture everything inside { ... }
    #   \s*           # optional whitespace
    #   \$            # closing '$'
    #
    # The question mark after \\ matches "zero or one" backslash, capturing either "\boxed" or "\\boxed".
    # If your model always produces double backslashes, you can change this to '\\boxed'.
    #
    pattern = r"\\boxed\{((?:[^{}]|{[^{}]*})*)\}"
    matches = re.findall(pattern, response)
    # logger.info(f"response: {response}")
    logger.info(f"matches: {matches}")
    # exit()
    if matches:
        # Return the last match if multiple appear
        return matches[-1].strip()

    return None


def tidal_inference_single(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """
    Single Tidal inference call. Feeds the custom MATH prompt as the user's text.
    """
    full_prompt = f"USER: {prompt}\nASSISTANT: "
    input_tensor = tokenizer(full_prompt, return_tensors="pt", return_attention_mask=False)

    do_sample = (temperature > 0.0)
    outputs = model.generate(
        **input_tensor,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
    )
    new_tokens = outputs[0, input_tensor["input_ids"].shape[-1]:]
    out_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    logger.info(f"Generated {len(new_tokens)} tokens. (temp={temperature}, sampling={do_sample})")
    return out_text.strip()


def make_n_attempts(
    problem_text: str,
    model,
    tokenizer,
    n: int,
    max_tokens: int,
    temperature: float
) -> List[Dict]:
    """
    Make n attempts (pass@N) for a single MATH-500 problem. 
    Returns a list of dicts, each with 'attempt_number', 'response', 'predicted_answer'.
    """
    attempts = []
    for attempt_index in range(n):
        prompt = get_math_prompt(problem_text)
        response = tidal_inference_single(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        predicted = extract_boxed_answer(response)

        # For debugging, you can uncomment:
        # logger.debug(f"Full response:\n{response}")
        # logger.debug(f"Extracted answer: {predicted}")

        attempts.append(
            {
                "attempt_number": attempt_index + 1,
                "response": response,
                "predicted_answer": predicted,
            }
        )
    return attempts


def naive_compare(model_answer: str, gold_answer: str) -> bool:
    """
    Very naive string match that ignores whitespace. 
    MATH solutions can vary widely (equivalent expressions), 
    so you may want a more robust approach for real usage.
    """
    if not model_answer or not gold_answer:
        return False
    model_str = "".join(model_answer.split()).lower()
    gold_str = "".join(gold_answer.split()).lower()
    return model_str == gold_str


def evaluate_pass_at_n(attempts: List[Dict], gold_answer: str) -> (bool, Optional[int]):
    """
    Check if any attempt is exactly correct (naive string match).
    If so, return (True, attempt_number). Otherwise (False, None).
    """
    for att in attempts:
        if naive_compare(att["predicted_answer"], gold_answer):
            return True, att["attempt_number"]
    return False, None


def main(args):
    logger.info("Loading Tidal model and tokenizer...")
    model, tokenizer = load(
        args.model_name,
        attn_type=args.attn_type,
        top_k=args.top_k,
        sparse_layer_start=args.sparse_layer_start,
        correction_layer=args.correction_layer,
        attention_sink=args.attention_sink,
        most_recent_scale_factor=args.most_recent_scale_factor,
    )

    logger.info("Loading MATH-500 dataset (test split)...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", "default", split="test")
    logger.info(f"Loaded {len(dataset)} samples from MATH-500 test split.")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join(
        "results",
        f"math500_{args.attn_type}_{args.top_k}_{args.model_name.replace('/', '_')}_{args.temperature}_{args.most_recent_scale_factor}_{args.attention_sink}results.json"
    )

    # Load partial results if they exist
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        logger.info(f"Loaded existing partial results: {len(results)} entries")
    else:
        results = []

    processed_indices = set(r["index"] for r in results)

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating MATH-500"):
        if idx in processed_indices:
            # Skip if already done
            continue

        problem_text = item["problem"]
        gold_solution = item["solution"]  # The detailed solution text
        gold_answer = item["answer"]      # The short final "answer" field

        # Generate pass@N attempts
        attempts = make_n_attempts(
            problem_text,
            model,
            tokenizer,
            n=args.n,
            max_tokens=args.max_gen_len,
            temperature=args.temperature,
        )

        # Evaluate correctness (naive string match)
        is_correct, first_correct_attempt = evaluate_pass_at_n(attempts, gold_answer)

        # Store the result
        result = {
            "index": idx,
            "problem": problem_text,
            "gold_answer": gold_answer,
            "gold_solution": gold_solution,
            "attempts": attempts,
            "is_correct": is_correct,
            "first_correct_attempt": first_correct_attempt,
        }
        results.append(result)

        # Save partial results immediately, so we don't lose progress
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # Final analysis
    n_correct = sum(r["is_correct"] for r in results)
    total = len(results)
    acc = n_correct / total if total > 0 else 0.0

    logger.info(f"Pass@{args.n} accuracy (naive compare) on MATH-500: {acc:.2%}")
    print("\n=== MATH-500 Final Results ===")
    print(f"Evaluated {total} items, pass@{args.n} = {acc:.2%}  (Correct: {n_correct}/{total})")
    print(f"Results file: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Tidal model on MATH-500 dataset with partial results saving.")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or local path")
    parser.add_argument("--attn_type", type=str, default=None, help="Attention type (e.g. tidal)")
    parser.add_argument("--top_k", type=int, default=128, help="Top-k for Tidal decode")
    parser.add_argument("--sparse_layer_start", type=int, default=2, help="Sparse attention start layer")
    parser.add_argument("--correction_layer", type=int, default=13, help="Layer used for Tidal corrections")
    parser.add_argument("--attention_sink", type=int, default=0, help="Attention sink param")
    parser.add_argument("--most_recent_scale_factor", type=float, default=1, help="Scale factor param")
    parser.add_argument("--n", type=int, default=1, help="Number of attempts per problem (pass@N)")
    parser.add_argument("--max_gen_len", type=int, default=32768, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = greedy)")

    args = parser.parse_args()

    if args.attn_type == "None":
        args.attn_type = None

    main(args)
