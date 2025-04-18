import json

from transformers import (
            AutoTokenizer,
        )
from src.models.llama_tidaldecoding import (
                LlamaForCausalLM
        )

tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            trust_remote_code=True,
        )

def extract_correct_counts(json_file):
    """
    Reads the JSON file and for each question extracts how many predicted answers
    are considered correct. In this implementation we assume that if a question
    has 'is_correct'==True (and a non-null 'first_correct_attempt'), then exactly
    one attempt is correct.
    """
    with open(json_file, 'r') as f:
        questions = json.load(f)
    
    results = {}
    n = len(questions[0].get("attempts"))
    total_gen_token = 0
    total_correct_gen_token = 0
    global_correct_attempt_cnt = 0
    global_attempt_cnt = 0
    print(f"total no. {len(questions)}")
    for i, q in enumerate(questions):
        if i > 100: 
            break
        q_index = q.get("index")
        ground_truth = q.get("correct_answer")
        correct_count = 0
        for i, q_attempt in enumerate(q.get("attempts")):
            predicted = q_attempt.get("predicted_answer")
            response_text = q_attempt.get("response")
            tokenized = tokenizer(response_text)
            generated_len = len(tokenized["input_ids"])
            total_gen_token += generated_len
            if predicted == ground_truth:
                total_correct_gen_token += generated_len
                correct_count += 1
                global_correct_attempt_cnt += 1
            global_attempt_cnt += 1
        results[q_index] = correct_count/n
    print(global_attempt_cnt, global_correct_attempt_cnt)
    print(f"average_gen_len: {total_gen_token / global_attempt_cnt}; average_correct_gen_len: {total_correct_gen_token/ global_correct_attempt_cnt}")
    print()
    return results

if __name__ == "__main__":
    # Replace the filename below with the path to your attached JSON file.
    json_filenames_union = [
    'data/aime_None_1_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_0.6.json', 
    'data/aime_tidal_4096_base.json',
    'data/aime_tidal_512_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_0.6.json',
    'data/aime_tidal_1024_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_0.6.json',
    'data/aime_tidal_2048_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_0.6.json',
    'data/aime_tidal_4096_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_0.6.json',
    ]

    json_filenames_union_streaming = [
    'results/aime/aime_tidal_512_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_0.6_4_2.0.json',
    '/home/artij/TidalDecode/aime_tidal_1024_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_0.6_4_2.0.json',
    'results/aime/aime_tidal_2048_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_0.6_4_2.0.json',
    'results/aime/aime_tidal_4096_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_0.6_4_2.0.json',
    ]

    json_filenames = json_filenames_union if False else json_filenames_union_streaming


    for json_filename in json_filenames:
        print(json_filename[20:40])
        correct_predictions = extract_correct_counts(json_filename)
        
        
        # for q_index, count in correct_predictions.items():
        #     print(f"Question {q_index}: {count} correct predicted answer(s)")
        # print(f"overall avg accuracy: {round(sum(correct_predictions.values())/len(correct_predictions),4)}\n")
    