import json

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
    for i, q in enumerate(questions):
        if i >= 100: 
            break
        q_index = q.get("index")
        ground_truth = q.get("correct_answer")
        correct_count = 0
        for i, q_attempt in enumerate(q.get("attempts")):
            predicted = q_attempt.get("predicted_answer")
            if predicted == ground_truth:
                correct_count += 1
        results[q_index] = correct_count/n

    return results

if __name__ == "__main__":
    # Replace the filename below with the path to your attached JSON file.
    json_filename = 'aime_None_1_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_0.6.json'
    correct_predictions = extract_correct_counts(json_filename)
    
    
    # for q_index, count in correct_predictions.items():
    #     print(f"Question {q_index}: {count} correct predicted answer(s)")
    print(f"overall avg accuracy: {sum(correct_predictions.values())/len(correct_predictions)}")
    
