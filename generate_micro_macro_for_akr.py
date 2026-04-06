import pandas as pd
import requests
import json
from tqdm import tqdm

# ==========================================
# 1. OLLAMA CONFIGURATION
# ==========================================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-v3.2:cloud" # Change this if you use mistral, qwen, etc.

def ask_ollama_judge(question, expected_fact, chatbot_answer):
    """Sends the evaluation prompt to the local Ollama API."""
    prompt = f"""You are a strict exact-match grader for a Knowledge Graph Question Answering system.
    
    Question: {question}
    Expected Fact: {expected_fact}
    Chatbot Answer: {chatbot_answer}
    
    Does the Chatbot Answer contain the core meaning of the Expected Fact? 
    Respond ONLY with "1" for Yes, or "0" for No. Do not write any other text."""
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0 # Strict, reproducible answers
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result_text = response.json()['response'].strip()
        
        # Parse the result (handling potential weird LLM outputs)
        if "1" in result_text:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Ollama API Error: {e}")
        return 0

# ==========================================
# 2. THE EVALUATOR FUNCTION (HYBRID)
# ==========================================
def evaluate_row(row):
    """Evaluates a single row using string matching first, then Ollama."""
    expected = str(row['expected_fact']).lower()
    answer = str(row['chatbot_answer']).lower()
    
    # Track if the bot actually tried to answer (for False Negatives)
    bot_answered = True
    if pd.isna(row['chatbot_answer']) or answer.strip() in ["", "i don't know", "error"]:
        bot_answered = False
        return 0, bot_answered
        
    # STEP 1: Fast Exact Match (Saves Ollama compute time)
    if expected in answer:
        return 1, bot_answered
        
    # STEP 2: Smart LLM Judge (If exact match fails)
    judge_score = ask_ollama_judge(row['question'], row['expected_fact'], row['chatbot_answer'])
    return judge_score, bot_answered

# ==========================================
# 3. METRICS CALCULATION (MICRO & MACRO)
# ==========================================
def calculate_p_r_f1(df_subset):
    """Calculates Precision, Recall, and F1 for a given dataframe."""
    # TP: Judge says 1, bot answered
    tp = len(df_subset[(df_subset['judge_score'] == 1) & (df_subset['bot_answered'] == True)])
    # FP: Judge says 0, bot answered
    fp = len(df_subset[(df_subset['judge_score'] == 0) & (df_subset['bot_answered'] == True)])
    # FN: Bot failed to answer / abstained
    fn = len(df_subset[df_subset['bot_answered'] == False])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return pd.Series({'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'Total_Questions': len(df_subset)})

# ==========================================
# 4. MAIN EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    # Load your dataset (Replace with your actual filename)
    # MUST contain: question, expected_fact, chatbot_answer, and predicate
    df = pd.read_excel("simplequestions_text_eval (1) (1).xlsx",sheet_name="Sheet2")
    
    # Ensure predicate column exists for Macro average
    if 'predicate' not in df.columns:
        print("WARNING: 'predicate' column is missing! Macro average cannot be calculated.")
        # If missing, you might need to merge it back from the original gold_answers.jsonl
    
    print(f"Starting evaluation of {len(df)} questions...")
    
    # Run the evaluation with a progress bar
    tqdm.pandas(desc="Evaluating")
    df[['judge_score', 'bot_answered']] = df.progress_apply(
        lambda row: pd.Series(evaluate_row(row)), axis=1
    )
    
    print("\nEvaluation Complete! Calculating Metrics...\n")
    
    # --- MICRO AVERAGE ---
    print("-------------------------------------------------")
    print(" MICRO-AVERAGE (Overall System Performance)")
    print("-------------------------------------------------")
    micro_metrics = calculate_p_r_f1(df)
    print(micro_metrics.round(4))
    
    # --- MACRO AVERAGE ---
    if 'predicate' in df.columns:
        print("\n-------------------------------------------------")
        print(" MACRO-AVERAGE (Average across all Relations)")
        print("-------------------------------------------------")
        # 1. Calculate metrics for each specific relation (P19, P57, etc.)
        relation_metrics = df.groupby('predicate').apply(calculate_p_r_f1)
        
        # 2. Average those metrics together
        macro_metrics = relation_metrics[['Precision', 'Recall', 'F1-Score']].mean()
        print(macro_metrics.round(4))
        
        # Optional: Save the detailed per-relation metrics to see where the bot is failing
        relation_metrics.to_csv("relation_metrics_breakdown.csv")
        print("\n(Saved detailed per-relation breakdown to 'relation_metrics_breakdown.csv')")

    # Save the final judged dataframe
    df.to_csv("judged_results_akr.csv", index=False)