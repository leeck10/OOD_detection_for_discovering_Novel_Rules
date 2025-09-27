import os
import openai
from datetime import datetime

openai.api_key = "API_KEY"

# File paths
prompt_file_path = "prompt_generate_hypotheses.txt"  
# prompt_file_path = "prompt_discover_new_rule.txt"  
output_file_path = f"rule_analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"  #

try:
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()
except FileNotFoundError:
    print(f"Prompt file not found: {prompt_file_path}")
    exit()

try:
    response = openai.ChatCompletion.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": "You are an expert NLI rule analyst."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        request_timeout=120
    )

    # Extract result text
    result = response["choices"][0]["message"]["content"].strip()

    # Print and save result
    print("GPT Response:\n")
    print(result)

    with open(output_file_path, "w", encoding="utf-8") as out_f:
        out_f.write(result)

    print(f"\nResult saved to: {output_file_path}")

except openai.error.OpenAIError as api_err:
    print(f"OpenAI API error: {api_err}")
except Exception as e:
    print(f"Unexpected error: {e}")
    print(f"Unexpected error: {e}")


