import os
import re
import openai
import json

openai.api_key = "API KEY"  # Enter your OpenAI key

# Define file paths
data_file = 'NLI_dataset_using_LLM/data/snli/snli_train_original.json'  # SNLI dataset path
prompt_file = 'NLI_dataset_using_LLM/prompt/Rule_prompt.txt'  # Prompt file path
output_file = 'NLI_dataset_using_LLM/data/Rule.txt'  # Output file path

# Extract filename without extension for labeling
output_label = os.path.splitext(os.path.basename(output_file))[0]  

# Read data and prompt files
with open(data_file, 'r') as data_f, open(prompt_file, 'r') as prompt_f:
    json_data = json.load(data_f)
    prompt_lines = [line.strip() for line in prompt_f]

# Open output file in write mode
with open(output_file, 'w') as output_f:
    for i in range(0, 1000):  # Adjustable data range 
        try:
            # Clean and format the sentence
            senten = re.sub(r'[\n*.,"\'-?:!;]', '', str(json_data[i]))

            # Generate response using ChatCompletion
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Model to be used
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{' '.join(prompt_lines)} What is the answer when the input sentence is {senten}?"}
                ],
                temperature=0,
                max_tokens=300,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                request_timeout=120
            )

            print(f"Response for index {i}: {response}")

            # Process response
            result_text = response['choices'][0]['message']['content'].strip()
            
            # Extract text
            answer_match = re.search(r'\{(.+?)\}', result_text)
            if answer_match:
                answer = answer_match.group(1)
                # Write the result to the output file
                output_f.write(f"neutral_{output_label}\t{senten.replace('.', '')}\t{answer}\n")
            else:
                print(f"No valid answer found for index {i}")

        except openai.error.OpenAIError as api_error:
            print(f"OpenAI API error at index {i}: {api_error}")
        except Exception as e:
            print(f"General error at index {i}: {e}")