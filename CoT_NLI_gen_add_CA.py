import os
import re
import openai
import json

# API 키 설정 (환경 변수 사용 권장)
openai.api_key = "API key"  # 실제 키 

# 파일 경로 정의
data_file = '/home/hjy/NLI_dataset_using_LLM/data/snli/snli_train.json'  # SNLI 데이터셋 경로
prompt_file = '/home/hjy/NLI_dataset_using_LLM/prompt_add_hjy/CA_prompt.txt'  # 프롬프트 파일 경로
output_file = '/home/hjy/NLI_dataset_using_LLM/hjy_add/CA2.txt'  # 출력 파일 경로

# 레이블 변수 정의
entailment = 'entailment'
contradiction = 'contradiction'
neutral = 'neutral'

# 데이터 및 프롬프트 파일 읽기
with open(data_file, 'r') as data_f, open(prompt_file, 'r') as prompt_f:
    json_data = json.load(data_f)
    prompt_lines = [line.strip() for line in prompt_f]

# 출력 파일 열기 (쓰기 모드)
with open(output_file, 'w') as output_f:
    # 생성할 데이터 범위 반복
    for i in range(38000, 38050):  # 생성할 데이터 수 조정 가능 25000~26000, 38000~38050
        try:
            # 문장 정리 및 포맷팅
            senten = re.sub(r'[\n*.,"\'-?:!;]', '', str(json_data[i]))

            # ChatCompletion을 사용해 응답 생성
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # 사용하고자 하는 모델
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

            # 응답 출력 (디버깅용)
            print(f"Response for index {i}: {response}")

            # 응답 처리
            result_text = response['choices'][0]['message']['content'].strip()
            
            # {} 사이의 텍스트 추출
            answer_match = re.search(r'\{(.+?)\}', result_text)
            if answer_match:
                answer = answer_match.group(1)
                # 결과를 출력 파일에 쓰기
                output_f.write(f"{neutral}\t{senten.replace('.', '')}\t{answer}\n")
            else:
                print(f"No valid answer found for index {i}")

        except openai.error.OpenAIError as api_error:
            print(f"OpenAI API error processing index {i}: {api_error}")
        except Exception as e:
            print(f"General error processing index {i}: {e}")
