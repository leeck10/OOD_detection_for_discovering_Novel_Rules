import pandas as pd
from sklearn.model_selection import train_test_split

# 파일 읽기
file_path = "NLI_dataset_using_LLM/transformation_rules/19rules_ml_19000.txt"  # txt 파일 경로
data = pd.read_csv(file_path, sep="\t", header=None, names=["Label", "Premise", "Hypothesis"])

# 라벨 균형 확인
print("Original label distribution:")
print(data["Label"].value_counts())

# 데이터 라벨별로 (8:2 나누기)
train_data, val_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["Label"] 
)

print("\nTrain label distribution:")
print(train_data["Label"].value_counts())

print("\nValidation label distribution:")
print(val_data["Label"].value_counts())

# file save
train_data.to_csv("hjy/NLI_dataset_using_LLM/transformation_rules/19rules_ml_train_data.txt", sep="\t", index=False, header=False)
val_data.to_csv("hjy/NLI_dataset_using_LLM/transformation_rules/19rules_ml_val_data.txt", sep="\t", index=False, header=False)

print("\nDataset split completed. Files saved as 'train_dataset.txt' and 'validation_dataset.txt'.")
