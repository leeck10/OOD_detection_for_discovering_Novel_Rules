from sentence_transformers import SentenceTransformer, util
import numpy as np

original_hypotheses = [
    "A boy is bored outdoors",
    "The man is asian",
    "A man fell asleep on a bench because he was drunk",
    "The man has made a lot of money",
    "The man is fats",
    "boy is crying",
    "The man is homeless",
    "A boy confirms he finds rock climbing easy",
    "the children are white",
    "The child she is holding is not hers",
    "The woman is homeless",
    "She is angry",
    "The boy is hurt",
    "The boy only skateboards at night",
    "the horse is leaping to see his girlfriend",
    "A man is very athletic",
    "THe man is fishing",
    "The man is good at guitar",
    "the man is a spy",
    "The driver is bored"
]

generated_hypotheses = [
    "The boy is curious about nature",
    "The man is highly focused and skilled",
    "The man is exhausted from work",
    "The man is a street performer earning money",
    "The man is a construction worker",
    "The boy is shy and nervous",
    "The man is struggling financially",
    "The boy is confident in his climbing skills",
    "The children are energetic and playful",
    "The child feels safe in her arms",
    "The woman is living in poverty",
    "She is feeling anxious about her wedding",
    "The boy is emotionally distressed",
    "The boy enjoys the freedom of night rides",
    "The horse is well-trained and competitive",
    "The man is physically very fit",
    "The man is preparing to fish",
    "The man is an experienced street musician",
    "The man is a motorcycle enthusiast",
    "The driver is tired after a long shift"
]


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


original_embeddings = model.encode(original_hypotheses, convert_to_tensor=True)
generated_embeddings = model.encode(generated_hypotheses, convert_to_tensor=True)


semantic_similarities = [
    util.pytorch_cos_sim(original_embeddings[i], generated_embeddings[i]).item()
    for i in range(len(original_hypotheses))
]


print("개별 유사도 결과:")
for i in range(len(original_hypotheses)):
    print(f"{i+1}. \"{original_hypotheses[i]}\" <-> \"{generated_hypotheses[i]}\" => 유사도: {semantic_similarities[i]*100:.2f}%")

# 평균 유사도
average_similarity = np.mean(semantic_similarities) * 100
print(f"\n전체 평균 의미적 유사도: {average_similarity:.2f}%")
