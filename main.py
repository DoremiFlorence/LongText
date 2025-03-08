import openai

import spacy
from transformers import AutoTokenizer
from datasets import load_dataset

# 加载 spaCy 的预训练模型进行实体提取
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3000000
# 加载测试数据集
dataset = load_dataset("THUDM/LongBench-v2")

# 加载预训练的多选模型
model_name = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def extract_entities(context):
    """
    使用 spaCy 提取长文本中的实体。
    """
    doc = nlp(context)
    entities = [ent.text for ent in doc.ents]
    return entities


def process_data(sample):
    """
    根据数据样本，提取实体并简化输入。
    """
    # print(sample)
    context = sample["context"]
    question = sample["question"]
    choice_A = sample["choice_A"]
    choice_B = sample["choice_B"]
    choice_C = sample["choice_C"]
    choice_D = sample["choice_D"]

    # 提取实体
    entities = extract_entities(context)

    # 将提取的实体拼接成简化的上下文
    simplified_context = " ".join(entities) if entities else context[:512]
    # print("simplified:", simplified_context)
    # 构造模型输入
    choices = [choice_A, choice_B, choice_C, choice_D]
    inputs = tokenizer(
        [simplified_context] * len(choices),
        [question + " " + choice for choice in choices],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    messages = [{"role": "system",
                 "content": "you need to solve the question, and only answer your choice, ie. A or B or C or D."}] + [
                   {"role": "user", "content": question + " " + choice} for choice in choices
               ]
    # print("input", inputs)
    return messages


def evaluate_model(dataset):
    """
    在数据集上评估模型。
    """
    correct = 0
    total = 0

    for sample in dataset['train']:  # 假设使用测试集
        # print("**",sample)
        messages = process_data(sample)

        # 模型预测
        response = client.chat.completions.create(
            model='Meta-Llama-3.1-8B-Instruct',
            messages=messages,
            temperature=0.8,
            top_p=0.8
        )
        response = response.choices[0].message.content
        print("response:", response)
        print("correct:", sample["answer"])

        # 检查预测是否正确
        if response == sample["answer"]:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    client = openai.OpenAI(
        api_key="xxx",
        base_url="https://api.sambanova.ai/v1",
    )
    # 运行模型评估
    evaluate_model(dataset)
