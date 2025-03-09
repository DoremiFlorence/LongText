from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer
import faiss
import torch
import numpy as np
import openai

client = openai.OpenAI(
    api_key="xxx",
    base_url="https://api.sambanova.ai/v1",
)


class MiniLMEmbedder:
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.embedding_dim = 384  # 保持与原模型一致

    def encode(self, texts):
        # 手动实现与SentenceTransformer相同的处理流程
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 实现mean-pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs.attention_mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        # 归一化处理
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy()

# 使用示例
embedder = MiniLMEmbedder()


class LongDialogueRAG:
    def __init__(self, max_history=20, chunk_size=3):
        # 初始化模型
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.generator = AutoModelForCausalLM.from_pretrained("gpt2")

        self.embedder = MiniLMEmbedder()

        # 对话记录配置
        self.dialogue_history = []
        self.max_history = max_history  # 最大历史记录数
        self.chunk_size = chunk_size  # 每个块包含的对话轮次

        # FAISS索引初始化
        self.embedding_dim = 384
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.context_chunks = []

    def _chunk_history(self):
        """将对话历史分成块"""
        chunks = []
        for i in range(0, len(self.dialogue_history), self.chunk_size):
            chunk = self.dialogue_history[i:i + self.chunk_size]
            chunk_text = "\n".join([f"{role}: {text}" for role, text in chunk])
            chunks.append(chunk_text)
        return chunks

    def _update_index(self, chunks):
        """更新FAISS索引"""
        embeddings = self.embedder.encode(chunks)
        if len(self.context_chunks) == 0:
            self.index.add(embeddings)
        else:
            self.index.add(embeddings)
        self.context_chunks.extend(chunks)

    def _retrieve_context(self, query, k=2):
        """检索相关上下文"""
        query_embed = self.embedder.encode([query])
        distances, indices = self.index.search(query_embed, k)

        contexts = []
        for i in indices[0]:
            if i < len(self.context_chunks):
                contexts.append(self.context_chunks[i])
        return "\n\n".join(contexts)

    def generate_response(self, user_input):
        # 1. 检索相关上下文
        if len(self.dialogue_history) > 0:
            chunks = self._chunk_history()
            self._update_index(chunks[-3:])  # 只更新最新块
            context = self._retrieve_context(user_input)
        else:
            context = ""

        # 2. 构建增强提示
        prompt = [
            {"role": "system", "content": "以下是对话上下文：\n" + context},
            {"role": "user", "content": user_input}
        ]
        # 3. 生成响应
        response = client.chat.completions.create(
            model='Meta-Llama-3.1-8B-Instruct',
            messages=prompt,
            temperature=0.8,
            top_p=0.8
        )
        response = response.choices[0].message.content

        # 4. 更新对话历史
        self.dialogue_history.append(("用户", user_input))
        self.dialogue_history.append(("助手", response))

        # 保持历史记录长度
        if len(self.dialogue_history) > self.max_history:
            self.dialogue_history = self.dialogue_history[-self.max_history:]

        return response


# 使用示例
rag = LongDialogueRAG(max_history=20, chunk_size=3)

conversation = [
    "巴黎有什么著名景点？",
    "法国的首都是哪里？",
    "德国有哪些汽车品牌？",
    "请比较一下巴黎和柏林的气候",
    "卢浮宫最著名的藏品是什么？"
]

for query in conversation:
    response = rag.generate_response(query)
    print(f"用户：{query}")
    print(f"助手：{response}\n")
