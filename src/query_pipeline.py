import os
import json
import asyncio
import torch
import numpy as np
from typing import List, Dict
from llama_index.core import Settings, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.llms import OpenAILike as OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from easyrag.utils import get_yaml_data
from easyrag.custom.retrievers import QdrantRetriever

# 初始化配置
config_path = "configs/easyrag.yaml"
config = get_yaml_data(config_path)

# 初始化 LLM
llm = OpenAI(
    api_key=config['llm_keys'][0],
    model=config['llm_name'],
    api_base="https://open.bigmodel.cn/api/paas/v4/",
    is_chat_model=True,
)

# 初始化 Embedding 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    torch.cuda.empty_cache()

embedding = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-zh-v1.5",
    cache_folder="./cache/models",
    embed_batch_size=256 if device == "cuda" else 32,
    device=device,
)

# 设置全局 embedding 模型
Settings.embed_model = embedding

class CustomQdrantRetriever:
    def __init__(self, collection_name: str, top_k: int = 5):
        self.collection_name = collection_name
        self.top_k = top_k
        self.client = AsyncQdrantClient(url="http://localhost:6333")
        
    async def search(self, query: str) -> List[Dict]:
        try:
            # 生成查询向量
            query_embedding = embedding.get_text_embedding(query)
            query_vector = np.array(query_embedding).astype(np.float32)
            
            # 执行向量搜索
            search_result = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=self.top_k,
                with_payload=True,
                with_vectors=False
            )
            
            # 格式化结果
            formatted_results = []
            for scored_point in search_result:
                formatted_results.append({
                    "text": scored_point.payload.get("text", ""),
                    "metadata": scored_point.payload.get("metadata", {}),
                    "score": float(scored_point.score)
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in search: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return []

async def main():
    try:
        # 读取测试文件
        test_file = "/mnt/sda/users/andy/project/Rag/BUAA/EasyRAG/data/数据集A/testA.json"
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    test_data.append(json.loads(line))
        
        print(f"Loaded {len(test_data)} questions from test file")
        
        # 初始化检索器
        retriever = CustomQdrantRetriever(collection_name="aiops25", top_k=5)
        print("Retriever initialized")
        
        # 存储结果
        results = []
        
        # 处理每个查询
        for item in test_data[:10]:  # 先测试前10个问题
            query_id = item['id']
            query_text = item['question']
            print(f"\nProcessing query {query_id}: {query_text}")
            
            # 获取相似文档
            search_results = await retriever.search(query_text)
            
            # 构建回答
            answer = {
                "id": query_id,
                "question": query_text,
                "answer": "",  # 这里需要根据检索结果生成答案
                "sources": search_results  # 直接使用格式化后的结果
            }
            
            results.append(answer)
            print(f"Found {len(search_results)} relevant documents for query {query_id}")
        
        # 保存结果
        output_file = "results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main()) 