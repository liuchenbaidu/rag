import os
import asyncio
import torch
import uuid
from qdrant_client.http.models import OptimizersConfigDiff

from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import TitleExtractor
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.legacy.llms import OpenAILike as OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from easyrag.pipeline.pipeline import (
    read_data,
    build_vector_store,
    build_pipeline,
    build_preprocess_pipeline,
)
from easyrag.utils import get_yaml_data

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

# 设置全局 embedding 模型和 LLM
Settings.embed_model = embedding
Settings.llm = llm

# 示例配置
config = {
    'data_path': '/mnt/sda/users/andy/project/Rag/BUAA/EasyRAG/data/赛题制度文档',
    'chunk_size': 512,
    'chunk_overlap': 50,
    'collection_name': 'aiops25',
    'qdrant_url': 'http://localhost:6333',
    'cache_path': './cache',
    'reindex': True,
    'vector_size': 1024,  # bge-large-zh-v1.5 的输出维度
    'split_type': 1,
}

def generate_point_id(node_id: str) -> str:
    """生成有效的点 ID"""
    # 使用 UUID5 基于节点 ID 生成稳定的 UUID
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, node_id))

async def main():
    try:
        data_path = os.path.abspath(config['data_path'])
        chunk_size = config['chunk_size']
        chunk_overlap = config['chunk_overlap']
        
        # 读取文档
        data = read_data(data_path)
        print(f"文档读入完成，一共有{len(data)}个文档")
        
        # 初始化向量存储
        collection_name = config['collection_name']
        client, vector_store = await build_vector_store(
            qdrant_url=config['qdrant_url'],
            cache_path=config['cache_path'],
            reindex=config['reindex'],
            collection_name=collection_name,
            vector_size=config['vector_size'],
        )
        
        # 检查集合信息
        collection_info = await client.get_collection(collection_name=collection_name)
        if collection_info.points_count == 0:
            # 创建文档处理管道
            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
            # 创建处理管道，暂时不使用 TitleExtractor
            pipeline = IngestionPipeline(transformations=[node_parser])
            
            # 暂停实时索引以提高性能
            await client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfigDiff(indexing_threshold=0),
            )
            
            # 处理文档
            nodes = await pipeline.arun(documents=data, show_progress=True)
            print(f"文档处理完成，生成了 {len(nodes)} 个节点")
            
            # 批量处理节点
            batch_size = 100
            for i in range(0, len(nodes), batch_size):
                batch_nodes = nodes[i:i + batch_size]
                points = []
                
                for node in batch_nodes:
                    # 确保节点有正确的文本内容
                    if not hasattr(node, 'text') or not node.text.strip():
                        print(f"Warning: Empty node text for node_id: {node.node_id}")
                        continue
                    
                    try:
                        # 生成嵌入向量
                        embedding_vector = embedding.get_text_embedding(node.text)
                        
                        # 生成有效的点 ID
                        point_id = generate_point_id(node.node_id)
                        
                        # 准备存储点
                        points.append({
                            'id': point_id,  # 使用 UUID 作为点 ID
                            'vector': embedding_vector,
                            'payload': {
                                'text': node.text,
                                'metadata': {
                                    'file_path': node.metadata.get('file_path', ''),
                                    'file_name': node.metadata.get('file_name', ''),
                                    'page_label': node.metadata.get('page_label', ''),
                                    'node_id': node.node_id  # 保存原始节点 ID
                                }
                            }
                        })
                    except Exception as e:
                        print(f"Error processing node {node.node_id}: {str(e)}")
                        continue
                
                # 批量存储到 Qdrant
                if points:
                    try:
                        await client.upsert(
                            collection_name=collection_name,
                            points=points
                        )
                        print(f"已处理 {i + len(points)}/{len(nodes)} 个节点")
                    except Exception as e:
                        print(f"Error upserting batch: {str(e)}")
                        print("First point ID in batch:", points[0]['id'] if points else "No points")
                        continue
            
            # 恢复实时索引
            await client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfigDiff(indexing_threshold=20000),
            )
            
            print(f"索引建立完成，共索引了 {len(nodes)} 个文档片段")
        
        print("处理完成")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())