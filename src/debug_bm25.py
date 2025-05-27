#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import jieba
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import SimpleDirectoryReader
from easyrag.pipeline.ingestion import get_node_content
from easyrag.custom.retrievers import tokenize_and_remove_stopwords

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_stopwords(file_path):
    """加载停用词列表"""
    if not os.path.exists(file_path):
        logger.error(f"停用词文件不存在: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f.readlines()]
        logger.info(f"成功加载 {len(stopwords)} 个停用词")
        return stopwords
    except Exception as e:
        logger.error(f"加载停用词失败: {str(e)}")
        return []

def read_test_document(path):
    """读取测试文档"""
    try:
        reader = SimpleDirectoryReader(
            input_dir=path,
            recursive=True,
            required_exts=[".docx"],
        )
        docs = reader.load_data()
        logger.info(f"成功读取 {len(docs)} 个文档")
        
        # 检查文档内容
        if docs:
            for i, doc in enumerate(docs[:3]):  # 修复这里的错误
                logger.info(f"文档 {i} 预览: {doc.text[:100]}...")
                logger.info(f"文档 {i} 元数据: {doc.metadata}")
        else:
            logger.warning("没有读取到任何文档")
            
        return docs
    except Exception as e:
        logger.error(f"读取文档失败: {str(e)}")
        return []

def test_tokenizer(tokenizer, text, stopwords):
    """测试分词器"""
    if not text:
        logger.warning("空文本无法进行分词测试")
        return
    
    # 原始分词结果
    words = list(tokenizer.cut(text[:200]))
    logger.info(f"分词结果 (前20个): {words[:20]}")
    logger.info(f"分词数量: {len(words)}")
    
    # 过滤停用词后的结果
    filtered_words = [word for word in words if word not in stopwords and word != ' ']
    logger.info(f"过滤停用词后 (前20个): {filtered_words[:20]}")
    logger.info(f"过滤后词数: {len(filtered_words)}")
    
    # 检查是否所有词都被过滤
    if not filtered_words:
        logger.error("所有词都被过滤掉了！")
        # 分析被过滤的词
        for word in words[:20]:
            if word in stopwords:
                logger.info(f"'{word}' 是停用词")
            elif word == ' ':
                logger.info(f"'{word}' 是空格")
            else:
                logger.info(f"'{word}' 被过滤的原因未知")

def create_test_nodes(docs):
    """从文档创建测试节点"""
    nodes = []
    for i, doc in enumerate(docs):
        node = TextNode(
            text=doc.text,
            metadata={
                "file_path": doc.metadata.get("file_path", ""),
                "file_name": doc.metadata.get("file_name", ""),
            }
        )
        nodes.append(node)
        if i < 3:  # 只记录前3个节点的日志
            logger.info(f"创建节点 {i}: 内容长度 {len(node.text)}")
    
    logger.info(f"总共创建了 {len(nodes)} 个节点")
    return nodes

def test_node_content_extraction(nodes, embed_type=0):
    """测试节点内容提取"""
    for i, node in enumerate(nodes[:3]):  # 只测试前3个节点
        node_with_score = NodeWithScore(node=node, score=1.0)
        content = get_node_content(node_with_score, embed_type)
        logger.info(f"节点 {i} 提取内容长度: {len(content)}")
        logger.info(f"节点 {i} 内容预览: {content[:100]}...")

def main():
    # 参数
    data_path = "../data/赛题制度文档"
    stopwords_path = "./data/hit_stopwords.txt"
    embed_type = 0  # 默认使用原始内容
    
    # 1. 加载停用词
    stopwords = load_stopwords(stopwords_path)
    
    # 2. 初始化分词器
    tokenizer = jieba.Tokenizer()
    
    # 3. 读取测试文档
    docs = read_test_document(data_path)
    if not docs:
        logger.error("没有文档可供测试")
        return
    
    # 4. 创建测试节点
    nodes = create_test_nodes(docs)
    
    # 5. 测试节点内容提取
    test_node_content_extraction(nodes, embed_type)
    
    # 6. 测试分词器
    if nodes:
        node_with_score = NodeWithScore(node=nodes[0], score=1.0)
        content = get_node_content(node_with_score, embed_type)
        test_tokenizer(tokenizer, content, stopwords)
    
    # 7. 测试完整的分词和过滤流程
    corpus = []
    empty_count = 0
    empty_content_count = 0
    empty_tokens_count = 0
    
    for i, node in enumerate(nodes):
        node_with_score = NodeWithScore(node=node, score=1.0)
        content = get_node_content(node_with_score, embed_type)
        if not content:
            empty_content_count += 1
            empty_count += 1
            continue
            
        tokens = tokenize_and_remove_stopwords(tokenizer, content, stopwords=stopwords)
        if not tokens:
            empty_tokens_count += 1
            empty_count += 1
        corpus.append(tokens)
        
        if i < 3:  # 只记录前3个节点的日志
            logger.info(f"节点 {i} 分词结果: {tokens[:20] if tokens else 'EMPTY'}")
    
    non_empty_docs = len([c for c in corpus if c])
    
    logger.info(f"处理后空文档数: {empty_count}/{len(nodes)}")
    logger.info(f"其中空内容文档数: {empty_content_count}")
    logger.info(f"其中空分词结果文档数: {empty_tokens_count}")
    logger.info(f"处理后非空文档数: {non_empty_docs}/{len(corpus)}")
    logger.info(f"非空文档百分比: {non_empty_docs/len(corpus)*100:.2f}%")
    
    # 8. 分析停用词问题
    if empty_tokens_count > 0:
        logger.info("分析停用词问题...")
        # 随机选择10个节点进行分析
        import random
        sample_nodes = random.sample(nodes, min(10, len(nodes)))
        
        for i, node in enumerate(sample_nodes):
            node_with_score = NodeWithScore(node=node, score=1.0)
            content = get_node_content(node_with_score, embed_type)
            if not content:
                continue
                
            words = list(tokenizer.cut(content[:200]))
            filtered_words = [word for word in words if word not in stopwords and word != ' ']
            
            logger.info(f"样本 {i}: 原始词数 {len(words)}, 过滤后词数 {len(filtered_words)}")
            if len(filtered_words) == 0 and len(words) > 0:
                logger.info(f"样本 {i} 所有词都被过滤: {words[:20]}")

if __name__ == "__main__":
    main() 