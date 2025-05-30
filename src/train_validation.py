import json
import os

from easyrag.pipeline.pipeline import EasyRAGPipeline
from submit import submit
import fire
from tqdm.asyncio import tqdm
from easyrag.pipeline.qa import read_jsonl, save_answers, write_jsonl
from easyrag.utils import get_yaml_data


def get_train_data(limit=None):
    """读取训练数据和答案"""
    # 读取训练问题
    with open("/mnt/sda/users/andy/project/Rag/BUAA/EasyRAG/data/数据集A/train.json", 'r', encoding='utf-8') as f:
        train_questions = []
        for line in f:
            train_questions.append(json.loads(line.strip()))
    
    # 读取训练答案
    with open("/mnt/sda/users/andy/project/Rag/BUAA/EasyRAG/data/数据集A/train_answer.json", 'r', encoding='utf-8') as f:
        train_answers = []
        for line in f:
            train_answers.append(json.loads(line.strip()))
    
    # 创建答案字典，方便查找
    answer_dict = {ans['id']: ans for ans in train_answers}
    
    # 合并问题和答案
    queries = []
    for question in train_questions:
        query_id = question['id']
        if query_id in answer_dict:
            answer_info = answer_dict[query_id]
            query = {
                'id': query_id,
                'category': question['category'],
                'query': question['question'],
                'content': question.get('content', ''),
                'answer': answer_info['answer']
            }
            queries.append(query)
            
            # 如果设置了限制，达到限制后停止
            if limit and len(queries) >= limit:
                break
    
    return queries


def calculate_accuracy(predicted_answer, ground_truth, category):
    """计算准确率"""
    if category == "选择题":
        # 对于选择题，比较选项是否完全匹配
        if isinstance(ground_truth, list) and isinstance(predicted_answer, str):
            # 从预测答案中提取选项
            predicted_options = []
            for option in ['A', 'B', 'C', 'D', 'E']:
                if option in predicted_answer:
                    predicted_options.append(option)
            
            # 检查是否完全匹配
            return set(predicted_options) == set(ground_truth)
        return False
    else:
        # 对于问答题，使用关键词匹配
        if isinstance(ground_truth, str) and isinstance(predicted_answer, str):
            # 简单的关键词匹配策略
            gt_keywords = ground_truth.split()
            match_count = 0
            for keyword in gt_keywords:
                if len(keyword) > 1 and keyword in predicted_answer:
                    match_count += 1
            
            # 如果匹配的关键词超过一定比例，认为正确
            return match_count / len(gt_keywords) > 0.3
        return False


async def main(
        re_only=False,
        push=False,  # 是否直接提交这次test结果
        save_inter=True,  # 是否保存检索结果等中间结果
        note="train_validation",  # 中间结果保存路径的备注名字
        config_path="configs/easyrag.yaml",  # 配置文件
        limit=50,  # 限制处理的数据量，用于快速测试
):
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 如果config_path是相对路径，则相对于当前脚本目录
    if not os.path.isabs(config_path):
        config_path = os.path.join(current_dir, config_path)
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"脚本所在目录: {current_dir}")
        return
    
    # 读入配置文件
    config = get_yaml_data(config_path)
    config['re_only'] = re_only
    for key in config:
        print(f"{key}: {config[key]}")

    # 创建RAG流程
    rag_pipeline = EasyRAGPipeline(
        config
    )

    # 读入训练数据（限制数量以提高速度）
    queries = get_train_data(limit=limit)
    print(f"加载了 {len(queries)} 条训练数据")

    # 生成答案
    print("开始生成答案...")
    answers = []
    all_nodes = []
    all_contexts = []
    for query in tqdm(queries, total=len(queries)):
        res = await rag_pipeline.run(query)
        answers.append(res['answer'])
        all_nodes.append(res['nodes'])
        all_contexts.append(res['contexts'])

    # 处理结果
    print("处理生成内容...")
    os.makedirs("outputs", exist_ok=True)

    # 本地提交
    answer_file = f"outputs/submit_result_train_{note}.jsonl"
    formatted_answers = save_answers(queries, answers, answer_file)
    print(f"保存结果至 {answer_file}")

    # 计算准确率
    print("计算准确率...")
    total_correct = 0
    choice_correct = 0
    qa_correct = 0
    choice_total = 0
    qa_total = 0
    
    for i, (query, predicted) in enumerate(zip(queries, answers)):
        ground_truth = query['answer']
        category = query['category']
        
        is_correct = calculate_accuracy(predicted, ground_truth, category)
        
        if is_correct:
            total_correct += 1
            if category == "选择题":
                choice_correct += 1
            else:
                qa_correct += 1
        
        if category == "选择题":
            choice_total += 1
        else:
            qa_total += 1
    
    # 输出结果
    total_accuracy = total_correct / len(queries) * 100
    choice_accuracy = choice_correct / choice_total * 100 if choice_total > 0 else 0
    qa_accuracy = qa_correct / qa_total * 100 if qa_total > 0 else 0
    
    print(f"\n=== 验证结果 ===")
    print(f"总体准确率: {total_accuracy:.2f}% ({total_correct}/{len(queries)})")
    print(f"选择题准确率: {choice_accuracy:.2f}% ({choice_correct}/{choice_total})")
    print(f"问答题准确率: {qa_accuracy:.2f}% ({qa_correct}/{qa_total})")

    # 保存中间结果
    if save_inter:
        print("保存中间结果...")
        inter_res_list = []
        for i, (query, answer, nodes, contexts) in enumerate(zip(queries, answers, all_nodes, all_contexts)):
            # 安全地获取路径信息，避免KeyError
            paths = []
            know_paths = []
            
            for node in nodes:
                # 安全地获取file_path
                file_path = node.metadata.get('file_path', 'unknown')
                paths.append(file_path)
                
                # 安全地获取know_path，如果不存在则使用file_path
                know_path = node.metadata.get('know_path', file_path)
                know_paths.append(know_path)
            
            # 计算该条数据的准确性
            is_correct = calculate_accuracy(answer, query['answer'], query['category'])
            
            # 清理predicted_answer中的转义字符
            cleaned_answer = answer
            if query['category'] == "选择题":
                # 如果是选择题，尝试解析并清理答案格式
                try:
                    # 移除可能的转义字符
                    if cleaned_answer.startswith('"') and cleaned_answer.endswith('"'):
                        cleaned_answer = cleaned_answer[1:-1]  # 移除外层引号
                    cleaned_answer = cleaned_answer.replace('\\"', '"')  # 移除转义字符
                    
                    # 尝试解析为JSON以验证格式
                    import json as json_module
                    try:
                        parsed = json_module.loads(cleaned_answer)
                        if isinstance(parsed, list):
                            cleaned_answer = parsed  # 直接使用解析后的列表
                    except:
                        pass  # 如果解析失败，保持原样
                except:
                    pass  # 如果处理失败，保持原样
            
            inter_res = {
                "id": query['id'],
                "category": query['category'],
                "query": query['query'],
                "content": query.get('content', ''),
                "predicted_answer": cleaned_answer,
                "ground_truth": query['answer'],
                "is_correct": is_correct,
                "candidates": contexts,
                "paths": paths,
                "know_paths": know_paths,
                "quality": [0 for _ in range(len(contexts))],
                "score": 1 if is_correct else 0,
                "duplicate": 0,
            }
            inter_res_list.append(inter_res)
        
        inter_file = f"inter/train_{note}.json"
        os.makedirs("inter", exist_ok=True)
        with open(f"{inter_file}", 'w', encoding='utf-8') as f:
            f.write(json.dumps(inter_res_list, ensure_ascii=False, indent=4))
        print(f"保存中间结果至 {inter_file}")


if __name__ == "__main__":
    fire.Fire(main) 