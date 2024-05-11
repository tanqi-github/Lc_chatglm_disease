from neo4j import GraphDatabase
import torch
from transformers import AutoTokenizer, AutoModel
import jieba
import jieba.posseg as pseg

# 提取关键词
question = "腰椎间盘突出的别名是什么"

# Neo4j连接配置
uri = "bolt://localhost:7687"
user = "neo4j"
password = "xueyan134679"


# Neo4j连接
class Neo4jConnector:
    def __init__(self):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def run_query(self, query, **parameters):
        with self._driver.session() as session:
            result = session.run(query, **parameters)
            return result.data()


# 整合问题和查询结果
def integrate_question_and_results(question, results):
    integrated_prompt = f"问题：{question}\n"
    integrated_prompt += "回答：\n"
    for record in results:
        # integrated_prompt += f"{record['e1.name']} - {record['r'][1]} - {record['e2.name']}\n"
        integrated_prompt += f"{record['e2.name']}\n"
    return integrated_prompt


# 实例化Neo4j连接对象
neo4j_connector = Neo4jConnector()

# 加载模型
model_path = "/mnt/workspace/Langchain-Chatchat/chatglm3-6b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(device)
model = model.eval()


# 一个简单的示例函数，用于筛选名词、动词、形容词
def filter_useful_words(keywords):
    useful_words = []
    noun_words = []  # 名词
    for word, flag in keywords:
        if flag.startswith("n"):  # 名词
            useful_words.append(word)
            noun_words.append(word)
        elif flag.startswith("v"):  # 动词
            if word == "有" or word == "是":
                continue
            useful_words.append(word)
        elif flag.startswith("a"):  # 形容词
            useful_words.append(word)
    return useful_words, noun_words


def extract_keywords(tokenizer, device, model, question):
    # 使用标记器对问题进行标记和编码
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 获取模型输出
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 使用中文分词器分词
    keywords = list(jieba.cut(generated_text))

    # 使用词性标注工具进行词性标注
    words_with_flag = pseg.cut("".join(keywords))

    # 筛选出名词、动词、形容词
    useful_words, noun_words = filter_useful_words(words_with_flag)

    # 使用set去重
    useful_words = list(set(useful_words))
    noun_words = list(set(noun_words))
    return useful_words, noun_words


useful_words, noun_words = extract_keywords(tokenizer, device, model, question)

# 打印筛选出的关键词
print("筛选出的关键词：", useful_words)
print("筛选出的名词关键词：", noun_words)

# 构建查询语句
query = """
MATCH (e1)-[r]->(e2)
WHERE e1.name CONTAINS $entity1 AND type(r) CONTAINS $entity2
RETURN e1.name,r,e2.name
"""
for i in noun_words:
    for j in useful_words:
        if i == j:
            continue
        # 执行查询
        result = neo4j_connector.run_query(query, entity1=i, entity2=j)
        for record in result:
            relationship_type = record["e2.name"]
            question = f"{i}是否具有{j}？"
            answer = relationship_type
            # print(question)
            # print(answer)

        # 构造提示字符串
        prompt = integrate_question_and_results(question, result)
        # input ='{}\n\n{}'.format(question,prompt)
        # # print(input)
        # response, history = model.chat(tokenizer, input, history=[])
        # print(response)

        # 将输入文本编码成tokens
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # 生成回答
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

        # 解码生成的文本
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        print(response)

        if result != None:
            break

#         # 将提示字符串传递给模型，并获取答案
#         inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
#         inputs = {key: value.to(device) for key, value in inputs.items()}
#         with torch.no_grad():
#             output = model.generate(**inputs)

#         # 解码生成的文本
#         response = tokenizer.decode(output[0], skip_special_tokens=True)

#         # 打印模型生成的答案
#         print("模型生成的答案：", response)


