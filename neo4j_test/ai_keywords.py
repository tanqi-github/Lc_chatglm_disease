from transformers import AutoTokenizer, AutoModel
import torch
import jieba
import jieba.posseg as pseg


def extract_keywords(question):
    # 加载模型
    model_path = "/mnt/workspace/Langchain-Chatchat/chatglm3-6b"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(device)
    model = model.eval()

    # question="我有点腰疼，是什么原因导致的？"

    # prompt='从上文中，提取"信息"(keyword,content)，包括:"疾病名称"、"症状"、"要求"等类型的实体，输出json格式内容'
    # input ='{}\n\n{}'.format(question,prompt)
    # print(input)
    # response, history = model.chat(tokenizer, input, history=[])
    # print(response)

    # # 要提取关键词的问题
    # question = "这是一个关于腰椎间盘突出的问题，如何治疗？"

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

    # 使用词性标注工具进行词性标注
    words_with_flag = pseg.cut("".join(keywords))

    # 筛选出名词、动词、形容词
    useful_words, noun_words = filter_useful_words(words_with_flag)

    # 使用set去重
    useful_words = list(set(useful_words))
    noun_words = list(set(noun_words))

    # 打印筛选出的关键词
    print("筛选出的关键词：", useful_words)
    print("筛选出的名词关键词：", noun_words)
    return useful_words, noun_words

#     # 一个简单的示例函数，用于筛选名词、动词、形容词
#     def filter_useful_words(keywords):
#         noun_words = []#名词
#         verb_words = []#动词
#         for word, flag in keywords:
#             if flag.startswith("n"):  # 名词
#                 noun_words.append(word)
#             elif flag.startswith("v"):  # 动词
#                 verb_words.append(word)
#             # elif flag.startswith("a"):  # 形容词
#             #     useful_words.append(word)
#         return noun_words,verb_words

#     # 使用词性标注工具进行词性标注
#     words_with_flag = pseg.cut("".join(keywords))

#     # 筛选出名词、动词、形容词
#     noun_words,verb_words = filter_useful_words(words_with_flag)

#     # 使用set去重
#     noun_words = list(set(noun_words))
#     verb_words = list(set(verb_words))

#     # 打印筛选出的关键词
#     print("筛选出的名词关键词：", noun_words)
#     print("筛选出的动词关键词：", verb_words)
#     return noun_words,verb_words