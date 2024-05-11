# import jieba.analyse

# question = "腰椎间盘突出的别名是什么？"

# #名词 (n)人名 (nr)地名 (ns)机构名 (nt)其他专名 (nz)动词 (v)形容词 (a)副词 (d)区别词 (b)时间词 (t)数词 (m)数量词 (q)代词 (r)处所词 (s)方位词 (f)介词 (p)连词 (c)助词 (u)语气词 (y)叹词 (e)拟声词 (o)状态词 (z)简称略语 (j)习用语 (l)成语 (i)前接成分 (h)后接成分 (k)符号 (x)
# # 使用TF-IDF算法提取关键词
# #名词（'n'）、人名（'nr'）、地名（'ns'）、机构名（'nt'）、其他专名（'nz'）
# extracted_keywords = jieba.analyse.extract_tags(question, topK=5, withWeight=False, allowPOS=('n', 'nz', 'v', 'a'))
# print(extracted_keywords)


import jieba
from jieba import analyse

# # 加载自定义词典
# jieba.load_userdict("/mnt/workspace/Langchain-Chatchat/webui_pages/custom_dict.txt")

# # 示例文本
# text = "我患有腰椎间盘突出别名椎间盘突出，挂号科室是骨科中医科？"

# # 使用 TF-IDF 提取关键词
# keywords = analyse.extract_tags(text, topK=5, withWeight=False, allowPOS=())

# print("提取的关键词：", keywords)

# 加载自定义词典
# jieba.load_userdict("/mnt/workspace/Langchain-Chatchat/webui_pages/custom_dict_simple.txt")

# # 示例文本
# text = "腰椎间盘突出的别名是什么？"

# # 使用 TF-IDF 提取关键词
# keywords = analyse.extract_tags(text, topK=5, withWeight=False, allowPOS=())

# print("提取的关键词：", keywords)

# 加载自定义词典
jieba.load_userdict("/mnt/workspace/Langchain-Chatchat/webui_pages/custom_dict_simple.txt")

# 示例文本
text = "我患有腰椎间盘突出别名椎间盘突出，挂号科室是骨科中医科"

# 使用 TF-IDF 提取关键词
keywords = analyse.extract_tags(text, topK=2, withWeight=False, allowPOS=())

print("提取的关键词：", keywords)

