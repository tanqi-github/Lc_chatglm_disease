import jieba.analyse

question = "什么是心脏病？"

# 使用TextRank算法提取关键词
extracted_keywords = jieba.analyse.textrank(question, topK=5, withWeight=False, allowPOS=('n', 'nr', 'ns', 'nt', 'nz'))
print(extracted_keywords)
