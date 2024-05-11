import jieba.posseg as pseg

question = "心脏病和高血压有什么区别？"

# 使用jieba进行分词和词性标注
words = pseg.cut(question)

# 提取命名实体作为关键词
extracted_keywords = [word.word for word in words if word.flag.startswith("nr")]
print(extracted_keywords)
