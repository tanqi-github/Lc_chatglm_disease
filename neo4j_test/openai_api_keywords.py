# import openai
# openai.api_key = "EMPTY" # Not support yet
# openai.api_base = "http://localhost:8888/v1"

# model = "chatglm3-6b"

# # create a chat completion
# completion = openai.Completion.create(
#   model=model,
#   messages=[{"role": "user", "content": "Hello! What is your name?"}]
# )
# # print the completion
# print(completion.choices[0].message.content)

import openai

# 设置你的 OpenAI API 密钥
openai.api_key = 'your-api-key'

question = "心脏病的预防方法有哪些？"

# 使用 OpenAI API 请求生成文本
response = openai.Completion.create(
    engine="text-davinci-002",  # 选择一个中文支持的引擎
    prompt=question,
    max_tokens=50,  # 控制生成文本的长度
    stop="\n",  # 指定停止字符，确保生成文本的完整性
)

# 从生成的文本中提取关键词
generated_text = response.choices[0].text.strip()
# 这里可以使用任何适合中文文本的关键词提取方法，比如jieba库等
keywords = jieba.analyse.extract_tags(generated_text, topK=5)
print(keywords)
