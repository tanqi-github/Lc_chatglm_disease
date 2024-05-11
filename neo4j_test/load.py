import csv

# 全局变量保存自定义词典内容
custom_dict = set()

# 读取自定义词典文件并加载到内存中
def load_custom_dict(csv_file):
    global custom_dict
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key, value in row.items():
                if value:  # 如果值不为空，则将其作为关键词添加到自定义词典中
                    custom_dict.add(value)
    print("自定义词典文件已加载到内存中")

# 加载自定义词典文件到内存中
load_custom_dict('data.csv')

# 从内存中读取自定义词典内容
def get_custom_dict():
    return custom_dict

# 示例使用
def main():
    # 获取自定义词典内容
    words = get_custom_dict()
    print("自定义词典内容：", words)

if __name__ == "__main__":
    main()
