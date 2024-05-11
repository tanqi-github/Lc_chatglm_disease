from neo4j import GraphDatabase
from ai_keywords import extract_keywords

# Neo4j连接配置
uri = "bolt://localhost:7687"
user = "neo4j"
password = "xueyan134679"


class Neo4jConnector:
    def __init__(self):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def run_query(self, query, **parameters):
        with self._driver.session() as session:
            result = session.run(query, **parameters)
            return result.data()

# 实例化Neo4j连接对象
neo4j_connector = Neo4jConnector()

# 提取关键词
question = "腰椎间盘突出的别名是什么"
keywords,noun_words = extract_keywords(question)

# 构建查询语句
query = """
MATCH (e1)-[r]->(e2)
WHERE e1.name CONTAINS $entity1 AND type(r) CONTAINS $entity2
RETURN e2
"""
for i in noun_words:
    for j in keywords:
        if i==j:
            continue
        # 执行查询
        result = neo4j_connector.run_query(query, entity1=i,entity2=j)
        for record in result:
            relationship_type = record["e2"]
            print(f"{i}是否具有{j}？")
            print(relationship_type["name"])
