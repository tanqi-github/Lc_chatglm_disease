from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def query(self, question):
        with self._driver.session() as session:
            result = session.run(
                "MATCH (n:my_entity)-[:RELATIONSHIP]->(m:my_entity) WHERE n.property CONTAINS $label RETURN m.property",
                label=question
            )
            b=[record["m.property"] for record in result]
            return b
    def dele(self):
        with self._driver.session() as session:
            result = session.run(
                "MATCH (a)-[r]->(b) WITH a, b, TAIL (COLLECT (r)) as rr WHERE size(rr)>0 FOREACH (r IN rr | DELETE r)"
            )

# 使用示例
# knowledge_graph = KnowledgeGraph("bolt://localhost:7687", "neo4j", "xueyan134679")
#
# knowledge_graph.dele()
#
# question = "腰椎间盘突出"
# results = knowledge_graph.query(question)
# for result in results:
#     print(result)
