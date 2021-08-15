from pprint import pprint
from sklearn.metrics import precision_score, recall_score
from utils import get_rel_score_word2vecbase, load_graph, load_queries

threshold = 0.25
graph = load_graph()
queries = load_queries()

def graph_traversal(query, topic_node):
    answers = set()
    visited_nodes = set()
    nodes_to_visit = [topic_node]
    for current_node in nodes_to_visit:
        visited_nodes.add(current_node)
        for items in graph[current_node]:
            relation, node = items[0], items[1]
            if node not in visited_nodes:
                nodes_to_visit.append(node)
                visited_nodes.add(node)
            rel_score = get_rel_score_word2vecbase('ns:' + relation, query)
            if threshold < rel_score:
                answers.add(node)
    return answers

def check_performance(queries):
    total_scores, total_recalls = 0, 0
    for query in queries:
        q, topic_node, answers_json = query[1], query[2], query[5]
        actual_answers = [v['AnswerArgument'] for v in answers_json]

        answers = graph_traversal(q, topic_node)

        predicted_answers = [1 if a in answers else 0 for a in actual_answers]
        answers_vectorized = [1] * len(actual_answers)

        score = precision_score(predicted_answers, answers_vectorized)
        r_score = recall_score(predicted_answers, answers_vectorized)
        total_scores += score
        total_recalls += r_score
        print(score, r_score, len(answers_vectorized), f'\t{q}')
    avg_precision = total_scores / len(queries)
    avg_recall = total_recalls / len(queries)
    return avg_precision, avg_recall

avg_precision, avg_recall = check_performance(queries)

print(f'\naverage precision: {avg_precision}, average recall: {avg_recall}')