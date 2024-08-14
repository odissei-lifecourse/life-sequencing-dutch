from src.evaluation.get_hops_from_ground_truth import bfs

def test_bfs():
  adjacency_dict = {
  'A': ['B', 'C'],
  'B': ['A', 'C'],
  'C': ['B', 'A'],
  }

  result_old = bfs('A', 3, adjacency_dict, use_old=True)
  result_new = bfs('A', 3, adjacency_dict, use_old=False)
  print(result_old)
  print(result_new)

test_bfs()