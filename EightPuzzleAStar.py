import numpy as np
from queue import PriorityQueue

class State:
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent

    def __lt__(self, other):
        return False

class Puzzle:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state

    def print_state(self, state):
        print(state)
        print()

    def is_goal(self, state):
        return np.array_equal(state, self.goal_state)

    def get_possible_moves(self, state):
        possible_moves = []
        zero_pos = np.where(state == 0)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # L R U D

        for d in directions:
            new_pos = (zero_pos[0] + d[0], zero_pos[1] + d[1])
            if 0 <= new_pos[0] < 3 and 0 <= new_pos[1] < 3:
                new_state = np.copy(state)
                new_state[zero_pos], new_state[new_pos] = new_state[new_pos], new_state[zero_pos]
                possible_moves.append(new_state)

        return possible_moves

    def heuristic(self, state):
        return np.count_nonzero(state != self.goal_state)

    def solve(self):
        pq = PriorityQueue()
        pq.put((0, State(self.initial_state, None)))
        visited = set()

        while not pq.empty():
            _, current = pq.get()

            if self.is_goal(current.state):
                return current

            for move in self.get_possible_moves(current.state):
                move_str = str(move)
                if move_str not in visited:
                    visited.add(move_str)
                    pq.put((self.heuristic(move), State(move, current)))
        return None

# ---------------- USER INPUT ----------------
def get_matrix(name):
    print(f"\nEnter {name} state (use space, 0 for blank):")
    matrix = []
    for i in range(3):
        row = list(map(int, input(f"Row {i+1}: ").split()))
        matrix.append(row)
    return np.array(matrix)

initial_state = get_matrix("INITIAL")
goal_state = get_matrix("GOAL")

puzzle = Puzzle(initial_state, goal_state)
solution = puzzle.solve()

# ---------------- OUTPUT ----------------
if solution:
    steps = []
    while solution:
        steps.append(solution.state)
        solution = solution.parent

    print("\n✅ Solution Path:\n")
    for i, step in enumerate(reversed(steps)):
        print(f"Move {i}:")
        puzzle.print_state(step)

    print("Total number of moves:", len(steps) - 1)
else:
    print("❌ No solution found.")

