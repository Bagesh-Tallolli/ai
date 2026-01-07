import numpy as np

def hill_climbing(func, start, step_size=0.01, max_iterations=1000):
    current_position = start
    current_value = func(current_position)

    for _ in range(max_iterations):
        next_pos = current_position + step_size
        prev_pos = current_position - step_size

        next_val = func(next_pos)
        prev_val = func(prev_pos)

        if next_val > current_value:
            current_position = next_pos
            current_value = next_val
        elif prev_val > current_value:
            current_position = prev_pos
            current_value = prev_val
        else:
            break

    return current_position, current_value

# ---------------- USER INPUT ----------------
print("Hill Climbing Algorithm")
print("Allowed functions: sin, cos, tan, exp, log, x**2, etc.")
print("Example: -(x-2)**2 + 4\n")

while True:
    func_str = input("Enter a function of x: ")

    try:
        # Safe evaluation environment
        def func(x):
            return eval(func_str, {"x": x, "np": np,
                                  "sin": np.sin, "cos": np.cos,
                                  "tan": np.tan, "exp": np.exp,
                                  "log": np.log})

        # Test function
        func(0)
        break
    except Exception:
        print("❌ Invalid function. Try again.\n")

while True:
    try:
        start = float(input("Enter starting value: "))
        break
    except ValueError:
        print("❌ Please enter a valid number.")

# Run hill climbing
maxima, max_value = hill_climbing(func, start)

print("\n✅ Result")
print(f"Maxima at x = {maxima:.4f}")
print(f"Maximum value = {max_value:.4f}")

