# Total 20 Game: Pick 1, 2, or 3; whoever reaches/exceeds 20 loses
def minimax(total, turn, alpha, beta):
    # Base cases
    if total == 20:
        return 0
    elif total > 20:
        return -1 if turn else 1

    if turn:  # AI's turn, maximize score
        max_eval = -float('inf')
        for i in range(1, 4):
            eval = minimax(total + i, False, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:  # Human's turn, minimize score
        min_eval = float('inf')
        for i in range(1, 4):
            eval = minimax(total + i, True, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# ---------------- GAME LOOP ----------------
print("Welcome to the Total 20 Game!")
print("Rules: Take turns adding 1, 2, or 3 to the total. Whoever reaches or exceeds 20 loses.\n")

total = 0

while True:
    # -------- Human Move --------
    while True:
        try:
            human_move = int(input("Your move (1, 2, or 3): "))
            if human_move not in [1, 2, 3]:
                print("❌ Invalid move. Enter 1, 2, or 3.")
                continue
            break
        except ValueError:
            print("❌ Invalid input. Enter a number 1, 2, or 3.")

    total += human_move
    print(f"After your move, total = {total}")

    if total >= 20:
        print("🎉 You win!")
        break

    # -------- AI Move --------
    print("AI is thinking...")
    ai_move = 1
    max_eval = -float('inf')

    for i in range(1, 4):
        eval = minimax(total + i, False, -float('inf'), float('inf'))
        if eval > max_eval:
            max_eval = eval
            ai_move = i

    total += ai_move
    print(f"AI adds {ai_move}. Total = {total}")

    if total >= 20:
        print("💻 AI wins!")
        break

