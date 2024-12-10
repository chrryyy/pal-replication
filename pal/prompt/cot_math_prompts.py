MATH_PROMPT = '''
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

# Step-by-step reasoning:
# 1. Olivia starts with $23.
# 2. She buys 5 bagels, and each costs $3.
# 3. The total cost of the bagels is 5 * 3 = $15.
# 4. To find how much money she has left, subtract the total cost from her initial amount: 23 - 15.
# 5. The remaining money is $8.
Final answer is: 8


Q: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?

# Step-by-step reasoning:
# 1. Michael starts with 58 golf balls.
# 2. On Tuesday, he loses 23 golf balls.
# 3. On Wednesday, he loses 2 more golf balls.
# 4. The total number of golf balls he loses is 23 + 2 = 25.
# 5. To find how many golf balls he has left, subtract the total lost from the initial number: 58 - 25.
# 6. The remaining golf balls are 33.
Final answer is: 33


Q: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?

# Step-by-step reasoning:
# 1. Initially, there are 9 computers in the server room.
# 2. Five more computers are installed each day.
# 3. The installation happens over 4 days (Monday to Thursday).
# 4. The total number of computers installed is 5 * 4 = 20.
# 5. To find the total number of computers now in the server room, add the installed computers to the initial count: 9 + 20.
# 6. The final count of computers is 29.
Final answer is 29


Q: {question}

# Step-by-step reasoning:

'''.strip() + '\n\n\n'