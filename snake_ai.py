import streamlit as st
import numpy as np
import random
import time

# --- SETUP & SIDEBAR ---
st.set_page_config(page_title="Snake AI Trainer", layout="wide")
st.sidebar.title("📊 AI Brain Dashboard")
train_speed = st.sidebar.slider("Training Speed (Seconds)", 0.01, 0.5, 0.1)

# Initialize Session States
if 'q_table' not in st.session_state:
    st.session_state.q_table = {} # Memory bank: State -> Actions
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'high_score' not in st.session_state:
    st.session_state.high_score = 0

# --- THE AI BRAIN (Q-Learning) ---
def get_state(snake_head, food_pos, grid_size):
    # The AI "sees" 4 things: Is food Up/Down/Left/Right?
    # And 4 more things: Is there a wall Up/Down/Left/Right?
    # This creates a "State" string like "Food_Up_Wall_Left"
    relative_food = (food_pos[0] - snake_head[0], food_pos[1] - snake_head[1])
    state = [
        1 if relative_food[0] < 0 else 0, # Food is Up
        1 if relative_food[0] > 0 else 0, # Food is Down
        1 if relative_food[1] < 0 else 0, # Food is Left
        1 if relative_food[1] > 0 else 0, # Food is Right
        1 if snake_head[0] == 0 else 0,   # Wall is Up
        1 if snake_head[0] == grid_size-1 else 0 # Wall is Down
    ]
    return tuple(state)

def choose_action(state, epsilon=0.1):
    # Exploration vs Exploitation
    if state not in st.session_state.q_table or random.random() < epsilon:
        return random.randint(0, 3) # Random move
    return np.argmax(st.session_state.q_table[state])

# --- THE GAME ENGINE ---
grid_size = 10
placeholder = st.empty() # This is where the game is "drawn"

if st.sidebar.button("Start AI Training Loop"):
    snake = [[5, 5]]
    food = [random.randint(0, 9), random.randint(0, 9)]
    
    while True:
        state = get_state(snake[0], food, grid_size)
        action = choose_action(state)
        
        # Move Logic
        new_head = list(snake[0])
        if action == 0: new_head[0] -= 1 # Up
        elif action == 1: new_head[0] += 1 # Down
        elif action == 2: new_head[1] -= 1 # Left
        elif action == 3: new_head[1] += 1 # Right
        
        # Reward Logic
        reward = 0
        if new_head == food:
            reward = 10 # BIG REWARD!
            st.session_state.score += 1
            food = [random.randint(0, 9), random.randint(0, 9)]
        elif new_head[0] < 0 or new_head[0] >= grid_size or new_head[1] < 0 or new_head[1] >= grid_size or new_head in snake:
            reward = -10 # PENALTY!
            if len(snake) > 1: snake.pop() # Shrink tail
            st.session_state.score = max(0, st.session_state.score - 1)
            new_head = [5, 5] # Reset to center
        else:
            reward = -0.1 # Small penalty to keep moving
            snake.pop() # Normal movement
            
        snake.insert(0, new_head)
        
        # Q-Table Update (Learning)
        next_state = get_state(new_head, food, grid_size)
        if state not in st.session_state.q_table: st.session_state.q_table[state] = np.zeros(4)
        if next_state not in st.session_state.q_table: st.session_state.q_table[next_state] = np.zeros(4)
        
        # The Bellman Equation
        st.session_state.q_table[state][action] += 0.1 * (reward + 0.9 * np.max(st.session_state.q_table[next_state]) - st.session_state.q_table[state][action])
        
        # --- RENDER THE GRID ---
        grid = [["⬜" for _ in range(grid_size)] for _ in range(grid_size)]
        grid[food[0]][food[1]] = "🍎"
        for seg in snake: grid[seg[0]][seg[1]] = "🟩"
        grid[snake[0][0]][snake[0][1]] = "🐍"
        
        with placeholder.container():
            st.write(f"### Current Score: {st.session_state.score} | High Score: {st.session_state.high_score}")
            st.text("\n".join(["".join(row) for row in grid]))
            st.sidebar.metric("Brain Memory Size", len(st.session_state.q_table))
            
        if st.session_state.score > st.session_state.high_score:
            st.session_state.high_score = st.session_state.score
            
        time.sleep(train_speed)
