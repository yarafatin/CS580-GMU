# Initialize Markov Decision Process model
actions = (0, 1 , 2, 3 )  # actions (0 - up, 1 down , 2=left, 3=right)
states = (0, 1, 2, 3)  # states (s1, s2, s3. s4)
rewards = [-0.04, -0.04, -0.04, 1]  # Direct rewards per state
gamma = 0.5  # discount factor
# Transition probabilities per state-action pair

probs = [
    [[0.9, 0, 0.0, 0.1], [0.1, 0.8, 0.0, 0.1], [0.9, 0.1, 0.0, 0], [0.1, 0.1,0.0, 0.8]], #s1 - up down left right
    [[0.8, 0.1, 0.1, 0], [0, 0.9, 0.1, 0], [0.1, 0.9, 0.0, 0.0], [0.1, 0.1, 0.8, 0.0]],
    [[0., 0.1, 0.1, 0.8], [0, 0.1, 0.9, 0.0], [0, 0.8, 0.1, 0.1], [0, 0, 0.9, 0.1]],  # Terminating state (all probs 0)
    [[0.1, 0, 0.0, 0.9], [0.1, 0, 0.8, 0.1], [0.8, 0, 0.1, 0.1], [0, 0, 0.1, 0.9]]
]

# Set value iteration parameters
max_iter = 1000  # Maximum number of iterations
delta = 0.0001  # Error tolerance
V = [0.1, 0.1, 0.1, 0.1]  # Initialize values
pi = [None, None, None, None, None]  # Initialize policy


# Start value iteration
for i in range(1,max_iter+1):
    print('-------start iteration:' , i)
    max_diff = 0  # Initialize max difference
    V_new = [0.1, 0.1, 0.1, 0.1]  # Initialize values
    for s in states:
        max_val = 0
        for a in actions:

            # Compute state value
            val = rewards[s]  # Get direct reward
            for s_next in states:
                #print("state:" , s, ",s_next :", s_next, ",action:", a, ",V[s_next]=",V[s_next],"probs[s][s_next][a]=",probs[s][s_next][a]
                #      ,",val=", val)
                val += probs[s][a][s_next] * (gamma * V[s_next]
                )  # Add discounted downstream values

            # Store value best action so far
            max_val = max(max_val, val)

            # Update best policy
            if V[s] < val:
                pi[s] = actions[a]  # Store action with highest value

        V_new[s] = max_val  # Update value with highest value


        # Update maximum difference
        max_diff = max(max_diff, abs(V[s] - V_new[s]))

    # Update value functions
    print("U values :" , V_new)
    V = V_new

    # If diff smaller than threshold delta for all states, algorithm terminates
    if max_diff < delta:
        print('Threashold :', delta , ', Stopped at iteration ', i )
        break

    print('End of iteration ', i )