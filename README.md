# Cart-Pole-Agent-v1

### Structure of the code
- Each of the learning algorithms are written in seperate classes in the file "main.py".
- Three classes namely "Policy_Iterator", "MCLearning", "Sarsa" are mentioned, each signifying the code corresponding to the named learning methods.
- The hyperparameters can be adjusted during the object declaration itself for both training and visualization.

### For visualizing the agent's performance
- For Policy Iteration, training=False (for best results).
- For Monte-Carlo and SARSA(1), use_pretrained=True (for best results).
- The trained optimal action files are provided along with the code, and are loaded automatically during the visualization.
- To check the best results, it is recommended to epsilon value as 0.
- To check the agent's performance under sub-optimal conditions (might be some external disturbance), set the epsilon value as something greater than 0. 

### For training the agent
- For Policy Iteration, training=True.
- For Monte-Carlo and SARSA(1), use_pretrained=False.
- The provided pre_trained policy files will be OVERWRITTEN once a new training loop starts.
