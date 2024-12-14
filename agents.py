# Urn-Learning Agent
class UrnAgent:
    def __init__(self, n_signaling_actions, n_final_actions, learning_rate=0.1,
                 discount_factor=0.99, exploration_rate=1.0,
                 exploration_decay=0.995, min_exploration_rate=0.01,
                 n_observed_features = 2,initialize_urns = False):
        """
        Initialize a Q-learning agent for signaling and final actions.

        :param n_signaling_actions: Number of possible signaling actions
        :param n_final_actions: Number of possible final actions
        :param learning_rate: Learning rate for updating Q-values
        :param discount_factor: Discount factor for future rewards
        :param exploration_rate: Initial exploration rate (epsilon)
        :param exploration_decay: Factor by which exploration rate decays
        :param min_exploration_rate: Minimum exploration rate
        """
        self.n_signaling_actions = n_signaling_actions
        self.n_final_actions = n_final_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Urns
        if not initialize_urns:
          self.signalling_urns = {}
        else:
          self.signalling_urns = create_initial_signals(n_observed_features=n_observed_features,
                                                        n_signals=n_signaling_actions,n=100,m=0)
        self.action_urns = {}

    def reset_urns(self):
        self.signalling_urns = {}
        self.action_urns = {}
        self.signalling_urns_history = []

    def get_action(self, state, is_signaling=True):
        """
        :param state: Current state (tuple)
        :param is_signaling: If True, choose a signaling action; otherwise, choose a final action
        :return: Chosen action (int)
        """
        n_actions = self.n_signaling_actions if is_signaling else self.n_final_actions

        if is_signaling:
          if state not in self.signalling_urns:
              # initiate urns with ones
              self.signalling_urns[state] = np.ones(n_actions)
        else:
          if state not in self.action_urns:
              self.action_urns[state] = np.ones(n_actions)

        # No epsilon greedy here!!
        if is_signaling:
          probability_weights = self.signalling_urns[state] / np.sum(self.signalling_urns[state])
          return np.random.choice(np.arange(len(probability_weights)), p=probability_weights)
        else:
          probability_weights = self.action_urns[state] / np.sum(self.action_urns[state])
          return np.random.choice(np.arange(len(probability_weights)), p=probability_weights)
            
    def update_urns(self, state, action, reward, is_signaling=True):

        n_actions = self.n_signaling_actions if is_signaling else self.n_final_actions

        if is_signaling:
          if state not in self.signalling_urns:
              self.signalling_urns[state] = np.ones(n_actions)
        else:
          if state not in self.action_urns:
              self.action_urns[state] = np.ones(n_actions)

        # Urn Update and history update
        if is_signaling:
          self.signalling_urns[state][action]+=int(reward)
          #self.signalling_urns_history.append(self.signalling_urns.copy())
        else:
          self.action_urns[state][action]+=int(reward)
          #self.action_urns_history.append(self.action_urns.copy())

    def decay_exploration(self):
        """
        Decay the exploration rate (epsilon).
        """
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)



# Q-Learning Agent
class QLearningAgent:
    def __init__(self, n_signaling_actions, n_final_actions, learning_rate=0.1,
                 discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01,
                initialize_urns = False):
        """
        Initialize a Q-learning agent for signaling and final actions.

        :param n_signaling_actions: Number of possible signaling actions
        :param n_final_actions: Number of possible final actions
        :param learning_rate: Learning rate for updating Q-values
        :param discount_factor: Discount factor for future rewards
        :param exploration_rate: Initial exploration rate (epsilon)
        :param exploration_decay: Factor by which exploration rate decays
        :param min_exploration_rate: Minimum exploration rate
        """
        self.n_signaling_actions = n_signaling_actions
        self.n_final_actions = n_final_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Q-tables for signaling and final actions
        if not initialize_urns:
          self.q_table_signaling = {}
        else:
          self.q_table_signaling = create_initial_signals(n_observed_features=n_observed_features,
                                                        n_signals=n_signaling_actions,n=100,m=0)
        self.q_table_action = {}

    def get_action(self, state, is_signaling=True):
        """
        Choose an action based on the exploration-exploitation trade-off.

        :param state: Current state (tuple)
        :param is_signaling: If True, choose a signaling action; otherwise, choose a final action
        :return: Chosen action (int)
        """
        #q_table = self.q_table_signaling if is_signaling else self.q_table_action
        n_actions = self.n_signaling_actions if is_signaling else self.n_final_actions

        if is_signaling:
          if state not in self.q_table_signaling:
              self.q_table_signaling[state] = np.zeros(n_actions)
        else:
          if state not in self.q_table_action:
              self.q_table_action[state] = np.zeros(n_actions)

        if random.uniform(0, 1) < self.exploration_rate:
            # Exploration: choose a random action
            return random.randint(0, n_actions - 1)
        else:      # Exploitation: choose the action with the highest Q-value
          if is_signaling:
            return np.argmax(self.q_table_signaling[state])
          else:
            return np.argmax(self.q_table_action[state])

    def update_q_table(self, state, action, reward, is_signaling=True):
        """
        Update the Q-value for the given state-action pair.

        :param state: Current state (tuple)
        :param action: Action taken (int)
        :param reward: Reward received (float)
        :param is_signaling: If True, update the signaling Q-table; otherwise, update the final action Q-table
        """
        n_actions = self.n_signaling_actions if is_signaling else self.n_final_actions

        if is_signaling:
          if state not in self.q_table_signaling:
              self.q_table_signaling[state] = np.zeros(n_actions)
        else:
          if state not in self.q_table_action:
              self.q_table_action[state] = np.zeros(n_actions)

        # Q-learning update rule
        td_target = reward #+ self.discount_factor * future_value
        if is_signaling:
          td_error = td_target - self.q_table_signaling[state][action]
          self.q_table_signaling[state][action]+= self.learning_rate * td_error
        else:
          td_error = td_target - self.q_table_action[state][action]
          self.q_table_action[state][action] += self.learning_rate * td_error

    def decay_exploration(self):
        """
        Decay the exploration rate (epsilon).
        """
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)


# Q-Network class remains unchanged
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# DEEPQ Agent
class NeuralQLearningAgent:
    def __init__(self, n_signaling_actions, n_final_actions, learning_rate=0.01,
                 exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01,
                 encoder_input_dim=2, decoder_input_dim=3, memory_size=5000, batch_size=64):
        self.n_signaling_actions = n_signaling_actions
        self.n_final_actions = n_final_actions
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.batch_size = batch_size

        self.network_signaling = QNetwork(encoder_input_dim, n_signaling_actions)
        self.network_final = QNetwork(decoder_input_dim, n_final_actions)
        self.optimizer_signaling = optim.Adam(self.network_signaling.parameters(), lr=learning_rate)
        self.optimizer_final = optim.Adam(self.network_final.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Experience replay memory
        # When the deque is full:
        # The oldest experience (the one added first) is automatically removed.
        # New Addition: The new experience is added to the end of the deque.
        self.signal_memory = deque(maxlen=memory_size)
        self.action_memory = deque(maxlen=memory_size)

    def get_action(self, state, is_signaling=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if random.uniform(0, 1) < self.exploration_rate:
            n_actions = self.n_signaling_actions if is_signaling else self.n_final_actions
            return random.randint(0, n_actions - 1)
        else:
            network = self.network_signaling if is_signaling else self.network_final
            with torch.no_grad():
                q_values = network(state_tensor)
            return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, done,is_signaling=True):
        """
        Store a single experience tuple in the replay memory.
        """
        if is_signaling:
          self.signal_memory.append((state, action, reward, done))
        else:
          self.action_memory.append((state, action, reward, done))

    def update_network(self, state, action, reward, done=True,is_signaling=True):
        """
        Train the network using a batch of experiences from the replay memory.
        """
        # first add the new experience to the memory
        self.store_experience(state, action, reward, done,is_signaling=is_signaling)

        memory = self.signal_memory if is_signaling else self.action_memory
        # this is so that they can have a batch even if it is small, but not larger
        batch = random.sample(memory, min(len(memory),self.batch_size))
        states, actions, rewards, dones = zip(*batch)

        states_tensor = torch.FloatTensor(states)
        rewards_tensor = torch.FloatTensor(rewards)
        actions_tensor = torch.LongTensor(actions)

        network = self.network_signaling if is_signaling else self.network_final
        optimizer = self.optimizer_signaling if is_signaling else self.optimizer_final

        # Current Q-values
        network.train()
        q_values = network(states_tensor)
        current_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        # OR
        # target_q_values[0, action] = updated_q_value

        # Target Q-values
        target_q_values = rewards_tensor

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize the network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
