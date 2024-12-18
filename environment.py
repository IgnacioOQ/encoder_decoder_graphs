# Multi-Agent Environment
class MultiAgentEnv:
    def __init__(self, n_agents=2, n_features=2, n_signaling_actions=2, n_final_actions=4,
                 full_information = False,game_dict=random_game_dicts,
                 observed_variables = agents_observed_variables):
        """
        Initialize the multi-agent environment with separate signaling and final actions.

        :param n_agents: Number of agents
        :param n_features: Length of the binary vector generated by nature
        :param n_signaling_actions: Number of possible signaling actions for agents
        :param n_final_actions: Number of possible final actions for agents
        """
        self.n_agents = n_agents
        self.n_features = n_features
        self.n_signaling_actions = n_signaling_actions
        self.n_final_actions = n_final_actions
        self.current_step = 0
        self.full_information = full_information

        # Internal state variables
        self.nature_vector = None
        self.signals = None
        self.final_actions = None

        # One random game dictionary per agent
        self.internal_game_dicts = game_dict

        # Init which variables each agent observes
        self.agents_observed_variables = agents_observed_variables
        # Store the relevant histories (signals and rewards)
        # Tracking metrics
        # # FOR EACH AGENT I WANT, FOR EACH STATE OF THE WORLD (OR OBSERVATION), WHICH SIGNAL WAS USED
        # INDEPENDENTLY I ALSO WANT THE REWARD HISTORY FOR BOTH AGENTS
        # self.world_states = set(product([0, 1], repeat=self.n_features))
        self.rewards_history = [[] for _ in range(self.n_agents)]  # Store rewards per episode
        self.signal_usage = [{} for _ in range(self.n_agents)]  # Track signal counts for each agent
        self.signal_information_history = [[] for _ in range(self.n_agents)]

    def reset(self):
        """
        Reset the environment for a new episode.

        :return: Initial observation (binary vector from nature)
        """
        self.current_step = 0
        self.nature_vector = np.random.randint(0, 2, size=self.n_features)  # Binary nature vector
        self.signals = [None] * self.n_agents  # Reset signals
        self.final_actions = [None] * self.n_agents  # Reset final actions
        return self.nature_vector

    def step(self, actions):
        """
        Execute a step in the environment based on the current phase.

        :param actions: Actions taken by the agents
        :return: Tuple (observation, rewards, done)
            - observation: The environment state observed by agents
            - rewards: A list of rewards for each agent
            - done: Boolean indicating if the episode has ended
        """

        if self.current_step == 0:
            # Step 0: Agents perform signaling actions
            self.signals = actions
            self.current_step += 1
            assigned_observations = self.assign_observations()
            # update signal history
            for i in range(self.n_agents):
              agent_observation = assigned_observations[i]
              #agent_observation = agents_observations[i]
              if agent_observation not in self.signal_usage[i]:
                # I initialize with one so that I measure the Normalized Mutual Information NMI
                  self.signal_usage[i][agent_observation] = [1] * n_signaling_actions
              self.signal_usage[i][agent_observation][self.signals[i]] += 1
              #new_observations = [obs+(signal,) for obs,signal in zip(self.nature_vector,self.signals)]
            return False

        elif self.current_step == 1:
            # Step 1: Agents perform final actions based on signals
            self.final_actions = actions
            #rewards = self._calculate_rewards()
            rewards = self.calculate_rewards()
            # Record rewards for this episode
            for i in range(self.n_agents):
              self.rewards_history[i].append(rewards[i])
            # Record the information that the signals have at that step
            for i in range(self.n_agents):
              mutual_info, normalized_mutual_info = compute_mutual_information(self.signal_usage[i])
              self.signal_information_history[i].append(normalized_mutual_info)
            # self.current_step += 1 # no need for more steps in simulation we reset after this step
            # reseting does not change the game dictionaries, just the underlying state of the world
            return rewards, True
        else:
            raise ValueError("Environment has already completed two steps. Reset before reusing.")

    def report_metrics(self):
      # Plot results
      return self.signal_usage, self.rewards_history, self.signal_information_history

    def calculate_rewards(self):
      rewards = []
      for i in range(self.n_agents):
        agent_action = self.final_actions[i]
        a_rew = self.internal_game_dicts[i][tuple(self.nature_vector)][agent_action]
        rewards.append(a_rew)
      return rewards

    def render(self):
        """
        Print the current state of the environment for debugging purposes.
        """
        print(f"Step: {self.current_step}")
        print(f"Nature Vector: {self.nature_vector}")
        print(f"Signals: {self.signals}")
        print(f"Final Actions: {self.final_actions}")

    def assign_observations(self):
      agents_observations = []
      if self.full_information:
        for i in range(self.n_agents):
          agents_observations.append(tuple(self.nature_vector))
      else:
        for i in range(self.n_agents):
          observed_indexes = self.agents_observed_variables[i]
          vector = self.nature_vector
          subset = tuple([vector[j] for j in observed_indexes])
          agents_observations.append(subset)
      return agents_observations
