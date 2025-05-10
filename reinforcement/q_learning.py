import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any
import random
from collections import defaultdict, deque

from blackjack.environment import BlackjackEnv


class BlackjackQTable:
    """Q-learning implementation for Blackjack using a lookup table."""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, 
                 exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Initialize the Q-learning agent.
        
        Args:
            learning_rate: Learning rate (alpha) for updating Q-values
            discount_factor: Discount factor (gamma) for future rewards
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Rate at which exploration decreases
            min_exploration_rate: Minimum exploration rate
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table
        # State format: (player_sum, dealer_card, usable_ace, is_split, has_pair, pair_card, count_level)
        # pair_card is the rank of the paired cards if has_pair is True, otherwise None
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Actions: hit, stand, double, split, surrender
        self.actions = ['hit', 'stand', 'double', 'split', 'surrender']
        
        # For tracking learning progress
        self.episode_rewards = []
    
    def get_state_key(self, state: Dict) -> Tuple:
        """Convert the game state to a tuple key for the Q-table."""
        # Use current hand
        if not state['player_hands'] or state['current_hand_index'] >= len(state['player_hands']):
            return None
        
        current_hand = state['player_hands'][state['current_hand_index']]
        player_sum = current_hand['value']
        
        # Check for usable ace
        usable_ace = False
        for card_value in current_hand['cards']:
            if card_value == 11:  # Card is an ace with value 11
                usable_ace = True
                break
        
        # Check for pairs
        has_pair = False
        pair_card = None
        cards = current_hand['cards']
        
        if len(cards) == 2 and cards[0] == cards[1]:
            has_pair = True
            pair_card = cards[0]
        
        dealer_card = state['dealer_up_card']
        is_split = current_hand['is_split']
        
        # Add true count for card counting
        true_count = int(state['true_count'])  # Discretize
        
        return (player_sum, dealer_card, usable_ace, is_split, has_pair, pair_card, true_count)
    
    def choose_action(self, state: Dict) -> str:
        """Choose an action using epsilon-greedy policy."""
        state_key = self.get_state_key(state)
        valid_actions = state['valid_actions']
        
        if not valid_actions or state_key is None:
            return 'stand'  # Default action if no valid actions or invalid state
        
        # Exploration: choose a random action
        if np.random.random() < self.exploration_rate:
            return np.random.choice(valid_actions)
        
        # Exploitation: choose the best action based on Q-values
        q_values = {action: self.q_table[state_key][action] for action in valid_actions}
        
        # If all Q-values are the same, choose randomly
        if len(set(q_values.values())) == 1:
            return np.random.choice(valid_actions)
        
        return max(q_values, key=q_values.get)
    
    def learn(self, state: Dict, action: str, reward: float, next_state: Dict, done: bool):
        """Update Q-values based on the observed transition."""
        state_key = self.get_state_key(state)
        
        if state_key is None:
            return
        
        if done:
            # Terminal state
            target = reward
        else:
            next_state_key = self.get_state_key(next_state)
            if next_state_key is None:
                target = reward
            else:
                # Get max Q-value for next state
                valid_next_actions = next_state['valid_actions']
                if not valid_next_actions:
                    max_next_q = 0.0
                else:
                    max_next_q = max(
                        self.q_table[next_state_key][next_action] 
                        for next_action in valid_next_actions
                    )
                target = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        current_q = self.q_table[state_key][action]
        self.q_table[state_key][action] = current_q + self.learning_rate * (target - current_q)
    
    def decay_exploration(self):
        """Decay the exploration rate."""
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )
    
    def train(self, env: BlackjackEnv, num_episodes: int, verbose: bool = False):
        """Train the agent for a specified number of episodes."""
        total_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)
                
                self.learn(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
            
            # If there are rewards for multiple hands, sum them
            if 'all_rewards' in info:
                episode_reward = sum(info['all_rewards'])
            
            total_rewards.append(episode_reward)
            self.decay_exploration()
            
            if verbose and (episode + 1) % (num_episodes // 10) == 0:
                avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
                print(f"Episode: {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.4f}, Exploration Rate: {self.exploration_rate:.4f}")
        
        self.episode_rewards = total_rewards
        return total_rewards
    
    def get_best_action(self, state: Dict) -> str:
        """Get the best action for a given state based on learned Q-values."""
        state_key = self.get_state_key(state)
        valid_actions = state['valid_actions']
        
        if not valid_actions or state_key is None:
            return 'stand'  # Default action
        
        q_values = {action: self.q_table[state_key][action] for action in valid_actions}
        return max(q_values, key=q_values.get)
    
    def generate_strategy_table(self, count_level: int = 0) -> pd.DataFrame:
        """Generate a strategy table based on the learned Q-values."""
        # Player sums from 4 to 21
        player_sums = range(4, 22)
        # Dealer up cards from 2 to 11 (where 11 represents an Ace)
        dealer_cards = range(2, 12)
        # Pair cards (2-11 where 11 is Ace)
        pair_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        
        strategy_data = []
        
        # Handle non-pair hands first
        for player_sum in player_sums:
            for dealer_card in dealer_cards:
                for usable_ace in [False, True]:
                    for is_split in [False, True]:
                        state_key = (player_sum, dealer_card, usable_ace, is_split, False, None, count_level)
                        
                        # Skip states that are not in the Q-table
                        if state_key not in self.q_table:
                            continue
                        
                        # Find the best action
                        best_action = max(self.q_table[state_key], key=self.q_table[state_key].get)
                        
                        strategy_data.append({
                            'player_sum': player_sum,
                            'dealer_card': dealer_card,
                            'usable_ace': usable_ace,
                            'is_split': is_split,
                            'has_pair': False,
                            'pair_card': None,
                            'count_level': count_level,
                            'best_action': best_action,
                            'q_value': self.q_table[state_key][best_action]
                        })
        
        # Handle pair hands
        for pair_value in pair_values:
            player_sum = pair_value * 2  # Sum of the paired cards
            if pair_value == 11:  # Pair of aces is 12, not 22
                player_sum = 12
                usable_ace = True
            else:
                usable_ace = False
                
            for dealer_card in dealer_cards:
                # We're only interested in initial pairs, not split hands
                is_split = False
                state_key = (player_sum, dealer_card, usable_ace, is_split, True, pair_value, count_level)
                
                # Skip states that are not in the Q-table
                if state_key not in self.q_table:
                    continue
                
                # Find the best action
                best_action = max(self.q_table[state_key], key=self.q_table[state_key].get)
                
                strategy_data.append({
                    'player_sum': player_sum,
                    'dealer_card': dealer_card,
                    'usable_ace': usable_ace,
                    'is_split': is_split,
                    'has_pair': True,
                    'pair_card': pair_value,
                    'count_level': count_level,
                    'best_action': best_action,
                    'q_value': self.q_table[state_key][best_action]
                })
        
        return pd.DataFrame(strategy_data)


class DQNAgent:
    """Deep Q-Network agent for Blackjack."""
    
    def __init__(self, state_dim: int = None, action_dim: int = 5, learning_rate=0.001, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01,
                 memory_size=10000, batch_size=64):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions
            learning_rate: Learning rate for the neural network
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Rate at which exploration decreases
            min_exploration_rate: Minimum exploration rate
            memory_size: Size of the replay memory
            batch_size: Number of samples to use for each training step
        """
        if state_dim is None:
            # player_sum (1) + dealer_one_hot (10) + usable_ace (1) + is_split (1) + has_pair (1) + pair_one_hot (10) + true_count (1)
            self.state_dim = 25
        else:
            self.state_dim = state_dim
            
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Initialize Q-network and target network
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize loss function
        self.loss_fn = nn.MSELoss()
        
        # For tracking learning progress
        self.episode_rewards = []
    
    def _build_model(self):
        """Build a neural network model for Q-value prediction."""
        model = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def encode_state(self, state: Dict) -> torch.Tensor:
        """Encode the state as a tensor for the neural network."""
        if not state['player_hands'] or state['current_hand_index'] >= len(state['player_hands']):
            return torch.zeros(self.state_dim)
        
        current_hand = state['player_hands'][state['current_hand_index']]
        player_sum = current_hand['value'] / 21.0  # Normalize
        
        # One-hot encoding of dealer's up card
        dealer_card = state['dealer_up_card']
        dealer_one_hot = [0] * 10  # 10 possible cards (2-11)
        if dealer_card is not None:
            dealer_one_hot[min(dealer_card - 2, 9)] = 1
        
        # Usable ace
        usable_ace = 0
        for card_value in current_hand['cards']:
            if card_value == 11:  # Card is an ace with value 11
                usable_ace = 1
                break
        
        # Check for pairs
        has_pair = 0
        pair_one_hot = [0] * 10  # One-hot for the pair card value (2-11)
        cards = current_hand['cards']
        if len(cards) == 2 and cards[0] == cards[1]:
            has_pair = 1
            pair_value = cards[0]
            pair_one_hot[min(pair_value - 2, 9)] = 1
        
        # Other state features
        is_split = 1 if current_hand['is_split'] else 0
        true_count = state['true_count'] / 10.0  # Normalize
        
        state_tensor = torch.tensor(
            [player_sum] + dealer_one_hot + [usable_ace, is_split, has_pair] + pair_one_hot + [true_count],
            dtype=torch.float32
        )
        
        return state_tensor
    
    def choose_action(self, state: Dict) -> str:
        """Choose an action using epsilon-greedy policy."""
        valid_actions = state['valid_actions']
        
        if not valid_actions:
            return 'stand'  # Default action if no valid actions
        
        # Map valid actions to indices
        action_mapping = {'hit': 0, 'stand': 1, 'double': 2, 'split': 3, 'surrender': 4}
        valid_indices = [action_mapping[action] for action in valid_actions]
        
        # Exploration: choose a random action
        if np.random.random() < self.exploration_rate:
            return np.random.choice(valid_actions)
        
        # Exploitation: choose the best action based on Q-values
        state_tensor = self.encode_state(state).unsqueeze(0)  # Add batch dimension
        q_values = self.q_network(state_tensor).detach().numpy()[0]
        
        # Filter Q-values for valid actions only
        valid_q_values = {idx: q_values[idx] for idx in valid_indices}
        best_idx = max(valid_q_values, key=valid_q_values.get)
        
        # Map index back to action
        reverse_mapping = {0: 'hit', 1: 'stand', 2: 'double', 3: 'split', 4: 'surrender'}
        return reverse_mapping[best_idx]
    
    def train_batch(self):
        """Train the network on a batch of experiences from memory."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_tensors = torch.stack([self.encode_state(s) for s in states])
        next_state_tensors = torch.stack([self.encode_state(s) for s in next_states])
        action_indices = torch.tensor([{'hit': 0, 'stand': 1, 'double': 2, 'split': 3, 'surrender': 4}[a] for a in actions])
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        
        # Compute Q-values for current states and actions
        current_q_values = self.q_network(state_tensors).gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Compute next state Q-values using target network
        next_q_values = self.target_network(next_state_tensors).max(1)[0]
        
        # Compute target Q-values
        target_q_values = rewards_tensor + (1 - dones_tensor) * self.discount_factor * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values.detach())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update the target network with the current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_exploration(self):
        """Decay the exploration rate."""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
    
    def train(self, env: BlackjackEnv, num_episodes: int, target_update_frequency=10, verbose=False):
        """Train the agent for a specified number of episodes."""
        total_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)
                
                self.remember(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
            
            # If there are rewards for multiple hands, sum them
            if 'all_rewards' in info:
                episode_reward = sum(info['all_rewards'])
            
            total_rewards.append(episode_reward)
            
            # Train on a batch
            if len(self.memory) >= self.batch_size:
                self.train_batch()
            
            # Update target network periodically
            if episode % target_update_frequency == 0:
                self.update_target_network()
            
            self.decay_exploration()
            
            if verbose and (episode + 1) % (num_episodes // 10) == 0:
                avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
                print(f"Episode: {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.4f}, Exploration Rate: {self.exploration_rate:.4f}")
        
        self.episode_rewards = total_rewards
        return total_rewards
    
    def get_best_action(self, state: Dict) -> str:
        """Get the best action for a given state based on learned Q-values."""
        valid_actions = state['valid_actions']
        
        if not valid_actions:
            return 'stand'  # Default action if no valid actions
        
        # Map valid actions to indices
        action_mapping = {'hit': 0, 'stand': 1, 'double': 2, 'split': 3, 'surrender': 4}
        valid_indices = [action_mapping[action] for action in valid_actions]
        
        # Get Q-values
        state_tensor = self.encode_state(state).unsqueeze(0)  # Add batch dimension
        q_values = self.q_network(state_tensor).detach().numpy()[0]
        
        # Filter Q-values for valid actions only
        valid_q_values = {idx: q_values[idx] for idx in valid_indices}
        best_idx = max(valid_q_values, key=valid_q_values.get)
        
        # Map index back to action
        reverse_mapping = {0: 'hit', 1: 'stand', 2: 'double', 3: 'split', 4: 'surrender'}
        return reverse_mapping[best_idx]
    
    def generate_strategy_table(self, env: BlackjackEnv, count_level: int = 0) -> pd.DataFrame:
        """Generate a strategy table based on the learned Q-values."""
        # Player sums from 4 to 21
        player_sums = range(4, 22)
        # Dealer up cards from 2 to 11 (where 11 represents an Ace)
        dealer_cards = range(2, 12)
        # Pair cards (2-11 where 11 is Ace)
        pair_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        
        strategy_data = []
        
        # Generate non-pair strategies
        for player_sum in player_sums:
            for dealer_card in dealer_cards:
                for usable_ace in [False, True]:
                    for is_split in [False, True]:
                        # Skip invalid combinations
                        if usable_ace and player_sum < 12:
                            continue
                        if not usable_ace and player_sum < 5:
                            continue
                            
                        # Create a synthetic state
                        cards = []
                        if usable_ace:
                            cards = [player_sum - 11, 11]  # One card is an ace
                        else:
                            # Distribute value across two cards for non-paired hands
                            if player_sum <= 10:
                                cards = [player_sum - 2, 2]
                            else:
                                cards = [player_sum - 10, 10]
                        
                        state = {
                            'dealer_up_card': dealer_card,
                            'dealer_hand': [dealer_card],
                            'player_hands': [
                                {
                                    'cards': cards,
                                    'card_ranks': ['X', 'X'],  # Not needed for encoding
                                    'value': player_sum,
                                    'is_split': is_split,
                                    'doubled': False,
                                    'stood': False,
                                    'busted': False
                                }
                            ],
                            'current_hand_index': 0,
                            'running_count': count_level,
                            'true_count': count_level,
                            'cards_remaining': 312,  # 6 decks
                            'valid_actions': ['hit', 'stand']
                        }
                        
                        # Add 'double' to valid actions if appropriate
                        if player_sum in [9, 10, 11] and len(state['player_hands'][0]['cards']) == 2:
                            state['valid_actions'].append('double')
                        
                        # Get the best action
                        best_action = self.get_best_action(state)
                        
                        strategy_data.append({
                            'player_sum': player_sum,
                            'dealer_card': dealer_card,
                            'usable_ace': usable_ace,
                            'is_split': is_split,
                            'has_pair': False,
                            'pair_card': None,
                            'count_level': count_level,
                            'best_action': best_action
                        })
        
        # Generate pair strategies
        for pair_value in pair_values:
            player_sum = pair_value * 2  # Sum of the paired cards
            if pair_value == 11:  # Pair of aces is 12, not 22
                player_sum = 12
                usable_ace = True
            else:
                usable_ace = False
                
            for dealer_card in dealer_cards:
                # Create a synthetic state
                state = {
                    'dealer_up_card': dealer_card,
                    'dealer_hand': [dealer_card],
                    'player_hands': [
                        {
                            'cards': [pair_value, pair_value],
                            'card_ranks': ['X', 'X'],  # Not needed for encoding
                            'value': player_sum,
                            'is_split': False,  # Initial pair, not a split hand
                            'doubled': False,
                            'stood': False,
                            'busted': False
                        }
                    ],
                    'current_hand_index': 0,
                    'running_count': count_level,
                    'true_count': count_level,
                    'cards_remaining': 312,  # 6 decks
                    'valid_actions': ['hit', 'stand', 'split']
                }
                
                # Add 'double' to valid actions if appropriate
                if player_sum in [9, 10, 11]:
                    state['valid_actions'].append('double')
                
                # Get the best action
                best_action = self.get_best_action(state)
                
                strategy_data.append({
                    'player_sum': player_sum,
                    'dealer_card': dealer_card,
                    'usable_ace': usable_ace,
                    'is_split': False,
                    'has_pair': True,
                    'pair_card': pair_value,
                    'count_level': count_level,
                    'best_action': best_action
                })
        
        return pd.DataFrame(strategy_data)