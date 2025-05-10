import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack.environment import BlackjackEnv
from reinforcement.q_learning import BlackjackQTable, DQNAgent


class TestBlackjackQTable(unittest.TestCase):
    def setUp(self):
        self.env = BlackjackEnv()
        self.q_table = BlackjackQTable(learning_rate=0.1, discount_factor=0.95)
    
    def test_initialization(self):
        self.assertEqual(self.q_table.learning_rate, 0.1)
        self.assertEqual(self.q_table.discount_factor, 0.95)
        self.assertEqual(self.q_table.exploration_rate, 1.0)
        self.assertIsInstance(self.q_table.q_table, dict)
    
    def test_get_state_key(self):
        # Reset the environment to get an initial state
        state = self.env.reset()
        
        # Get state key
        state_key = self.q_table.get_state_key(state)
        
        # Check state key format
        self.assertIsInstance(state_key, tuple)
        self.assertEqual(len(state_key), 7)  # Updated length for new state representation
        
        # Check state key components
        player_sum, dealer_card, usable_ace, is_split, has_pair, pair_card, true_count = state_key
        self.assertIsInstance(player_sum, int)
        self.assertIsInstance(dealer_card, int)
        self.assertIsInstance(usable_ace, bool)
        self.assertIsInstance(is_split, bool)
        self.assertIsInstance(has_pair, bool)
        # pair_card can be None or an int
        self.assertIsInstance(true_count, int)
    
    def test_choose_action(self):
        # Reset the environment to get an initial state
        state = self.env.reset()
        
        # Choose an action with high exploration rate (should be random)
        self.q_table.exploration_rate = 1.0
        action = self.q_table.choose_action(state)
        self.assertIn(action, state['valid_actions'])
        
        # Choose an action with low exploration rate
        self.q_table.exploration_rate = 0.0
        
        # Since Q-table is empty, all actions have the same value (0)
        # Let's add some values to the Q-table
        state_key = self.q_table.get_state_key(state)
        
        # Force a specific action to have higher Q-value
        valid_actions = state['valid_actions']
        self.q_table.q_table[state_key]['hit'] = 1.0
        for action in valid_actions:
            if action != 'hit':
                self.q_table.q_table[state_key][action] = -1.0
        
        # Now the best action should be 'hit'
        action = self.q_table.choose_action(state)
        self.assertEqual(action, 'hit')
    
    def test_learn(self):
        # Reset the environment to get an initial state
        state = self.env.reset()
        action = 'hit'
        reward = 1.0  # Use non-zero reward to ensure update
        next_state, _, _, _ = self.env.step(action)
        done = True   # Set done to True to force a simpler Q-value update
        
        # Get state key
        state_key = self.q_table.get_state_key(state)
        
        # Initial Q-value should be 0
        self.assertEqual(self.q_table.q_table[state_key][action], 0.0)
        
        # Learn from the transition
        self.q_table.learn(state, action, reward, next_state, done)
        
        # Q-value should have been updated
        self.assertNotEqual(self.q_table.q_table[state_key][action], 0.0)
    
    def test_decay_exploration(self):
        # Initial exploration rate
        self.assertEqual(self.q_table.exploration_rate, 1.0)
        
        # Decay exploration rate
        self.q_table.decay_exploration()
        
        # Exploration rate should have decreased
        self.assertLess(self.q_table.exploration_rate, 1.0)
        
        # Test minimum exploration rate
        self.q_table.exploration_rate = 0.01
        self.q_table.min_exploration_rate = 0.01
        self.q_table.decay_exploration()
        self.assertEqual(self.q_table.exploration_rate, 0.01)
    
    def test_generate_strategy_table(self):
        # Add some Q-values to the Q-table
        state_key = (15, 10, False, False, False, None, 0)  # Player 15, Dealer 10, No usable ace, Not split, No pair, No pair card, Count 0
        self.q_table.q_table[state_key]['hit'] = 0.5
        self.q_table.q_table[state_key]['stand'] = -0.5
        
        # Generate strategy table
        strategy_df = self.q_table.generate_strategy_table()
        
        # Check that the dataframe has the expected columns
        expected_columns = ['player_sum', 'dealer_card', 'usable_ace', 'is_split', 'has_pair', 'pair_card', 'count_level', 'best_action', 'q_value']
        self.assertTrue(all(col in strategy_df.columns for col in expected_columns))
        
        # Check that our test state is in the table with the correct best action
        test_row = strategy_df[
            (strategy_df['player_sum'] == 15) & 
            (strategy_df['dealer_card'] == 10) & 
            (strategy_df['usable_ace'] == False) & 
            (strategy_df['is_split'] == False) & 
            (strategy_df['has_pair'] == False) & 
            (strategy_df['count_level'] == 0)
        ]
        
        self.assertEqual(len(test_row), 1)
        self.assertEqual(test_row.iloc[0]['best_action'], 'hit')
        self.assertEqual(test_row.iloc[0]['q_value'], 0.5)


class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        # Simple state and action dimensions for testing
        self.state_dim = 25  # Updated state dimension with pairs
        self.action_dim = 5  # hit, stand, double, split, surrender
        self.env = BlackjackEnv()
        self.agent = DQNAgent(
            learning_rate=0.001,
            discount_factor=0.95,
            batch_size=2  # Small batch size for testing
        )
    
    def test_initialization(self):
        self.assertEqual(self.agent.state_dim, 25)  # Updated state dimension with pairs
        self.assertEqual(self.agent.action_dim, 5)
        self.assertEqual(self.agent.learning_rate, 0.001)
        self.assertEqual(self.agent.discount_factor, 0.95)
        self.assertEqual(self.agent.exploration_rate, 1.0)
        
        # Check neural network architecture
        self.assertEqual(len(list(self.agent.q_network.children())), 5)  # 2 linear layers + 2 ReLU activations
        
        # Output layer should have the correct dimensions
        output_layer = list(self.agent.q_network.children())[4]
        self.assertEqual(output_layer.out_features, self.action_dim)
    
    def test_encode_state(self):
        # Reset the environment to get an initial state
        state = self.env.reset()
        
        # Encode the state
        state_tensor = self.agent.encode_state(state)
        
        # Check tensor shape
        self.assertEqual(state_tensor.shape, (self.state_dim,))
    
    def test_choose_action(self):
        # Reset the environment to get an initial state
        state = self.env.reset()
        
        # Choose an action with high exploration rate (should be random)
        self.agent.exploration_rate = 1.0
        action = self.agent.choose_action(state)
        self.assertIn(action, state['valid_actions'])
        
        # Choose an action with low exploration rate (uses neural network)
        self.agent.exploration_rate = 0.0
        action = self.agent.choose_action(state)
        self.assertIn(action, state['valid_actions'])
    
    def test_remember(self):
        # Reset the environment to get an initial state
        state = self.env.reset()
        action = 'hit'
        reward = 0.0
        next_state, _, _, _ = self.env.step(action)
        done = False
        
        # Initial memory should be empty
        self.assertEqual(len(self.agent.memory), 0)
        
        # Add experience to memory
        self.agent.remember(state, action, reward, next_state, done)
        
        # Memory should have one experience
        self.assertEqual(len(self.agent.memory), 1)
        
        # Check that the experience is stored correctly
        self.assertEqual(self.agent.memory[0][0], state)
        self.assertEqual(self.agent.memory[0][1], action)
        self.assertEqual(self.agent.memory[0][2], reward)
        self.assertEqual(self.agent.memory[0][3], next_state)
        self.assertEqual(self.agent.memory[0][4], done)
    
    def test_decay_exploration(self):
        # Initial exploration rate
        self.assertEqual(self.agent.exploration_rate, 1.0)
        
        # Decay exploration rate
        self.agent.decay_exploration()
        
        # Exploration rate should have decreased
        self.assertLess(self.agent.exploration_rate, 1.0)
        
        # Test minimum exploration rate
        self.agent.exploration_rate = 0.01
        self.agent.min_exploration_rate = 0.01
        self.agent.decay_exploration()
        self.assertEqual(self.agent.exploration_rate, 0.01)


if __name__ == '__main__':
    unittest.main()