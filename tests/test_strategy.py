import sys
import os
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack.strategy import StrategyTable


class TestStrategyTable(unittest.TestCase):
    def setUp(self):
        # Create test data
        data = [
            {'player_sum': 15, 'dealer_card': 10, 'usable_ace': False, 'is_split': False, 'count_level': 0, 'best_action': 'hit'},
            {'player_sum': 15, 'dealer_card': 6, 'usable_ace': False, 'is_split': False, 'count_level': 0, 'best_action': 'stand'},
            {'player_sum': 16, 'dealer_card': 10, 'usable_ace': False, 'is_split': False, 'count_level': 0, 'best_action': 'hit'},
            {'player_sum': 16, 'dealer_card': 6, 'usable_ace': False, 'is_split': False, 'count_level': 0, 'best_action': 'stand'},
            {'player_sum': 17, 'dealer_card': 10, 'usable_ace': True, 'is_split': False, 'count_level': 0, 'best_action': 'hit'},
            {'player_sum': 18, 'dealer_card': 6, 'usable_ace': True, 'is_split': False, 'count_level': 0, 'best_action': 'double'},
            {'player_sum': 19, 'dealer_card': 10, 'usable_ace': True, 'is_split': False, 'count_level': 0, 'best_action': 'stand'},
            {'player_sum': 8, 'dealer_card': 6, 'usable_ace': False, 'is_split': True, 'count_level': 0, 'best_action': 'hit'},
            {'player_sum': 16, 'dealer_card': 10, 'usable_ace': False, 'is_split': True, 'count_level': 0, 'best_action': 'split'},
            {'player_sum': 20, 'dealer_card': 6, 'usable_ace': False, 'is_split': True, 'count_level': 0, 'best_action': 'stand'},
            {'player_sum': 15, 'dealer_card': 10, 'usable_ace': False, 'is_split': False, 'count_level': 1, 'best_action': 'stand'},
            {'player_sum': 16, 'dealer_card': 10, 'usable_ace': False, 'is_split': False, 'count_level': 1, 'best_action': 'stand'},
        ]
        self.test_df = pd.DataFrame(data)
        self.strategy_table = StrategyTable(self.test_df)
    
    def test_initialization(self):
        self.assertEqual(len(self.strategy_table.data), 12)
        self.assertTrue(all(col in self.strategy_table.data.columns for col in [
            'player_sum', 'dealer_card', 'usable_ace', 'is_split', 'count_level', 'best_action']))
    
    def test_filter_strategy(self):
        # Filter for hard totals (no usable ace, not split)
        hard_totals = self.strategy_table.filter_strategy(usable_ace=False, is_split=False, count_level=0)
        self.assertEqual(len(hard_totals), 4)
        
        # Filter for soft totals (usable ace, not split)
        soft_totals = self.strategy_table.filter_strategy(usable_ace=True, is_split=False, count_level=0)
        self.assertEqual(len(soft_totals), 3)
        
        # Filter for pair splitting (is split)
        pair_splitting = self.strategy_table.filter_strategy(usable_ace=False, is_split=True, count_level=0)
        self.assertEqual(len(pair_splitting), 3)
        
        # Filter for count level 1
        count_1 = self.strategy_table.filter_strategy(usable_ace=False, is_split=False, count_level=1)
        self.assertEqual(len(count_1), 2)
    
    def test_create_strategy_matrix(self):
        # Create matrix for hard totals
        hard_matrix = self.strategy_table.create_strategy_matrix(usable_ace=False, is_split=False, count_level=0)
        
        # Check matrix dimensions
        self.assertEqual(hard_matrix.shape, (2, 2))  # 2 player sums (15, 16) x 2 dealer cards (6, 10)
        
        # Check specific values
        self.assertEqual(hard_matrix.loc[15, 10], 'hit')
        self.assertEqual(hard_matrix.loc[15, 6], 'stand')
        self.assertEqual(hard_matrix.loc[16, 10], 'hit')
        self.assertEqual(hard_matrix.loc[16, 6], 'stand')
    
    def test_action_to_color(self):
        # Check color mappings
        self.assertEqual(self.strategy_table.action_to_color('hit'), 'red')
        self.assertEqual(self.strategy_table.action_to_color('stand'), 'green')
        self.assertEqual(self.strategy_table.action_to_color('double'), 'orange')
        self.assertEqual(self.strategy_table.action_to_color('split'), 'blue')
        self.assertEqual(self.strategy_table.action_to_color('surrender'), 'purple')
        self.assertEqual(self.strategy_table.action_to_color('unknown'), 'gray')  # Default for unknown actions
    
    def test_visualize_strategy(self):
        # Test that visualization runs without errors
        fig = self.strategy_table.visualize_strategy(usable_ace=False, is_split=False, count_level=0)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_compare_with_basic_strategy(self):
        # Create a basic strategy for comparison
        basic_strategy_data = self.test_df.copy()
        # Modify only one row's action to create a difference
        # Find the first row with count_level=0, player_sum=15, dealer_card=10
        idx = basic_strategy_data[(basic_strategy_data['count_level'] == 0) & 
                                 (basic_strategy_data['player_sum'] == 15) & 
                                 (basic_strategy_data['dealer_card'] == 10)].index[0]
        basic_strategy_data.loc[idx, 'best_action'] = 'stand'  # Change from 'hit' to 'stand'
        basic_strategy = StrategyTable(basic_strategy_data)
        
        # Compare strategies
        differences = self.strategy_table.compare_with_basic_strategy(basic_strategy)
        
        # Check that differences were detected
        self.assertEqual(len(differences), 1)
        self.assertEqual(differences.iloc[0]['player_sum'], 15)
        self.assertEqual(differences.iloc[0]['dealer_card'], 10)
        self.assertEqual(differences.iloc[0]['count_level'], 0)
        self.assertEqual(differences.iloc[0]['best_action_learned'], 'hit')
        self.assertEqual(differences.iloc[0]['best_action_basic'], 'stand')
    
    def test_get_summary(self):
        # Get strategy summary
        summary = self.strategy_table.get_summary()
        
        # Check summary content
        self.assertEqual(summary['total_states'], 12)
        self.assertEqual(sum(summary['action_counts'].values()), 12)
        self.assertIn('hit', summary['action_counts'])
        self.assertIn('stand', summary['action_counts'])
        self.assertIn('double', summary['action_counts'])
        self.assertIn('split', summary['action_counts'])
    
    def test_create_basic_strategy(self):
        # Create basic strategy
        basic_strategy = StrategyTable.create_basic_strategy()
        
        # Check that it contains data
        self.assertGreater(len(basic_strategy.data), 0)
        
        # Check that it has the expected columns
        self.assertTrue(all(col in basic_strategy.data.columns for col in [
            'player_sum', 'dealer_card', 'usable_ace', 'is_split', 'count_level', 'best_action']))
        
        # Check some specific basic strategy rules
        
        # Hard 16 vs 10 should be hit
        hard_16_vs_10 = basic_strategy.data[
            (basic_strategy.data['player_sum'] == 16) & 
            (basic_strategy.data['dealer_card'] == 10) & 
            (basic_strategy.data['usable_ace'] == False) & 
            (basic_strategy.data['is_split'] == False)
        ]
        self.assertEqual(len(hard_16_vs_10), 1)
        self.assertEqual(hard_16_vs_10.iloc[0]['best_action'], 'hit')
        
        # Hard 16 vs 6 should be stand
        hard_16_vs_6 = basic_strategy.data[
            (basic_strategy.data['player_sum'] == 16) & 
            (basic_strategy.data['dealer_card'] == 6) & 
            (basic_strategy.data['usable_ace'] == False) & 
            (basic_strategy.data['is_split'] == False)
        ]
        self.assertEqual(len(hard_16_vs_6), 1)
        self.assertEqual(hard_16_vs_6.iloc[0]['best_action'], 'stand')
        
        # A,8 (19) vs any should be stand
        soft_19 = basic_strategy.data[
            (basic_strategy.data['player_sum'] == 19) & 
            (basic_strategy.data['usable_ace'] == True) & 
            (basic_strategy.data['is_split'] == False)
        ]
        self.assertTrue(all(row['best_action'] == 'stand' for _, row in soft_19.iterrows()))
        
        # 8,8 should always be split
        pair_8s = basic_strategy.data[
            (basic_strategy.data['player_sum'] == 16) & 
            (basic_strategy.data['usable_ace'] == False) & 
            (basic_strategy.data['is_split'] == True)
        ]
        self.assertTrue(all(row['best_action'] == 'split' for _, row in pair_8s.iterrows()))


if __name__ == '__main__':
    unittest.main()