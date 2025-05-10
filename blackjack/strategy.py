import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class StrategyTable:
    """Generate and visualize blackjack strategy tables."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with strategy data.
        
        Args:
            data: DataFrame with strategy information including:
                - player_sum: The player's hand total
                - dealer_card: The dealer's up card
                - usable_ace: Whether the player has a usable ace
                - is_split: Whether this is a split hand
                - count_level: The true count level
                - best_action: The recommended action
        """
        self.data = data
    
    def filter_strategy(self, usable_ace: bool = False, is_split: bool = False, has_pair: bool = False, count_level: int = 0) -> pd.DataFrame:
        """Filter the strategy for specific conditions."""
        if 'has_pair' in self.data.columns:
            filtered = self.data[
                (self.data['usable_ace'] == usable_ace) & 
                (self.data['is_split'] == is_split) &
                (self.data['has_pair'] == has_pair) &
                (self.data['count_level'] == count_level)
            ]
        else:
            # For backward compatibility with old data format
            filtered = self.data[
                (self.data['usable_ace'] == usable_ace) & 
                (self.data['is_split'] == is_split) &
                (self.data['count_level'] == count_level)
            ]
        return filtered
    
    def create_strategy_matrix(self, usable_ace: bool = False, is_split: bool = False, has_pair: bool = False, count_level: int = 0) -> pd.DataFrame:
        """Create a matrix representation of the strategy for visualization."""
        filtered = self.filter_strategy(usable_ace, is_split, has_pair, count_level)
        
        if has_pair and 'pair_card' in self.data.columns and not filtered.empty:
            # For pair strategies, use pair_card as index instead of player_sum
            pair_names = {
                2: '2,2', 3: '3,3', 4: '4,4', 5: '5,5', 6: '6,6',
                7: '7,7', 8: '8,8', 9: '9,9', 10: '10,10', 11: 'A,A'
            }
            
            # Create a pivot table with pair_card (showing pairs) as rows and dealer_card as columns
            strategy_matrix = filtered.pivot_table(
                values='best_action',
                index='pair_card',
                columns='dealer_card',
                aggfunc='first'  # Just take the first value in case of duplicates
            )
            
            # Rename indices to show pairs
            strategy_matrix.index = [pair_names.get(card, str(card)) for card in strategy_matrix.index]
        else:
            # Create a pivot table with player_sum as rows and dealer_card as columns
            strategy_matrix = filtered.pivot_table(
                values='best_action',
                index='player_sum',
                columns='dealer_card',
                aggfunc='first'  # Just take the first value in case of duplicates
            )
        
        return strategy_matrix
    
    def action_to_color(self, action: str) -> str:
        """Map an action to a color for visualization."""
        color_map = {
            'hit': 'red',
            'stand': 'green',
            'double': 'orange',
            'split': 'blue',
            'surrender': 'purple'
        }
        return color_map.get(action, 'gray')
    
    def visualize_strategy(self, usable_ace: bool = False, is_split: bool = False, count_level: int = 0,
                           title: Optional[str] = None, figsize: tuple = (10, 8)):
        """Visualize the strategy as a colored table."""
        strategy_matrix = self.create_strategy_matrix(usable_ace, is_split, count_level)
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='#1B1C20')
        ax.set_facecolor('#1B1C20')
        
        # Create a color matrix from the strategy matrix
        color_matrix = np.zeros((len(strategy_matrix), len(strategy_matrix.columns), 3))
        action_abbr = {
            'hit': 'H',
            'stand': 'S',
            'double': 'D',
            'split': 'P',
            'surrender': 'R'
        }
        
        for i, (idx, row) in enumerate(strategy_matrix.iterrows()):
            for j, action in enumerate(row):
                if pd.isna(action):
                    continue
                
                # Get color for the action
                color = self.action_to_color(action)
                
                # Create a rectangle for each cell with the corresponding color
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color, alpha=0.7)
                ax.add_patch(rect)
                
                # Add the action abbreviation to the cell
                ax.text(j, i, action_abbr.get(action, '?'), ha='center', va='center', color='white', fontweight='bold')
        
        # Set axis labels and ticks
        dealer_labels = [str(c) if c != 11 else 'A' for c in strategy_matrix.columns]
        ax.set_xticks(range(len(dealer_labels)))
        ax.set_xticklabels(dealer_labels, color='white')
        ax.set_yticks(range(len(strategy_matrix.index)))
        ax.set_yticklabels(strategy_matrix.index, color='white')
        
        ax.set_xlabel('Dealer Upcard', color='white')
        ax.set_ylabel('Player Sum', color='white')
        
        # Set title
        if title is None:
            title_parts = []
            if usable_ace:
                title_parts.append('Soft Totals')
            else:
                title_parts.append('Hard Totals')
            if is_split:
                title_parts = ['Pair Splitting']
            if count_level != 0:
                title_parts.append(f'Count: {count_level}')
            
            title = ' - '.join(title_parts) + ' Strategy'
        
        ax.set_title(title, color='white', fontsize=14)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color=self.action_to_color('hit'), alpha=0.7, label='Hit (H)'),
            plt.Rectangle((0, 0), 1, 1, color=self.action_to_color('stand'), alpha=0.7, label='Stand (S)'),
            plt.Rectangle((0, 0), 1, 1, color=self.action_to_color('double'), alpha=0.7, label='Double (D)'),
            plt.Rectangle((0, 0), 1, 1, color=self.action_to_color('split'), alpha=0.7, label='Split (P)'),
            plt.Rectangle((0, 0), 1, 1, color=self.action_to_color('surrender'), alpha=0.7, label='Surrender (R)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', facecolor='#1B1C20', labelcolor='white')
        
        # Fix the aspect ratio and layout
        plt.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        
        plt.tight_layout()
        return fig
    
    def generate_all_strategy_tables(self, count_levels: List[int] = [0], save_path: Optional[str] = None):
        """Generate all strategy tables for different scenarios."""
        scenarios = [
            {'usable_ace': False, 'is_split': False, 'name': 'hard_totals'},
            {'usable_ace': True, 'is_split': False, 'name': 'soft_totals'},
            {'usable_ace': False, 'is_split': True, 'name': 'pair_splitting'}
        ]
        
        for count_level in count_levels:
            for scenario in scenarios:
                fig = self.visualize_strategy(
                    usable_ace=scenario['usable_ace'],
                    is_split=scenario['is_split'],
                    count_level=count_level,
                    title=f"{scenario['name'].replace('_', ' ').title()} - Count {count_level}"
                )
                
                if save_path:
                    filename = f"{save_path}/{scenario['name']}_count_{count_level}.png"
                    fig.savefig(filename, facecolor='#1B1C20', dpi=300, bbox_inches='tight')
                
                plt.show()
    
    def compare_with_basic_strategy(self, basic_strategy: 'StrategyTable') -> pd.DataFrame:
        """Compare with a basic strategy table and identify differences."""
        merged = pd.merge(
            self.data, 
            basic_strategy.data, 
            on=['player_sum', 'dealer_card', 'usable_ace', 'is_split', 'count_level'],
            suffixes=('_learned', '_basic')
        )
        
        # Find differences
        merged['different'] = merged['best_action_learned'] != merged['best_action_basic']
        differences = merged[merged['different']]
        
        return differences
    
    def get_summary(self) -> Dict:
        """Get a summary of the strategy table."""
        total_states = len(self.data)
        action_counts = self.data['best_action'].value_counts()
        action_percentages = (action_counts / total_states * 100).round(2)
        
        usable_ace_summary = self.data.groupby('usable_ace')['best_action'].value_counts()
        is_split_summary = self.data.groupby('is_split')['best_action'].value_counts()
        
        return {
            'total_states': total_states,
            'action_counts': action_counts.to_dict(),
            'action_percentages': action_percentages.to_dict(),
            'usable_ace_summary': usable_ace_summary.to_dict(),
            'is_split_summary': is_split_summary.to_dict()
        }
    
    @classmethod
    def create_basic_strategy(cls) -> 'StrategyTable':
        """Create a standard basic strategy table for comparison."""
        # Hard totals strategy
        hard_strategy = []
        
        # 5-8: always hit
        for player_sum in range(5, 9):
            for dealer_card in range(2, 12):
                hard_strategy.append({
                    'player_sum': player_sum,
                    'dealer_card': dealer_card,
                    'usable_ace': False,
                    'is_split': False,
                    'count_level': 0,
                    'best_action': 'hit'
                })
        
        # 9: double against 3-6, otherwise hit
        for dealer_card in range(2, 12):
            if 3 <= dealer_card <= 6:
                action = 'double'
            else:
                action = 'hit'
            hard_strategy.append({
                'player_sum': 9,
                'dealer_card': dealer_card,
                'usable_ace': False,
                'is_split': False,
                'count_level': 0,
                'best_action': action
            })
        
        # 10: double against 2-9, otherwise hit
        for dealer_card in range(2, 12):
            if 2 <= dealer_card <= 9:
                action = 'double'
            else:
                action = 'hit'
            hard_strategy.append({
                'player_sum': 10,
                'dealer_card': dealer_card,
                'usable_ace': False,
                'is_split': False,
                'count_level': 0,
                'best_action': action
            })
        
        # 11: double against 2-10, otherwise hit
        for dealer_card in range(2, 12):
            if 2 <= dealer_card <= 10:
                action = 'double'
            else:
                action = 'hit'
            hard_strategy.append({
                'player_sum': 11,
                'dealer_card': dealer_card,
                'usable_ace': False,
                'is_split': False,
                'count_level': 0,
                'best_action': action
            })
        
        # 12: stand against 4-6, otherwise hit
        for dealer_card in range(2, 12):
            if 4 <= dealer_card <= 6:
                action = 'stand'
            else:
                action = 'hit'
            hard_strategy.append({
                'player_sum': 12,
                'dealer_card': dealer_card,
                'usable_ace': False,
                'is_split': False,
                'count_level': 0,
                'best_action': action
            })
        
        # 13-16: stand against 2-6, otherwise hit
        for player_sum in range(13, 17):
            for dealer_card in range(2, 12):
                if 2 <= dealer_card <= 6:
                    action = 'stand'
                else:
                    action = 'hit'
                hard_strategy.append({
                    'player_sum': player_sum,
                    'dealer_card': dealer_card,
                    'usable_ace': False,
                    'is_split': False,
                    'count_level': 0,
                    'best_action': action
                })
        
        # 17-21: always stand
        for player_sum in range(17, 22):
            for dealer_card in range(2, 12):
                hard_strategy.append({
                    'player_sum': player_sum,
                    'dealer_card': dealer_card,
                    'usable_ace': False,
                    'is_split': False,
                    'count_level': 0,
                    'best_action': 'stand'
                })
        
        # Soft totals strategy
        soft_strategy = []
        
        # A,2 (13) and A,3 (14): double against 5-6, otherwise hit
        for player_sum in [13, 14]:
            for dealer_card in range(2, 12):
                if 5 <= dealer_card <= 6:
                    action = 'double'
                else:
                    action = 'hit'
                soft_strategy.append({
                    'player_sum': player_sum,
                    'dealer_card': dealer_card,
                    'usable_ace': True,
                    'is_split': False,
                    'count_level': 0,
                    'best_action': action
                })
        
        # A,4 (15) and A,5 (16): double against 4-6, otherwise hit
        for player_sum in [15, 16]:
            for dealer_card in range(2, 12):
                if 4 <= dealer_card <= 6:
                    action = 'double'
                else:
                    action = 'hit'
                soft_strategy.append({
                    'player_sum': player_sum,
                    'dealer_card': dealer_card,
                    'usable_ace': True,
                    'is_split': False,
                    'count_level': 0,
                    'best_action': action
                })
        
        # A,6 (17): double against 3-6, otherwise hit
        for dealer_card in range(2, 12):
            if 3 <= dealer_card <= 6:
                action = 'double'
            else:
                action = 'hit'
            soft_strategy.append({
                'player_sum': 17,
                'dealer_card': dealer_card,
                'usable_ace': True,
                'is_split': False,
                'count_level': 0,
                'best_action': action
            })
        
        # A,7 (18): double against 2-6, stand against 7-8, hit against 9-A
        for dealer_card in range(2, 12):
            if 2 <= dealer_card <= 6:
                action = 'double'
            elif 7 <= dealer_card <= 8:
                action = 'stand'
            else:
                action = 'hit'
            soft_strategy.append({
                'player_sum': 18,
                'dealer_card': dealer_card,
                'usable_ace': True,
                'is_split': False,
                'count_level': 0,
                'best_action': action
            })
        
        # A,8 (19) and higher: always stand
        for player_sum in range(19, 22):
            for dealer_card in range(2, 12):
                soft_strategy.append({
                    'player_sum': player_sum,
                    'dealer_card': dealer_card,
                    'usable_ace': True,
                    'is_split': False,
                    'count_level': 0,
                    'best_action': 'stand'
                })
        
        # Pair splitting strategy
        pair_strategy = []
        
        # For player_sum values, we're representing pairs by their total - e.g., a pair of 5s has player_sum 10
        
        # 2,2 and 3,3: split against 2-7, otherwise hit
        for player_sum in [4, 6]:
            for dealer_card in range(2, 12):
                if 2 <= dealer_card <= 7:
                    action = 'split'
                else:
                    action = 'hit'
                pair_strategy.append({
                    'player_sum': player_sum,
                    'dealer_card': dealer_card,
                    'usable_ace': False,
                    'is_split': True,
                    'count_level': 0,
                    'best_action': action
                })
        
        # 4,4: never split
        for dealer_card in range(2, 12):
            pair_strategy.append({
                'player_sum': 8,
                'dealer_card': dealer_card,
                'usable_ace': False,
                'is_split': True,
                'count_level': 0,
                'best_action': 'hit'
            })
        
        # 5,5: treat as hard 10
        for dealer_card in range(2, 12):
            if 2 <= dealer_card <= 9:
                action = 'double'
            else:
                action = 'hit'
            pair_strategy.append({
                'player_sum': 10,
                'dealer_card': dealer_card,
                'usable_ace': False,
                'is_split': True,
                'count_level': 0,
                'best_action': action
            })
        
        # 6,6: split against 2-6, otherwise hit
        for dealer_card in range(2, 12):
            if 2 <= dealer_card <= 6:
                action = 'split'
            else:
                action = 'hit'
            pair_strategy.append({
                'player_sum': 12,
                'dealer_card': dealer_card,
                'usable_ace': False,
                'is_split': True,
                'count_level': 0,
                'best_action': action
            })
        
        # 7,7: split against 2-7, otherwise hit
        for dealer_card in range(2, 12):
            if 2 <= dealer_card <= 7:
                action = 'split'
            else:
                action = 'hit'
            pair_strategy.append({
                'player_sum': 14,
                'dealer_card': dealer_card,
                'usable_ace': False,
                'is_split': True,
                'count_level': 0,
                'best_action': action
            })
        
        # 8,8: always split
        for dealer_card in range(2, 12):
            pair_strategy.append({
                'player_sum': 16,
                'dealer_card': dealer_card,
                'usable_ace': False,
                'is_split': True,
                'count_level': 0,
                'best_action': 'split'
            })
        
        # 9,9: split against 2-6, 8-9, otherwise stand
        for dealer_card in range(2, 12):
            if dealer_card in [2, 3, 4, 5, 6, 8, 9]:
                action = 'split'
            else:
                action = 'stand'
            pair_strategy.append({
                'player_sum': 18,
                'dealer_card': dealer_card,
                'usable_ace': False,
                'is_split': True,
                'count_level': 0,
                'best_action': action
            })
        
        # 10,10: always stand
        for dealer_card in range(2, 12):
            pair_strategy.append({
                'player_sum': 20,
                'dealer_card': dealer_card,
                'usable_ace': False,
                'is_split': True,
                'count_level': 0,
                'best_action': 'stand'
            })
        
        # A,A: always split
        for dealer_card in range(2, 12):
            pair_strategy.append({
                'player_sum': 12,  # Two aces value initially as 12 (1+11)
                'dealer_card': dealer_card,
                'usable_ace': True,
                'is_split': True,
                'count_level': 0,
                'best_action': 'split'
            })
        
        # Combine all strategies
        all_strategy = hard_strategy + soft_strategy + pair_strategy
        df = pd.DataFrame(all_strategy)
        
        return cls(df)