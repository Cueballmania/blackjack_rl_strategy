import numpy as np
from typing import List, Dict, Tuple, Optional, Union


class Card:
    """Representation of a playing card."""
    
    def __init__(self, suit: str, rank: str):
        """
        Initialize a card with a suit and rank.
        
        Args:
            suit: The suit of the card ('hearts', 'diamonds', 'clubs', 'spades')
            rank: The rank of the card ('2', '3', ..., '10', 'J', 'Q', 'K', 'A')
        """
        self.suit = suit
        self.rank = rank
        
        # Map card values
        if rank in ['J', 'Q', 'K']:
            self.value = 10
        elif rank == 'A':
            self.value = 11  # Ace is initially 11, can be changed to 1
        else:
            self.value = int(rank)
    
    def __str__(self) -> str:
        return f"{self.rank} of {self.suit}"
    
    def __repr__(self) -> str:
        return self.__str__()


class Deck:
    """Representation of a deck of cards."""
    
    def __init__(self, num_decks: int = 1):
        """
        Initialize a deck of cards.
        
        Args:
            num_decks: Number of standard decks to use
        """
        self.num_decks = num_decks
        self.cards = []
        self.reset()
    
    def reset(self):
        """Reset the deck to its initial state."""
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        self.cards = []
        for _ in range(self.num_decks):
            for suit in suits:
                for rank in ranks:
                    self.cards.append(Card(suit, rank))
        
        self.shuffle()
    
    def shuffle(self):
        """Shuffle the deck."""
        np.random.shuffle(self.cards)
    
    def deal(self) -> Optional[Card]:
        """Deal a card from the deck."""
        if not self.cards:
            return None
        return self.cards.pop()
    
    def cards_remaining(self) -> int:
        """Get the number of cards remaining in the deck."""
        return len(self.cards)
    
    def __len__(self) -> int:
        return len(self.cards)


class Hand:
    """Representation of a blackjack hand."""
    
    def __init__(self, cards: List[Card] = None):
        """
        Initialize a hand with optional starting cards.
        
        Args:
            cards: Initial list of cards (default: None)
        """
        self.cards = cards or []
        self.is_split = False
        self.doubled = False
        self.stood = False
        self.busted = False
    
    def add_card(self, card: Card):
        """Add a card to the hand."""
        self.cards.append(card)
        if self.get_value() > 21:
            self.busted = True
    
    def get_value(self) -> int:
        """Calculate the value of the hand, accounting for aces."""
        value = sum(card.value for card in self.cards)
        num_aces = sum(1 for card in self.cards if card.rank == 'A')
        
        # Adjust for aces if needed
        while value > 21 and num_aces > 0:
            value -= 10  # Change an ace from 11 to 1
            num_aces -= 1
        
        return value
    
    def is_blackjack(self) -> bool:
        """Check if the hand is a blackjack (an ace and a 10-value card)."""
        return len(self.cards) == 2 and self.get_value() == 21 and not self.is_split
    
    def can_split(self) -> bool:
        """Check if the hand can be split."""
        return (len(self.cards) == 2 and 
                self.cards[0].value == self.cards[1].value and 
                not self.is_split)
    
    def can_double(self) -> bool:
        """Check if the hand can be doubled."""
        return len(self.cards) == 2 and not self.doubled
    
    def split(self) -> 'Hand':
        """Split the hand into two hands."""
        if not self.can_split():
            raise ValueError("Cannot split this hand")
        
        new_hand = Hand([self.cards.pop()])
        new_hand.is_split = True
        self.is_split = True
        return new_hand
    
    def __str__(self) -> str:
        cards_str = ", ".join(str(card) for card in self.cards)
        return f"Hand ({self.get_value()}): {cards_str}"
    
    def __repr__(self) -> str:
        return self.__str__()


class BlackjackEnv:
    """Blackjack game environment."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the blackjack environment with configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                - num_decks: Number of decks to use (default: 6)
                - dealer_hit_soft_17: Whether dealer hits on soft 17 (default: True)
                - double_after_split: Whether doubling is allowed after splitting (default: True)
                - max_splits: Maximum number of splits allowed (default: 3)
                - allow_double_on: List of hand values where doubling is allowed (default: [9, 10, 11])
                - allow_surrender: Whether surrender is allowed (default: False)
        """
        default_config = {
            'num_decks': 6,
            'dealer_hit_soft_17': True,
            'double_after_split': True,
            'max_splits': 3,
            'allow_double_on': [9, 10, 11],  # Hand values where doubling is allowed
            'allow_surrender': False,
            'resplit_aces': False,  # Whether aces can be split more than once
            'hits_on_split_aces': False,  # Whether split aces can receive additional cards
        }
        
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        self.deck = Deck(self.config['num_decks'])
        self.dealer_hand = None
        self.player_hands = []
        self.current_hand_index = 0
        self.card_count = 0  # For card counting
    
    def reset(self) -> Dict:
        """Reset the environment and deal initial cards."""
        if self.deck.cards_remaining() < 52:  # Reshuffle if less than 1 deck remains
            self.deck.reset()
            self.card_count = 0
        
        self.dealer_hand = Hand()
        self.player_hands = [Hand()]
        self.current_hand_index = 0
        self.dealer_hole_card = None  # Store the dealer's hole card separately
        
        # Deal initial cards to player (both face up)
        for _ in range(2):
            card = self.deck.deal()
            self.player_hands[0].add_card(card)
            self.update_count(card)  # Both player cards are counted
        
        # Deal first card to dealer (face up)
        card = self.deck.deal()
        self.dealer_hand.add_card(card)
        self.update_count(card)  # Dealer's up card is counted
        
        # Deal second card to dealer (face down) - don't update count yet
        self.dealer_hole_card = self.deck.deal()
        self.dealer_hand.add_card(self.dealer_hole_card)  # Add to hand but don't count
        
        return self.get_state()
    
    def update_count(self, card: Card):
        """Update the running count based on the dealt card."""
        # Simple high-low count strategy
        if card.value >= 10 or card.rank == 'A':
            self.card_count -= 1
        elif card.value <= 6:
            self.card_count += 1
        # Cards 7, 8, 9 are considered neutral (0)
    
    def get_true_count(self) -> float:
        """Get the true count (running count divided by decks remaining)."""
        decks_remaining = max(1, self.deck.cards_remaining() / 52)
        return self.card_count / decks_remaining
    
    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment based on the action.
        
        Args:
            action: One of 'hit', 'stand', 'double', 'split', 'surrender'
        
        Returns:
            Tuple of (new_state, reward, done, info)
        """
        current_hand = self.player_hands[self.current_hand_index]
        reward = 0.0
        done = False
        info = {}
        
        if action == 'hit':
            card = self.deck.deal()
            current_hand.add_card(card)
            self.update_count(card)
            
            if current_hand.busted:
                reward = -1.0
                self.current_hand_index += 1
        
        elif action == 'stand':
            current_hand.stood = True
            self.current_hand_index += 1
        
        elif action == 'double':
            if current_hand.can_double() and \
               (not current_hand.is_split or self.config['double_after_split']) and \
               current_hand.get_value() in self.config['allow_double_on']:
                current_hand.doubled = True
                card = self.deck.deal()
                current_hand.add_card(card)
                self.update_count(card)
                
                if current_hand.busted:
                    reward = -2.0  # Double the loss
                else:
                    reward = 0.0  # Will be calculated at the end
                
                self.current_hand_index += 1
            else:
                # Invalid move, treat as stand
                current_hand.stood = True
                self.current_hand_index += 1
        
        elif action == 'split':
            can_split = current_hand.can_split()
            # Check for ace restrictions
            has_ace = any(card.rank == 'A' for card in current_hand.cards)
            already_split = any(hand.is_split for hand in self.player_hands)
            max_splits_reached = len(self.player_hands) >= self.config['max_splits'] + 1
            
            # Cannot split aces again if resplit_aces is False
            if has_ace and already_split and not self.config['resplit_aces']:
                can_split = False
            
            if can_split and not max_splits_reached:
                new_hand = current_hand.split()
                
                # Deal a card to each hand
                card = self.deck.deal()
                current_hand.add_card(card)
                self.update_count(card)
                
                card = self.deck.deal()
                new_hand.add_card(card)
                self.update_count(card)
                
                # If splitting aces and hits not allowed, mark them as stood
                if has_ace and not self.config['hits_on_split_aces']:
                    current_hand.stood = True
                    new_hand.stood = True
                
                self.player_hands.insert(self.current_hand_index + 1, new_hand)
            else:
                # Invalid move, treat as stand
                current_hand.stood = True
                self.current_hand_index += 1
        
        elif action == 'surrender':
            if self.config['allow_surrender'] and len(current_hand.cards) == 2:
                reward = -0.5  # Half the bet is lost
                current_hand.stood = True
                current_hand.surrendered = True
                self.current_hand_index += 1
            else:
                # Invalid move, treat as stand
                current_hand.stood = True
                self.current_hand_index += 1
        
        # Check if all hands are played
        if self.current_hand_index >= len(self.player_hands):
            done = True
            rewards = self.resolve_dealer_and_get_rewards()
            reward = rewards[0]  # Return reward for the initial hand
            info['all_rewards'] = rewards
        
        return self.get_state(), reward, done, info
    
    def resolve_dealer_and_get_rewards(self) -> List[float]:
        """Resolve the dealer's hand and calculate rewards for all player hands."""
        # First, count the dealer's hole card now that it's revealed
        if self.dealer_hole_card:
            self.update_count(self.dealer_hole_card)
            self.dealer_hole_card = None
        
        # Play dealer's hand
        while self.dealer_plays():
            card = self.deck.deal()
            self.dealer_hand.add_card(card)
            self.update_count(card)
        
        dealer_value = self.dealer_hand.get_value()
        dealer_blackjack = self.dealer_hand.is_blackjack()
        
        rewards = []
        for hand in self.player_hands:
            if hasattr(hand, 'surrendered') and hand.surrendered:
                rewards.append(-0.5)
                continue
                
            if hand.busted:
                reward = -1.0
            elif dealer_blackjack and not hand.is_blackjack():
                reward = -1.0
            elif hand.is_blackjack() and not dealer_blackjack:
                reward = 1.5  # Blackjack pays 3:2
            elif hand.is_blackjack() and dealer_blackjack:
                reward = 0.0  # Push
            elif self.dealer_hand.busted:
                reward = 1.0
            elif hand.get_value() > dealer_value:
                reward = 1.0
            elif hand.get_value() < dealer_value:
                reward = -1.0
            else:
                reward = 0.0  # Push
            
            # Double the reward if the hand was doubled
            if hand.doubled:
                reward *= 2.0
                
            rewards.append(reward)
        
        return rewards
    
    def dealer_plays(self) -> bool:
        """Determine if the dealer should hit according to the rules."""
        value = self.dealer_hand.get_value()
        
        if value < 17:
            return True
        
        if value == 17:
            # Check for soft 17
            has_ace = any(card.rank == 'A' for card in self.dealer_hand.cards)
            aces_as_11 = sum(1 for card in self.dealer_hand.cards if card.rank == 'A' and card.value == 11)
            
            if aces_as_11 > 0 and self.config['dealer_hit_soft_17']:
                return True
        
        return False
    
    def get_state(self) -> Dict:
        """Get the current state of the game."""
        # Dealer's up card is the first card (index 0 in standard blackjack)
        dealer_up_card = self.dealer_hand.cards[0].value if self.dealer_hand and self.dealer_hand.cards else None
        
        current_hand = self.player_hands[self.current_hand_index] if self.current_hand_index < len(self.player_hands) else None
        
        # For the state representation, only show the dealer's up card unless game is done
        # Dealer's up card is the first card (index 0 in standard blackjack)
        dealer_up_card_obj = self.dealer_hand.cards[0] if self.dealer_hand and self.dealer_hand.cards else None
        dealer_up_card_info = (dealer_up_card_obj.suit, dealer_up_card_obj.rank) if dealer_up_card_obj else None

        dealer_visible_cards_info = []
        # If all player hands are done, or game is over, show the complete dealer hand with suits and ranks
        # The check for all hands being stood or busted implies game is effectively over for player interaction.
        # The actual reveal of dealer's hole card happens in resolve_dealer_and_get_rewards.
        # For state representation, we can show full dealer hand if it's "dealer's turn" or game ended.
        
        # Simplified: if current_hand_index >= len(self.player_hands), it's dealer's turn or game over.
        # More robust: check a game_over flag if available, or if all player hands are resolved.
        # For now, let's assume if player is done, dealer cards are revealed for state.
        # The actual game logic ensures hole card is not used in counting until revealed.
        
        is_player_phase_done = self.current_hand_index >= len(self.player_hands) or \
                               all(hand.stood or hand.busted or (hasattr(hand, 'surrendered') and hand.surrendered) 
                                   for hand in self.player_hands)

        if is_player_phase_done:
            # Show all dealer cards (suit, rank)
            dealer_visible_cards_info = [(card.suit, card.rank) for card in self.dealer_hand.cards] if self.dealer_hand else []
        elif dealer_up_card_obj:
            # Show only dealer's up card (suit, rank) and a placeholder for the hole card
            dealer_visible_cards_info = [dealer_up_card_info, ('hidden', 'hidden')] # Placeholder for hole card
        
        return {
            'dealer_up_card': dealer_up_card_info, # Tuple (suit, rank) or None
            'dealer_hand': dealer_visible_cards_info, # List of (suit, rank) tuples, or includes hidden placeholder
            'player_hands': [
                {
                    # 'cards_values': [card.value for card in hand.cards], # Retain for compatibility or remove
                    'cards': [(card.suit, card.rank) for card in hand.cards], # List of (suit, rank) tuples
                    'card_ranks': [card.rank for card in hand.cards], # Keep for convenience in JS if needed
                    'value': hand.get_value(),
                    'is_split': hand.is_split,
                    'doubled': hand.doubled,
                    'stood': hand.stood,
                    'busted': hand.busted
                } for hand in self.player_hands
            ],
            'current_hand_index': self.current_hand_index,
            'running_count': self.card_count,
            'true_count': self.get_true_count(),
            'cards_remaining': self.deck.cards_remaining(),
            'valid_actions': self.get_valid_actions(current_hand) if current_hand else []
        }
    
    def get_valid_actions(self, hand: Hand) -> List[str]:
        """Get the list of valid actions for the current hand."""
        valid_actions = ['hit', 'stand']
        
        if hand.can_double() and \
           (not hand.is_split or self.config['double_after_split']) and \
           hand.get_value() in self.config['allow_double_on']:
            valid_actions.append('double')
        
        if hand.can_split() and len(self.player_hands) < self.config['max_splits'] + 1:
            valid_actions.append('split')
        
        if self.config['allow_surrender'] and len(hand.cards) == 2:
            valid_actions.append('surrender')
        
        return valid_actions