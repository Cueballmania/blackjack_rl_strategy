import sys
import os
import unittest
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack.environment import Card, Deck, Hand, BlackjackEnv


class TestCard(unittest.TestCase):
    def test_card_initialization(self):
        # Test numeric card
        card = Card('hearts', '7')
        self.assertEqual(card.suit, 'hearts')
        self.assertEqual(card.rank, '7')
        self.assertEqual(card.value, 7)
        
        # Test face card
        card = Card('spades', 'K')
        self.assertEqual(card.suit, 'spades')
        self.assertEqual(card.rank, 'K')
        self.assertEqual(card.value, 10)
        
        # Test ace
        card = Card('diamonds', 'A')
        self.assertEqual(card.suit, 'diamonds')
        self.assertEqual(card.rank, 'A')
        self.assertEqual(card.value, 11)
    
    def test_card_string_representation(self):
        card = Card('clubs', 'Q')
        self.assertEqual(str(card), 'Q of clubs')


class TestDeck(unittest.TestCase):
    def test_deck_initialization(self):
        # Test single deck
        deck = Deck(1)
        self.assertEqual(len(deck.cards), 52)
        
        # Test multiple decks
        deck = Deck(6)
        self.assertEqual(len(deck.cards), 6 * 52)
    
    def test_deck_deal(self):
        deck = Deck(1)
        initial_length = len(deck.cards)
        
        # Deal a card
        card = deck.deal()
        self.assertIsInstance(card, Card)
        self.assertEqual(len(deck.cards), initial_length - 1)
        
        # Deal all cards
        for _ in range(initial_length - 1):
            deck.deal()
        
        self.assertEqual(len(deck.cards), 0)
        
        # Deal from empty deck
        self.assertIsNone(deck.deal())
    
    def test_deck_shuffle(self):
        # Note: This test is probabilistic and might occasionally fail
        deck = Deck(1)
        original_order = deck.cards.copy()
        
        # Shuffle the deck
        deck.shuffle()
        
        # Check if order has changed
        is_different = False
        for i in range(len(deck.cards)):
            if i < len(original_order) and deck.cards[i] != original_order[i]:
                is_different = True
                break
        
        self.assertTrue(is_different, "Deck did not shuffle properly")
    
    def test_deck_reset(self):
        deck = Deck(1)
        
        # Deal some cards
        for _ in range(10):
            deck.deal()
        
        self.assertEqual(len(deck.cards), 42)
        
        # Reset the deck
        deck.reset()
        self.assertEqual(len(deck.cards), 52)


class TestHand(unittest.TestCase):
    def test_hand_initialization(self):
        # Test empty hand
        hand = Hand()
        self.assertEqual(len(hand.cards), 0)
        self.assertEqual(hand.get_value(), 0)
        
        # Test hand with cards
        cards = [Card('hearts', '5'), Card('clubs', 'K')]
        hand = Hand(cards)
        self.assertEqual(len(hand.cards), 2)
        self.assertEqual(hand.get_value(), 15)
    
    def test_hand_add_card(self):
        hand = Hand()
        
        # Add a card
        hand.add_card(Card('diamonds', '7'))
        self.assertEqual(len(hand.cards), 1)
        self.assertEqual(hand.get_value(), 7)
        
        # Add another card
        hand.add_card(Card('spades', 'J'))
        self.assertEqual(len(hand.cards), 2)
        self.assertEqual(hand.get_value(), 17)
    
    def test_hand_value_with_aces(self):
        # Test hand with one ace
        hand = Hand([Card('hearts', 'A'), Card('clubs', '5')])
        self.assertEqual(hand.get_value(), 16)
        
        # Test hand with multiple aces
        hand = Hand([Card('hearts', 'A'), Card('diamonds', 'A'), Card('clubs', '5')])
        self.assertEqual(hand.get_value(), 17)
        
        # Test hand where aces need to be valued at 1
        hand = Hand([Card('hearts', 'A'), Card('clubs', 'K')])
        self.assertEqual(hand.get_value(), 21)
        
        # Test bust with aces
        hand = Hand([Card('hearts', 'A'), Card('clubs', '10'), Card('diamonds', 'K')])
        self.assertEqual(hand.get_value(), 21)
        hand.add_card(Card('spades', '5'))
        self.assertEqual(hand.get_value(), 26)
        self.assertTrue(hand.busted)
    
    def test_hand_blackjack(self):
        # Test blackjack
        hand = Hand([Card('hearts', 'A'), Card('clubs', 'K')])
        self.assertTrue(hand.is_blackjack())
        
        # Test not blackjack (more than 2 cards)
        hand = Hand([Card('hearts', '5'), Card('clubs', '6'), Card('diamonds', '10')])
        self.assertEqual(hand.get_value(), 21)
        self.assertFalse(hand.is_blackjack())
        
        # Test not blackjack (not 21)
        hand = Hand([Card('hearts', 'Q'), Card('clubs', '9')])
        self.assertEqual(hand.get_value(), 19)
        self.assertFalse(hand.is_blackjack())
        
        # Test not blackjack (split hand)
        hand = Hand([Card('hearts', 'A'), Card('clubs', '10')])
        hand.is_split = True
        self.assertEqual(hand.get_value(), 21)
        self.assertFalse(hand.is_blackjack())
    
    def test_hand_split(self):
        # Test valid split
        hand = Hand([Card('hearts', '8'), Card('clubs', '8')])
        self.assertTrue(hand.can_split())
        
        new_hand = hand.split()
        self.assertEqual(len(hand.cards), 1)
        self.assertEqual(len(new_hand.cards), 1)
        self.assertEqual(hand.get_value(), 8)
        self.assertEqual(new_hand.get_value(), 8)
        self.assertTrue(hand.is_split)
        self.assertTrue(new_hand.is_split)
        
        # Test invalid split (different values)
        hand = Hand([Card('hearts', '8'), Card('clubs', '9')])
        self.assertFalse(hand.can_split())
        
        # Test invalid split (already split)
        hand = Hand([Card('hearts', '8'), Card('clubs', '8')])
        hand.is_split = True
        self.assertFalse(hand.can_split())
    
    def test_hand_double(self):
        # Test valid double
        hand = Hand([Card('hearts', '5'), Card('clubs', '6')])
        self.assertTrue(hand.can_double())
        
        # Test invalid double (more than 2 cards)
        hand = Hand([Card('hearts', '5'), Card('clubs', '6'), Card('diamonds', '2')])
        self.assertFalse(hand.can_double())
        
        # Test invalid double (already doubled)
        hand = Hand([Card('hearts', '5'), Card('clubs', '6')])
        hand.doubled = True
        self.assertFalse(hand.can_double())


class TestBlackjackEnv(unittest.TestCase):
    def test_environment_initialization(self):
        # Test with default config
        env = BlackjackEnv()
        self.assertEqual(env.config['num_decks'], 6)
        self.assertTrue(env.config['dealer_hit_soft_17'])
        
        # Test with custom config
        custom_config = {
            'num_decks': 2,
            'dealer_hit_soft_17': False,
            'max_splits': 2
        }
        env = BlackjackEnv(custom_config)
        self.assertEqual(env.config['num_decks'], 2)
        self.assertFalse(env.config['dealer_hit_soft_17'])
        self.assertEqual(env.config['max_splits'], 2)
    
    def test_environment_reset(self):
        env = BlackjackEnv()
        state = env.reset()
        
        # Check initial state
        self.assertIsNotNone(state['dealer_up_card'])
        # Dealer hand should have only the up card visible in the state
        self.assertEqual(len(state['dealer_hand']), 1)
        # But the actual dealer hand should have 2 cards
        self.assertEqual(len(env.dealer_hand.cards), 2)
        # Check that the hole card is stored
        self.assertIsNotNone(env.dealer_hole_card)
        
        self.assertEqual(len(state['player_hands']), 1)
        self.assertEqual(len(state['player_hands'][0]['cards']), 2)
        self.assertEqual(state['current_hand_index'], 0)
    
    def test_environment_step_hit(self):
        env = BlackjackEnv()
        env.reset()
        
        # Take a hit action
        next_state, reward, done, info = env.step('hit')
        
        # Check that a card was added to the player's hand
        self.assertEqual(len(next_state['player_hands'][0]['cards']), 3)
    
    def test_environment_step_stand(self):
        env = BlackjackEnv()
        env.reset()
        
        # Take a stand action
        next_state, reward, done, info = env.step('stand')
        
        # Check that the player's hand is marked as stood
        self.assertTrue(next_state['player_hands'][0]['stood'])
        # Game should be done since we only have one hand
        self.assertTrue(done)
    
    def test_environment_dealer_plays(self):
        env = BlackjackEnv()
        env.reset()
        
        # Force dealer to have a soft 17
        env.dealer_hand = Hand([Card('hearts', 'A'), Card('clubs', '6')])
        
        # Check dealer plays with hit soft 17 rule
        self.assertTrue(env.dealer_plays())
        
        # Change rule and check again
        env.config['dealer_hit_soft_17'] = False
        self.assertFalse(env.dealer_plays())
        
        # Test with hard 17
        env.dealer_hand = Hand([Card('hearts', '10'), Card('clubs', '7')])
        self.assertFalse(env.dealer_plays())
        
        # Test with 16
        env.dealer_hand = Hand([Card('hearts', '10'), Card('clubs', '6')])
        self.assertTrue(env.dealer_plays())
    
    def test_card_counting(self):
        env = BlackjackEnv()
        env.reset()
        env.card_count = 0  # Reset count
        
        # High cards should decrease the count
        env.update_count(Card('hearts', 'K'))
        self.assertEqual(env.card_count, -1)
        env.update_count(Card('clubs', 'A'))
        self.assertEqual(env.card_count, -2)
        
        # Low cards should increase the count
        env.update_count(Card('diamonds', '2'))
        self.assertEqual(env.card_count, -1)
        env.update_count(Card('spades', '6'))
        self.assertEqual(env.card_count, 0)
        
        # Neutral cards should not change the count
        env.update_count(Card('hearts', '7'))
        self.assertEqual(env.card_count, 0)
        env.update_count(Card('clubs', '9'))
        self.assertEqual(env.card_count, 0)
        
    def test_dealer_hole_card_counting(self):
        # Test that the dealer's hole card is only counted when revealed
        env = BlackjackEnv()
        env.reset()
        
        # Reset the count for predictable testing
        env.card_count = 0
        initial_count = env.card_count
        
        # Set a known hole card
        env.dealer_hole_card = Card('hearts', 'K')
        
        # Count should not change yet
        self.assertEqual(env.card_count, initial_count)
        
        # Resolve dealer's hand which reveals the hole card
        env.resolve_dealer_and_get_rewards()
        
        # Now the count should be updated (K is a high card, so -1)
        self.assertEqual(env.card_count, initial_count - 1)


if __name__ == '__main__':
    unittest.main()