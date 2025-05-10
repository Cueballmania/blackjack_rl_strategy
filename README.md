# Blackjack RL Strategy Generator

This project is a reinforcement learning-based blackjack strategy table generator. It allows users to customize various blackjack rules and generate optimal strategy tables based on card counting.

## Features

- Custom blackjack environment with configurable rules:
  - Number of decks
  - Dealer hitting on soft 17
  - Rules for doubling down
  - Rules for splitting
  - Surrender option
- Reinforcement learning algorithms:
  - Q-learning with lookup tables
  - Deep Q-Network (DQN)
- Web interface for customizing rules and visualizing strategies
- Card counting strategy generation for different count levels
- Comparison with basic strategy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/blackjack_rl_strategy.git
   cd blackjack_rl_strategy
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Interface

```bash
python main.py web
```

This will start a web server at http://127.0.0.1:5000/ where you can access the user interface.

Options:
- `--host`: Host to run the web app on (default: 127.0.0.1)
- `--port`: Port to run the web app on (default: 5000)
- `--debug`: Run in debug mode

### Running Unit Tests

```bash
python main.py test
```

## Project Structure

- `blackjack/`: Blackjack game environment
  - `environment.py`: Implementation of the blackjack game
  - `strategy.py`: Strategy table generation and visualization
- `reinforcement/`: Reinforcement learning algorithms
  - `q_learning.py`: Q-learning and DQN implementations
- `web/`: Web interface
  - `app.py`: Flask web application
  - `templates/`: HTML templates
  - `static/`: Static files (CSS, JavaScript)
- `tests/`: Unit tests
- `main.py`: Main script to run the application

## Customizable Rules

- **Number of Decks**: 1, 2, 4, 6, or 8 decks
- **Dealer Hits on Soft 17**: Whether the dealer hits or stands on soft 17
- **Double After Split**: Whether doubling down is allowed after splitting
- **Maximum Splits**: Maximum number of splits allowed
- **Allow Double On**: Which hand values allow doubling down
- **Allow Surrender**: Whether surrender is allowed

## Reinforcement Learning Settings

- **Algorithm**: Q-Learning or Deep Q-Network (DQN)
- **Training Episodes**: Number of episodes to train for
- **Learning Rate**: Rate at which the agent learns
- **Discount Factor**: How much future rewards are valued
- **Count Levels**: Which count levels to generate strategies for

## Strategy Tables

The application generates three types of strategy tables:

1. **Hard Totals**: Strategy for hands without usable aces
2. **Soft Totals**: Strategy for hands with usable aces
3. **Pair Splitting**: Strategy for pairs

Each table shows the recommended action for each player hand value against each dealer up card. The actions are color-coded for easy reference:

- **Hit (H)**: Red
- **Stand (S)**: Green
- **Double (D)**: Orange
- **Split (P)**: Blue
- **Surrender (R)**: Purple

## License

This project is licensed under the MIT License - see the LICENSE file for details.