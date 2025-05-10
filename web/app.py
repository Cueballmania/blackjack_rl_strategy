from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import sys
import time
from threading import Thread

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack.environment import BlackjackEnv
from reinforcement.q_learning import BlackjackQTable, DQNAgent
from blackjack.strategy import StrategyTable

app = Flask(__name__)

# Global variables to store the model and results
model = None
training_thread = None
training_progress = {
    'status': 'idle',
    'progress': 0,
    'message': '',
    'results': None
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/train', methods=['POST'])
def train():
    global model, training_thread, training_progress
    
    # Check if training is already in progress
    if training_progress['status'] == 'running':
        return jsonify({'error': 'Training already in progress'}), 400
    
    # Get configuration from request
    config = request.json
    
    # Update training status
    training_progress['status'] = 'running'
    training_progress['progress'] = 0
    training_progress['message'] = 'Starting training...'
    training_progress['results'] = None
    
    # Start training in a separate thread
    training_thread = Thread(target=run_training, args=(config,))
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'status': 'started'})

def run_training(config):
    global model, training_progress
    
    try:
        # Create environment with the specified configuration
        blackjack_config = {
            'num_decks': config.get('num_decks', 6),
            'dealer_hit_soft_17': config.get('dealer_hit_soft_17', True),
            'double_after_split': config.get('double_after_split', True),
            'max_splits': config.get('max_splits', 3),
            'allow_double_on': config.get('allow_double_on', [9, 10, 11]),
            'allow_surrender': config.get('allow_surrender', False),
            'resplit_aces': config.get('resplit_aces', False),
            'hits_on_split_aces': config.get('hits_on_split_aces', False),
        }
        
        env = BlackjackEnv(blackjack_config)
        
        # Choose algorithm: Q-table or DQN
        algorithm = config.get('algorithm', 'q_table')
        episodes = config.get('episodes', 100000)
        
        if algorithm == 'q_table':
            # Create Q-learning agent
            model = BlackjackQTable(
                learning_rate=config.get('learning_rate', 0.1),
                discount_factor=config.get('discount_factor', 0.95),
                exploration_rate=config.get('exploration_rate', 1.0),
                exploration_decay=config.get('exploration_decay', 0.9999),
                min_exploration_rate=config.get('min_exploration_rate', 0.01)
            )
        else:  # DQN
            # State dimensions will be automatically set in the DQNAgent constructor
            model = DQNAgent(
                learning_rate=config.get('learning_rate', 0.001),
                discount_factor=config.get('discount_factor', 0.95),
                exploration_rate=config.get('exploration_rate', 1.0),
                exploration_decay=config.get('exploration_decay', 0.9999),
                min_exploration_rate=config.get('min_exploration_rate', 0.01),
                memory_size=config.get('memory_size', 10000),
                batch_size=config.get('batch_size', 64)
            )
        
        # Training loop
        chunk_size = 1000  # Update progress every 1000 episodes
        total_chunks = episodes // chunk_size
        
        for i in range(total_chunks):
            training_progress['progress'] = (i / total_chunks) * 100
            training_progress['message'] = f'Training episode {i*chunk_size}/{episodes}...'
            
            # Train for a chunk of episodes
            if algorithm == 'q_table':
                model.train(env, chunk_size)
            else:  # DQN
                model.train(env, chunk_size, target_update_frequency=10)
            
            # Sleep briefly to prevent hogging CPU
            time.sleep(0.1)
        
        # Generate strategy tables for different count levels
        count_levels = config.get('count_levels', [0])
        strategy_tables = {}
        
        try:
            print("Generating strategy tables for count levels:", count_levels)
            
            # Get initial strategy dataframe for the first count level
            if algorithm == 'q_table':
                strategy_df = model.generate_strategy_table(count_levels[0])
            else:  # DQN
                strategy_df = model.generate_strategy_table(env, count_levels[0])
            
            print(f"Strategy dataframe columns: {strategy_df.columns}")
            print(f"Strategy dataframe shape: {strategy_df.shape}")
            
            # Store all strategy tables
            for count_level in count_levels:
                if algorithm == 'q_table':
                    level_df = model.generate_strategy_table(count_level)
                else:  # DQN
                    level_df = model.generate_strategy_table(env, count_level)
                
                strategy_tables[str(count_level)] = level_df.to_dict(orient='records')
            
            # Calculate average reward over last 1000 episodes
            avg_reward = np.mean(model.episode_rewards[-1000:])
            
            # Create learned strategy object
            learned_strategy = StrategyTable(strategy_df)
            
            # Generate strategy visualizations
            print("Generating strategy visualizations...")
            strategy_images = generate_strategy_visualizations(learned_strategy, count_levels)
            
        except Exception as e:
            import traceback
            print(f"Error during strategy generation: {str(e)}")
            print(traceback.format_exc())
            
            # Provide simplified data if error occurs
            avg_reward = np.mean(model.episode_rewards[-1000:]) if model.episode_rewards else 0
            strategy_images = {str(level): {'error': True, 'message': str(e)} for level in count_levels}
        
        # Update progress
        training_progress['status'] = 'complete'
        training_progress['progress'] = 100
        training_progress['message'] = f'Training complete. Average reward: {avg_reward:.4f}'
        training_progress['results'] = {
            'avg_reward': float(avg_reward),
            'strategy_tables': strategy_tables,
            'strategy_images': strategy_images
        }
        
        print('Training complete with average reward:', float(avg_reward))
    
    except Exception as e:
        # Update progress with error
        training_progress['status'] = 'error'
        training_progress['message'] = f'Error: {str(e)}'

def generate_strategy_visualizations(strategy_table, count_levels):
    """Generate raw strategy data for different scenarios."""
    strategy_data = {}
    scenarios = [
        {'usable_ace': False, 'is_split': False, 'has_pair': False, 'name': 'hard_totals', 'title': 'Hard Totals Strategy'},
        {'usable_ace': True, 'is_split': False, 'has_pair': False, 'name': 'soft_totals', 'title': 'Soft Totals Strategy'},
        {'usable_ace': False, 'is_split': False, 'has_pair': True, 'name': 'pair_splitting', 'title': 'Pair Splitting Strategy'}
    ]
    
    # Store action colors for client-side rendering
    action_info = {
        'hit': {'color': '#ff6b6b', 'text': 'white', 'abbr': 'H', 'full': 'Hit'},
        'stand': {'color': '#51cf66', 'text': 'white', 'abbr': 'S', 'full': 'Stand'},
        'double': {'color': '#ffa94d', 'text': 'white', 'abbr': 'D', 'full': 'Double'},
        'split': {'color': '#339af0', 'text': 'white', 'abbr': 'P', 'full': 'Split'},
        'surrender': {'color': '#cc5de8', 'text': 'white', 'abbr': 'R', 'full': 'Surrender'}
    }
    
    for count_level in count_levels:
        strategy_data[str(count_level)] = {'action_info': action_info}
        
        for scenario in scenarios:
            try:
                # Print debug info
                print(f"Generating strategy for {scenario['name']}, count={count_level}")
                
                # Create strategy matrix
                matrix = strategy_table.create_strategy_matrix(
                    usable_ace=scenario['usable_ace'],
                    is_split=scenario['is_split'],
                    has_pair=scenario['has_pair'],
                    count_level=count_level
                )
                
                # Check if the matrix is empty
                if matrix.empty:
                    print(f"Warning: Empty matrix for {scenario['name']}, count={count_level}")
                    strategy_data[str(count_level)][scenario['name']] = {
                        'error': True,
                        'message': 'No data available for this strategy/count combination'
                    }
                    continue
                
                # Print matrix shape
                print(f"Matrix shape: {matrix.shape}, columns: {matrix.columns}")
                
                # Convert to a simple data structure for JSON
                table_data = {
                    'title': scenario['title'],
                    'count_level': count_level,
                    'columns': [str(col) if col != 11 else 'A' for col in matrix.columns],
                    'rows': []
                }
                
                # Add each row of data
                for idx, row in matrix.iterrows():
                    row_data = {
                        'label': str(idx),
                        'cells': []
                    }
                    
                    for action in row:
                        if pd.isna(action):
                            row_data['cells'].append(None)
                        else:
                            row_data['cells'].append(str(action))  # Ensure action is a string
                    
                    table_data['rows'].append(row_data)
                
                strategy_data[str(count_level)][scenario['name']] = table_data
                print(f"Successfully generated strategy for {scenario['name']}, count={count_level}")
                
            except Exception as e:
                # Detailed error logging
                import traceback
                print(f"Error generating strategy for {scenario['name']}, count={count_level}: {str(e)}")
                print(traceback.format_exc())
                
                # If visualization fails, return error flag
                strategy_data[str(count_level)][scenario['name']] = {
                    'error': True,
                    'message': str(e)
                }
    
    return strategy_data

@app.route('/api/training_status', methods=['GET'])
def training_status():
    global training_progress
    return jsonify({
        'status': training_progress['status'],
        'progress': training_progress['progress'],
        'message': training_progress['message']
    })

@app.route('/api/training_results', methods=['GET'])
def training_results():
    global training_progress
    
    if training_progress['status'] != 'complete':
        return jsonify({'error': 'Training not complete'}), 400
    
    try:
        # Return just the metadata and available count levels
        results = training_progress['results']
        count_levels = []
        
        if 'strategy_images' in results:
            count_levels = list(results['strategy_images'].keys())
        
        # Create a minimal response
        minimal_response = {
            'avg_reward': results.get('avg_reward', 0),
            'count_levels': count_levels
        }
        
        return jsonify(minimal_response)
    except Exception as e:
        import traceback
        print(f"Error in training_results: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            'error': f'Error generating results: {str(e)}',
            'avg_reward': 0,
            'count_levels': [],
            'simplified_response': True
        })

@app.route('/api/strategy_table', methods=['GET'])
def strategy_table():
    global training_progress
    
    if training_progress['status'] != 'complete' or 'results' not in training_progress:
        return jsonify({'error': 'Training not complete'}), 400
    
    # Get query parameters
    count_level = request.args.get('count_level', '0')
    table_type = request.args.get('type', 'hard_totals')  # Options: hard_totals, soft_totals, pair_splitting
    
    try:
        # Extract just the requested strategy table
        results = training_progress['results']
        
        if 'strategy_images' not in results or count_level not in results['strategy_images']:
            return jsonify({'error': 'Requested strategy table not found'}), 404
        
        strategy_data = results['strategy_images'][count_level]
        
        # Get just the requested table type and the action info
        response = {
            'action_info': strategy_data.get('action_info', {}),
            'table_data': strategy_data.get(table_type, {'error': True, 'message': 'Table not found'})
        }
        
        return jsonify(response)
    except Exception as e:
        import traceback
        print(f"Error in strategy_table: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            'error': f'Error retrieving strategy table: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)