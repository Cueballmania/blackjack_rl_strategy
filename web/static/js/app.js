document.addEventListener('DOMContentLoaded', function() {
    const configForm = document.getElementById('configForm');
    const statusText = document.getElementById('statusText');
    const progressBar = document.getElementById('progressBar');
    const trainingMessage = document.getElementById('trainingMessage');
    const resultsSection = document.getElementById('resultsSection');
    const avgReward = document.getElementById('avgReward');
    const countSelect = document.getElementById('countSelect');
    const hardImage = document.getElementById('hardImage');
    const softImage = document.getElementById('softImage');
    const pairsImage = document.getElementById('pairsImage');
    
    let trainingResults = null;
    let pollingInterval = null;
    
    // Handle form submission
    configForm.addEventListener('submit', function(e) {
        e.preventDefault();
        startTraining();
    });
    
    function startTraining() {
        // Get form data
        const formData = new FormData(configForm);
        const config = {};
        
        // Parse form data
        config.num_decks = parseInt(formData.get('num_decks'));
        config.dealer_hit_soft_17 = formData.get('dealer_hit_soft_17') === 'on';
        config.double_after_split = formData.get('double_after_split') === 'on';
        config.max_splits = parseInt(formData.get('max_splits'));
        config.allow_surrender = formData.get('allow_surrender') === 'on';
        config.resplit_aces = formData.get('resplit_aces') === 'on';
        config.hits_on_split_aces = formData.get('hits_on_split_aces') === 'on';
        
        // Parse doubling rules
        const allow_double_on = [];
        if (formData.get('double_9') === 'on') allow_double_on.push(9);
        if (formData.get('double_10') === 'on') allow_double_on.push(10);
        if (formData.get('double_11') === 'on') allow_double_on.push(11);
        if (formData.get('double_any') === 'on') {
            // Allow doubling on any value (2-21)
            for (let i = 2; i <= 21; i++) {
                if (!allow_double_on.includes(i)) {
                    allow_double_on.push(i);
                }
            }
        }
        config.allow_double_on = allow_double_on;
        
        // Parse training settings
        config.algorithm = formData.get('algorithm');
        config.episodes = parseInt(formData.get('episodes'));
        config.learning_rate = parseFloat(formData.get('learning_rate'));
        config.discount_factor = parseFloat(formData.get('discount_factor'));
        
        // Parse count levels
        const count_levels = [];
        if (formData.get('count_0') === 'on') count_levels.push(0);
        if (formData.get('count_neg2') === 'on') count_levels.push(-2);
        if (formData.get('count_neg1') === 'on') count_levels.push(-1);
        if (formData.get('count_1') === 'on') count_levels.push(1);
        if (formData.get('count_2') === 'on') count_levels.push(2);
        config.count_levels = count_levels;
        
        // Disable form
        const formElements = configForm.elements;
        for (let i = 0; i < formElements.length; i++) {
            formElements[i].disabled = true;
        }
        
        // Reset UI
        statusText.textContent = 'Starting...';
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';
        trainingMessage.textContent = 'Initializing training...';
        resultsSection.style.display = 'none';
        
        // Send training request
        fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'started') {
                // Start polling for status updates
                statusText.textContent = 'Running';
                startPolling();
            } else {
                alert('Failed to start training: ' + data.error);
                resetForm();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred: ' + error.message);
            resetForm();
        });
    }
    
    function startPolling() {
        // Clear any existing interval
        if (pollingInterval) {
            clearInterval(pollingInterval);
        }
        
        // Poll for status updates every second
        pollingInterval = setInterval(function() {
            fetch('/api/training_status')
                .then(response => response.json())
                .then(data => {
                    // Update UI
                    statusText.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
                    progressBar.style.width = data.progress + '%';
                    progressBar.textContent = Math.round(data.progress) + '%';
                    trainingMessage.textContent = data.message;
                    
                    // Check if training is complete
                    if (data.status === 'complete') {
                        clearInterval(pollingInterval);
                        fetchResults();
                    } else if (data.status === 'error') {
                        clearInterval(pollingInterval);
                        alert('Training error: ' + data.message);
                        resetForm();
                    }
                })
                .catch(error => {
                    console.error('Error polling status:', error);
                });
        }, 1000);
    }
    
    function fetchResults() {
        // First fetch just the metadata (average reward and count levels)
        fetch('/api/training_results')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(metadata => {
                console.log('Metadata received:', metadata);
                
                // Initialize results object with metadata
                trainingResults = {
                    avg_reward: metadata.avg_reward,
                    strategy_images: {}
                };
                
                // If no count levels, display what we have
                if (!metadata.count_levels || metadata.count_levels.length === 0) {
                    displayResults();
                    resetForm();
                    return;
                }
                
                // Fetch each strategy table separately
                const fetchPromises = [];
                const tableTypes = ['hard_totals', 'soft_totals', 'pair_splitting'];
                
                // For each count level, fetch each table type
                metadata.count_levels.forEach(countLevel => {
                    // Initialize this count level
                    trainingResults.strategy_images[countLevel] = {};
                    
                    tableTypes.forEach(tableType => {
                        const fetchPromise = fetch(`/api/strategy_table?count_level=${countLevel}&type=${tableType}`)
                            .then(response => {
                                if (!response.ok) {
                                    throw new Error(`HTTP error fetching ${tableType} for count ${countLevel}!`);
                                }
                                return response.json();
                            })
                            .then(data => {
                                // Store the action info if we don't have it yet
                                if (!trainingResults.strategy_images[countLevel].action_info && data.action_info) {
                                    trainingResults.strategy_images[countLevel].action_info = data.action_info;
                                }
                                
                                // Store the table data
                                trainingResults.strategy_images[countLevel][tableType] = data.table_data;
                                console.log(`Loaded ${tableType} for count ${countLevel}`);
                            })
                            .catch(error => {
                                console.error(`Error loading ${tableType} for count ${countLevel}:`, error);
                                trainingResults.strategy_images[countLevel][tableType] = {
                                    error: true,
                                    message: error.message
                                };
                            });
                        
                        fetchPromises.push(fetchPromise);
                    });
                });
                
                // Wait for all fetches to complete, then display results
                Promise.allSettled(fetchPromises).then(() => {
                    console.log('All strategy tables loaded');
                    displayResults();
                    resetForm();
                });
            })
            .catch(error => {
                console.error('Error fetching training results:', error);
                alert('Failed to fetch results: ' + error.message);
                resetForm();
            });
    }
    
    function displayResults() {
        if (!trainingResults) return;
        
        // Show results section
        resultsSection.style.display = 'block';
        
        try {
            // Display average reward if available
            if (trainingResults.avg_reward !== undefined) {
                avgReward.textContent = trainingResults.avg_reward.toFixed(4);
            } else {
                avgReward.textContent = 'N/A';
            }
            
            // Check if strategy images exist
            if (!trainingResults.strategy_images || Object.keys(trainingResults.strategy_images).length === 0) {
                document.getElementById('strategyTabsContent').innerHTML = 
                    '<div class="alert alert-warning">No strategy tables were generated</div>';
                return;
            }
            
            // Populate count levels dropdown
            countSelect.innerHTML = '';
            Object.keys(trainingResults.strategy_images).forEach(count => {
                const option = document.createElement('option');
                option.value = count;
                option.textContent = `Count: ${count}`;
                countSelect.appendChild(option);
            });
            
            // Display initial strategy images
            const firstCount = Object.keys(trainingResults.strategy_images)[0];
            if (firstCount) {
                updateStrategyImages(firstCount);
            }
            
            // Add event listener for count level changes
            countSelect.addEventListener('change', function() {
                updateStrategyImages(this.value);
            });
        } catch (error) {
            console.error('Error displaying results:', error);
            document.getElementById('strategyTabsContent').innerHTML = 
                `<div class="alert alert-danger">Error displaying strategy tables: ${error.message}</div>`;
        }
    }
    
    function updateStrategyImages(countLevel) {
        try {
            const strategyData = trainingResults.strategy_images[countLevel];
            if (!strategyData) {
                console.error('No strategy data for count level:', countLevel);
                return;
            }
            
            // Get action colors and info
            const actionInfo = strategyData.action_info || {
                'hit': {'color': '#ff6b6b', 'text': 'white', 'abbr': 'H', 'full': 'Hit'},
                'stand': {'color': '#51cf66', 'text': 'white', 'abbr': 'S', 'full': 'Stand'},
                'double': {'color': '#ffa94d', 'text': 'white', 'abbr': 'D', 'full': 'Double'},
                'split': {'color': '#339af0', 'text': 'white', 'abbr': 'P', 'full': 'Split'},
                'surrender': {'color': '#cc5de8', 'text': 'white', 'abbr': 'R', 'full': 'Surrender'}
            };
            
            // Generate tables for each strategy type
            const strategyTypes = ['hard_totals', 'soft_totals', 'pair_splitting'];
            const targetIds = ['hard', 'soft', 'pairs'];
            
            strategyTypes.forEach((type, index) => {
                const targetElement = document.getElementById(targetIds[index]);
                const tableData = strategyData[type];
                
                if (!tableData) {
                    targetElement.innerHTML = `<div class="alert alert-warning">No ${type.replace('_', ' ')} strategy available</div>`;
                    return;
                }
                
                if (tableData.error) {
                    targetElement.innerHTML = `<div class="alert alert-danger">Error: ${tableData.message}</div>`;
                    return;
                }
                
                // Generate HTML for the table
                let html = `<h4>${tableData.title} - Count: ${tableData.count_level}</h4>`;
                html += "<p>Player's Hand Value (Rows) vs Dealer's Upcard (Columns)</p>";
                html += "<div class='table-responsive'>";
                html += "<table class='table table-bordered strategy-table'>";
                
                // Header row with dealer cards
                html += "<thead><tr><th>Player</th>";
                tableData.columns.forEach(col => {
                    html += `<th>${col}</th>`;
                });
                html += "</tr></thead>";
                
                // Body with player hands and actions
                html += "<tbody>";
                tableData.rows.forEach(row => {
                    html += `<tr><th>${row.label}</th>`;
                    row.cells.forEach(action => {
                        if (!action) {
                            html += "<td></td>";
                        } else {
                            const style = actionInfo[action] ? 
                                `background-color: ${actionInfo[action].color}; color: ${actionInfo[action].text};` : 
                                '';
                            const abbr = actionInfo[action] ? 
                                `${actionInfo[action].full} (${actionInfo[action].abbr})` : 
                                action;
                            html += `<td style="${style}">${actionInfo[action] ? actionInfo[action].abbr : action}</td>`;
                        }
                    });
                    html += "</tr>";
                });
                html += "</tbody>";
                html += "</table></div>";
                
                // Add legend
                html += "<div class='strategy-legend mt-2 mb-4'><strong>Legend:</strong> ";
                Object.entries(actionInfo).forEach(([action, info]) => {
                    const style = `background-color: ${info.color}; color: ${info.text};`;
                    html += `<span class='legend-item' style='${style}'>${info.full} (${info.abbr})</span> `;
                });
                html += "</div>";
                
                targetElement.innerHTML = html;
            });
        } catch (error) {
            console.error('Error updating strategy images:', error);
            const errorHtml = `<div class="alert alert-danger">Error displaying strategy tables: ${error.message}</div>`;
            document.getElementById('hard').innerHTML = errorHtml;
            document.getElementById('soft').innerHTML = errorHtml;
            document.getElementById('pairs').innerHTML = errorHtml;
        }
    }
    
    function resetForm() {
        // Enable form
        const formElements = configForm.elements;
        for (let i = 0; i < formElements.length; i++) {
            formElements[i].disabled = false;
        }
    }
});