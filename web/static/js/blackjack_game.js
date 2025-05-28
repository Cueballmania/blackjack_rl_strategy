document.addEventListener('DOMContentLoaded', () => {
    // Get DOM Elements
    const dealerHandDisplay = document.getElementById('dealer-hand'); // This is the div with class="hand-cards"
    const playerHandDisplay = document.getElementById('player-hand'); // This is the main div for all player hands
    
    // Score Spans within the new HTML structure
    const dealerScoreEl = document.getElementById('dealer-score'); 
    const playerScoreEl = document.getElementById('player-score'); 
    
    const gameMessagesDiv = document.getElementById('game-messages');

    const newGameButton = document.getElementById('new-game-button');
    const hitButton = document.getElementById('hit-button');
    const standButton = document.getElementById('stand-button');
    const doubleButton = document.getElementById('double-button');
    const splitButton = document.getElementById('split-button');

    // Action mapping
    const ACTION_MAP = {
        'hit': 0,
        'stand': 1,
        'double': 2,
        'split': 3,
    };

    let currentGameState = null; 

    function mapRankToFilenameRank(backendRank) {
        if (backendRank === 'A') return 'ace';
        if (backendRank === 'K') return 'king';
        if (backendRank === 'Q') return 'queen';
        if (backendRank === 'J') return 'jack';
        return backendRank; 
    }

    function getCardImageFilename(card) { 
        let suit, rank;
        if (Array.isArray(card)) {
            [suit, rank] = card;
        } else {
            // Fallback or error handling if card is not an array
            console.error("Invalid card format for getCardImageFilename:", card);
            return 'back_blue.svg'; // Default to a card back if format is wrong
        }

        // Handle the 'hidden' card case first
        if (suit === 'hidden' && rank === 'hidden') {
            return 'back_blue.svg';
        }

        const filenameRank = mapRankToFilenameRank(rank);
        let suitPrefix = '';

        switch (suit.toLowerCase()) { // Ensure consistent case for suit names
            case 'spades': suitPrefix = 'spade'; break;
            case 'hearts': suitPrefix = 'heart'; break;
            case 'clubs': suitPrefix = 'club_green'; break;
            case 'diamonds': suitPrefix = 'diamond_blue'; break;
            default: 
                console.warn(`Unknown suit: ${suit}. Defaulting to card back.`);
                return 'back_blue.svg'; 
        }
        return `${suitPrefix}_${filenameRank}.svg`;
    }
    
    function displayMessage(message, type = 'info') {
        gameMessagesDiv.innerHTML = ''; // Clear previous messages
        // Reset all possible message classes by setting className to its base and only class
        gameMessagesDiv.className = 'game-messages'; 
        // Add the new specific message class
        gameMessagesDiv.classList.add(`message-${type}`);
        
        const msgEl = document.createElement('p'); 
        msgEl.textContent = message;
        gameMessagesDiv.appendChild(msgEl);
    }


    function startGame() {
        displayMessage("Dealing new hand...", "info");

        fetch('/blackjack/new_game', { method: 'POST' })
            .then(response => response.json())
            .then(state => {
                currentGameState = state;
                renderGameState(state);
                setButtonsState(state.valid_actions || [], false); 
                newGameButton.disabled = true; 
                
                let initialMessage = "Game started. Your turn!";
                if (state.info && state.info.message) { // Message from backend at game start
                    initialMessage = state.info.message;
                }
                
                let playerBlackjack = false;
                let messageType = "info";

                if (state.player_hands && state.player_hands.length > 0 && state.player_hands.some(h => h.has_blackjack)) {
                    playerBlackjack = true;
                    initialMessage = "Player Blackjack!";
                    messageType = "win"; // Tentatively a win
                }

                // If backend immediately flags game as over (e.g. player BJ vs dealer no BJ)
                if (state.is_game_over) {
                    if (state.info && state.info.final_outcome) {
                        initialMessage = state.info.final_outcome;
                    }
                    // Determine message type from reward
                    if (state.reward > 0) messageType = "win";
                    else if (state.reward < 0) messageType = "lose";
                    else if (state.reward === 0) messageType = "push";
                    
                    setButtonsState([], true); // Disable action buttons
                    newGameButton.disabled = false; // Enable new game
                }
                displayMessage(initialMessage, messageType);
            })
            .catch(error => {
                console.error('Error starting new game:', error);
                displayMessage('Error starting new game. See console.', 'lose');
            });
    }

    function handleAction(actionName) {
        const actionCode = ACTION_MAP[actionName];
        if (actionCode === undefined) {
            console.error('Unknown action:', actionName);
            return;
        }
        setButtonsState([], true); 

        fetch('/blackjack/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: actionCode })
        })
        .then(response => response.json())
        .then(result => {
            if (result.error) {
                console.error('Action error:', result.error);
                displayMessage(`Error: ${result.error}`, 'lose');
                setButtonsState(currentGameState ? (currentGameState.valid_actions || []) : [], false);
                newGameButton.disabled = !(currentGameState && currentGameState.is_game_over); // Re-evaluate based on current state
                return;
            }

            currentGameState = result.new_state; 
            renderGameState(result.new_state); 
            
            let messageText = "";
            let messageType = "info"; 

            if (result.info) {
                if (result.info.message) { 
                    messageText = result.info.message;
                }
                if (result.info.hand_messages && result.info.hand_messages.length > 0) {
                    // If there's already a general message, append hand messages. Otherwise, use hand messages.
                    messageText = messageText ? `${messageText} ${result.info.hand_messages.join(" / ")}` : result.info.hand_messages.join(" / ");
                }
            }
            
            if (result.done) { 
                let finalOutcomeMessage = "Game Over.";
                if (result.info && result.info.final_outcome) { 
                    finalOutcomeMessage = result.info.final_outcome;
                }
                messageText = messageText ? `${messageText}. ${finalOutcomeMessage}` : finalOutcomeMessage;
                messageText += ` (Overall Reward: ${result.reward})`;

                if (result.reward > 0) messageType = 'win';
                else if (result.reward < 0) messageType = 'lose';
                else messageType = 'push';
                
                setButtonsState([], true); 
                newGameButton.disabled = false;
            } else { 
                setButtonsState(result.new_state.valid_actions || [], false);
                newGameButton.disabled = true;
                if (messageText.trim() === "" && result.new_state.player_hands[result.new_state.current_hand_index] && 
                    !result.new_state.player_hands[result.new_state.current_hand_index].is_busted &&
                    !result.new_state.player_hands[result.new_state.current_hand_index].stood) {
                     messageText = `Hand ${result.new_state.current_hand_index + 1}: Your turn.`;
                } else if (messageText.trim() === "" && result.new_state.current_hand_index < result.new_state.player_hands.length) {
                     messageText = `Next action for Hand ${result.new_state.current_hand_index + 1}.`;
                } else if (messageText.trim() === "") {
                    messageText = "Your turn."; // Generic fallback for ongoing game
                }
            }
            displayMessage(messageText.trim(), messageType);
        })
        .catch(error => {
            console.error('Error performing action:', error);
            displayMessage('Error performing action. See console.', 'lose');
            setButtonsState(currentGameState ? (currentGameState.valid_actions || []) : [], false);
            newGameButton.disabled = !(currentGameState && currentGameState.is_game_over);
        });
    }

    function renderGameState(state) {
        // --- Dealer's Hand & Score ---
        dealerHandDisplay.innerHTML = ''; 
        if (state.dealer_hand && state.dealer_hand.length > 0) {
            state.dealer_hand.forEach(cardTuple => {
                const img = document.createElement('img');
                img.src = `/static/images/cards/${getCardImageFilename(cardTuple)}`;
                img.alt = (cardTuple[0] === 'hidden') ? 'Hidden Card' : `${mapRankToFilenameRank(cardTuple[1])} of ${cardTuple[0]}`;
                img.className = 'card-image';
                dealerHandDisplay.appendChild(img);
            });
        }
        
        if (state.is_game_over) {
            dealerScoreEl.textContent = state.dealer_score !== undefined ? state.dealer_score : 'N/A';
        } else {
            if (state.dealer_hand && state.dealer_hand.length > 0 && state.dealer_hand[0][0] !== 'hidden') {
                const upCard = state.dealer_hand[0]; // [suit, rank]
                const rank = upCard[1];
                let upCardValue = 0;
                if (rank === 'A') upCardValue = 11;
                else if (['K', 'Q', 'J'].includes(rank)) upCardValue = 10;
                else upCardValue = parseInt(rank); // Assumes rank is '2'-'10'
                dealerScoreEl.textContent = upCardValue;
            } else {
                dealerScoreEl.textContent = '?'; 
            }
        }

        // --- Player's Hands & Score ---
        playerHandDisplay.innerHTML = ''; 
        
        state.player_hands.forEach((hand, index) => {
            const handContainer = document.createElement('div');
            handContainer.className = 'hand-container';
            if (index === state.current_hand_index && !state.is_game_over) {
                handContainer.classList.add('active-hand');
            } else {
                handContainer.classList.remove('active-hand');
            }

            const infoDiv = document.createElement('div');
            infoDiv.className = 'hand-info';
            let handStatus = '';
            if (hand.has_blackjack) handStatus = ' - BLACKJACK!';
            else if (hand.is_busted) handStatus = ' - BUSTED!';
            else if (hand.stood) handStatus = ' - Stood';
            
            let doubledText = hand.doubled || hand.doubled_down ? ' (Doubled)' : '';
            
            infoDiv.textContent = `Hand ${index + 1} (Value: ${hand.value})${handStatus}${doubledText}`;
            handContainer.appendChild(infoDiv);

            const cardsDisplay = document.createElement('div');
            cardsDisplay.className = 'hand-cards';
            if (hand.cards) { 
                hand.cards.forEach(cardDetails => { 
                    const img = document.createElement('img');
                    img.src = '/static/images/cards/' + getCardImageFilename(cardDetails);
                    img.alt = `${mapRankToFilenameRank(cardDetails[1])} of ${cardDetails[0]}`;
                    img.className = 'card-image';
                    cardsDisplay.appendChild(img);
                });
            }
            handContainer.appendChild(cardsDisplay);
            playerHandDisplay.appendChild(handContainer);
        });

        // Update general player score display (span id="player-score")
        if (state.is_game_over) {
            // Display sum of all non-busted hand values or a general "Game Over" message for score
            let totalPlayerValue = 0;
            let allBusted = true;
            state.player_hands.forEach(h => { 
                if (!h.is_busted) {
                    totalPlayerValue += h.value;
                    allBusted = false;
                }
            });
            playerScoreEl.textContent = allBusted ? 'All Bust' : totalPlayerValue;

        } else if (state.player_hands.length > 0 && state.current_hand_index < state.player_hands.length) {
            playerScoreEl.textContent = state.player_hands[state.current_hand_index].value || '0';
        } else if (state.player_hands.length > 0) {
            playerScoreEl.textContent = state.player_hands[0].value || '0'; 
        }
         else {
            playerScoreEl.textContent = '0';
        }
    }

    function setButtonsState(validActions, disableAll = false) {
        const actionsArray = Array.isArray(validActions) ? validActions : [];
        hitButton.disabled = disableAll || !actionsArray.includes(ACTION_MAP['hit']);
        standButton.disabled = disableAll || !actionsArray.includes(ACTION_MAP['stand']);
        doubleButton.disabled = disableAll || !actionsArray.includes(ACTION_MAP['double']);
        splitButton.disabled = disableAll || !actionsArray.includes(ACTION_MAP['split']);
    }

    // Event Listeners
    newGameButton.addEventListener('click', startGame);
    hitButton.addEventListener('click', () => handleAction('hit'));
    standButton.addEventListener('click', () => handleAction('stand'));
    doubleButton.addEventListener('click', () => handleAction('double'));
    splitButton.addEventListener('click', () => handleAction('split'));

    // Initial Call to start the game
    startGame();
});
