# Card Games Package - TODO

## Code Structure
- [ ] **Correct pathing** - Verify imports use `rl_card_lib_core` and cross-game references
- [ ] **Game base validation** - Ensure all games properly implement CardGame interface
- [ ] **State serialization** - Implement consistent state serialization for all games

## Klondike Solitaire
- [ ] **Test game rules** - Comprehensive tests for game logic
- [ ] **Test win conditions** - Verify win/loss detection
- [ ] **Test action validity** - Ensure legal_actions are correct
- [ ] **Performance** - Optimize hot-path operations
- [ ] **State space analysis** - Document state space characteristics

## Macao Game
- [ ] **Test game rules** - Comprehensive tests for bidding and play
- [ ] **Test multi-player** - Verify N-player interactions
- [ ] **Test edge cases** - Corner cases in bidding/play
- [ ] **State normalization** - Ensure deterministic state encoding

## New Games Framework
- [ ] **Template/example** - Create starter template for new games
- [ ] **Validation helpers** - Utility functions for game validation

## Testing
- [ ] **Unit tests** - All game rule implementations
- [ ] **Integration tests** - Games with different agents
- [ ] **Regression tests** - Known issue prevention

## Documentation
- [ ] **Game rules** - Document rules for each game
- [ ] **State format** - Document state representation
- [ ] **Action space** - Document action encoding
- [ ] **Reward design** - Document reward structure
