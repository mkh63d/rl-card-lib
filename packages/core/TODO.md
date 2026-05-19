# Core Package - TODO

## Code Structure
- [ ] **Correct pathing** - Ensure all imports use `rl_card_lib_core` namespace correctly
- [ ] **Circular imports** - Verify no circular dependencies between modules
- [ ] **__init__.py exports** - Ensure public API is properly exported

## Testing
- [ ] **Unit tests for Card** - Card creation, comparison, serialization
- [ ] **Unit tests for Deck** - Shuffling, dealing, composition
- [ ] **Unit tests for Player** - Hand management, state
- [ ] **Unit tests for CardGame base** - Abstract class contract
- [ ] **Unit tests for Agent base** - Agent interface
- [ ] **Unit tests for Trainer** - Training loop, metrics
- [ ] **Type hints** - Add comprehensive type hints to all classes

## Documentation
- [ ] **Docstrings** - Add detailed docstrings to all public methods
- [ ] **Type hints in signatures** - Ensure type hints for all functions
- [ ] **Example code** - Add usage examples in docstrings

## API Stability
- [ ] **Deprecation markers** - Mark any unstable APIs with @deprecated
- [ ] **Versioning** - Lock version compatibility

## Publishing
- [ ] **Package metadata** - Verify pyproject.toml is complete
- [ ] **License headers** - Add MIT license headers to source files
- [ ] **CHANGELOG** - Create CHANGELOG for tracking changes
