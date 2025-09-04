# Coding Standards

## Python Code Standards

### Style Guidelines
- **PEP 8 compliance**: Enforced via PyCharm formatting
- **Type hints**: Required for all function parameters and returns
- **Docstrings**: Google style for all public functions
- **Line length**: 100 characters maximum
- **Import organization**: Standard library, third-party, local imports

### Naming Conventions
- **Functions**: snake_case
- **Classes**: PascalCase  
- **Constants**: UPPER_SNAKE_CASE
- **Variables**: snake_case
- **Private methods**: _leading_underscore

### Language Usage
- **Existing code**: Keep Dutch comments intact
- **New code**: English comments and documentation
- **CLI interface**: Dutch help text (existing pattern)
- **Error messages**: English for logs, Dutch for CLI

## Testing Standards

### Test Structure
- **Location**: All tests in `test/` directory
- **Naming**: `test_*.py` files, `test_*` functions
- **Framework**: pytest with fixtures
- **Coverage**: Aim for >80% code coverage

### Test Categories
- **Unit tests**: Individual function testing
- **Integration tests**: Component interaction testing  
- **Smoke tests**: Basic functionality validation
- **Critical tests**: Look-ahead bias detection

## Documentation Standards

### Code Documentation
- **Docstrings**: All public functions and classes
- **Inline comments**: For complex business logic
- **Type hints**: Complete parameter and return typing
- **README files**: For each major component

### Architecture Documentation
- **Decision records**: Document major architectural choices
- **API documentation**: For all public interfaces
- **Setup instructions**: Clear installation and configuration
- **Usage examples**: Working code samples

## Git Standards

### Commit Messages
- **Format**: `type(scope): description`
- **Types**: feat, fix, docs, style, refactor, test, chore
- **Scope**: Component being modified
- **Description**: Clear, concise action description

### Branching Strategy
- **Main branch**: Production-ready code
- **Feature branches**: For new development
- **Hot fixes**: Critical bug fixes
- **Review process**: All changes reviewed before merge

## Performance Standards

### Code Performance
- **Vectorization**: Use pandas/numpy operations
- **Memory efficiency**: Avoid unnecessary data copies
- **Profiling**: Measure before optimizing
- **Caching**: Cache expensive computations

### Testing Performance
- **Fast tests**: Unit tests under 1 second
- **Integration tests**: Under 10 seconds each
- **Full suite**: Complete in under 2 minutes
- **Parallel execution**: Use pytest-xdist when needed