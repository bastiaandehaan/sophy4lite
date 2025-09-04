# CLAUDE.md - AI-Driven Development Workflow

This file provides comprehensive guidance to Claude Code for working with this repository using an AI-driven development approach.

## 🎯 AI-Driven Development Workflow

### Phase 1: Documentation & Standards Setup

#### Knowledge Base Foundation
- **CLAUDE.md**: Central knowledge base (this file)
- **Architecture principles**: Modular, testable, maintainable
- **Test strategies**: Pytest-based with comprehensive coverage
- **Deployment procedures**: Local development → testing → production

#### Memory & Behavior Configuration
- **Small incremental steps**: Maximum 1 component at a time
- **Always test before proceeding**: No exceptions
- **Document decisions**: Every architectural choice explained
- **Validate with user**: Get approval before major changes

### Phase 2: Project Execution Protocol

#### Step 1: Functional Document Creation
When user requests: "I want to build functionality X"
1. Create detailed functional document including:
   - Clear functionality description
   - Technical implementation approach  
   - Required file structure changes
   - Database/data model modifications
   - Comprehensive test scenarios
   - Dependencies and integration points

#### Step 2: Document Review Cycle
1. User reviews document thoroughly
2. User asks questions about unclear parts
3. Claude refines and corrects document
4. Repeat until document is complete and approved
5. **NO IMPLEMENTATION** until document is approved

#### Step 3: Implementation in Small Steps
1. Claude announces: "Implementing step X: [specific task]"
2. Claude implements and tests single component
3. Claude reports: "Step X complete, please review"
4. User validates result
5. **Next step ONLY after user approval**

### Phase 3: Quality & Monitoring

#### Continuous Quality Checks
- Run tests after every significant change: `pytest test/ -v`
- Use TodoWrite tool for progress tracking
- Validate imports and syntax regularly
- Check git status before and after changes

#### Feedback Loops
- Use "stop" if direction is incorrect
- Ask for explanations when implementation unclear
- Regular architecture validation
- Document all decisions and reasoning

## 🏗️ Framework Overview

Sophy4Lite is a lean trading research framework focused on ATR-based breakout strategies with realistic backtesting and FTMO compliance guards. It emphasizes falsifiability over optimization.

### Core Architecture

- **backtest/**: Backtesting engines and data loading
  - `data_loader.py`: CSV data ingestion with timezone handling
  - `mtf_exec_fast.py`: Fast vectorized multi-timeframe confluence execution engine
  - `runner.py`: Central backtest runner supporting multiple strategies
- **strategies/**: Trading strategy implementations
  - `base.py`: Base strategy interface (currently minimal)
  - `mtf_confluence.py`: Multi-timeframe confluence strategy
  - `order_block.py`: Order block strategy implementation
- **risk/**: Risk management and compliance
  - `ftmo_guard.py`: FTMO challenge compliance rules (daily/total loss limits)
- **cli/**: Command-line interface using Typer
  - `main.py`: Primary CLI entry point with confluence command
- **utils/**: Shared utilities for metrics, plotting, and data health
- **live/**: Live trading integration (MT5 on Windows)
- **config.py**: Global configuration and logging setup

## 💻 PyCharm Workflow Instructions

### Code Style & Formatting
- Use PyCharm's auto-formatting: `Ctrl+Alt+L`
- Follow PEP 8 standards strictly
- Use type hints for all function parameters and returns
- Keep Dutch comments/docstrings where they exist
- Use English for new code and comments

### Testing Workflow
- Run tests via PyCharm's test runner or `pytest test/ -v`
- Always run tests before committing changes
- Write tests for new functionality immediately
- Use pytest configuration in `pyproject.toml`

### Debugging & Development
- Use PyCharm debugger for complex issues
- Set strategic breakpoints for problem analysis
- Inspect variables in debug mode
- Use PyCharm's integrated terminal for CLI commands

### Git Integration
- Use PyCharm's built-in git tools
- Always check `git status` before committing
- Write meaningful commit messages
- Never commit to main without user approval

## 🔧 Development Commands

### Running Backtests
```bash
# Primary CLI - Multi-timeframe confluence backtest
python -m cli.main <SYMBOL> --csv <PATH_TO_M1_CSV> [options]

# Specific confluence command  
python -m cli.main confluence <SYMBOL> --csv <PATH_TO_M1_CSV>

# Framework validation
python validate_framework.py
```

### Testing Commands
```bash
# Run all tests with verbose output
pytest test/ -v

# Run specific test file
pytest test/test_mtf_confluence.py -v

# Run tests with coverage
pytest test/ --cov=strategies --cov=backtest
```

### Common Parameters
- `--start YYYY-MM-DD` / `--end YYYY-MM-DD`: Date range
- `--server-tz`: Server timezone (default: Europe/Athens)
- `--session-start` / `--session-end`: Trading session hours
- `--risk-pct`: Risk percentage per trade (default: 1.0%)
- `--min-score`: Minimum confluence score threshold
- `--max-daily`: Maximum trades per day

## ⚙️ Key Configuration

### Risk Management (config.py)
- `INITIAL_CAPITAL`: 20000.0
- `RISK_PER_TRADE`: 0.01 (1%)
- `MAX_DAILY_LOSS`: 0.05 (5%)
- `MAX_TOTAL_LOSS`: 0.10 (10%)

### Dependencies
Core requirements: pandas, numpy, typer, rich, pytest
Optional: MetaTrader5 (Windows only, for live trading)

## 📋 Session Routine

### Per Session Checklist
1. **Status Check**: "Where did we leave off?"
2. **Goal Definition**: "What are we implementing today?"
3. **Plan Creation**: Use TodoWrite for progress tracking
4. **Small Steps**: Maximum 1 component at a time
5. **Validation**: Test and review before next step

### Quality Gates
- ✅ All tests pass before proceeding
- ✅ Code follows established patterns
- ✅ User approves before major changes
- ✅ Documentation updated when needed

## ⚠️ Critical Guidelines

### What Works Well
- Rapid prototyping and boilerplate generation
- Architecture discussions and code reviews
- Test automation and debugging support
- Documentation and standards enforcement

### Where to Be Careful
- **Complex business logic**: User must lead and validate
- **New technologies**: Research together first
- **Large refactoring**: Break into smallest possible steps
- **Security-sensitive code**: Extra review required

### Role Distribution

**User's Role:**
- Define functional requirements clearly
- Make architectural decisions
- Review and approve code changes
- Validate business logic correctness

**Claude's Role:**
- Execute implementation following standards
- Create comprehensive documentation
- Automate testing procedures
- Apply best practices consistently
- Ask clarifying questions when uncertain

## 📁 Project Structure Standards

```
/project
├── CLAUDE.md              # This file - central knowledge base
├── docs/                  # Project documentation
│   ├── architecture.md    # System architecture
│   ├── standards.md       # Coding standards
│   └── implementation-log.md  # Decision history
├── templates/             # Reusable code templates
├── test/                  # Test suite
└── src/                   # Source code
```

## 🔍 Important Framework Notes

### Language & Comments
- Primary codebase: Mix of Dutch and English
- Keep existing Dutch comments intact
- Use English for new code and documentation
- CLI help text remains in Dutch

### Timezone Handling
- Framework expects UTC-normalized data
- Server timezone conversion in data loading
- Critical for multi-session instruments

### FTMO Compliance
- Pre-trade risk checking via `FtmoGuard`
- Daily and total drawdown limits enforced
- No look-ahead bias in backtesting

### Data Requirements
- M1 OHLC CSV format expected
- Automatic column detection and normalization
- Required columns: open, high, low, close (case-insensitive)