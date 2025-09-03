#!/usr/bin/env python3
"""
Sophy4Lite Repository Validator
Controleert het hele framework op fouten en biedt fixes.
"""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import List, Dict


class FrameworkValidator:
    def __init__(self, root_path: Path = Path.cwd()):
        self.root = root_path
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []

    def validate_all(self) -> bool:
        """Hoofdfunctie die alle checks uitvoert."""
        print(">> Starting Sophy4Lite Framework Validation...\n")

        # 1. Python syntax check
        self._check_syntax()

        # 2. Import validation
        self._check_imports()

        # 3. Missing files check
        self._check_required_files()

        # 4. Type hints check (optioneel met mypy)
        self._check_types()

        # 5. Framework-specifieke checks
        self._check_framework_specific()

        # Report
        self._print_report()

        return len(self.errors) == 0

    def _check_syntax(self):
        """Controleert Python syntax in alle .py files."""
        print("1. Checking Python syntax...")

        for py_file in self.root.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                self.errors.append(
                    {'file': str(py_file.relative_to(self.root)), 'line': e.lineno,
                        'type': 'SyntaxError', 'msg': str(e.msg)})

        print(f"   >> Checked {len(list(self.root.rglob('*.py')))} files\n")

    def _check_imports(self):
        """Valideert alle imports."""
        print("2. Checking imports...")

        # KRITIEKE FIX voor live_trading.py
        live_trading = self.root / "live" / "live_trading.py"
        if live_trading.exists():
            with open(live_trading, 'r') as f:
                content = f.read()
                if "from risk.ftmo import" in content:
                    self.errors.append({'file': 'live/live_trading.py', 'line': 6,
                        'type': 'ImportError',
                        'msg': "Module 'risk.ftmo' does not exist! Use 'risk.ftmo_guard' instead",
                        'fix': "Remove line 6 or create risk/ftmo.py with fixed_percent_sizing function"})

        # Check alle andere imports
        for py_file in self.root.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            self._validate_file_imports(py_file)

        print(f"   >> Import validation complete\n")

    def _validate_file_imports(self, file_path: Path):
        """Valideert imports in een specifiek bestand."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._check_module_exists(alias.name, file_path, node.lineno)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Check relatieve imports binnen project
                        if not node.level and not node.module.startswith('.'):
                            if node.module.startswith(
                                    ('backtest', 'cli', 'risk', 'strategies', 'utils',
                                     'live')):
                                module_path = self.root / node.module.replace('.', '/')
                                if not module_path.exists() and not (
                                module_path.with_suffix('.py')).exists():
                                    self.errors.append(
                                        {'file': str(file_path.relative_to(self.root)),
                                            'line': node.lineno, 'type': 'ImportError',
                                            'msg': f"Local module '{node.module}' not found"})
        except Exception as e:
            self.warnings.append({'file': str(file_path.relative_to(self.root)),
                'msg': f"Could not parse imports: {e}"})

    def _check_module_exists(self, module_name: str, file_path: Path, line: int):
        """Controleert of een module bestaat."""
        try:
            if module_name in ['backtest', 'cli', 'risk', 'strategies', 'utils',
                               'live']:
                # Lokale modules
                module_path = self.root / module_name
                if not module_path.exists():
                    self.errors.append(
                        {'file': str(file_path.relative_to(self.root)), 'line': line,
                            'type': 'ImportError',
                            'msg': f"Local module '{module_name}' directory missing"})
            else:
                # External modules
                spec = importlib.util.find_spec(module_name.split('.')[0])
                if spec is None and module_name not in ['MetaTrader5', 'vectorbt']:
                    self.warnings.append(
                        {'file': str(file_path.relative_to(self.root)), 'line': line,
                            'msg': f"External module '{module_name}' not installed"})
        except (ImportError, ModuleNotFoundError, ValueError):
            pass  # Optionele dependencies

    def _check_required_files(self):
        """Controleert of alle vereiste bestanden aanwezig zijn."""
        print("3. Checking required files...")

        required = ['backtest/__init__.py', 'cli/__init__.py', 'risk/__init__.py',
            'strategies/__init__.py', 'utils/__init__.py', 'config.py',
            'requirements.txt']

        for req_file in required:
            if not (self.root / req_file).exists():
                self.errors.append({'file': req_file, 'type': 'MissingFile',
                    'msg': f"Required file '{req_file}' is missing"})

        # Check of output directory bestaat
        output_dir = self.root / 'output'
        if not output_dir.exists():
            self.warnings.append(
                {'msg': "Output directory missing - will cause crash in config.py",
                    'fix': "Run: mkdir output"})

        print(f"   >> File structure validated\n")

    def _check_types(self):
        """Optioneel: run mypy voor type checking."""
        print("4. Checking types (if mypy available)...")

        try:
            result = subprocess.run(
                ['mypy', '--ignore-missing-imports', '--no-error-summary',
                 str(self.root)], capture_output=True, text=True, timeout=10)
            if result.returncode != 0 and result.stdout:
                # Parse alleen de belangrijkste type errors
                for line in result.stdout.split('\n')[:10]:  # Max 10 errors
                    if 'error:' in line:
                        self.warnings.append({'msg': line.strip()})
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("   WARNING: mypy not available, skipping type checks")

        print()

    def _check_framework_specific(self):
        """Sophy4Lite specifieke validaties."""
        print("5. Checking Sophy4Lite specific requirements...")

        # Check timezone handling
        breakout_exec = self.root / "backtest" / "breakout_exec.py"
        if breakout_exec.exists():
            with open(breakout_exec, 'r') as f:
                content = f.read()
                if 'tz_localize(tz)' in content and 'df.index.tz is None' in content:
                    self.warnings.append(
                        {'file': 'backtest/breakout_exec.py', 'line': 24,
                            'msg': "Dangerous timezone handling - blindly localizes to UTC",
                            'fix': "Check original timezone before localizing"})

        # Check ATR calculation for look-ahead bias
        breakout_signals = self.root / "strategies" / "breakout_signals.py"
        if breakout_signals.exists():
            with open(breakout_signals, 'r') as f:
                content = f.read()
                if 'df.loc[:w.index.max()]' in content:
                    self.warnings.append(
                        {'file': 'strategies/breakout_signals.py', 'line': 91,
                            'msg': "Potential look-ahead bias in ATR calculation",
                            'fix': "Verify that w.index.max() doesn't include future data"})

        print(f"   >> Framework validation complete\n")

    def _print_report(self):
        """Print het eindrapport."""
        print("=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        if self.errors:
            print(f"\nCRITICAL ERRORS ({len(self.errors)}):")
            for i, err in enumerate(self.errors, 1):
                print(f"  {i}. [{err['type']}] {err['file']}")
                if 'line' in err:
                    print(f"     Line {err['line']}: {err['msg']}")
                else:
                    print(f"     {err['msg']}")
                if 'fix' in err:
                    print(f"     💡 FIX: {err['fix']}")
                print()

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for i, warn in enumerate(self.warnings, 1):
                if 'file' in warn:
                    print(f"  {i}. {warn['file']}")
                    if 'line' in warn:
                        print(f"     Line {warn['line']}")
                print(f"     {warn['msg']}")
                if 'fix' in warn:
                    print(f"     💡 FIX: {warn['fix']}")
                print()

        if not self.errors and not self.warnings:
            print("\n>> All checks passed! Framework is ready to run.\n")
        elif self.errors:
            print("\n🛑 Fix critical errors before running the framework!\n")
        else:
            print("\n>> No critical errors, but review warnings.\n")

        print("=" * 60)

        # Quick fixes sectie
        if self.errors:
            print("\n>> QUICK FIXES TO APPLY:")
            print("-" * 40)
            print("1. Fix import in live/live_trading.py:")
            print("   DELETE line 6: from risk.ftmo import fixed_percent_sizing")
            print("\n2. Create output directory:")
            print("   RUN: mkdir output")
            print("\n3. Install missing dependencies:")
            print("   RUN: pip install -r requirements.txt")
            print("-" * 40)


def auto_fix_critical_issues():
    """Automatisch fixen van de meest kritieke issues."""
    print("\n>> Attempting auto-fixes...\n")

    # Fix 1: Maak output directory
    output_dir = Path.cwd() / 'output'
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)
        print(">> Created output/ directory")

    # Fix 2: Comment out broken import
    live_trading = Path.cwd() / "live" / "live_trading.py"
    if live_trading.exists():
        with open(live_trading, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "from risk.ftmo import fixed_percent_sizing" in line:
                lines[
                    i] = "# " + line + "# TODO: Create risk/ftmo.py or remove this import\n"
                with open(live_trading, 'w') as f:
                    f.writelines(lines)
                print(">> Commented out broken import in live/live_trading.py")
                break

    # Fix 3: Create missing __init__.py files
    for module in ['backtest', 'cli', 'risk', 'strategies', 'utils', 'live', 'test',
                   'scripts']:
        init_file = Path.cwd() / module / '__init__.py'
        if (Path.cwd() / module).exists() and not init_file.exists():
            init_file.touch()
            print(f">> Created {module}/__init__.py")

    print("\n>> Auto-fixes applied!\n")


if __name__ == "__main__":
    validator = FrameworkValidator()

    # Vraag om auto-fix
    print(">> Sophy4Lite Framework Validator\n")
    response = input(
        "Apply automatic fixes for critical issues? (y/n): ").strip().lower()

    if response == 'y':
        auto_fix_critical_issues()

    # Run validation
    is_valid = validator.validate_all()

    sys.exit(0 if is_valid else 1)