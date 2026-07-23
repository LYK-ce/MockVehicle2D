# Contributing to MockVehicle2D

## Getting Started

```bash
git clone git@github.com:LYK-ce/MockVehicle2D.git
cd MockVehicle2D
bash bootstrap.sh
source .venv/bin/activate
mockvehicle2d test          # verify everything works
```

## Development Workflow

1. **Create a branch** off `main`:
   ```bash
   git checkout -b feature/my-feature
   # or: git checkout -b fix/my-bug
   ```

2. **Write code.** Keep it minimal — solve the problem, nothing more.
   - Library code goes in `src/mockvehicle2d/`
   - Tests go in `tests/`
   - CLI commands go in `src/mockvehicle2d/cli/`

3. **Run tests** before committing:
   ```bash
   mockvehicle2d test
   ```

4. **Commit** with a descriptive message:
   ```bash
   git commit -m "Fix command timeout handling"
   ```

5. **Push and open a Pull Request.**

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints where helpful
- No dead code, TODOs, or commented-out code
- New features require tests

## PR Checklist

Every pull request must:
- [ ] Pass all tests (`mockvehicle2d test`)
- [ ] Include tests for new functionality
- [ ] Update relevant documentation
- [ ] Be free of dead code and TODOs

## Project Structure

```
MockVehicle2D/
├── src/mockvehicle2d/       ← library code
│   ├── cli/                 ← CLI entry points
│   ├── map_grid.py          ← 2D grid map
│   ├── collision.py         ← collision detection
│   ├── navigation.py        ← direct go-to-goal control
│   ├── safety.py            ← local safety sensing and limits
│   ├── server.py            ← WebSocket server
│   └── visual.py            ← Pygame visualization
├── tests/                   ← test suite
├── docs/                    ← design documents
└── bootstrap.sh             ← one-shot setup script
```
