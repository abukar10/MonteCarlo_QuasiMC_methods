## MonteCarlo and Quasi MonteCarlo methods for pricing Derivative Pricing (Python)

This project translates several MATLAB MonteCarlo based code into Python, including RNGs, GBM simulations, vanilla options (crude and antithetic MC), importance sampling for a straddle, a stochastic-vol antithetic demo, Asian options with control variates, and QMC via Halton/Sobol.

### Setup

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\pip install -r requirements.txt
```

### Run everything

```bash
.\.venv\Scripts\python run_all.py
```

Outputs (charts and a results summary) are saved in `outputs/`.

### Structure

- `mc_pricing/` modules mirror the MATLAB snippets
- `run_all.py` executes all modules and writes charts/prices
- `outputs/` contains generated images and `results.json`

### Publish to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Monte Carlo pricing demos"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Make sure `.gitignore` excludes `.venv/` and `outputs/`.



