from __future__ import annotations

import json
from pathlib import Path

from mc_pricing.utils import ensure_dir
from mc_pricing import pi_mc, lcg, ibm_rng, accept_reject, gbm_ftse, vanilla_mc, straddle_is, antithetic_stoch_vol, asian_options, qmc_asian


def main():
    outputs = ensure_dir("outputs")
    results = {}

    results.update(pi_mc.run(outputs))
    results.update(lcg.run(outputs))
    results.update(ibm_rng.run(outputs))
    results.update(accept_reject.run(outputs))
    results.update(gbm_ftse.run(outputs))
    results.update(vanilla_mc.run(outputs))
    results.update(straddle_is.run(outputs))
    results.update(antithetic_stoch_vol.run(outputs))
    results.update(asian_options.run(outputs))
    results.update(qmc_asian.run(outputs))

    Path(outputs).mkdir(parents=True, exist_ok=True)
    with open(Path(outputs) / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("All demos completed. Results written to outputs/results.json and charts saved to outputs/.")


if __name__ == "__main__":
    main()


