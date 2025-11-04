"""
Legacy entrypoint retained for backwards compatibility.

Parameter-noise robustness scripts now live in:
  * `noise_in_parameters_vi.py`
  * `noise_in_parameters_mcmc.py`
"""


if __name__ == "__main__":
    raise SystemExit(
        "Please run 'noise_in_parameters_vi.py' or 'noise_in_parameters_mcmc.py' instead."
    )
