# Normative Uncertainty

Code for the paper "Reinforcement Learning Under Normative Uncertainty".

The simplest way to run the experiments is to run:
`python run_experiments.py --n_on_track=300 --n_credences=300 --timesteps=10000000 --processes=10`

- `--processes` is the number of parallel processes to use when training, and should depend on the number of resources available on the machine.
- `--timesteps` is the number of training timesteps. The paper used 10 million, which takes around 10 hours for each experiment. Most experiments actually converge well before that, so you could use 1 million or less instead for quick experimentation.
- `--n_credences` is the granularity of the x axis when producing the plots. The paper uses 300 to get a smooth picture, but it is usually possible to get a general idea of what's going on with values as low as 100 or even 50, which can be much faster.
- `--n_on_track` is the granularity of the y axis when producing the plots. Likewise the paper uses 300 but values as low as 100 or 50 are often good enough.

Note that if you have previously run the experiments for a given number of timesteps but want to change th plot granularity, re-running `run_experiments.py` with the same number of timesteps but different `--n_on_track` and `--n_credences` will regenerate the plots but not retrain, potentially saving a lot of time.

