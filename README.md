# house_price_predict

Modelling Experiments
We've got some historical data, to model it, we should run a series of modelling experiments and see which model
performs best.

Terms to be familiar with:

Horizon = number of timesteps into the future we're going to predict
Window size = number of timesteps we're going to use to predict horizon Modelling experiments we're running:

- Naïve model (baseline)
- Dense model, horizon = 1, window = 7
- Same as 1, horizon = 1, window = 30
- Same as 1, horizon = 7, window = 30
- Conv1D
- LSTM
- Same as 1 (but with multivariate data)
- N-BEATs Algorithm
- Ensemble (multiple models optimized on different loss functions)
- Future prediction model (model to predict future values)
- Same as 1 (but with turkey 🦃 data introduced)