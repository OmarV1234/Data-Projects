# BSTS for Demand Forecasting
In the following notebook, we build a Bayesian Structural Time Series (BSTS) model predicting electricity demand. 
A BSTS model is one where the parameters are built using Bayesian Spike and Slab methods (in essence, we put a prior over the parameters with a high likelihood of having value 0) and a Kalman Filter (an algorithm to produce a prediction which includes a modeling of noise). It ultimately produces a collection of predictions based on probabilities.
Its a fairly popular model used in time series predictions in the marketing realm. Usually to determine the causal impact of specific actions on a given day/week/etc. For reference, the original paper from Google can be found here: https://research.google/pubs/pub41854/
