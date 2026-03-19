# Local autoregressive forecast and basic Kalman filtering

## Introduction and purpose

Most marine datasets are what we called a 2DVAR analysis - the current analysis is based on the present observations only (2D as in x, y; in contrast of 3DVAR seen in historical weather forecasting literature: x, y, z). 

This limits the amount of the observations one can used, resulting in higher uncertainty. In many cases, this is mitigated by temporally aggregating the observations (like doing monthly analysis instead of daily analysis) or blending multiple input sources (like high-level remote sensing data product). Sometimes one is stuck with such spotty observations; like one really wants a daily analysis based purely on daily observations.

This leads to the introduction of time dimension into the analysis. This is what we called 4DVAR (x, y, z, t) weather forecasting (or 3DVAR if one is dealing with only surface variables - x, y, t).

A popular way to do so is via Kalman Filter -- a recursive method that blends a forecast to the present state (aka prior) with present observations, resulting a posterior analysis that is more accurate with just the forecast or the observations. The method is widely used in autopilot, navigation, and positioning; they are in every satellite navigation device and flight cockpit. Nowadays, it is found in Earth sciences in particular with weather forecasts (citation). Since XXXX, Kalman filter has been used in producing more accurate weather forecasts and historical reanalysis by the ECWMF and the UK Met Office (citation).

# Producing the prior using a simple autoregressive statistical model

This lead to the question how would one produce a forecast priors. The priors used in weather forecasts are very high dimension and are based on actual computer simulations of the weather (aka numerical weather prediction). Such kind of dynamical modelling is generally not used in producing gridded observations, but there are some exceptions (XXX; https://soda.umd.edu/). 

A simple statistical approach is implemented here to produce such forecast, basing on the simple idea is that current marine/atmospheric condition will always tend toward climatology (), providing the latter can be defined. The simpliest way to model that behavior is the local 1st order autoregressive model, in which the expected value of the forecast is that anomalies (aka departures from climatology) observed now will tend toward climatology following an exponential decay, in which the decay rate is a function of the observed lag-1 correlation.

# Uncertainty weighted average (Kalman Filter)

## Code and usage



