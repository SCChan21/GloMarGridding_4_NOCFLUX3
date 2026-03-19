# Local autoregressive forecast and basic Kalman filtering

## Introduction and purpose

Most marine datasets are what we called a 2DVAR analysis - the current analysis is based on the present observations only (2D as in x, y; in contrast of 3DVAR seen in historical weather forecasting literature: x, y, z). 

This limits the amount of the observations one can used, resulting in higher uncertainty. In many cases, this is mitigated by temporally aggregating the observations (like doing monthly analysis instead of daily analysis) or blending multiple input sources (like high-level remote sensing data product). Sometimes one is stuck with such spotty observations; like one really wants a daily analysis based purely on daily observations.

This leads to the introduction of time dimension into the analysis. This is what we called 4DVAR (x, y, z, t) weather forecasting (or 3DVAR if one is dealing with only surface variables - x, y, t).

A popular way to do so is via Kalman Filter -- a recursive method that blends a forecast to the present state (aka prior) with present observations, resulting a posterior analysis that is more accurate with just the forecast or the observations. The method is widely used in autopilot, navigation, and positioning; they are in every satellite navigation device and flight cockpit. Nowadays, it is found in Earth sciences in particular with weather forecasts (citation). Since XXXX, Kalman filter has been used in producing more accurate weather forecasts and historical reanalysis by the ECWMF and the UK Met Office (citation).

# Producing the prior using a simple autoregressive statistical model

This lead to the question how would one produce a forecast priors. The priors used in weather forecasts are based on numerical weather prediction. Such kind of dynamical modelling is generally not used in producing gridded observations, but there are exceptions (XXX; https://soda.umd.edu/). 

A simple statistical approach is implemented here to produce such forecast, basing on the simple idea is that current condition (described by the variable `a` below) will generally tend toward climatology (`E(a)`) (). The simplest way to model that behavior is the local 1st order autoregressive model, in which the expected value for `a(t)` is a linear function of the observed lag-1 correlation Phi and its pervious observed value (`a(t-1)`) , resulting in exponential relaxation toward `E(a)`, plus a normally distributed uncertainty epsilon: 

$$ 
a_{t} = \Phi (a_{t-1} - E(a)) + \epsilon
$$

This is often referred as the AR1 model. More sophisticated approach can be used like multi-lag or autoregressive moving average (ARMA). The AR1 model produces its own uncertainty and input uncertainties can also be incorporated (i.e. epsilon). This is essential as uncertainty lie in the heart of Kalman Filter. 

Let say Phi and sigma(climatology) are diagonal matrices with local lag-1 correlation and climatological variance (standard deviation squared), Sigma t is the uncertainty for the current observations, the uncertainty to epsilon is,:

$$
\Sigma_{\text{AR1}} \sim MVN(0, \sqrt{\Phi}\Sigma_{t}\sqrt{\Phi} + \sqrt{(\textbf{1}-\Phi\Phi)}\Sigma_\text{climatology}\sqrt{(\textbf{1}-\Phi\Phi)} )
$$

For the purpose of this exercise, uncertainty for the current observations is the full output Kriging covariance, using only present observations.

# Uncertainty weighted average (Kalman Filter)

Kalman Filter, in its heart, is a weighted average -- an average for two input streams: the prior/first-guess/forecast what the current state probably be, and what the most recent observations say. The weights are determined only by uncertainties. In simple English, the weight favours the more certain input stream.

Let say the uncertainty of the forecast and observations can be described by two different uni- or multi-variate zero-averaged normal distribution with some scalar variance / error covariance matrix. 

$$
\epsilon_{\text{forecast}} \sim N(0, \sigma_{\text{forecast}}^2) \text{or} MVN(0, \Sigma_{\text{forecast}})
$$

$$
\epsilon_{\text{obs}} \sim N(0, \sigma_{\text{obs}}^2) \text{or} MVN(0, \Sigma_{\text{obs}})
$$

Assuming observations are on the same grid as the forecast (true for the exercise), the Kalman gain (weight for new observations) is:

Univariate: 

$$
K = \frac{\sigma_{\text{forecast}}^2  (sigma_{\text{forecast}}^2 + \sigma_{\text{obs}}^2)} 
$$

Multi-variate: 

$$
K = \Sigma_{\text{forecast}} (\Sigma_{\text{forecast}} + \sigma_{\text{obs}})^{-1} = \sigma_{\text{obs}}^{-1} (\Sigma_{\text{forecast}}^{-1} + \sigma_{\text{obs}}^{-1})^{-1}
$$

in which K is solution to the following linear equation, this is how it is solved in the code using `numpy.linalg.solve`. No actual error covariance inversions is needed.

$$
(\Sigma_{\text{forecast}} + \sigma_{\text{obs}}) K^{T} = \Sigma_{\text{forecast}}
$$

noting that Sigma is symmetrical but K is not (similar to Kriging weights).

Once the weights are known, the posterior analysis is a weighted average of forecast and observations

$$
\tilde{a}(t) = K a_{\text{obs}} + (I - K)  a_{\text{forecast}} = K (a_{\text{obs}} - a_{\text{forecast}} ) +  a_{\text{forecast}}
$$

with posterior error covariance of:

$$
\Sigma_{\text{analysis}} = (\textbf{I} - \textbf{K}) \Sigma_{\text{forecast}}}
$$

If observations are not on the same grid as the forecast, additional mapping multiplications are needed; that leads the usual form one will see in text books or Wikipedia.

## Code and usage



