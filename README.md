# laplace-lts

A [Laplace](https://github.com/mlatinov/laplace) library of latent time series components for Stan — local level, local linear trend, stochastic and trigonometric seasonality, stochastic volatility, and latent-Gaussian counts. These are the building blocks of structural time series models: each one is a path that evolves over time and is never observed directly. Import it into any `.laplace` model and call it with namespaced calls (`lts::function_name(...)`).

Like all Laplace libraries, `lts` compiles down to plain, readable Stan functions. Nothing about how you use it hides what actually ends up in your `.stan` file.

This is the companion to [`ts`](https://github.com/mlatinov/laplace-ts), which covers the observation-driven models — ARIMA, GARCH, exponential smoothing. The split is deliberate: everything in `ts` computes its recursion from the data, and everything here samples a hidden path.

## How it's organised

[#how-its-organised](#how-its-organised)

A structural time series model is a **sum of components**, and the library keeps them separate:

$$
y_t = \mu_t + s_t + x_t^\top\beta + \varepsilon_t
$$

Each component is an independent function returning a `vector[T]`. You add them with ordinary Stan arithmetic and write the observation model yourself:

```
vector[T] mu   = lts::local_level(mu1, z_mu, sigma_mu);
vector[T] seas = lts::stochastic_seasonal(s_init, z_s, sigma_s);

y ~ normal(mu + seas + X * beta, sigma_y);
```

There is no `bsts()` function that does all of it at once, because the useful thing about structural models is that you choose which components are in them. A trend alone, a trend plus seasonality, a seasonality on its own added to an ARIMA mean from `ts` — all of these are the same three lines with different terms.

### Centred and non-centred

[#centred-and-non-centred](#centred-and-non-centred)

Every component comes in two parameterizations, and **you use exactly one of them**.

The **non-centred primary** is the default. It takes standard normal innovations and builds the path from them:

```
parameters { vector[T - 1] z_mu; real<lower=0> sigma_mu; }
transformed parameters { vector[T] mu = lts::local_level(mu1, z_mu, sigma_mu); }
model { z_mu ~ std_normal(); }
```

The **centred alternative** declares the path itself as a parameter and scores it with `_state_lpdf`:

```
parameters { vector[T] mu; real<lower=0> sigma_mu; }
model { target += lts::local_level_state_lpdf(mu | sigma_mu); }
```

Running the primary *and then* adding the state lpdf counts the state prior twice. It will compile, it will sample, and the answer will be wrong in a way nothing warns you about. Pick one.

Non-centred is the default because the centred version has a funnel: when $\sigma_\mu$ is small the path is squeezed into a narrow neck the sampler can't turn in, and you get divergences that no amount of `adapt_delta` will fix. Reach for `_state_lpdf` only when the data pin $\sigma$ down hard enough that the funnel closes, which in practice means long series with a clearly visible signal.

## The five forms

[#the-five-forms](#the-five-forms)

For a component called `<name>`:

| Form | Shape | Where it goes |
| --- | --- | --- |
| `<name>(...)` | Takes innovations, returns the path | `transformed parameters` |
| `<name>_state_lpdf(path \| ...)` | Log density of the latent path | `model`, as the centred alternative |
| `<name>_lpdf` / `_lpmf` | Observation density, where the component defines one | `model` |
| `<name>_rng(..., T)` | A simulated path of length $T$ | `transformed data` or `generated quantities` |
| `<name>_forecast_rng(state_last, ..., H)` | $H$ draws continuing from the final state | `generated quantities` |

Only `stochastic_volatility` and `latent_gaussian_counts` define observation densities, because only those two carry their own likelihood. The other four are pure components: you add them to a mean and write the likelihood yourself.

Note what `_forecast_rng` takes: the **final state**, not the series. A latent component has no residuals to replay, so continuing it needs only where it ended — `mu[T]` for the local level, `mu[T]` and `delta[T]` for the trend, the last $S-1$ effects for the seasonal.

Every function carries `@brief`, `@param`, `@return`, `@math`, and (where useful) `@example` documentation, so you can read it from the terminal without leaving your model:

```
laplace doc lts::trigonometric_seasonal
```

## Trend components

[#trend-components](#trend-components)

| Component | Innovations | State equations |
| --- | --- | --- |
| `local_level` | `z`, length $T-1$ | $\mu_t = \mu_{t-1} + \sigma_\mu z_t$ |
| `local_linear_trend` | `zeta_mu`, `zeta_delta`, both length $T-1$ | see below |

The local level is a random walk observed with noise — the simplest thing that is not a constant. Its one real subtlety is that $\sigma_\mu$ and the observation noise $\sigma_y$ are only jointly identified through their ratio. That signal-to-noise ratio is what the data actually see: a large ratio gives a level that tracks every wiggle, a small one gives something close to a flat line. Vague priors on both scales leave a ridge, so prior them on the same scale as the data.

The local linear trend adds a slope that itself drifts:

$$
\delta_t = \rho\,\delta_{t-1} + \sigma_\delta\,\zeta^{\delta}_t, \qquad
\mu_t = \mu_{t-1} + \delta_{t-1} + \sigma_\mu\,\zeta^{\mu}_t
$$

$\rho = 1$ is the classical local linear trend, whose forecast variance grows cubically in the horizon and whose predictive intervals become useless within a couple of seasons. $\rho \in (0,1)$ damps the slope, the trend flattens out instead of extrapolating forever, and long-horizon forecasts get dramatically better. Use damping unless you have a reason not to. $\sigma_\delta = 0$ collapses to a fixed slope, which is a straight-line trend with a wandering level.

The primary returns only $\mu$; the slope path stays internal. The centred `_state_lpdf` needs both, so declare `delta` as a parameter when you use it.

## Seasonal components

[#seasonal-components](#seasonal-components)

| Component | Arguments | What it does |
| --- | --- | --- |
| `stochastic_seasonal` | `s_init` (length $S-1$), `zs`, `sigma_s` | One effect per period, constrained to sum to zero |
| `trigonometric_prep` | `period`, `K` | Precomputes $\cos\lambda_k, \sin\lambda_k$. Run once in `transformed data` |
| `trigonometric_seasonal` | `g1`, `g1_star`, `z_g`, `z_gs`, `sigma_gamma`, `cos_sin` | $K$ harmonics, each a slowly rotating pair |

The **dummy** seasonal encodes its identification constraint in the mean:

$$
s_t \sim \mathcal{N}\left(-\sum_{j=1}^{S-1} s_{t-j},\ \sigma_s\right)
$$

which says any $S$ consecutive effects sum to about zero. Without that, the seasonal component and the level are not separately identified and the level drifts to absorb whatever the seasonal doesn't.

The **trigonometric** seasonal rotates each harmonic pair by its own frequency $\lambda_k = 2\pi k / \text{period}$:

```math
\begin{pmatrix}\gamma_{k,t}\\ \gamma^{*}_{k,t}\end{pmatrix} \sim
\mathcal{N}\left(
\begin{pmatrix}\cos\lambda_k & \sin\lambda_k\\ -\sin\lambda_k & \cos\lambda_k\end{pmatrix}
\begin{pmatrix}\gamma_{k,t-1}\\ \gamma^{*}_{k,t-1}\end{pmatrix},\ \sigma_\gamma\right),
\qquad s_t = \sum_{k=1}^{K}\gamma_{k,t}
```

In the library the conjugate $\gamma^{\ast}$ is called `g_star` and its innovations `z_gs`.

The difference that matters: the dummy seasonal spends $S-1$ parameters per time point and imposes no smoothness, while the trigonometric one spends $2K$ and lets you control smoothness through $K$ independently of the period. For monthly data with $S = 12$ they are comparable. For daily data with a yearly cycle, $S = 365.25$ is unusable as a dummy seasonal and perfectly ordinary with $K = 6$ harmonics. Either becomes a *fixed* seasonal pattern when its innovation scale is zero, at which point you may as well use plain Fourier regressors and skip the component entirely.

## Components with their own likelihood

[#components-with-their-own-likelihood](#components-with-their-own-likelihood)

| Component | State | Observation |
| --- | --- | --- |
| `stochastic_volatility` | $h_t = \mu_h + \rho(h_{t-1} - \mu_h) + \sigma_h z_t$ | $y_t \sim \mathcal{N}(\mu_t,\ e^{h_t/2})$ |
| `latent_gaussian_counts` | $u_t = \rho\,u_{t-1} + \sigma_u z_t$ | $\log\lambda_t = x_t^\top\beta + u_t$, $y_t \sim \text{Poisson}(\lambda_t)$ |

Both are AR(1) latent paths started from their stationary distribution, which for a scalar state is available in closed form and needs no Lyapunov solve:

$$
h_1 \sim \mathcal{N}\left(\mu_h,\ \frac{\sigma_h}{\sqrt{1 - \rho^2}}\right)
$$

`stochastic_volatility` is the latent-state answer to the same question GARCH answers observation-driven. GARCH makes the variance a deterministic function of past returns; this gives the variance its own noise. It fits worse per parameter and forecasts volatility better, and it is the harder of the two to sample — non-centred is not optional here.

`latent_gaussian_counts` is deliberately zero-mean, so the level lives in the regression. Its no-covariate overload therefore has no intercept at all, which implies counts centred on $\lambda = 1$. Include an intercept column in `X` unless that's really what you mean.

## Installation

[#installation](#installation)

`lts` is distributed as a git-hosted Laplace library — there's no published registry entry yet, so it's added by pointing `laplace` (or `cmdlaplacer`, if you're working from R) directly at the repository. The package lives in the repository's `laplace/` subdirectory, so pass it as the subdir.

### Via the `laplace` CLI

[#via-the-laplace-cli](#via-the-laplace-cli)

From inside a Laplace project (a directory with its own `laplace.toml`):

```
laplace add lts --git https://github.com/mlatinov/laplace-lts --tag 0.1.0 --subdir laplace
```

### Via R (`cmdlaplacer`)

[#via-r-cmdlaplacer](#via-r-cmdlaplacer)

```
library(cmdlaplacer)

laplace_install_git(
  "lts",
  "https://github.com/mlatinov/laplace-lts",
  tag = "0.1.0",
  subdir = "laplace"
)
```

Either way, this pins the dependency in your project's `laplace.toml`/`laplace.lock` at tag `0.1.0`. Check the [tags](https://github.com/mlatinov/laplace-lts/tags) for newer versions as they become available.

## Usage

[#usage](#usage)

Import the library in a `library { }` block and call its functions with the `lts::` namespace prefix.

### A structural model: level, seasonality and regressors

[#a-structural-model-level-seasonality-and-regressors](#a-structural-model-level-seasonality-and-regressors)

The canonical BSTS decomposition. Each component is built separately, summed into the mean, and forecast separately from its own final state.

```
library {
    import lts
}

data {
  int<lower=1> T;
  vector[T] y;
  int<lower=2> S;                       // seasonal period, e.g. 12
  int<lower=1> K;                       // number of covariates
  matrix[T, K] X;
  int<lower=1> H;                       // forecast horizon
  matrix[H, K] X_future;
}

parameters {
  real mu1;                             // initial level
  vector[S - 1] s_init;                 // initial seasonal effects
  vector[T - 1] z_mu;                   // non-centred level innovations
  vector[T - S + 1] z_s;                // non-centred seasonal innovations
  vector[K] beta;

  real<lower=0> sigma_mu;               // level drift
  real<lower=0> sigma_s;                // seasonal drift
  real<lower=0> sigma_y;                // observation noise
}

transformed parameters {
  vector[T] level    = lts::local_level(mu1, z_mu, sigma_mu);
  vector[T] seasonal = lts::stochastic_seasonal(s_init, z_s, sigma_s);
}

model {
  mu1    ~ normal(0, 5);
  s_init ~ normal(0, 1);
  beta   ~ normal(0, 1);

  z_mu ~ std_normal();
  z_s  ~ std_normal();

  sigma_mu ~ exponential(2);            // prior the drift relative to sigma_y
  sigma_s  ~ exponential(5);            // seasonality usually drifts slower than the level
  sigma_y  ~ exponential(1);

  y ~ normal(level + seasonal + X * beta, sigma_y);
}

generated quantities {
  vector[H] level_f    = lts::local_level_forecast_rng(level[T], sigma_mu, H);
  vector[H] seasonal_f = lts::stochastic_seasonal_forecast_rng(
                           seasonal[(T - S + 2):T], sigma_s, H);

  vector[H] y_forecast = to_vector(
    normal_rng(level_f + seasonal_f + X_future * beta, sigma_y)
  );
}
```

Swap `local_level` for `local_linear_trend` to add a slope, or `stochastic_seasonal` for `trigonometric_seasonal` when the period is long. The observation line doesn't change either way — that's the point of keeping the components separate.

The `exponential(2)` / `exponential(5)` / `exponential(1)` ordering is doing real work here. It says the observation noise is the largest scale, the level drifts more slowly, and the seasonal shape drifts more slowly still. Without that ordering the ridge between $\sigma_\mu$ and $\sigma_y$ is wide open and the sampler will wander along it.

### Stochastic volatility on returns

[#stochastic-volatility-on-returns](#stochastic-volatility-on-returns)

The mean is zero, the variance is the latent path, and the observation density comes from the library.

```
library {
    import lts
}

data {
  int<lower=1> T;
  vector[T] y;                          // log returns, already centred
  int<lower=1> H;
}

parameters {
  real mu_h;                            // mean log variance
  real<lower=-1, upper=1> rho;          // persistence, usually near 1
  real<lower=0> sigma_h;
  vector[T] z;                          // non-centred innovations
}

transformed parameters {
  vector[T] h = lts::stochastic_volatility(mu_h, sigma_h, rho, z);
}

model {
  mu_h    ~ normal(0, 5);
  rho     ~ beta(20, 1.5);              // shifted to (0,1); see the note below
  sigma_h ~ exponential(5);
  z       ~ std_normal();

  target += lts::stochastic_volatility_lpdf(y | h, rep_vector(0, T));
}

generated quantities {
  vector[H] h_forecast = lts::stochastic_volatility_forecast_rng(h[T], mu_h, sigma_h, rho, H);
  vector[H] y_forecast = to_vector(normal_rng(rep_vector(0, H), exp(h_forecast / 2)));
}
```

A `beta(20, 1.5)` prior is on $(0,1)$, so either declare `rho` as `real<lower=0, upper=1>` or transform it — `rho = 2 * rho_raw - 1` with `rho_raw ~ beta(20, 1.5)` — if you want to allow negative persistence. In practice volatility persistence is positive and close to 1, and the prior is there to keep the sampler away from the $\rho \to 1$ boundary where the stationary initialisation blows up.

### Counts with an AR(1) latent effect

[#counts-with-an-ar1-latent-effect](#counts-with-an-ar1-latent-effect)

Overdispersed, autocorrelated counts. `X` carries the intercept, since the latent path is zero-mean.

```
library {
    import lts
}

data {
  int<lower=1> T;
  array[T] int<lower=0> y;
  int<lower=1> K;
  matrix[T, K] X;                       // FIRST COLUMN SHOULD BE ONES
  int<lower=1> H;
  matrix[H, K] X_future;
}

parameters {
  vector[K] beta;
  real<lower=-1, upper=1> rho;
  real<lower=0> sigma_u;
  vector[T] z;
}

transformed parameters {
  vector[T] u = lts::latent_gaussian_counts(rho, sigma_u, z);
}

model {
  beta    ~ normal(0, 2);
  rho     ~ normal(0, 0.5);
  sigma_u ~ exponential(2);
  z       ~ std_normal();

  target += lts::latent_gaussian_counts_lpmf(y | u, X, beta);
}

generated quantities {
  vector[H] u_forecast = lts::latent_gaussian_counts_forecast_rng(u[T], rho, sigma_u, H);
  array[H] int y_forecast = poisson_log_rng(X_future * beta + u_forecast);
}
```

Swapping `poisson_log_rng` for `neg_binomial_2_log_rng` and writing the likelihood by hand gives you a negative binomial version; the latent component doesn't change.

### From R

[#from-r](#from-r)

With `cmdlaplacer`, the `.laplace` file compiles straight to a `cmdstanr` model, and the generated `.stan` file stays on disk next to it:

```
library(cmdlaplacer)

mod <- laplace_model("structural.laplace")
fit <- mod$sample(data = list(T = length(y), y = y, S = 12, K = ncol(X),
                              X = X, H = 24, X_future = X_future))

fit$draws(c("level", "seasonal", "y_forecast"))
```

## Things to know

[#things-to-know](#things-to-know)

- **Use `target +=`, not `~`.** Laplace rewrites `lts::func(` calls, so write `target += lts::stochastic_volatility_lpdf(y | h, mu)`. The `y ~ lts::stochastic_volatility(...)` form won't resolve.
- **Never use a primary and its `_state_lpdf` together.** One is non-centred, the other centred. Using both counts the state prior twice, compiles cleanly, and gives a silently wrong posterior.
- **Non-centred is the default for a reason.** The centred parameterization funnels when $\sigma$ is small. If you see divergences that survive `adapt_delta = 0.99`, check which parameterization you're on before touching anything else.
- **Every component costs one parameter per time point.** There is no Kalman filter here, so the latent path is sampled rather than marginalised. A two-component model on 2000 points is roughly 4000 parameters, which HMC handles but not instantly. Marginalisation is planned, not present.
- **Scales are only jointly identified.** $\sigma_\mu$ against $\sigma_y$, $\sigma_s$ against both. Put priors on all of them on the data's scale, and prefer an ordering that says what you believe about which component moves fastest.
- **Seasonal initial effects need a prior and want to sum to zero.** `s_init` has length $S-1$ and the recursion supplies the $S$-th implicitly. If you leave `s_init` flat the level will absorb a constant offset.
- **Covariates in `local_level` and `local_linear_trend` enter the increment.** The covariate overloads accumulate: a constant covariate produces a linear trend, and `beta` is a drift rather than a level shift. For an ordinary level-shift regression put `X * beta` in the observation mean, as the structural example does, and use the plain overload.
- **`_forecast_rng` takes the final state, not the series.** `level[T]`, or `mu[T]` and `delta[T]` for the trend, or the last $S-1$ seasonal effects. Forecasting a summed model means forecasting each component and adding the results.
- **Forecast a damped trend, not an undamped one.** With $\rho = 1$ the local linear trend's predictive intervals grow cubically and stop meaning anything past a season or two.
- **`_rng` functions are restricted by Stan.** They can only be called in `transformed data` or `generated quantities`.
- **`latent_gaussian_counts` has no intercept.** Its latent path is zero-mean by construction, so the single-argument `_lpmf` overload implies counts centred on $\lambda = 1$. Put an intercept column in `X`.
- **Storing component paths in `transformed parameters` is usually right here.** Unlike a covariance matrix, a `vector[T]` per component is what you actually want in the output — the decomposition into level, seasonality and regression is the reason to fit a structural model at all.
- **Composes with [`ts`](https://github.com/mlatinov/laplace-ts).** A latent seasonal added to an ARMA mean, or a local level with GARCH errors, is just two imports and one sum. The two libraries share no state and don't need to know about each other.

## License

[#license](#license)

See [LICENSE](https://github.com/mlatinov/laplace-lts/blob/main/LICENSE).
