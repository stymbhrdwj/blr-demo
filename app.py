import numpy as np
import numpyro
from numpyro.distributions import Normal, StudentT, Laplace, Uniform
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random
import arviz as az
import streamlit as st
import matplotlib.pyplot as plt

import pickle

plt.style.use("seaborn-v0_8")


def phi(X, degree=2):
    return np.concatenate([X**i for i in range(1, degree + 1)], axis=1)


st.set_page_config(layout="wide")
st.title("Bayesian Linear Regression")
st.markdown(
    "This app shows the effect of changing the prior and the likelihood distributions used in Bayesian Linear Regression. Since they do not always form a conjugate pair, we use ``` numpyro ``` to numerically sample from the posterior distribution using MCMC with the No-U-Turn Sampler (NUTS) algorithm."
)

left, right = st.columns([0.3, 0.7])

with left:
    st.write("---")
    d = st.select_slider(
        label="Degree of polynomial features", options=np.arange(1, 11), value=1
    )
    weight_prior_type = st.selectbox(
        r"##### Weight prior $p(\theta)$",
        ["Normal", "Laplace", "Uniform"],
    )

    if weight_prior_type == "Normal":
        ll, rr = st.columns(2)
        with ll:
            mu = st.select_slider(
                label=r"$\mu$", options=np.arange(-5.0, 6.0), value=0.0
            )

        with rr:
            sigma = st.slider(
                label=r"$\sigma$",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
            )
        weight_prior = Normal(mu, sigma)
    elif weight_prior_type == "Laplace":
        ll, rr = st.columns(2)
        with ll:
            mu = st.slider(
                label=r"$\mu$", min_value=-5.0, max_value=5.0, value=0.0, step=0.1
            )
        with rr:
            bw = st.slider(
                label=r"$b$", min_value=0.1, max_value=10.0, value=1.0, step=0.1
            )
        weight_prior = Laplace(mu, bw)
    elif weight_prior_type == "Uniform":
        ll, rr = st.columns(2)
        with ll:
            a = st.slider(
                label=r"$a$", min_value=-6.0, max_value=5.0, value=-6.0, step=0.1
            )
        with rr:
            b = st.slider(
                label=r"$b$",
                min_value=a + 1e-3,
                max_value=6.0 + 1e-3,
                value=6.0,
                step=0.1,
            )
        if a >= b:
            st.error("Lower bound must be less than upper bound")
        weight_prior = Uniform(a, b)

    same_bias_prior = st.checkbox(r"Same prior on bias $\theta_0$", value=True)

    if not same_bias_prior:
        bias_prior_type = st.selectbox(
            r"##### Bias prior  $p(\mathcal{b})$",
            ["Normal", "Laplace", "Uniform"],
        )

        if bias_prior_type == "Normal":
            ll, rr = st.columns(2)
            with ll:
                mu = st.slider(
                    label=r"$\mu$",
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.1,
                    key="bias_mu",
                )
            with rr:
                sigma = st.slider(
                    label=r"$\sigma$",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key="bias_sigma",
                )
            bias_prior = Normal(mu, sigma)
        elif bias_prior_type == "Laplace":
            ll, rr = st.columns(2)
            with ll:
                mu = st.slider(
                    label=r"$\mu$",
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.1,
                    key="bias_mu",
                )
            with rr:
                bw = st.slider(
                    label=r"$b$",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key="bias_bw",
                )
            bias_prior = Laplace(mu, bw)
        elif bias_prior_type == "Uniform":
            ll, rr = st.columns(2)
            with ll:
                a = st.slider(
                    label="Lower bound",
                    min_value=-6.0,
                    max_value=5.0,
                    value=-6.0,
                    step=0.1,
                    key="bias_a",
                )
            with rr:
                b = st.slider(
                    label="Upper bound",
                    min_value=a + 1e-3,
                    max_value=6.0 + 1e-3,
                    value=6.0,
                    step=0.1,
                    key="bias_b",
                )
            if a >= b:
                st.error("Lower bound must be less than upper bound")
            bias_prior = Uniform(a, b)

    else:
        bias_prior = weight_prior

    st.write("---")

    ll, rr = st.columns(2)
    with ll:
        likelihood_type = st.selectbox(
            r"##### Likelihood $p(\mathcal{D} | \theta)$",
            ["Normal", "StudentT", "Laplace"],
        )
    with rr:
        noise_sigma = st.slider(
            label="Aleatoric noise $\sigma$",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
        )

    if likelihood_type == "StudentT":
        likelihood_df = st.select_slider(
            label=r"Degrees of freedom $\nu$",
            options=list(range(1, 21)),
            value=3,
            key="likelihood_df",
        )

    st.write("---")
    st.write("##### Sampling parameters")

    ll, rr = st.columns(2)
    with ll:
        num_samples = st.slider(
            label="Number of samples",
            min_value=500,
            max_value=10000,
            value=2000,
            step=500,
        )
    with rr:
        num_warmup = st.slider(
            label="Number of warmup steps",
            min_value=100,
            max_value=1000,
            value=500,
            step=100,
        )

    st.write("---")
    st.write("##### Dataset parameters")

    ll, rr = st.columns(2)
    with ll:
        dataset_type = st.selectbox("Select Dataset", ["Sin", "Log", "Exp"])
    with rr:
        dataset_noise_sigma = st.slider(
            label="Dataset noise $\sigma$",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
        )


with right:
    np.random.seed(42)

    if dataset_type == "Sin":
        X = np.sort(2 * np.random.rand(100)).reshape(-1, 1)
        X_lin = np.linspace(0, 2, 100).reshape(-1, 1)
        y = X * np.sin(2 * np.pi * X) + dataset_noise_sigma * np.random.randn(
            100
        ).reshape(-1, 1)
    elif dataset_type == "Log":
        X = np.sort(2 * np.random.rand(100)).reshape(-1, 1)
        X_lin = np.linspace(0, 2, 100).reshape(-1, 1)
        y = np.log(X) + dataset_noise_sigma * np.random.randn(100).reshape(-1, 1)
    elif dataset_type == "Exp":
        X = np.sort(2 * np.random.rand(100)).reshape(-1, 1)
        X_lin = np.linspace(0, 2, 100).reshape(-1, 1)
        y = np.exp(X) + dataset_noise_sigma * np.random.randn(100).reshape(-1, 1)

    X = phi(X, d)
    X_lin = phi(X_lin, d)

    def model(X=None, y=None):
        w = numpyro.sample("w", weight_prior.expand([X.shape[1], 1]))
        b = numpyro.sample("b", bias_prior.expand([1, 1]))

        y_hat = X @ w + b

        if likelihood_type == "Normal":
            return numpyro.sample(
                "y_pred",
                Normal(y_hat, noise_sigma),
                obs=y,
            )
        elif likelihood_type == "StudentT":
            return numpyro.sample(
                "y_pred",
                StudentT(likelihood_df, y_hat, noise_sigma),
                obs=y,
            )
        elif likelihood_type == "Laplace":
            return numpyro.sample(
                "y_pred",
                Laplace(y_hat, noise_sigma),
                obs=y,
            )

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_samples=num_samples,
        num_warmup=num_warmup,
    )

    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, X=X, y=y)

    posterior_samples = mcmc.get_samples()

    rng_key = random.PRNGKey(1)
    posterior_predictive = Predictive(model, posterior_samples)
    y_pred = posterior_predictive(rng_key, X=X_lin, y=None)["y_pred"]

    mean = y_pred.mean(0)
    std = y_pred.std(0)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    for i in range(1, 21):
        ax.fill_between(
            X_lin[:, 0],
            (mean - (3 * i / 20) * std).reshape(-1),
            (mean + (3 * i / 20) * std).reshape(-1),
            color="C0",
            alpha=0.05,
            edgecolor=None,
        )
    ax.plot(X_lin[:, 0], mean, "r", label="Mean", linewidth=1)
    ax.scatter(
        X[:, 0],
        y.ravel(),
        c="k",
        label="Datapoints",
        marker="x",
        s=8,
        linewidth=0.5,
    )
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.set_title("Posterior Predictive", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)

    axes = az.plot_trace(mcmc, compact=True)
    fig = axes.ravel()[0].figure

    plt.tight_layout()
    st.pyplot(fig)


file = open("description.md", "r")
st.markdown(file.read())
