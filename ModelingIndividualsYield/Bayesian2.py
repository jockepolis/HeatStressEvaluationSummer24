import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Union, List, Optional
from itertools import cycle
from matplotlib.pyplot import cm
from matplotlib.figure import Figure
import re

class Gaussian:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample(self, N):
        return stats.multivariate_normal(
            mean=self.mean.reshape(-1), cov=self.cov, allow_singular=True
        ).rvs(N)

class BayesianLinearRegression:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        subject_name: str,
        selected_features: Union[str, List[str]],
        target: str,
        subject_type: str = None,
        prior_mean: Optional[np.ndarray] = None,
        prior_cov: Optional[np.ndarray] = None,
        beta: Optional[float] = None
    ) -> None:
        self.subject_type = subject_type
        self.dataframe = dataframe
        self.subject_name = subject_name
        self.selected_features = selected_features
        self.target = target

        # Select relevant data
        self.data_subset = self.dataframe[self.selected_features + [self.target]].dropna()

        self.y = self.data_subset[self.target].values.astype(np.float64)
        self.X = self.data_subset[self.selected_features].values.astype(np.float64)
        self.Phi = np.concatenate([np.ones((len(self.X), 1)), self.X], axis=1)

        # Initialize prior_mean, prior_cov, and beta
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.beta = beta

        # Set default priors if not provided
        if self.prior_mean is None:
            self.prior_mean = np.zeros((self.Phi.shape[1], 1))
            self.prior_mean[0] = 1  # Set prior mean for the intercept to 1
        if self.prior_cov is None:
            self.prior_cov = np.eye(self.Phi.shape[1]) * 0.1  # Stronger regularization
        if self.beta is None:
            self.beta = 1 / np.var(self.y + 1e-6)

    def fit(self, X, y):
        self.X = X.astype(np.float64)
        self.y = y.reshape(-1, 1).astype(np.float64)
        self.Phi = np.concatenate([np.ones((len(self.X), 1)), self.X], axis=1).astype(np.float64)

        self.prior_cov_inv = np.linalg.inv(self.prior_cov)

        self.posterior_cov = np.linalg.inv(
            np.linalg.inv(self.prior_cov) + self.beta * self.Phi.T @ self.Phi
        )

        self.posterior_mean = self.posterior_cov @ (
            np.linalg.inv(self.prior_cov) @ self.prior_mean
            + self.beta * self.Phi.T @ self.y
        )

        self.posterior = Gaussian(self.posterior_mean, self.posterior_cov)

        return self

    def predict(self, X):
        Phi = np.concatenate([np.ones((len(X), 1)), X], axis=1).astype(np.float64)
        return Phi @ self.posterior_mean

    def fit_model(
        self,
        prior_mean: Optional[np.ndarray] = None,
        prior_cov: Optional[np.ndarray] = None,
        beta: Optional[float] = None
    ) -> dict:
        if prior_mean is None:
            self.prior_mean = np.zeros((self.Phi.shape[1], 1))
            self.prior_mean[0] = 1  # Set prior mean for the intercept to 1
        else:
            self.prior_mean = prior_mean
        if prior_cov is None:
            self.prior_cov = np.eye(self.Phi.shape[1]) * 0.1  # Stronger regularization
        else:
            self.prior_cov = prior_cov

        if beta is None:
            variance_y = np.var(self.y + 1e-6)
            if variance_y < 1e-6:
                variance_y = 1e-6
            self.beta = 1 / variance_y

        if self.prior_mean.shape != (len(self.selected_features) + 1, 1):
            raise ValueError(
                f"Invalid shape for prior_mean. Expected shape: {(len(self.selected_features) + 1, 1)}, got: {self.prior_mean.shape}."
            )

        if self.prior_cov.shape != np.identity(self.Phi.shape[1]).shape:
            raise ValueError(
                f"Invalid shape for prior_cov. Expected shape: {np.identity(self.Phi.shape[1]).shape}, got: {self.prior_cov.shape}."
            )

        self.prior_cov_inv = np.linalg.inv(self.prior_cov)
        self.y_reshaped = self.y.reshape(-1, 1)

        self.posterior_cov = np.linalg.inv(
            np.linalg.inv(self.prior_cov) + self.beta * self.Phi.T @ self.Phi
        )

        self.posterior_mean = self.posterior_cov @ (
            np.linalg.inv(self.prior_cov) @ self.prior_mean
            + self.beta * self.Phi.T @ self.y_reshaped
        )

        self.posterior = Gaussian(self.posterior_mean, self.posterior_cov)

        self.result = {}
        for i in range(self.Phi.shape[1]):
            mu = self.posterior_mean[i]
            variance = self.posterior_cov[i, i]
            sigma = math.sqrt(variance)

            feature_name = "Off-set" if i == 0 else self.selected_features[i - 1]
            self.result[feature_name] = {"mu": mu[0], "sigma": sigma}

        return self.result

    def plot_posterior_distributions(self) -> List[Figure]:
        """
        Plot the posterior distributions of the model parameters.

        Returns:
        - List[Figure]: List of figures.
        """
        figures = []
        for i in range(len(self.selected_features) + 1):
            mu = self.posterior.mean[i]
            variance = self.posterior.cov[i, i]
            sigma = math.sqrt(variance)
            fig, ax = plt.subplots(figsize=(6, 5))
            plt.title(
                f"Posterior Distribution - {self.selected_features[i - 1] if i != 0 else 'Off-set'}.",
                fontsize=16
            )
            xx = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            sns.lineplot(
                x=xx.flatten(),
                y=stats.norm.pdf(xx, mu, sigma).flatten(),
                label = f"PDF: $\mu = {mu.item():8.3f}$,\n \t $\sigma = {variance:8.3f}$",
                color="black",
                ax=ax,
            )
            sns.histplot(
                self.posterior.sample(500)[:, i],
                bins=30,
                kde=True,
                stat="density",
                alpha=0.5,
                color="blue",
                label="Samples",
                ax=ax,
            )

            plt.xlabel(f"Parameter Value", fontsize=14)
            plt.ylabel("Density", fontsize=14)
            plt.legend(fontsize=12, loc='upper right')
            plt.tight_layout()
            figures.append(fig)
        return figures

    def assess_x_vector(self, ordered_by: str) -> np.ndarray:
        """
        Function which checks and gets the desired x-vector ('ordered_by') to do plotting.

        Args:
        - ordered_by (str): Feature or target variable to use for ordering.

        Returns:
        - np.ndarray: X-vector for plotting.
        """
        if ordered_by not in self.selected_features + [self.target]:
            raise ValueError(
                f"{ordered_by} is not a valid feature, target, 'Date', or 'DateTime'."
            )
        x_vector = self.data_subset[ordered_by].values
        return x_vector

    def plot_model_samples(
        self, n_samples: int = 3, ordered_by: str = "Temperature"
    ) -> Figure:
        """
        Sample and plot 'n_samples' models from the posterior.

        Args:
        - n_samples (int): Number of samples to generate and plot.
        - ordered_by (str): Feature or target variable to use for ordering.

        Returns:
        - Figure: Matplotlib Figure object representing the plot.
        """
        #For samples of w \theta, f(x) = phi(x)^T \theta
        samples = np.array([self.Phi @ t for t in self.posterior.sample(n_samples)]) # draw n_samples from the posterior
        x_vector = self.assess_x_vector(ordered_by=ordered_by)

        ordered_indices = np.argsort(x_vector) # sort the indices by the desired 'ordered_by' string
        ordered_x = x_vector[ordered_indices]
        ordered_samples = samples[:, ordered_indices]
        ordered_y = self.y[ordered_indices]
        # plot
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot(111)
        plt.scatter(ordered_x, ordered_y, zorder=1)
        color = cycle(cm.rainbow(np.linspace(0, 1, n_samples)))
        i = 1
        for sample, c in zip(ordered_samples, color):
            ax.plot(ordered_x, sample, color=c, alpha=0.7, label=f"Sample {i}")
            i += 1

        plt.xlabel('Daily '+ re.sub(r'([a-z])([A-Z])', r'\1 \2', ordered_by), fontsize=14)
        plt.ylabel("Normalized Daily Yield", fontsize=14)
        plt.title(f"Posterior - samples for {self.subject_name}", fontsize=16)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return fig #return the figure

    def plot_model_uncertainty(
        self, ordered_by: str = "Temperature", every_th: int = None
    ) -> plt.Figure:
        """
        Plots the mean-model and 1, 2, and 3 - standard deviation uncertainties.

        Args:
        - ordered_by (str): Feature or target variable to use for ordering.
        - every_th (int): Plot every-th data point for computational efficiency.

        Returns:
        - plt.Figure: Matplotlib Figure object representing the plot.
        """
        @nb.njit(parallel=True)
        def compute_uncertainty_matrix(Phi, posterior_cov, beta, y_shape):
            # init uncertainty matrix
            uncs = np.zeros((y_shape, y_shape), dtype=np.float32)
            # first term: Phi @ posterior_cov @ Phi.T
            for i in nb.prange(Phi.shape[0]):
                for j in nb.prange(Phi.shape[1]):
                    for k in nb.prange(Phi.shape[1]):
                        uncs[i, j] += Phi[i, k] * posterior_cov[k, j]

            # add second term: beta**(-1) * np.eye(y_shape)
            for i in range(y_shape):
                uncs[i, i] += beta ** (-1)

            return np.sqrt(np.diag(uncs))

        x_vector = self.assess_x_vector(ordered_by=ordered_by)
        ordered_indices = np.argsort(x_vector) # sort the indices by the desired 'ordered_by' string
        ordered_x = x_vector[ordered_indices]

        ordered_y = self.y[ordered_indices]
        mean = self.Phi @ self.posterior_mean
        mean = mean.flatten()
        mean = mean[ordered_indices]

        uncs = compute_uncertainty_matrix(
            self.Phi, self.posterior_cov, self.beta, self.y.shape[0]
        )
        uncs = uncs[ordered_indices]
        # create a temporary dataframe for making plotting this easier
        plot_data = pd.DataFrame(
            {
                ordered_by: ordered_x,
                "Yield": ordered_y.flatten(),
                "Mean": mean,
                "Uncertainty": uncs,
            }
        )

        plot_data = plot_data.sort_values(by=ordered_by)
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot(111)
        if every_th is None: #Sometimes data is huge, then you set every_th to e.g. 100 to only plot every 100th sample
            every_th = 1

        sns.scatterplot(
            x=ordered_by,
            y="Yield",
            data=plot_data.iloc[::every_th],
            zorder=10,
            alpha=0.4,
            color = '#183B87',
            ax = ax
        )

        sns.lineplot(
            x=ordered_by,
            y="Mean",
            data=plot_data.iloc[::every_th],
            color="black",
            label="Mean",
            ax=ax
        )

        ax.fill_between(
            plot_data[ordered_by].iloc[::every_th],
            (
                plot_data["Mean"].iloc[::every_th] - 3 * plot_data["Uncertainty"].iloc[::every_th]
            ),
            (
                plot_data["Mean"].iloc[::every_th] + 3 * plot_data["Uncertainty"].iloc[::every_th]
            ),
            color="lightgray",
            label=f'$Mean \pm 3\\sigma$',
            alpha=0.7,
        )

        ax.fill_between(
            plot_data[ordered_by].iloc[::every_th],
            (
                plot_data["Mean"].iloc[::every_th] - 2 * plot_data["Uncertainty"].iloc[::every_th]
            ),
            (
                plot_data["Mean"].iloc[::every_th] + 2 * plot_data["Uncertainty"].iloc[::every_th]
            ),
            color="darkgray",
            label=f'$Mean \pm 2\\sigma$',
            alpha=0.7,
        )
        ax.fill_between(
            plot_data[ordered_by].iloc[::every_th],
            (
                plot_data["Mean"].iloc[::every_th] - 1 * plot_data["Uncertainty"].iloc[::every_th]
            ),
            (
                plot_data["Mean"].iloc[::every_th] + 1 * plot_data["Uncertainty"].iloc[::every_th]
            ),
            color="gray",
            label=f'$Mean \pm 1\\sigma$',
            alpha=0.7,
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.xlabel('Daily '+ re.sub(r'([a-z])([A-Z])', r'\1 \2', ordered_by), fontsize=14)
        plt.ylabel("Normalized Daily Yield", fontsize=14)
        plt.title(f"Bayesian Regression - {self.subject_name}", fontsize=16)

        return fig

class SklearnBayesianLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, dataframe, subject_name, selected_features, target, subject_type=None, prior_mean=None, prior_cov=None, beta=None):
        self.dataframe = dataframe
        self.subject_name = subject_name
        self.selected_features = selected_features
        self.target = target
        self.subject_type = subject_type
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.beta = beta

    def fit(self, X, y):
        self.model = BayesianLinearRegression(
            dataframe=self.dataframe,
            subject_name=self.subject_name,
            selected_features=self.selected_features,
            target=self.target,
            subject_type=self.subject_type,
            prior_mean=self.prior_mean,
            prior_cov=self.prior_cov,
            beta=self.beta
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            'dataframe': self.dataframe,
            'subject_name': self.subject_name,
            'selected_features': self.selected_features,
            'target': self.target,
            'subject_type': self.subject_type,
            'prior_mean': self.prior_mean,
            'prior_cov': self.prior_cov,
            'beta': self.beta
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
