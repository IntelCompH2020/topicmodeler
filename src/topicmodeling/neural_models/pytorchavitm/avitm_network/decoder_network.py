# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F

# Local imports
from .inference_network import InferenceNetwork


class DecoderNetwork(nn.Module):
    """PyTorch class for feed forward AVITM network."""

    def __init__(self, input_size, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
                 learn_priors=True, topic_prior_mean=0.0, topic_prior_variance=None):
        """
        Initializes DecoderNetwork.

        Parameters
        ----------
        input_size : int
            Dimension of the input
        n_components : int (default=10)
            Number of topic components
        model_type : string (default='prodLDA')
            Type of the model that is going to be trained, 'prodLDA' or 'LDA'
        hidden_sizes : tuple, length = n_layers (default=(100,100))
            Size of the hidden layer
        activation : string (default='softplus')
            Activation function to be used, chosen from 'softplus', 'relu', 'sigmoid', 'leakyrelu', 'rrelu', 'elu', 'selu' or 'tanh'
        dropout : float (default=0.2)
            Percent of neurons to drop out.
        learn_priors : bool, (default=True)
            If true, priors are made learnable parameters
        topic_prior_mean: double (default=0.0)
            Mean parameter of the prior
        topic_prior_variance: double (default=None)
            Variance parameter of the prior
        """

        super(DecoderNetwork, self).__init__()

        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(n_components, int) and n_components > 0, \
            "n_components must be type int > 0."
        assert model_type in ['prodLDA', 'LDA'], \
            "model type must be 'prodLDA' or 'LDA'"
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'sigmoid', 'tanh', 'leakyrelu',
                              'rrelu', 'elu', 'selu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'leakyrelu'," \
            " 'rrelu', 'elu', 'selu' or 'tanh'."
        assert dropout >= 0, "dropout must be >= 0."
        assert isinstance(topic_prior_mean, float), \
            "topic_prior_mean must be type float"

        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors

        self.inf_net = InferenceNetwork(
            input_size, n_components, hidden_sizes, activation)
        if torch.cuda.is_available():
            self.inf_net = self.inf_net.cuda()

        # init prior parameters
        # \mu_1k = log \alpha_k + 1/K \sum_i log \alpha_i;
        # \alpha = 1 \forall \alpha
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        if topic_prior_variance is None:
            topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta = torch.Tensor(n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)

        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterizes the theta distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """Forward pass."""
        # batch_size x n_components
        posterior_mu, posterior_log_sigma = self.inf_net(x)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        topic_doc = theta
        theta = self.drop_theta(theta)

        # prodLDA vs LDA
        if self.model_type == 'prodLDA':
            # in: batch_size x input_size x n_components
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
            topic_word = self.beta
            # word_dist: batch_size x input_size
            self.topic_word_matrix = self.beta
        elif self.model_type == 'LDA':
            # simplex constrain on Beta
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            topic_word = beta
            word_dist = torch.matmul(theta, beta)
            # word_dist: batch_size x input_size

        return self.prior_mean, self.prior_variance, posterior_mu, \
               posterior_sigma, posterior_log_sigma, word_dist

    def get_theta(self, x):
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.inf_net(x)
            posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta
