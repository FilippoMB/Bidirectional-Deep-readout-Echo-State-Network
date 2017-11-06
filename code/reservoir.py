import numpy as np
from scipy import sparse
from scipy.sparse import linalg as slinalg
from sklearn.decomposition import KernelPCA, PCA


class Reservoir(object):
    '''
    This class contains methods to generate and retrieve states form a reservoir.
    
    CONSTRUCTOR:
        generates a random reservoir parametrized by the following inputs
        - n_internal_units: number of processing units (neurons) in the reservoir
        - spectral_radius: largest eigenvalue of the reservoir
        - connectivity: number of nonzero connections in the reservoir
        - input_scaling: the constant value reservoir inputs are multiplied with
        - noise_level: deviation of the Gaussian noise 
    
    GET_STATES:
        returns the last internal reservoir state generated after processing an input time series.
        - X: input time series
        - n_drop: not used
        - embedding: dimensionality reduction procedure applied to the last state of the reservoir.
            options are "identity" (no reduction), "pca" and "kpca"
        - bidir: the reservoir process X also in reverse order. A new final state is generated and
            and is concatenated to the original final state.
        - train: if True, the embedding method is fitted on the training data.
    '''
    def __init__(self, n_internal_units=100, spectral_radius=0.9,
                 connectivity=0.3, input_scaling=0.5, noise_level=0.01):
        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._spectral_radius = spectral_radius
        self._connectivity = connectivity
        self._input_scaling = input_scaling
        self._noise_level = noise_level

        # The weights will be set later, when data is provided
        self._input_weights = None
        self._feedback_weights = None

        # Regression method. Initialized to None for now.
        # Will be set in get_states.
        self._embedding_method = None

        # Generate internal weights
        self._internal_weights = self._initialize_internal_weights(
            n_internal_units,
            connectivity,
            spectral_radius)

    def _initialize_internal_weights(self, n_internal_units,
                                     connectivity, spectral_radius):
        # The eigs function might not converge. Attempt until it does.
        convergence = False
        while (not convergence):
            # Generate sparse, uniformly distributed weights.
            internal_weights = sparse.rand(n_internal_units,
                                           n_internal_units,
                                           density=connectivity).todense()

            # Ensure that the nonzero values are
            # uniformly distributed in [-0.5, 0.5]
            internal_weights[np.where(internal_weights > 0)] -= 0.5

            try:
                # Get the largest eigenvalue
                w, _ = slinalg.eigs(internal_weights, k=1, which='LM')

                convergence = True

            except:
                continue

        # Adjust the spectral radius.
        internal_weights /= np.abs(w)/spectral_radius

        return internal_weights

    def get_states(self, X, n_drop=0, embedding='identity', n_dim=3,
                   embedding_params=None, bidir=True, train=True):
        N, T, V = X.shape
        if self._input_weights is None:
            self._input_weights = \
                2.0*np.random.rand(self._n_internal_units, V) - 1.0

        # last reservoir state
        states = self._compute_state_matrix(X, n_drop)
        last_states = states[:, -1, :]
        if bidir is True:
            # last reservoir state on time reversed input
            X_r = X[:, ::-1, :]
            states_r = self._compute_state_matrix(X_r, n_drop)
            last_states = np.concatenate(
                (states[:, -1, :],
                 states_r[:, -1, :]),
                axis=1)

        # initialize embedding method
        if train:
            if embedding.lower() == 'kpca':
                self._embedding_method = KernelPCA(n_components=n_dim,
                                                   kernel='rbf',
                                                   gamma=embedding_params)
            elif embedding.lower() == 'pca':
                self._embedding_method = PCA(n_components=n_dim)

        # compute embedding
        if self._embedding_method is not None:
            if train:
                embedded_states = \
                    self._embedding_method.fit_transform(last_states)
            else:
                embedded_states = self._embedding_method.transform(last_states)
        else:
            embedded_states = last_states

        return embedded_states

    def _compute_state_matrix(self, X, n_drop=0):
        N, T, _ = X.shape
        previous_state = np.zeros((N, self._n_internal_units), dtype=float)

        # Storage
        state_matrix = np.empty((N, T - n_drop, self._n_internal_units),
                                dtype=float)

        for t in range(T):
            current_input = X[:, t, :]*self._input_scaling

            # Calculate state. Add noise and apply nonlinearity.
            state_before_tanh = \
                self._internal_weights.dot(previous_state.T) \
                + self._input_weights.dot(current_input.T)

            state_before_tanh += \
                np.random.rand(self._n_internal_units, N)*self._noise_level

            previous_state = np.tanh(state_before_tanh).T

            # Store everything after the dropout period
            if (t > n_drop - 1):
                state_matrix[:, t - n_drop, :] = previous_state

        return state_matrix

