import numpy as np
import tensorflow as tf
from tqdm import tqdm

tfkl = tf.keras.layers

class Ising(object):
    def __init__(self, J, h, simulation_mode='MCMC'):
        """
        :param simulation_mode: String. 
            'MCMC': Markov-Chain Monte Carlo
            'langevin': Hackey langevin
        """

        self.N = J.shape[0]
        self.J = J
        self.h = h

        # Starting estimate for number of flips to make on system in MCMC step.
        self.candidate_step_fraction = 0.05
        self.data = {}

    def get_params(self, extras=None, exclude=None):
        if extras is None:
            extras = {}
        keys = {'N', 'J', 'h'}
        keys.update(extras)
        return {s: self.__dict__[s] for s in keys if s in self.__dict__.keys()}

    def K_random_flips(self, K=0, N=None):
        """
        Produce a vector of N entries populated by zeros and approximately K instances of 1, chosen randomly.
        """
        if N is None:
            N = self.N
        prob = K/N
        return 1 - (rd.rand(N)>prob)

    def energy(self, state):
        """
        Energy as a function of state
        :return: Energy of state
        """
        return  - state @ self.J @ state - self.h * np.sum(state)

    def force(self, state):
        return 2 * J @ state + h * np.ones(self.N)

    def run_sim(self, time_steps, initial_state=None, key=None):
        if initial_state is None:
            initial_state = self.K_random_flips(self.N/2)
        self.run_sim_MCMC(time_steps, initial_state, key=key)

    def run_sim_MCMC(self, time_steps, initial_state, key=None):
        start_timestamp = strftime("%a, %d %b %Y %H:%M:%S", gmtime())
        N = self.N
        s = np.zeros((time_steps, N))
        old_s = initial_state
        old_energy = math.inf

        # step update initial params
        num_flips = np.zeros(time_steps)
        cur_num_flips = int(self.candidate_step_fraction * self.N)
        step_update_window = min(20, int(time_steps / 100))
        accepted_counter = 0  # increment every time a move is accepted, decrement at end of loop.

        for t in tqdm(range(time_steps)):
            new_s = np.abs(old_s - self.K_random_flips(K=cur_num_flips))
            new_energy = self.energy(new_s)
            alpha = np.exp(old_energy - new_energy)
            # print(alpha)
            if alpha > rd.uniform():  # accept step
                # add two to counter if window is moving (accept: +2-1 = +1, fail: +0-1 = -1). Otherwise, if not
                # scrolling yet, there's no decrement, so accept: +1-0 = +1, fail: +0 = 0.
                if t >= step_update_window:
                    accepted_counter = min(step_update_window, accepted_counter + 2)
                else:
                    accepted_counter += 1

                # update field and energy.
                s[t, :] = new_s
                old_s = new_s
                old_energy = new_energy

            else:
                s[t, :] = old_s

            # code to update proposal step size. Take average over last ten time steps.
            if t >= step_update_window:
                accepted_counter = max(0, accepted_counter - 1)
                if accepted_counter > step_update_window * 7 / 10:
                    cur_num_flips = min(self.N/2, cur_num_flips+1)
                elif accepted_counter < step_update_window * 3 / 10:
                    cur_num_flips = max(1, cur_num_flips-1)

            if t < time_steps - 1:
                num_flips[t] = cur_num_flips

        # Store data from simulation in dictionary under key.
        if key is None:
            key = len(self.data)

        self.data.update({
            key:
                {'params': self.get_params(),
                 'start_time': start_timestamp,
                 'steps': num_flips,
                 's': s}
        })


def mlp(hidden_dim, output_dim, layers, activation, dropout_rate=0):
    """Constructs multi-layer perceptron (MLP) critic with given number of hidden layers.

    Keyword arguments:
    hidden_dim (int) -- dimensionality of hidden dense layers
    output_dim (int) -- dimensionality of the output tensor
    layers (int) -- number of hidden dense layers
    activation -- activation function of the neurons
    """

    # return tf.keras.Sequential(
    #     [tfkl.Dense(hidden_dim, activation) for _ in range(layers)]
    #     + [tfkl.Dense(output_dim)]
    # )
    
    return tf.keras.Sequential(
        [DenseWithDropout(hidden_dim, activation, dropout_rate) for _ in range(layers)] + [tfkl.Dense(output_dim)]
    )


def polynomial(output_dim, order):
    return tf.keras.Sequential(PolynomialLayer(output_dim, order))


# @tf.keras.saving.register_keras_serializable(name="n_outer")
def n_outer(x, order):
    """takes outer product (order) times, along the last axis. This means first axes are broadcast together.

    For example:
      >>> a = tf.ones((2,3,4))
      >>> b = n_outer(a, 3) # b[i,j,k,l,m] = a[i,j,k] * a[i,j,l] * a[i,j,m]
      >>> b.shape
      TensorShape([2, 3, 4, 4, 4])

    The choice to not have an axis argument and reshaping steps is deliberate, for speed.

    In the case that order < 1, return a tf.ones tensor, with a feature dimensionality of 1.
    """
    if order < 1:
        newshape = tf.concat([tf.shape(x)[:-1], tf.ones(1, dtype=tf.int32)], axis=0)
        return tf.ones(
            newshape, dtype=x.dtype
        )  # replace feature dimension with singleton dim, set all entries to 1.
        # return tf.ones(tuple(tf.shape(x)[:-1]) + (1,), dtype=x.dtype)  # replace feature dimension with singleton dim, set all entries to 1.
    inds = "abcdefghijklmnopqrstuvwxyz"  # not that we'll ever go to 26th order :)
    to_inds = inds[:order]
    from_inds = "..." + ",...".join(list(to_inds))  # e.g. '...a,...b,...c,
    return tf.einsum(from_inds + "->..." + to_inds, *[x for _ in range(order)])

class DenseWithDropout(tfkl.Layer):
    """"""
    def __init__(self, output_size, activation, dropout_rate):
        super(DenseWithDropout, self).__init__()
        self.dense_layer = tfkl.Dense(output_size, activation)
        self.dropout_layer = tfkl.Dropout(dropout_rate)
        self.init_kwargs = {"output_size":output_size, "activation":activation, "dropout_rate":dropout_rate}

    def get_config(self):
        return self.init_kwargs

    def call(self, inputs, training=False):
        return self.dropout_layer(
            self.dense_layer(inputs), training=training
        )


class StackedLayer(tf.keras.layers.Layer):
    """
    This implements a layer that passes the input to each in a set of sub-layers and returns their concatenated outputs. 
    """
    def __init__(self, sub_layers):
        super(StackedLayer, self).__init__()
        self.sub_layers = sub_layers
    
    def get_config(self):
        return {"layers":layers}
    
    def call(self, inputs):
        return tf.stack([f(inputs) for f in self.sub_layers], axis=-1)


class LinearCombinationLayer(tf.keras.layers.Layer):
    """
    Takes basis_vectors, which has shape (N, M). N is the dimensionality of inputs to this model and M is the rank. The output is a linear combination of these M column vectors 
    """
    def __init__(self, basis_vectors, **kwargs):
        super(LinearCombinationLayer, self).__init__(**kwargs)
        self.basis_vectors = basis_vectors  # Store the provided basis vectors

        # Create a trainable weight for coefficients of the basis vectors
        self.coefficients = self.add_weight(
            shape=(self.basis_vectors.shape[1],),
            initializer="random_normal",
            trainable=True,
            name="coefficients",
        )

        # Set the basis vectors as a non-trainable weight
        self.set_basis_vectors = self.add_weight(
            shape=self.basis_vectors.shape,
            initializer=tf.constant_initializer(self.basis_vectors),
            trainable=False,
            name="basis_vectors",
        )

    def call(self, inputs):
        w = tf.einsum("ij,j->i", self.set_basis_vectors, self.coefficients)
        return tf.einsum("...i,i->...", inputs, w)


# @tf.keras.saving.register_keras_serializable()
class ConstantLayer(tf.keras.Model):
    def __init__(self, output_dim):
        """
        A layer designed as a component of the PolynomialLayer class via SimpleOuterLayer. It's role is to be a zeroth-order layer in the polynomial.

        Its output is a tensor of shape (batch_dim_1, batch_dim_2,...,output_dim) where the input is of shape (batch_dim_1, batch_dim_2,...,batch_dim_K, N_features). The output does not depend on the input, and is constant across batch_dim indices. In other words, the output is a tensor of shape (output_dim,) broadcast across any batch dimensions.
        """
        super(ConstantLayer, self).__init__()
        self.out = int(output_dim)
        self.c = tf.Variable(tf.zeros(self.out))

    # def build(self, input_shape):
    #   self.batch_dims = tuple(input_shape[:-1])
    #   self.c = tf.Variable(tf.zeros(self.out))
    #   # self.c = tf.Variable(tf.zeros(tuple(input_shape[:-1]) + (self.out,)))

    @tf.function
    def call(self, inputs):
        batch_dims = tf.shape(inputs)[:-1]
        new_shape = tf.concat([batch_dims, tf.constant([self.out])], axis=0)
        return tf.zeros(new_shape) + self.c[..., :]
        # return tf.zeros(batch_dims + (self.out,)) + self.c[...,:]


# @tf.keras.saving.register_keras_serializable()
class SimpleOuterLayer(tf.keras.Model):
    """
    A layer designed for use in the PolynomialLayer class. It has a fixed output dimension of (output_dim,), including broadcasting over batch dimensions in the input. The input is assumed to be structured as (batch_dim_1,...,batch_dim_K, N_features). The layer itself is shaped (N_features,) x order + (output_dim,).
    """

    def __init__(self, output_dim, order):
        super(SimpleOuterLayer, self).__init__()
        self.output_dim = int(output_dim)
        self.order = int(order)

        if order < 1:
            self.w = ConstantLayer(output_dim=self.output_dim)
            # self.w = tfkl.EinsumDense(tuple(x.shape[:-1]) + (1,), dtype=x.dtype)  # replace feature dimension with singleton dim, set all entries to 1.
        else:
            inds = (
                "abcdefghijklmnopqrstuvwxyz"  # not that we'll ever go to 26th order :)
            )
            input_inds = inds[:order]
            layer_inds = inds[: order + 1]
            # first half, i.e. '...ab,abc'
            from_inds = (
                "..." + "".join(list(input_inds)) + "," + "".join(list(layer_inds))
            )
            to_inds = "..." + inds[order]
            self.w = tfkl.EinsumDense(
                equation=from_inds + "->" + to_inds, output_shape=output_dim
            )

    def get_config(self):
        return {"output_dim": self.output_dim, "order": self.order}

    @tf.function
    def call(self, inputs):
        x = n_outer(inputs, self.order)
        return self.w(x)


# @tf.keras.saving.register_keras_serializable()
class PolynomialLayer(tf.keras.Model):
    def __init__(self, output_dim, order):
        # super(PolynomialLayer2, self)__init__()
        """May want to recreate this class with a collection of layers of the shape (batch, N, N, ..., N) which will be much easier to use"""

        super(PolynomialLayer, self).__init__()
        self.output_dim = int(output_dim)
        self.order = int(order)
        self.sub_layers = [
            SimpleOuterLayer(self.output_dim, order=n) for n in range(self.order + 1)
        ]

    def get_config(self):
        return {"output_dim": self.output_dim, "order": self.order}

    @tf.function
    def call(self, inputs):
        return tfkl.add([f(inputs) for f in self.sub_layers])


# @tf.keras.saving.register_keras_serializable()
class PolynomialLayer2(tf.keras.Model):
    def __init__(self, input_dim, output_dim, order, init_rule=None):
        super(PolynomialLayer2, self).__init__()

        self.N = int(input_dim)
        self.order = int(order)
        self.init_rule = init_rule
        full_input_dim = self.geometric_sum(self.N, self.order)

        if isinstance(init_rule, np.ndarray):
            init = init_rule
        else:
            w_init = tf.random_normal_initializer()
            init = w_init(shape=(full_input_dim, int(output_dim)), dtype="float32")
        self.ws = tf.Variable(initial_value=init, trainable=True)

    def geometric_sum(self, x, order, dtype="int"):
        """return 1 + x + x**2 + ... + x**order"""
        return np.sum(np.array([x**n for n in range(order)])).astype(dtype)

    def get_weights_by_order(self, order=None, *args, **kwargs):
        ws = self.get_weights(*args, **kwargs)
        if order is None:
            return [
                self.get_weights_by_order(order=n, *args, **kwargs)[0]
                for n in range(self.order)
            ]
        else:
            start_ind = self.geometric_sum(self.N, order=order)
            end_ind = self.geometric_sum(self.N, order=order + 1)
            return [
                ws[0][start_ind:end_ind, :]
            ]  # wrapping in list to be consistent with typical get_weights() behvior

    def get_config(self):
        return {
            "input_dim": self.N,
            "output_dim": self.output_dim,
            "order": self.order,
            "init_rule": self.init_rule,
        }

    @tf.function
    def call(self, inputs):
        if inputs.dtype != self.ws.dtype:
            inputs = tf.cast(inputs, self.ws.dtype)
        extended_inputs = [(n_outer(inputs, order=n)) for n in range(self.order)]
        x = [
            tf.reshape(extended_inputs[n], (-1,) + (self.N**n,))
            for n in range(self.order)
        ]
        x = tfkl.concatenate(x, axis=-1)
        return tf.einsum("ij,...i->...j", self.ws, x)


# class PolynomialCoarseGrainer(tf.keras.Model):
#   def __init__(self, output_dim, order=1, **extra_kwargs):
#     super(PolynomialCoarseGrainer, self).__init__()


# @tf.keras.saving.register_keras_serializable()
class SeparableCustomCritic(tf.keras.Model):
    """
    Example input:
    pairs = ([0,2], [2,0], [1,1], [1,1], [1,1])

    gives xx, yy, xy, xy, xy.
    """

    def __init__(self, pairs, **extra_kwargs):
        super(SeparableCustomCritic, self).__init__()
        self.pairs = pairs
        self.embed_dim = len(pairs)
        self._g = []
        self._h = []
        for p in pairs:
            cur_output_dim_x = 1
            cur_output_dim_y = 1
            cur_order_x = p[0]
            cur_order_y = p[1]
            self._g.append(
                SimpleOuterLayer(output_dim=cur_output_dim_x, order=cur_order_x)
            )
            self._h.append(
                SimpleOuterLayer(output_dim=cur_output_dim_y, order=cur_order_y)
            )

    @tf.function
    def call(self, x, y):
        return tf.keras.layers.add(
            [
                tf.einsum("ij,kj->ik", self._h[i](y), self._g[i](x))
                for i in range(self.embed_dim)
            ]
        )


# @tf.keras.saving.register_keras_serializable()
class SeparablePolynomialCritic(tf.keras.Model):
    def __init__(self, embed_dim, order_x=1, order_y=1, **extra_kwargs):
        """
        Constructs critic function for x and y in terms of a polynomial basis.
        """

        super(SeparablePolynomialCritic, self).__init__()
        self._g = polynomial(int(embed_dim), int(order_x))
        self._h = polynomial(int(embed_dim), int(order_y))

    @tf.function
    def call(self, x, y):
        return tf.einsum("ij,kj->ik", self._h(y), self._g(x))


# @tf.keras.saving.register_keras_serializable()
class SeparableCritic(tf.keras.Model):
    """Separable ansatz (critic) for MI bound.

    Attributes:
    _g (_h) -- MLP ansatz for X (_Y) variable

    Methods:
    call(x, y) -- calls the ansatz as a function for samples x, y
    """

    def __init__(self, hidden_dim, embed_dim, layers, activation, dropout_rate=0):
        """Constructs two separate MLP critics of same structure for x and y data.

        Keyword arguments:
        hidden_dim (int) -- dimensionality of hidden dense layers
        embed_dim (int) -- dimension of contracted output of the dense nets _g and _h
        layers (int) -- number of hidden layers in dense nets _g and _h
        activation -- activation function of the neurons

        :Author:
          Ben Poole, Doruk Efe Gökmen, Copyright 2019 Google LLC.
        """

        super(SeparableCritic, self).__init__()
        self._g = mlp(hidden_dim, embed_dim, layers, activation, dropout_rate)
        self._h = mlp(hidden_dim, embed_dim, layers, activation, dropout_rate)


    @tf.function
    def call(self, x, y):
        """Constructs unnormalised likelihood matrix (or scores)
        from the two separate MLP's for x and y data.

        Keyword arguments:
        x -- a sample for the random variable X
        y -- a sample for the random variable Y
        """

        return tf.einsum("ij,kj->ik", self._h(y), self._g(x))
