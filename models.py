import tensorflow as tf

def unstack_on_t_axis(tensor):
    return tf.unstack(tensor, axis=1)


class LFADS:

    def build_weights(self):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        self.weights = {
            'w_g_mu': tf.get_variable('w_g_mu', [2 * self.encoder_hidden_dim, self.generator_dim], initializer=w_init),
            'w_g_sigma': tf.get_variable('w_g_sigma', [2 * self.encoder_hidden_dim, self.generator_dim], initializer=w_init),
            'b_g_mu': tf.get_variable('b_g_mu', [self.generator_dim], initializer=b_init),
            'b_g_sigma': tf.get_variable('b_g_sigma', [self.generator_dim], initializer=b_init),

            'w_c0': tf.get_variable('w_c0', [2 * self.encoder_hidden_dim, self.c_hidden_dim], initializer=w_init),
            'w_f0': tf.get_variable('w_f0', [self.generator_dim, self.factor_dim], initializer=w_init),
            'b_c0': tf.get_variable('b_c0', [self.c_hidden_dim], initializer=b_init),
            'b_f0': tf.get_variable('b_f0', [self.factor_dim], initializer=b_init),

            'w_c_mu': tf.get_variable('w_c_mu', [self.c_hidden_dim, self.infered_input_dim], initializer=w_init),
            'w_c_sigma': tf.get_variable('w_c_sigma', [self.c_hidden_dim, self.infered_input_dim], initializer=w_init),
            'b_c_mu': tf.get_variable('b_c_mu', [self.infered_input_dim], initializer=b_init),
            'b_c_sigma': tf.get_variable('b_c_sigma', [self.infered_input_dim], initializer=b_init),

            'w_rate': tf.get_variable('w_rate', [self.factor_dim, self.neuron_dim], initializer=w_init),
            'w_factor': tf.get_variable('w_factor', [self.generator_dim, self.factor_dim], initializer=w_init),
            'b_rate': tf.get_variable('b_rate', [self.neuron_dim], initializer=b_init),
            'b_factor': tf.get_variable('b_factor', [self.factor_dim], initializer=b_init),

            'g_e_b0': tf.get_variable('g_e_b0', [self.encoder_hidden_dim], initializer=w_init),
            'g_e_f0': tf.get_variable('g_e_f0', [self.encoder_hidden_dim], initializer=w_init),
            'u_e_b0': tf.get_variable('u_e_b0', [self.encoder_hidden_dim], initializer=w_init),
            'u_e_f0': tf.get_variable('u_e_f0', [self.encoder_hidden_dim], initializer=w_init)
        }

    def build_rnn_cells(self):
        self.rnn_cells = {
            'controller': tf.nn.rnn_cell.GRUCell(self.c_hidden_dim),
            'generator': tf.nn.rnn_cell.GRUCell(self.generator_dim),
            'u_backwards_cell': tf.nn.rnn_cell.GRUCell(self.encoder_hidden_dim),
            'u_forwards_cell': tf.nn.rnn_cell.GRUCell(self.encoder_hidden_dim),
            'g_backwards_cell': tf.nn.rnn_cell.GRUCell(self.encoder_hidden_dim),
            'g_forwards_cell': tf.nn.rnn_cell.GRUCell(self.encoder_hidden_dim),
        }

    def __init__(self, neuron_dim, c_hidden_dim=5, factor_dim=3,
        infered_input_dim=2, generator_dim=5, encoder_hidden_dim=3):
        
        self.neuron_dim = neuron_dim
        self.c_hidden_dim = c_hidden_dim
        self.factor_dim = factor_dim
        self.infered_input_dim = infered_input_dim
        self.generator_dim = generator_dim
        self.encoder_hidden_dim = encoder_hidden_dim

        self.build_weights()
        self.build_rnn_cells()

    def loss(self):
        pass

    def full_inference(self, X, A, sample_size):
        with tf.variable_scope("full_model") as scope:
            # Encodings
            c0_encoding, g_mu, g_sigma = self.g_encoder(X, A)
            c_encodings = self.u_encoder(X, A)

            marginal_g_KLs = []
            marginal_u_KLs = []
            marginal_log_likelihoods = []
            # This generates Monte Carlo estimate of the expectations of
            # the likelihood and KLs taken with respect to Z = {g, u}
            for _ in range(sample_size):
                # Sampling initial states
                g_state = g_mu + tf.sqrt(g_sigma) * \
                    tf.random_normal(tf.shape(g_mu), 0, 1, dtype=tf.float32)
                c_state = tf.matmul(c0_encoding, self.weights['w_c0']) + self.weights['b_c0']
                factor = tf.matmul(g_state, self.weights['w_f0']) + self.weights['b_f0']

                u_KLs = []
                log_likelihoods = []
                X_T, c_encodings_T = tuple(map(unstack_on_t_axis, [X, c_encodings]))

                with tf.variable_scope("rnn-scope") as rnn_scope:
                    for x, encoding in zip(X_T, c_encodings_T):
                        # controller
                        efactor = tf.concat([factor, encoding], axis=1)
                        c_output, c_state = self.rnn_cells['controller'](efactor, c_state, scope='controller')
                        u_mu, u_sigma = self.u_controller(c_output)
                        u = u_mu + u_sigma * \
                            tf.random_normal(tf.shape(u_mu), 0, 1, dtype=tf.float32)
                        u_KLs.append(self.std_mv_gaussian_KL(u_mu, u_sigma))

                        # decoder
                        g_output, g_state = self.rnn_cells['generator'](u, g_state, scope='generator')
                        factors = tf.matmul(
                            g_output, self.weights['w_factor']) + self.weights['b_factor']
                        rates = tf.exp(tf.matmul(factors, self.weights['w_rate']) + self.weights['b_rate'])
                        poisson = tf.contrib.distributions.Poisson(rates)
                        joint_prob = tf.reduce_prod(tf.log(poisson.prob(x)), axis=1)
                        log_likelihoods.append(joint_prob)

                        rnn_scope.reuse_variables()

                marginal_log_likelihoods.append(tf.add_n(log_likelihoods))
                marginal_u_KLs.append(tf.add_n(u_KLs))
                marginal_g_KLs.append(self.std_mv_gaussian_KL(g_mu, g_sigma))

        likelihood = tf.reduce_mean(tf.stack(marginal_log_likelihoods))
        u_KL = tf.reduce_mean(tf.stack(marginal_u_KLs))
        g_KL = tf.reduce_mean(tf.stack(marginal_g_KLs))
        elbo = likelihood - g_KL - u_KL
        loss = - elbo

        return loss, likelihood, u_KL + g_KL

    def build_initial_state(self, state, xa):
        batch_size = xa[0].get_shape()[0].value
        init_states = tf.tile(state, [batch_size])
        return tf.reshape(init_states, [batch_size, self.encoder_hidden_dim])

    def g_encoder(self, x, a):
        """
        Encoder for g0.
        """
        xa = tf.concat([x, a], axis=2)
        xa = tf.unstack(xa, axis=1)

        e_b, _ = tf.nn.static_rnn(
            self.rnn_cells['g_backwards_cell'], list(reversed(xa)), dtype=tf.float32, scope='g_backwards', 
            initial_state=self.build_initial_state(self.weights['g_e_b0'], xa))
        e_f, _ = tf.nn.static_rnn(
            self.rnn_cells['g_forwards_cell'], xa, dtype=tf.float32, scope='g_forwards', 
            initial_state=self.build_initial_state(self.weights['g_e_f0'], xa))

        e = tf.concat([e_b[-1], e_f[-1]], axis=1)

        mu = tf.matmul(e, self.weights['w_g_mu']) + self.weights['b_g_mu']
        sigma = tf.exp(tf.scalar_mul(
            1. / 2., tf.matmul(e, self.weights['w_g_sigma']) + self.weights['b_g_sigma']))

        return e, mu, sigma

    def u_encoder(self, x, a):
        """
        Forwards/backwards encodings to be passed to controller
        for use with previous factor to generate next inferred
        input.
        """
        xa = tf.concat([x, a], axis=2)
        xa = tf.unstack(xa, axis=1)

        # Forward passes
        e_b, _ = tf.nn.static_rnn(
            self.rnn_cells['u_backwards_cell'], list(reversed(xa)), dtype=tf.float32, scope='u_backwards', 
            initial_state=self.build_initial_state(self.weights['u_e_b0'], xa))
        e_f, _ = tf.nn.static_rnn(
            self.rnn_cells['u_forwards_cell'], xa, dtype=tf.float32, scope='u_forwards', 
            initial_state=self.build_initial_state(self.weights['u_e_f0'], xa))

        # Reshaping for concat
        e_b = tf.stack(e_b)
        e_b = tf.transpose(e_b, [1, 0, 2])

        e_f = tf.stack(e_f)
        e_f = tf.transpose(e_f, [1, 0, 2])

        encodings = tf.concat([e_b, e_f], axis=2)

        return encodings

    def u_controller(self, c):
        mu = tf.matmul(c, self.weights['w_c_mu']) + self.weights['b_c_mu']
        sigma = tf.exp(tf.matmul(c, self.weights['w_c_sigma']) + self.weights['b_c_mu'])

        return mu, sigma

    def std_mv_gaussian_KL(self, mu, sigma):
        """
        Calculates KL divergence between standard MV Normal and
        a MV normal with mean: mu, and diagonal covariance: sigma.
        """
        shape = sigma.get_shape()
        bdim = shape[0].value
        mu_dim = shape[1].value
        sigma = tf.reshape(sigma, [bdim, mu_dim, 1]) * tf.reshape(tf.eye(mu_dim), [1, mu_dim, mu_dim])
        return 0.5 * tf.trace(sigma) + tf.reduce_sum(mu*mu, axis=1) - tf.log(1e-8 + tf.matrix_determinant(sigma)) - mu_dim