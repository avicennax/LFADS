import tensorflow as tf

class LFADS:

    def __init__():
        pass

    def loss():
        pass

    def controller():
        pass

    def g_encoder():
        pass

    def u_encoder():
        pass

    def full_inference():
        pass

def build_weights(X, c_hidden_dim, factor_dim, inferred_input_dim, generator_dim, encoder_hidden_dim):
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.)

    weights = {
        'w_g_mu': tf.get_variable('w_g_mu', [2 * encoder_hidden_dim, generator_dim], initializer=w_init),
        'w_g_sigma': tf.get_variable('w_g_sigma', [2 * encoder_hidden_dim, generator_dim], initializer=w_init),
        'b_g_mu': tf.get_variable('b_g_mu', [generator_dim], initializer=b_init),
        'b_g_sigma': tf.get_variable('b_g_sigma', [generator_dim], initializer=b_init),

        'w_c0': tf.get_variable('w_c0', [2 * encoder_hidden_dim, c_hidden_dim], initializer=w_init),
        'w_f0': tf.get_variable('w_f0', [generator_dim, factor_dim], initializer=w_init),
        'b_c0': tf.get_variable('b_c0', [c_hidden_dim], initializer=b_init),
        'b_f0': tf.get_variable('b_f0', [factor_dim], initializer=b_init),

        'w_c_mu': tf.get_variable('w_c_mu', [c_hidden_dim, inferred_input_dim], initializer=w_init),
        'w_c_sigma': tf.get_variable('w_c_sigma', [c_hidden_dim, inferred_input_dim], initializer=w_init),
        'b_c_mu': tf.get_variable('b_c_mu', [inferred_input_dim], initializer=b_init),
        'b_c_sigma': tf.get_variable('b_c_sigma', [inferred_input_dim], initializer=b_init),

        'w_rate': tf.get_variable('w_rate', [factor_dim, X.get_shape()[2]], initializer=w_init),
        'w_factor': tf.get_variable('w_factor', [generator_dim, factor_dim], initializer=w_init),
        'b_rate': tf.get_variable('b_rate', [X.get_shape()[0], X.get_shape()[2]], initializer=b_init),
        'b_factor': tf.get_variable('b_factor', [factor_dim], initializer=b_init),

        'g_e_b0': tf.get_variable('g_e_b0', [encoder_hidden_dim], initializer=w_init),
        'g_e_f0': tf.get_variable('g_e_f0', [encoder_hidden_dim], initializer=w_init),
        'u_e_b0': tf.get_variable('u_e_b0', [encoder_hidden_dim], initializer=w_init),
        'u_e_f0': tf.get_variable('u_e_f0', [encoder_hidden_dim], initializer=w_init)
    }

    return weights


def build_initial_state(state, xa, encoder_hidden_dim):
    batch_size = xa[0].get_shape()[0].value
    init_states = tf.tile(state, [batch_size])
    return tf.reshape(init_states, [batch_size, encoder_hidden_dim])


def g_encoder(x, a, weights, encoder_hidden_dim):
    """
    Q(g|X, A)
    """
    backwards_cell = tf.nn.rnn_cell.GRUCell(encoder_hidden_dim)
    forwards_cell = tf.nn.rnn_cell.GRUCell(encoder_hidden_dim)

    xa = tf.concat([x, a], axis=2)
    xa = tf.unstack(xa, axis=1)

    e_b, _ = tf.nn.static_rnn(
        backwards_cell, list(reversed(xa)), dtype=tf.float32, scope='g_backwards', 
        initial_state=build_initial_state(weights['g_e_b0'], xa, encoder_hidden_dim))
    e_f, _ = tf.nn.static_rnn(
        forwards_cell, xa, dtype=tf.float32, scope='g_forwards', 
        initial_state=build_initial_state(weights['g_e_f0'], xa, encoder_hidden_dim))

    e = tf.concat([e_b[-1], e_f[-1]], axis=1)

    mu = tf.matmul(e, weights['w_g_mu']) + weights['b_g_mu']
    sigma = tf.exp(tf.scalar_mul(
        1. / 2., tf.matmul(e, weights['w_g_sigma']) + weights['b_g_sigma']))

    return e, mu, sigma


def u_encoder(x, a, weights, encoder_hidden_dim):
    """
    Forwards/backwards encodings to be passed to controller
    for use with previous factor to generate next inferred
    input.
    """
    backwards_cell = tf.nn.rnn_cell.GRUCell(encoder_hidden_dim)
    forwards_cell = tf.nn.rnn_cell.GRUCell(encoder_hidden_dim)

    xa = tf.concat([x, a], axis=2)
    xa = tf.unstack(xa, axis=1)

    # Forward passes
    e_b, _ = tf.nn.static_rnn(
        backwards_cell, list(reversed(xa)), dtype=tf.float32, scope='u_backwards', 
        initial_state=build_initial_state(weights['u_e_b0'], xa, encoder_hidden_dim))
    e_f, _ = tf.nn.static_rnn(
        forwards_cell, xa, dtype=tf.float32, scope='u_forwards', 
        initial_state=build_initial_state(weights['u_e_f0'], xa, encoder_hidden_dim))

    # Reshaping for concat
    e_b = tf.stack(e_b)
    e_b = tf.transpose(e_b, [1, 0, 2])

    e_f = tf.stack(e_f)
    e_f = tf.transpose(e_f, [1, 0, 2])

    encodings = tf.concat([e_b, e_f], axis=2)

    return encodings


def u_controller(c, weights):
    mu = tf.matmul(c, weights['w_c_mu']) + weights['b_c_mu']
    sigma = tf.exp(tf.matmul(c, weights['w_c_sigma']) + weights['b_c_mu'])

    return mu, sigma


def std_mv_gaussian_KL(mu, sigma):
    shape = sigma.get_shape()
    bdim = shape[0].value
    mu_dim = shape[1].value
    sigma = tf.reshape(sigma, [bdim, mu_dim, 1]) * tf.reshape(tf.eye(mu_dim), [1, mu_dim, mu_dim])
    return 0.5 * tf.trace(sigma) + tf.reduce_sum(mu*mu, axis=1) - tf.log(1e-8 + tf.matrix_determinant(sigma)) - mu_dim


def unstack_on_t_axis(tensor):
    return tf.unstack(tensor, axis=1)


def model(
    X, A, c_hidden_dim=5, factor_dim=3, sample_size=20,
    inferred_input_dim=2, generator_dim=5, encoder_hidden_dim=3):

    with tf.variable_scope("full_model") as scope:
        weights = build_weights(
            X, c_hidden_dim, factor_dim,inferred_input_dim, generator_dim, encoder_hidden_dim)

        controller = tf.nn.rnn_cell.GRUCell(c_hidden_dim)
        generator = tf.nn.rnn_cell.GRUCell(generator_dim)

        # Encodings
        c0_encoding, g_mu, g_sigma = g_encoder(X, A, weights, encoder_hidden_dim)
        c_encodings = u_encoder(X, A, weights, encoder_hidden_dim)

        marginal_g_KLs = []
        marginal_u_KLs = []
        marginal_log_likelihoods = []
        # This loop is akin to a Monte Carlo estimate of the expectations of
        # the likelihood and KLs taken with respect to Z = {g, u}
        for _ in range(sample_size):
            # Sampling initial states
            g_state = g_mu + tf.sqrt(g_sigma) * \
                tf.random_normal(tf.shape(g_mu), 0, 1, dtype=tf.float32)
            c_state = tf.matmul(c0_encoding, weights['w_c0']) + weights['b_c0']
            factor = tf.matmul(g_state, weights['w_f0']) + weights['b_f0']

            u_KLs = []
            log_likelihoods = []
            X_T, c_encodings_T = tuple(map(unstack_on_t_axis, [X, c_encodings]))

            with tf.variable_scope("rnn-scope") as rnn_scope:
                for x, encoding in zip(X_T, c_encodings_T):
                    # controller
                    efactor = tf.concat([factor, encoding], axis=1)
                    c_output, c_state = controller(efactor, c_state, scope='controller')
                    u_mu, u_sigma = u_controller(c_output, weights)
                    u = u_mu + u_sigma * \
                        tf.random_normal(tf.shape(u_mu), 0, 1, dtype=tf.float32)
                    u_KLs.append(std_mv_gaussian_KL(u_mu, u_sigma))

                    # decoder
                    g_output, g_state = generator(u, g_state, scope='generator')
                    factors = tf.matmul(
                        g_output, weights['w_factor']) + weights['b_factor']
                    rates = tf.exp(tf.matmul(factors, weights['w_rate']) + weights['b_rate'])
                    poisson = tf.contrib.distributions.Poisson(rates)
                    joint_prob = tf.reduce_prod(tf.log(poisson.prob(x)), axis=1)
                    log_likelihoods.append(joint_prob)

                    rnn_scope.reuse_variables()


            marginal_log_likelihoods.append(tf.add_n(log_likelihoods))
            marginal_u_KLs.append(tf.add_n(u_KLs))
            marginal_g_KLs.append(std_mv_gaussian_KL(g_mu, g_sigma))

    likelihood = tf.reduce_mean(tf.stack(marginal_log_likelihoods))
    u_KL = tf.reduce_mean(tf.stack(marginal_u_KLs))
    g_KL = tf.reduce_mean(tf.stack(marginal_g_KLs))
    elbo = likelihood - g_KL - u_KL
    loss = - elbo

    return loss, likelihood, u_KL + g_KL
