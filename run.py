import tensorflow as tf
import numpy as np
import os.path as op 
import os
import argparse
import models

from utils import sigmoid
from models import LFADS

def main(
    data_path, batch_size=128, neuron_dim=30, g_hidden_dim=32, c_hidden_dim=32, factor_dim=3, infered_input_dim=8, 
    generator_dim=32, encoder_hidden_dim=16, dropout_keep_p=0.9, kl_prior_var=0.1, state_clip_value=5, 
    learn_rate=1e-3, num_epochs=100, max_norm=50, z_sample_size=20):

    # Load training data
    training_spikes = np.load(data_path)
    n_samples, T, d = training_spikes.shape

    # Generating inputs
    inputs = np.random.randn(n_samples, T, 1)
    training_data = [(x, a) for x, a in zip(training_spikes, inputs)]


    # Initialize TF
    x = tf.placeholder(tf.float32, shape=[None, T, d], name='spikes')
    a = tf.placeholder(tf.float32, shape=[None, T, d], name='inputs')
    kl_weight = tf.placeholder(tf.float32, shape=[], name='kl_weight')

    # Model
    model = LFADS(
        neuron_dim=neuron_dim, c_hidden_dim=c_hidden_dim, factor_dim=factor_dim, 
        infered_input_dim=infered_input_dim, generator_dim=generator_dim, 
        encoder_hidden_dim=encoder_hidden_dim, dropout_keep_p=dropout_keep_p, 
        kl_prior_var=kl_prior_var, state_clip_value=state_clip_value)

    loss, lik, kl = model.loss(
        X=x, A=a, KL_weight=kl_weight, sample_size=z_sample_size)

    # Generate loss
    optimizer = tf.train.AdamOptimizer(learn_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, max_norm)
    train_op = optimizer.apply_gradients(zip(gradients, variables))

    writer = tf.summary.FileWriter('./graphs', sess.graph)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for epoch in range(num_epochs):
            np.random.shuffle(training_data) 

            # Approximate map of [0, num_epochs] -> [-4, 4]
            # Thus our KL weights spans from ~0 to ~1 during training.
            kl_weight = sigmoid((8 * epoch) / num_epochs - 4)

            for i in range(0, n_samples, batch_size):
                batch = training_data[i: i+total_batch]
                x_i, a_i = batch

            _, loss, loss_lik, loss_kl = sess.run(
                (train_op, loss, lik, kl),
                feed_dict={X: x_i, A: a_i, KL_weight: kl_weight})

        print("epoch: {i}/{n} - loss={loss:.4f}, log-likelihood={log_lik:.4f}, KL={loss_kl:.4f}".format(
            i=epoch, n=num_epochs, loss=less, log_lik=loss_lik, loss_kl=loss_kl))

    writer.close()

if __name__ == "__main__":
    desc = "LFADS: Latent Factor Analysis via Dynamical Systems"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data-path', type=str,  required=True)
    parser.add_argument('--g-hidden-dim', type=int, default=32, help='Dimension of latent vector')
    parser.add_argument('--c-hidden-dim', type=int, default=32, help='Dimension of controller hidden state')
    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Optimizer learning rate')
    parser.add_argument('--num-epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--neuron-dim', type=int, default=30, help='Dimension of latent vector')
    parser.add_argument('--factor-dim', type=float, default=3, help='Optimizer learning rate')
    parser.add_argument('--infered-input-dim', type=int, default=8, help='Training epochs')
    parser.add_argument('--generator-dim', type=int, default=32, help='Batch size')
    parser.add_argument('--encoder-hidden-dim', type=int, default=16, help='Dimension of controller hidden state')
    parser.add_argument('--dropout-keep-p', type=float, default=0.9, help='Optimizer learning rate')
    parser.add_argument('--kl-prior-var', type=float, default=0.1, help='Training epochs')
    parser.add_argument('--state-clip-value', type=float, default=5.0, help='Batch size')
    parser.add_argument('--max-norm', type=float, default=50, help='Batch size')
    args = vars(parser.parse_args())
    main(**args)