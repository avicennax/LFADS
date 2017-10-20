from scipy.integrate import odeint
import scipy.stats as stats
import numpy as np
import pickle
import inspect
import time

def get_func_params(frame):
	args, _, _, values = inspect.getargvalues(frame)
	return {arg_name: values[arg_name] for arg_name in args}


# https://jakevdp.github.io/blog/2013/02/16/animating-the-lorentz-system-in-3d/
def lorentz_deriv(var, t0, sigma=10., beta=8. / 3, rho=28.0):
    """Compute the time-derivative of a Lorenz system."""
    x, y, z = var
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def main(conditions_num=30, neuron_num=30, trials_num=20, seed_std_dev=30,
	weights_std_dev=0.125, rate_mean=0.01, T=5, steps=1000, burn_in=100):
	"""
	Generate spikes from Lorenz latent factors

	Parameters
	----------
	conditions_num : int
		Number initialize states to run Lorenz attractor from
	neuron_num : int
		Number neurons to use in each trial
	trials_num : int
		Number trials (spike trains) to draw samples for
		per condition
	seed_std_dev : float
		Lorenz initial states are drawn from standard normal,
		this parameter controls standard deviation of initial draw.
	weights_std_dev : 0.125
		Specifies standard deviation of random weight matrix.
		NOTE: this is a very sensitive parameter and can determine
		quality of generated spikes; default value recommended: 0.125.
	rate_mean : float
		Set mean for rate vector.
		NOTE: this is a very sensitive parameter and can determine
		quality of generated spikes; default value recommended: 0.01.
	T : float
		Length of time to integrate to generate Lorenz factors.
	steps : int
		Number of integration steps for odeint.
	burn_in : int
		The number of steps to ignore for latent factors;
		employed to wash-out initial state.
	"""

	# Time vector
	t0 = 0
	t = np.linspace(t0, T, steps)  # one thousand time steps

	# Our synthetic dataset consisted of 65 conditions, with 20 trials per condition
	seeds = np.random.randn(65, 3) * seed_std_dev

	# Generate weights
	weights_std_dev = 0.125
	weights = np.random.rand(3, neuron_num) * weights_std_dev

	# Init data dictionary
	data_dict = {}
	data_dict['weights'] = weights
	data_dict['seeds'] = []
	data_dict['params'] = get_func_params(inspect.currentframe())

	for seed in seeds:
		condition_dict = {'x0': seed}
		t = np.linspace(t0, T, steps)
		states = odeint(lorentz_deriv, seed, t) 

		# seed state burn-in 
		t = t[burn_in:]
		factors = states[burn_in:]

		# Generate weights
		weights = np.random.rand(3, neuron_num) * weights_std_dev
		linear_response = np.dot(factors, weights)
		# Set average rate per neuron over all neurons
		shift = np.mean(linear_response) - np.log(rate_mean) 

		# Generate rates
		rates = np.exp(linear_response - shift)

		spikes = []
		for _ in range(trials_num):
			x = np.array([np.random.poisson(rate) for rate in rates.flatten()]).reshape(steps - burn_in, neuron_num)
			spikes.append((x > 0).astype(float))

		condition_dict['spikes'] = spikes
		data_dict['seeds'].append(condition_dict)


	timestamp = "{:.0f}".format(time.time())
	with open('data/lorenz_data-' + timestamp + '.p', 'wb') as fo:
		pickle.dump(data_dict, fo)


if __name__ == "__main__":
	desc = "Lorenz latent variable driven spike generator"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--conditions-num', type=int, default=30,
    	help='Number initialize states to run Lorenz attractor from')
    parser.add_argument('--neuron-num', type=int, default=30, 
    	help='Number neurons to use in each trial')
    parser.add_argument('--trials-num', type=int, default=20, 
    	help='Number trials (spike trains) to draw samples for per condition')
    parser.add_argument('--seed-std-dev', type=float, default=30, 
    	help='Lorenz initial states are drawn from standard normal ' 
    	'this parameter controls standard deviation of initial draw')
    parser.add_argument('--weights-std-dev', type=float, default=0.125, 
    	help='Specifies standard deviation of random weight matrix')
    parser.add_argument('--rate-mean', type=float, default=0.01, 
    	help='Set mean for rate vector')
      parser.add_argument('--T', type=float, default=5, 
      	help='Length of time to integrate to generate Lorenz factors')
    parser.add_argument('--steps', type=int, default=1000, 
    	help='Number of integration steps for odeint')
    parser.add_argument('--burn-in', type=int, default=100, 
    	help='he number of steps to ignore for latent factors; ' 
    	'employed to wash-out initial state.')
    args = vars(parser.parse_args())
    main(**args)
