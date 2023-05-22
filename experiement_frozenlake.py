import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import random
import math
from util import saveGraph
import warnings
warnings.filterwarnings("ignore")

random.seed(100)

best_vi_time = 0
best_vi_iterations = 0
best_vi_score = 0
best_vi_gamma = 0
best_vi_delta = 0

best_pi_time = 0
best_pi_iterations = 0
best_pi_score = 0
best_pi_gamma = 0
best_pi_delta = 0

def evaluate_rewards_and_transitions(problem, mutate=False):
	# Enumerate state and action space sizes
	num_states = problem.observation_space.n
	num_actions = problem.action_space.n

	# Intiailize T and R matrices
	R = np.zeros((num_states, num_actions, num_states))
	T = np.zeros((num_states, num_actions, num_states))

	# Iterate over states, actions, and transitions
	for state in range(num_states):
		for action in range(num_actions):
			for transition in problem.env.P[state][action]:
				probability, next_state, reward, done = transition
				R[state, action, next_state] = reward
				T[state, action, next_state] = probability

			# Normalize T across state + action axes
			T[state, action, :] /= np.sum(T[state, action, :])

	# Conditionally mutate and return
	if mutate:
		problem.env.R = R
		problem.env.T = T
	return R, T

def value_iteration(problem, R=None, T=None, gamma=0.9, max_iterations=10**6, delta=10**-3):
	""" Runs Value Iteration on a gym problem """
	value_fn = np.zeros(problem.observation_space.n)
	if R is None or T is None:
		R, T = evaluate_rewards_and_transitions(problem)

	for i in range(max_iterations):
		previous_value_fn = value_fn.copy()
		Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
		value_fn = np.max(Q, axis=1)

		if np.max(np.abs(value_fn - previous_value_fn)) < delta:
			break

	# Get and return optimal policy
	policy = np.argmax(Q, axis=1)
	return policy, i + 1, value_fn

def encode_policy(policy, shape):
	""" One-hot encodes a policy """
	encoded_policy = np.zeros(shape)
	encoded_policy[np.arange(shape[0]), policy] = 1
	return encoded_policy

def policy_iteration(problem, R=None, T=None, gamma=0.9, max_iterations=10**6, delta=10**-3):
	""" Runs Policy Iteration on a gym problem """
	num_spaces = problem.observation_space.n
	num_actions = problem.action_space.n

	# Initialize with a random policy and initial value function
	policy = np.array([problem.action_space.sample() for _ in range(num_spaces)])
	value_fn = np.zeros(num_spaces)

	# Get transitions and rewards
	if R is None or T is None:
		R, T = evaluate_rewards_and_transitions(problem)

	# Iterate and improve policies
	for i in range(max_iterations):
		previous_policy = policy.copy()

		for j in range(max_iterations):
			previous_value_fn = value_fn.copy()
			Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
			value_fn = np.sum(encode_policy(policy, (num_spaces, num_actions)) * Q, 1)

			if np.max(np.abs(previous_value_fn - value_fn)) < delta:
				break

		Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
		policy = np.argmax(Q, axis=1)

		if np.array_equal(policy, previous_policy):
			break

	# Return optimal policy
	return policy, i + 1, value_fn

def run_policy(env, policy):
	obs = env.reset()
	total_reward = 0
	done = False
	while not done:
		#env.render()
		obs, reward, done , _ = env.step(int(policy[obs]))
		total_reward += reward
	return total_reward

def evaluate_policy(env, policy):
	trials = 100
	scores = [run_policy(env, policy) for _ in range(trials)]
	return np.mean(scores)

def print_policy(policy, mapping=None, shape=(0,)):
	print( np.array([mapping[action] for action in policy]).reshape(shape))

def runIterationOnGamma(environment):

	global best_vi_time, best_vi_iterations, best_vi_score, best_vi_gamma, best_vi_delta
	global best_pi_time, best_pi_iterations, best_pi_score, best_pi_gamma, best_pi_delta

	print('Running value and policy iteration on various gamma...')

	mapping = {0: "L", 1: "D", 2: "R", 3: "U"}
	shape = (4, 4)

	problem = gym.make(environment)
	print('number of states: ' + str(problem.observation_space.n))
	print('number of actions: ' + str(problem.action_space.n))

	vi_iters_array = []
	vi_score_array = []
	vi_time_array = []

	pi_iters_array = []
	pi_score_array = []
	pi_time_array = []

	discrepancy_array = []

	gamma_range = np.arange(0.05, 1 , 0.1)
	for gamma in gamma_range:

		start_time = time.time()
		best_policy_vi, iters, value_fn_vi = value_iteration(problem=problem, gamma=gamma)
		end_time = time.time()
		vi_iters_array.append(iters)
		score = evaluate_policy(problem, best_policy_vi)
		# print_policy(best_policy_vi, mapping, shape)
		# print(value_fn_vi.reshape(shape))
		# print('vi-score :' + str(score))
		vi_score_array.append(np.mean(score))
		vi_time_array.append(end_time - start_time)
		if score > best_vi_score:
			best_vi_score=score; best_vi_delta=10**-3; best_vi_gamma=gamma; best_vi_iterations=iters; best_vi_time=end_time-start_time

		start_time = time.time()
		best_policy_pi, iters, value_fn_pi = policy_iteration(problem=problem, gamma=gamma)
		end_time = time.time()
		pi_iters_array.append(iters)
		score = evaluate_policy(problem, best_policy_pi)
		# print_policy(best_policy_pi, mapping, shape)
		# print(value_fn_pi.reshape(shape))
		# print('pi-score :' + str(score))
		pi_score_array.append(np.mean(score))
		pi_time_array.append(end_time - start_time)
		if score > best_pi_score:
			best_pi_score=score; best_pi_delta=10**-3; best_pi_gamma=gamma; best_pi_iterations=iters; best_pi_time=end_time-start_time

		diff = sum([abs(x-y) for x, y in zip(best_policy_vi.flatten(), best_policy_pi.flatten())])
		discrepancy_array.append(diff)

	fig, ax = plt.subplots()
	ax.plot(gamma_range, vi_time_array, color="red", label="VI")
	ax.plot(gamma_range, pi_time_array, color="blue", label="PI")
	plt.xlabel('Gamma')
	plt.ylabel('Execution Time (s)')
	plt.title('{environment} - Value / Policy Iteration: Execution Time vs Gamma'.format(environment=environment))
	plt.legend()
	filename = '{environment}-Iteration-ExecutionVsGamma.png'.format(environment=environment)
	saveGraph(plt, filename)

	fig, ax = plt.subplots()
	ax.plot(gamma_range, vi_score_array, color="red", label="VI")
	ymax1 = max(vi_score_array);		xpos1 = vi_score_array.index(ymax1);		xmax1 = round(gamma_range[xpos1], 2)
	ax.annotate('({xmax}, {ymax})'.format(xmax=xmax1, ymax=round(ymax1,2)), xy=(xmax1, ymax1), ha='center', va='bottom', color='red')

	ax.plot(gamma_range, pi_score_array, color="blue", label="PI")
	ymax2 = max(pi_score_array);		xpos2 = pi_score_array.index(ymax2);		xmax2 = round(gamma_range[xpos2], 2)
	if xmax1 == xmax2:
		xmax2 += 10
	ax.annotate('({xmax}, {ymax})'.format(xmax=xmax2, ymax=round(ymax2,2)), xy=(xmax2, ymax2), ha='center', va='bottom', color='blue')
	plt.xlabel('Gammas')
	plt.ylabel('Average Rewards')
	plt.title('{environment} - Value / Policy Iteration: Average Rewards vs Gamma'.format(environment=environment))
	plt.legend()
	filename = '{environment}-Iteration-RewardsVsGamma.png'.format(environment=environment)
	saveGraph(plt, filename)

	fig, ax = plt.subplots()
	ax.plot(gamma_range, vi_iters_array, color="red", label="VI")
	ymax = max(vi_iters_array);		xpos = vi_iters_array.index(ymax);		xmax = round(gamma_range[xpos], 2)
	ax.annotate('({xmax}, {ymax})'.format(xmax=xmax, ymax=round(ymax,2)), xy=(xmax, ymax), ha='center', va='bottom', color='red')
	ax.plot(gamma_range, pi_iters_array, color="blue", label="PI")
	ymax = max(pi_iters_array);		xpos = pi_iters_array.index(ymax);		xmax = round(gamma_range[xpos], 2)
	ax.annotate('({xmax}, {ymax})'.format(xmax=xmax, ymax=round(ymax,2)), xy=(xmax, ymax), ha='center', va='bottom', color='blue')
	plt.xlabel('Gammas')
	plt.ylabel('Iterations to Converge')
	plt.title('{environment} - Value / Policy Iteration: Converge Iterations vs Gamma'.format(environment=environment))
	plt.legend()
	filename = '{environment}-Iteration-IterationsVsGamma.png'.format(environment=environment)
	saveGraph(plt, filename)

	fig, ax = plt.subplots()
	ax.plot(gamma_range, discrepancy_array, color="green")
	plt.xlabel('Gammas')
	plt.ylabel('Discrepancy')
	plt.title('{environment} - Value / Policy Iteration: Discrepancy vs Gamma'.format(environment=environment))
	filename = '{environment}-Iteration-DiscrepancyVsGamma.png'.format(environment=environment)
	saveGraph(plt, filename)


def runIterationOnDelta(environment):

	global best_vi_time, best_vi_iterations, best_vi_score, best_vi_gamma, best_vi_delta
	global best_pi_time, best_pi_iterations, best_pi_score, best_pi_gamma, best_pi_delta

	print('Running value and policy iteration on various delta...')

	mapping = {0: "L", 1: "D", 2: "R", 3: "U"}
	shape = (4, 4)

	problem = gym.make(environment)

	vi_iters_array = []
	vi_score_array = []
	vi_time_array = []

	pi_iters_array = []
	pi_score_array = []
	pi_time_array = []

	discrepancy_array = []

	delta_range = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
	for delta in delta_range:

		start_time = time.time()
		best_policy_vi, iters, value_fn_vi = value_iteration(problem=problem, delta=delta)
		end_time = time.time()
		vi_iters_array.append(iters)
		score = evaluate_policy(problem, best_policy_vi)
		vi_score_array.append(np.mean(score))
		vi_time_array.append(end_time - start_time)
		if score > best_vi_score:
			best_vi_score=score; best_vi_delta=delta; best_vi_gamma=0.9; best_vi_iterations=iters; best_vi_time=end_time-start_time

		start_time = time.time()
		best_policy_pi, iters, value_fn_pi = policy_iteration(problem=problem, delta=delta)
		end_time = time.time()
		pi_iters_array.append(iters)
		score = evaluate_policy(problem, best_policy_pi)
		pi_score_array.append(np.mean(score))
		pi_time_array.append(end_time - start_time)
		if score > best_pi_score:
			best_pi_score=score; best_pi_delta=delta; best_pi_gamma=0.9; best_pi_iterations=iters; best_pi_time=end_time-start_time

		diff = sum([abs(x-y) for x, y in zip(best_policy_vi.flatten(), best_policy_pi.flatten())])
		discrepancy_array.append(diff)

	fig, ax = plt.subplots()
	ax.plot(delta_range, vi_time_array, color="red", label="VI")
	ax.plot(delta_range, pi_time_array, color="blue", label="PI")
	plt.xscale('log')
	plt.xlabel('Delta')
	plt.ylabel('Execution Time (s)')
	plt.title('{environment} - Value / Policy Iteration: Execution Time vs Delta'.format(environment=environment))
	plt.legend()
	filename = '{environment}-Iteration-ExecutionVsDelta.png'.format(environment=environment)
	saveGraph(plt, filename)

	fig, ax = plt.subplots()
	ax.plot(delta_range, vi_score_array, color="red", label="VI")
	ax.plot(delta_range, pi_score_array, color="blue", label="PI")
	plt.xscale('log')
	plt.xlabel('Delta')
	plt.ylabel('Average Rewards')
	plt.title('{environment} - Value / Policy Iteration: Average Rewards vs Delta'.format(environment=environment))
	plt.legend()
	filename = '{environment}-Iteration-RewardsVsDelta.png'.format(environment=environment)
	saveGraph(plt, filename)

	fig, ax = plt.subplots()
	ax.plot(delta_range, vi_iters_array, color="red", label="VI")
	ax.plot(delta_range, pi_iters_array, color="blue", label="PI")
	plt.xscale('log')
	plt.xlabel('Delta')
	plt.ylabel('Iterations to Converge')
	plt.title('{environment} - Value / Policy Iteration: Converge Iterations vs Delta'.format(environment=environment))
	plt.legend()
	filename = '{environment}-Iteration-IterationsVsDelta.png'.format(environment=environment)
	saveGraph(plt, filename)

	fig, ax = plt.subplots()
	ax.plot(delta_range, discrepancy_array, color="green")
	plt.xscale('log')
	plt.xlabel('Delta')
	plt.ylabel('Discrepancy')
	plt.title('{environment} - Value / Policy Iteration: Discrepancy vs Delta'.format(environment=environment))
	filename = '{environment}-Iteration-DiscrepancyVsDelta.png'.format(environment=environment)
	saveGraph(plt, filename)

def decay_step_based(e):
	return 5.0 / float(e+1)

def decay_logarithmic(e, rar=0.99, radr=0.9):
	for i in range(0, e+1):
		rar *= radr
	return rar

def decay_exponential(e):
	return float(math.e**(-e*0.0005))

def chunk_list(l, n):
	for i in range(0, len(l), n):
		yield l[i:i + n]

def runQLearning(environment, alpha=0.75, discount=0.95, episodes=3000):

	env = gym.make(environment)
	print('> working with alpha: {alpha}, discount: {discount}'.format(alpha=alpha, discount=discount))

	decays = [decay_step_based, decay_logarithmic, decay_exponential]

	figure1 = plt.figure(1)
	ax1 = figure1.gca()

	figure2 = plt.figure(2)
	ax2 = figure2.gca()

	line = 0
	for decay in decays:

		print('>> working with decay: {decay}'.format(decay=decay.__name__))

		# Q and rewards
		Q = np.zeros((env.observation_space.n, env.action_space.n))
		rewards = []
		iterations = []
		aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
		colors = ['red', 'blue', 'green']

		start_time = time.time()
		# Episodes
		for episode in range(episodes):
			# Refresh state
			state = env.reset()
			done = False
			t_reward = 0
			max_steps = env._max_episode_steps

			# Run episode
			for i in range(max_steps):
				if done:
					break
				current = state
				action = np.argmax(Q[current, :] + np.random.randn(1, env.action_space.n) * decay_step_based(episode))
				state, reward, done, info = env.step(action)
				t_reward += reward
				Q[current, action] += alpha * (reward + discount * np.max(Q[state, :]) - Q[current, action])

			rewards.append(t_reward)
			iterations.append(i)

			STATS_EVERY = 50
			if not episode % STATS_EVERY:
				average_reward = sum(rewards[-STATS_EVERY:])/STATS_EVERY
				aggr_ep_rewards['ep'].append(episode)
				aggr_ep_rewards['avg'].append(average_reward)
				aggr_ep_rewards['max'].append(max(rewards[-STATS_EVERY:]))
				aggr_ep_rewards['min'].append(min(rewards[-STATS_EVERY:]))

		end_time = time.time()
		print('>> time spent: {time}'.format(time=round(end_time-start_time,4)))

		ax1.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label=decay.__name__, color=colors[line])
		size = int(episodes / 50)
		chunks = list(chunk_list(iterations, size))
		avg_iterations = [sum(chunk) / len(chunk) for chunk in chunks]
		ax2.plot(range(0, len(iterations), size), avg_iterations, label=decay.__name__, color=colors[line])
		line += 1

		figure3 = plt.figure(3)
		plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
		plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
		plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
		plt.legend()
		plt.xlabel('Episode')
		plt.ylabel('Reward')
		plt.suptitle('{environment} - Q-Learning with {decay}: Reward vs Episode'.format(environment=environment, decay=decay.__name__))
		plt.title('alpha: {alpha}, discount: {discount}'.format(alpha=alpha, discount=discount))
		filename = '{environment}-QL-RewardVsEpisode-{decay}-{alpha}-{discount}.png'.format(environment=environment, decay=decay.__name__ ,alpha=round(alpha/0.1), discount=round(discount/0.1))
		saveGraph(plt, filename)

		avg_rewards = np.mean(rewards[1000:])
		print(f'>>> Converging average reward: {avg_rewards:>4.2f}')
		print(f'>>> Converging average iteration: {np.mean(iterations[1000:]):>4.0f}')


	# Close environment
	env.close()

	# average awards
	figure1 = plt.figure(1)
	plt.legend()
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.suptitle('{environment} - Q-Learning : Average Reward vs Episode'.format(environment=environment))
	plt.title('alpha: {alpha}, discount: {discount}'.format(alpha=alpha, discount=discount))
	filename = '{environment}-QL-RewardVsEpisode-{alpha}-{discount}.png'.format(environment=environment, alpha=round(alpha/0.1), discount=round(discount/0.1))
	saveGraph(plt, filename)

	# iterations
	figure2 = plt.figure(2)
	plt.plot(range(0, len(iterations), size), avg_iterations)
	plt.legend()
	plt.xlabel('Episode')
	plt.ylabel('Number of iterations')
	plt.suptitle('{environment} - Q-Learning: Number of iterations vs Episode'.format(environment=environment))
	plt.title('alpha: {alpha}, discount: {discount}'.format(alpha=alpha, discount=discount))
	filename = '{environment}-QL-IterationsVsEpisode-{alpha}-{discount}.png'.format(environment=environment, alpha=round(alpha/0.1), discount=round(discount/0.1) )
	saveGraph(plt, filename)

def runQLearningAnalysis(environment):

	print('Running Q-Learning Analysis...')

	runQLearning(environment, alpha=0.75, discount=0.95, episodes=3000)
	runQLearning(environment, alpha=0.55, discount=0.95, episodes=3000)
	runQLearning(environment, alpha=0.75, discount=0.55, episodes=3000)

def print_iteration_stat():

	print('Best VI score: {score}'.format(score=round(best_vi_score,2)))
	print('Best VI execution time: {time}'.format(time=round(best_vi_time,4)))
	print('Best VI converging iterations: {iteration}'.format(iteration=round(best_vi_iterations,2)))
	print('Best VI gamma: {best_vi_gamma}'.format(best_vi_gamma=round(best_vi_gamma,2)))
	print('Best VI delta: {best_vi_delta}'.format(best_vi_delta=best_vi_delta))

	print('Best PI score: {score}'.format(score=round(best_pi_score,2)))
	print('Best PI execution time: {time}'.format(time=round(best_pi_time,4)))
	print('Best PI converging iterations: {iteration}'.format(iteration=round(best_pi_iterations,2)))
	print('Best PI gamma: {best_vi_gamma}'.format(best_vi_gamma=round(best_pi_gamma,2)))
	print('Best PI delta: {best_vi_delta}'.format(best_vi_delta=best_pi_delta))



def run(environment):
	print('-------{environment}-------'.format(environment=environment))
	runIterationOnGamma(environment)
	runIterationOnDelta(environment)
	print_iteration_stat()
	runQLearningAnalysis(environment)

run('FrozenLake-v0')




