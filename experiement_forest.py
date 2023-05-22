import math
import numpy as np
import gym
import matplotlib.pyplot as plt
import time
import seaborn as sns
import random
from hiive.mdptoolbox.example import forest
import matplotlib.pyplot as plt
import random
from util import saveGraph

random.seed(100)

def draw_policy(policy, V, env, size, file_name, plt_title):
	policy_flat = np.argmax(policy, axis=1)
	V = policy_flat
	policy_grid = np.copy(policy_flat)
	sns.set()
	policy_list = np.chararray((size), unicode=True)
	policy_list[np.where(policy_grid == 0)] = 'Wait'
	policy_list[np.where(policy_grid == 1)] = 'Cut'
	a4_dims = (3, 9)
	fig, ax = plt.subplots(figsize = a4_dims)
	V = V.reshape((size,1))
	policy_list = policy_list.reshape((size,1))
	print(V.shape,policy_list.shape)
	sns.heatmap(V, annot=policy_list, fmt='', ax=ax)
	plt.title(plt_title)
	plt.tight_layout()
	saveGraph(plt, file_name)
	return True

def value_iteration(env, theta=10e-8, discount_factor=1.0):

	def one_step_lookahead(state, V):
		A = np.zeros(env.nA)
		for a in range(env.nA):
			for prob, next_state, reward, done in env.P[state][a]:
				A[a] += prob * (reward + discount_factor * V[next_state])
		return A
	V = np.zeros(env.nS)
	DELTA_ARR = []
	V_ARR = []
	V_SUM = []
	while True:
		delta = 0
		for s in range(env.nS):
			# Do a one-step lookahead to find the best action
			A = one_step_lookahead(s, V)
			best_action_value = np.max(A)
			# Calculate delta across all states seen so farg
			delta = max(delta, np.abs(best_action_value - V[s]))
			# Update the value function. Ref: Sutton book eq. 4.10.
			V[s] = best_action_value
			# Check if we can stop
		DELTA_ARR.append(delta)
		V_ARR.append(V)
		V_SUM.append(V.sum())
		if delta < theta:
			break
	# Create a deterministic policy using the optimal value function
	policy = np.zeros([env.nS, env.nA])
	for s in range(env.nS):
		# One step lookahead to find the best action for this state
		A = one_step_lookahead(s, V)
		best_action = np.argmax(A)
		# Always take the best action
		policy[s, best_action] = 1.0
	return DELTA_ARR, V_ARR,V_SUM, policy


def Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
		   max_epsilon, min_epsilon, decay_rate, verbose= True):

	rewards = []
	qtable = np.zeros((env.nS, env.nA))
	time0		 = time.time()
	clean_episode = True
	episode_length = total_episodes
	time_length	= 10e6
	for episode in range(total_episodes):
		# Reset the environment
		state = np.random.randint(env.nS, size=1)[0]

		step = 0
		done = False
		total_rewards = 0
		REWARD_ARR = []
		for step in range(max_steps):
			# 3. Choose an action a in the current world state (s)
			## First we randomize a number
			exp_exp_tradeoff = random.uniform(0, 1)
			## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
			if exp_exp_tradeoff > epsilon:
				action = np.argmax(qtable[state, :])
			# Else doing a random choice --> exploration
			else:
				action = env.random_action()
			# Take the action (a) and observe the outcome state(s') and reward (r)
			new_state, reward, done, info = env.new_state(state, action)
			if reward > 0:
				total_rewards += reward
			# Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
			# qtable[new_state,:] : all the actions we can take from new state
			qtable[state, action] = qtable[state, action] + learning_rate * (
					reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
			# Our new state is state
			state = new_state


		# Reduce epsilon (because we need less and less exploration)
		epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

		if verbose:
			if math.fmod(episode,100)==0:
				print(episode, total_rewards, epsilon, decay_rate)
		rewards.append(total_rewards)

		if np.array(rewards)[-100:].mean() > 995 and clean_episode==True:
			episode_length = episode
			time_length = time.time() - time0
			clean_episode = False
			break

	return time_length, episode_length, qtable, rewards


class my_env:

	def __init__(self, n_states, n_actions):
		self.P =  [[[] for x in range(n_actions)] for y in range(n_states)]
		self.nS = n_states
		self.nA = n_actions

	def new_state(self,state,action):
		listy = self.P[state][action]
		p = []
		for item in listy:
			p.append(item[0])
		p = np.array(p)
		#print(p,state)
		chosen_index = np.random.choice(self.nS, 1, p=p)[0]
		chosen_item = listy[chosen_index]
		return chosen_item[1],chosen_item[0], chosen_item[2],chosen_item[3]

	def random_action(self):
		action = np.random.randint(2, size=1)[0]
		return action


def my_forest(size):

	n_states  = size
	n_actions = 2
	P, R = forest(S=n_states, r1=4, r2=50, p=0.9)

	env = my_env(n_states, n_actions)
	for action in range(0, n_actions):
		for state in range(0, n_states):
			for state_slash in range(0,n_states):
				reward = R[state][action]
				env.P[state][action].append([P[action][state][state_slash], state_slash, reward, False])
	return env

def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

def runExperiementQLearningDecayRate():

	size = 100
	env = my_forest(size)
	total_episodes = 10000
	learning_rate = 0.1		 # Learning rate
	max_steps = 1000		   # Max steps per episode
	gamma = 0.9		  # Discounting rate

	epsilon = 1.0				 # Exploration rate
	max_epsilon = 1.0			 # Exploration probability at start
	min_epsilon = 10e-9		   # Minimum exploration probability
	decay_rate = 1.0			   # Exponential decay rate for exploration prob

	decay_rates = [0.01, 0.05, 0.1, 0.5, 1]
	fig, ax = plt.subplots()
	time_array = []

	for decay_rate in decay_rates:

		#testing
		decay_rate = 1

		action_size = env.nA
		state_size = env.nS
		qtable = np.zeros((state_size, action_size))

		time_start = time.time()
		time_length, episode_length, qtable, rewards = \
			Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
				max_epsilon, min_epsilon, decay_rate, verbose = False)
		time_end = time.time()
		time_array.append(time_end - time_start)

		qbest = np.empty(env.nS)
		for state in range(env.nS):
			qbest[state] = np.argmax(qtable[state,:])

		rewards_moving = moving_average(rewards, 10)
		X1 = np.arange(1,len(rewards_moving)+1, 1)

		ax.plot(X1, rewards_moving, label=decay_rate, linestyle='-')

	plt.title('Forest - QLearning on various decay rate - Mean Reward vs Number of Episodes')
	plt.ylabel('Mean Reward (over 10 episodes)')
	plt.xlabel('Number of episodes')
	plt.legend()
	saveGraph(plt, 'forest_Qlearning_DecayRate.png')

	plt.title('Forest - QLearning - Time spent at various decay rate ')
	new_labels = [str(x) for x in decay_rates]
	plt.bar(new_labels, time_array)
	plt.ylabel('Time (s)')
	plt.xlabel('Decay Rate')
	saveGraph(plt, 'forest_Qlearning_DecayRate_Time.png')


def runExperiementQLearningEpsilon():

	size = 100
	env = my_forest(size)
	total_episodes = 10000
	learning_rate = 0.1		 # Learning rate
	max_steps = 1000		   # Max steps per episode
	gamma = 0.9		  # Discounting rate

	epsilon = 1.0				 # Exploration rate
	max_epsilon = 1.0			 # Exploration probability at start
	min_epsilon = 10e-9		   # Minimum exploration probability
	decay_rate = 1.0			   # Exponential decay rate for exploration prob

	epsilons = [0.05, 0.2, 0.5, 0.8, 1]
	fig, ax = plt.subplots()
	time_array = []

	for epsilon in epsilons:

		action_size = env.nA
		state_size = env.nS
		qtable = np.zeros((state_size, action_size))

		time_start = time.time()
		time_length, episode_length, qtable, rewards = \
			Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
				epsilon, min_epsilon, decay_rate, verbose = False)
		time_end = time.time()
		time_array.append(time_end - time_start)

		qbest = np.empty(env.nS)
		for state in range(env.nS):
			qbest[state] = np.argmax(qtable[state,:])

		rewards_moving = moving_average(rewards, 10)
		X1 = np.arange(1,len(rewards_moving)+1, 1)

		ax.plot(X1, rewards_moving, label=epsilon, linestyle='-')

	plt.title('Forest - QLearning on various epsilons - Mean Reward vs Number of Episodes')
	plt.ylabel('Mean Reward (over 10 episodes)')
	plt.xlabel(' Number of episodes')
	plt.legend()
	saveGraph(plt, 'forest_Qlearning_Epsilon.png')

	plt.title('Forest - QLearning - Time spent at various epsilon ')
	new_labels = [str(x) for x in epsilons]
	plt.bar(new_labels, time_array)
	plt.ylabel('Time (s)')
	plt.xlabel('Epsions')
	saveGraph(plt, 'forest_Qlearning_Epsilon_Time.png')

def runExperiementQLearningMinEpsilon():

	size = 100
	env = my_forest(size)
	total_episodes = 10000
	learning_rate = 0.1		 # Learning rate
	max_steps = 1000		   # Max steps per episode
	gamma = 0.9		  # Discounting rate

	epsilon = 1.0				 # Exploration rate
	max_epsilon = 1.0			 # Exploration probability at start
	min_epsilon = 10e-9		   # Minimum exploration probability
	decay_rate = 1.0			   # Exponential decay rate for exploration prob

	min_epsilons = [10e-1, 10e-2, 10e-5, 10e-7, 10e-9]
	fig, ax = plt.subplots()
	time_array = []

	for min_epsilon in min_epsilons:

		action_size = env.nA
		state_size = env.nS
		qtable = np.zeros((state_size, action_size))

		time_start = time.time()
		time_length, episode_length, qtable, rewards = \
			Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
				max_epsilon, min_epsilon, decay_rate, verbose = False)
		time_end = time.time()
		time_array.append(time_end - time_start)

		qbest = np.empty(env.nS)
		for state in range(env.nS):
			qbest[state] = np.argmax(qtable[state,:])

		rewards_moving = moving_average(rewards, 10)
		X1 = np.arange(1,len(rewards_moving)+1, 1)

		ax.plot(X1, rewards_moving, label=min_epsilon, linestyle='-')

	plt.title('Forest - QLearning on various min epsilons - Mean Reward vs Number of Episodes')
	plt.ylabel('Mean Reward (over 10 episodes)')
	plt.xlabel('Number of episodes')
	plt.legend()
	saveGraph(plt, 'forest_Qlearning_EpsilonMin.png')

	plt.title('Forest - QLearning - Time spent at various min epsilon ')
	new_labels = [str(x) for x in min_epsilons]
	plt.bar(new_labels, time_array)
	plt.ylabel('Time (s)')
	plt.xlabel('Min Epsions')
	saveGraph(plt, 'forest_Qlearning_EpsilonMin_Time.png')


def runExperiementQLearning():
	runExperiementQLearningDecayRate()
	runExperiementQLearningEpsilon()
	runExperiementQLearningMinEpsilon()

def runExperiementValueIteration():

	N_ITERS  = []
	SIZE = np.arange(2,15,1)
	TIME_ARR = []
	SUM_V_ARR = []
	for size in SIZE:

		env = my_forest(size)
		print("nStates  ", env.nS)
		time0 = time.time()
		DELTA_ARR, V_ARR, V_SUM, policy = value_iteration(env, 10e-10, 0.99)
		time1 = time.time()
		N_ITERS.append(len(V_ARR))
		TIME_ARR.append(time1 - time0)
		SUM_V_ARR.append(V_SUM[-1])

	fig, ax = plt.subplots()
	ax.plot(SIZE, N_ITERS , color='red', label="Number of iterations", linewidth=2.0, linestyle='-')
	ax.plot(SIZE, N_ITERS , color='red',  marker='o', markersize = 4)
	ax.legend(loc='best', frameon=True)
	plt.title('Forest - Value Iteration - Number of iterations to converge')
	plt.ylabel('Number of Iterations')
	plt.xlabel('Problem Size')
	plt.tight_layout()
	saveGraph(plt, 'forest_vi_iters.png')

	fig, ax = plt.subplots()
	ax.plot(SIZE, TIME_ARR , color='red', label="Time", linewidth=2.0, linestyle='-')
	ax.plot(SIZE, TIME_ARR , color='red',  marker='o', markersize = 4)
	ax.legend(loc='best', frameon=True)
	plt.title('Forest -  Value Iteration - Time VS Size')
	plt.ylabel('Time')
	plt.xlabel('Problem Size')
	plt.tight_layout()
	saveGraph(plt, 'forest_vi_time.png')

	fig, ax = plt.subplots()
	ax.plot(SIZE, SUM_V_ARR , color='red', linewidth=2.0, linestyle='-')
	ax.plot(SIZE, SUM_V_ARR , color='red',  marker='o', markersize = 4)
	ax.legend(loc='best', frameon=True)
	plt.title('Forest -  Value Iteration - Final V values')
	plt.ylabel('Final V Values')
	plt.xlabel('Problem Size')
	plt.tight_layout()
	saveGraph(plt, 'forest_vi_v.png')

	size = 100
	env = my_forest(size)

	DELTA_ARR99, V_ARR99, V_SUM99, policy99 = value_iteration(env,  10e-15,  0.99)
	DELTA_ARR9, V_ARR9, V_SUM9, policy9 = value_iteration(env,  10e-15,  0.9)
	DELTA_ARR7, V_ARR7, V_SUM7, policy7 = value_iteration(env,  10e-15,  0.7)

	X99 = np.arange(1,len(V_SUM99)+1,1)
	X9 = np.arange(1,len(V_SUM9)+1,1)
	X7 = np.arange(1,len(V_SUM7)+1,1)
	fig, ax = plt.subplots()
	ax.plot(X99, V_SUM99 , color='steelblue', label="gamma = 0.99", linewidth=2.0, linestyle='-')
	ax.plot(X9, V_SUM9 , color='red', label=" gamma = 0.9 ", linewidth=2.0, linestyle='-')
	ax.plot(X7, V_SUM7 , color='purple', label=" gamma = 0.7", linewidth=2.0, linestyle='-')
	ax.legend(loc='best', frameon=True)
	plt.grid(False, linestyle='--')
	plt.title('Forest - Value Iteration - Sum of V vs Iterations number, size = 100')
	plt.ylabel('Sum of V Values')
	plt.xlabel('Iterations')
	plt.xlim((0,500))
	plt.tight_layout()
	saveGraph(plt, 'forest_vi_plot.png')
	draw_policy(policy99, V_ARR99[len(V_ARR99)-1], env, size, "forest_value_iteration.png", "Policy visualization \n(value iteration), size = 10")

def policy_eval(policy, env, v_prev,  discount_factor=1.0, theta=0.0001 ):

	"""
	Evaluate a policy given an environment and a full description of the environment's dynamics.
	Args:
		policy: [S, A] shaped matrix representing the policy.
		env: OpenAI env. env.P represents the transition probabilities of the environment.
			env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
			env.nS is a number of states in the environment.
			env.nA is a number of actions in the environment.
		theta: We stop evaluation once our value function change is less than theta for all states.
		discount_factor: Gamma discount factor.
	Returns:
		Vector of length env.nS representing the value function.
	"""
	# Start with a random (all 0) value function

	V = v_prev
	num_iters = 0
	while True:
		num_iters = num_iters + 1
		delta = 0
		# For each state, perform a "full backup"
		for s in range(env.nS):
			v = 0
			# Look at the possible next actions
			for a, action_prob in enumerate(policy[s]):
				# For each action, look at the possible next states...
				for prob, next_state, reward, done in env.P[s][a]:
					# Calculate the expected value
					v += action_prob * prob * (reward + discount_factor * V[next_state])
			# How much our value function changed (across any states)
			delta = max(delta, np.abs(v - V[s]))
			V[s] = v
		# Stop evaluating once our value function change is below a threshold
		if delta < theta:
			break
	return np.array(V)

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
	"""
	Policy Improvement Algorithm. Iteratively evaluates and improves a policy
	until an optimal policy is found.
	Args:
		env: The OpenAI environment.
		policy_eval_fn: Policy Evaluation function that takes 3 arguments:
			policy, env, discount_factor.
		discount_factor: gamma discount factor.
	Returns:
		A tuple (policy, V).
		policy is the optimal policy, a matrix of shape [S, A] where each state s
		contains a valid probability distribution over actions.
		V is the value function for the optimal policy.
	"""

	def one_step_lookahead(state, V):
		"""
		Helper function to calculate the value for all action in a given state.
		Args:
			state: The state to consider (int)
			V: The value to use as an estimator, Vector of length env.nS
		Returns:
			A vector of length env.nA containing the expected value of each action.
		"""
		A = np.zeros(env.nA)
		for a in range(env.nA):
			for prob, next_state, reward, done in env.P[state][a]:
				A[a] += prob * (reward + discount_factor * V[next_state])
		return A

	# Start with a random policy
	policy = np.ones([env.nS, env.nA]) / env.nA
	V_ARR = []
	V_SUM = []
	V = np.zeros(env.nS)

	while True:

		V = policy_eval_fn(policy, env, V, discount_factor = discount_factor)

		# Will be set to false if we make any changes to the policy
		policy_stable = True

		# For each state...
		for s in range(env.nS):
			# The best action we would take under the current policy
			chosen_a = np.argmax(policy[s])
			# Find the best action by one-step lookahead
			# Ties are resolved arbitarily
			action_values = one_step_lookahead(s, V)
			best_a = np.argmax(action_values)
			# Greedily update the policy
			if chosen_a != best_a:
				policy_stable = False
			policy[s] = np.eye(env.nA)[best_a]

		# If the policy is stable we've found an optimal policy. Return it
		V_ARR.append(V)
		V_SUM.append(V.sum())
		if policy_stable:
			return policy, V_ARR, V_SUM

def runExperiementPolicyIteration():

	N_ITERS  = []
	SIZE	 = np.arange(2,15,1)
	TIME_ARR = []
	SUM_V_ARR = []
	for size in SIZE:
		env = my_forest(size)
		time0 = time.time()
		policy, V_ARR, V_SUM = policy_improvement(env, discount_factor= 0.99)
		time1 = time.time()
		N_ITERS.append(len(V_ARR))
		TIME_ARR.append(time1 - time0)
		SUM_V_ARR.append(V_SUM[-1])

	fig, ax = plt.subplots()
	ax.plot(SIZE, N_ITERS , color='red', label="Number of iterations", linewidth=2.0, linestyle='-')
	ax.plot(SIZE, N_ITERS , color='red',  marker='o', markersize = 2)

	ax.legend(loc='best', frameon=True)
	plt.title('Forest - Policy Iteration - Number of iterations to converge')
	plt.ylabel('Number of iterations')
	plt.xlabel('Problem Size')
	plt.tight_layout()
	saveGraph(plt, 'forest_pi_iters.png')

	fig, ax = plt.subplots()
	ax.plot(SIZE, TIME_ARR , color='red', label="Time", linewidth=2.0, linestyle='-')
	ax.plot(SIZE, TIME_ARR , color='red',  marker='o', markersize = 4)
	ax.legend(loc='best', frameon=True)
	plt.title('Forest -  Policy Iteration - Time VS Size')
	plt.ylabel('Time')
	plt.xlabel('Problem Size')
	plt.tight_layout()
	saveGraph(plt, 'forest_pi_time.png')

	fig, ax = plt.subplots()
	ax.plot(SIZE, SUM_V_ARR , color='red', linewidth=2.0, linestyle='-')
	ax.plot(SIZE, SUM_V_ARR , color='red',  marker='o', markersize = 4)
	ax.legend(loc='best', frameon=True)
	plt.title('Forest -  Policy Iteration - Final V values')
	plt.ylabel('Final V Values')
	plt.xlabel('Problem Size')
	plt.tight_layout()
	saveGraph(plt, 'forest_pi_v.png')


	size = 100
	env = my_forest(size)

	policy99,V_ARR99, V_SUM99 = policy_improvement(env,  discount_factor = 0.99)
	policy9,V_ARR9, V_SUM9 = policy_improvement(env,	discount_factor = 0.9)
	policy7, V_ARR7, V_SUM7 = policy_improvement(env,  discount_factor =0.7)
	X99 = np.arange(1,len(V_SUM99)+1,1)
	X9 = np.arange(1,len(V_SUM9)+1,1)
	X7 = np.arange(1,len(V_SUM7)+1,1)
	fig, ax = plt.subplots()
	ax.plot(X99, V_SUM99 , color='steelblue', label="gamma = 0.99", linewidth=2.0, linestyle='-')
	ax.plot(X9, V_SUM9 , color='red', label="gamma = 0.9 ", linewidth=2.0, linestyle='-')
	ax.plot(X7, V_SUM7 , color='purple', label="gamma = 0.7", linewidth=2.0, linestyle='-')
	ax.legend(loc='best', frameon=True)
	plt.grid(False, linestyle='--')
	plt.title('Forest - Policy Iteration - Sum of V vs Iterations number, size = 100')
	plt.ylabel('Sum of V values')
	plt.xlabel('Iterations')
	plt.tight_layout()
	saveGraph(plt, 'forest_pi_plot.png')

	draw_policy(policy99, V_ARR99[len(V_ARR99)-1], env, size, "forest_policy_iteration.png", "Policy visualization \n(policy iteration), size = 10")

runExperiementValueIteration()
runExperiementPolicyIteration()
runExperiementQLearning()
