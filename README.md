# <a name="_d9n8kug9zfwy"></a>Markov Decision Processes 
## <a name="_ck498m7vlico"></a>Problem 1
I have chosen a popular problem Frozen Lake (FrozenLake-v0) from OpenAI, where agents are rewarded by the environment when they accomplish a predefined goal. It has a simple structure and has a small number of states 16, that would help me understand how Markov Decision Processes (MDP) and reinforcement learning algorithms work by going step-by-step.
### <a name="_hepr1q7gula"></a>MDP
In the experiment I will be comparing the difference between value iteration (VI) and policy iteration (PI). In the first set of experiments I want to examine how the gamma (discounting factor) would affect the result, in terms of execution time, iterations to converge, and average award over 100 trials. 

From figure 1.1, PI is spending more execution time, which is reasonable as it is repeating the value iteration process but with different policy until the new policy does not further improve. The significant increase in execution time on both VI and PI shows that with a higher factor value of the future, it is harder to converge and conclude the best policy. 

Figure 1.2 shows the VI has a hard time to converge when gamma is high at 0.95, it requires 44 iterations so the value matrix is no longer changing to conclude a policy. On the other side PI requires a lower number of iteration, yet each iteration is more time consuming by working on the value function using Bellman operator recursively, until policy has no improvement. Supposedly the inner recursive value evaluation has a high number of iterations, which is not represented by the chart.


|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.001.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.002.png)|
| :- | :- |
|Figure 1.1|Figure 1.2|

Figure 1.3 shows VI is able to achieve an average award of 0.7 when gamma is above 0.5, implying a longer term consideration is needed to achieve a better award, or it may end up falling into the hole. PI on the other hand can achieve a generally high average award even at low gamma, showing that the initial random policy can work well in this simple problem. Note the best result with average 0.81 is found at gamma 0.25 on PI, that means randomness is crucial in identifying the best result. 

Figure 1.4 shows the result consistent with figure 1.3, with a large difference in the best policy when gamma is low, and it is diminishing when gamma is increasing, showing both achieve the same resulting policy.


|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.003.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.004.png)|
| :- | :- |
|Figure 1.3|Figure 1.4|

The next experiment is to examine the effect of delta (the threshold of a value function change, below which implies no change). As expected a higher threshold would result in lower execution time in figure 2.1. In general it should require a few number of iterations to converge, note in figure 2.2 the number of iterations required has not changed much across different delta for PI, as the value update in each policy is able to achieve a satisfactory result regardless of delta, though a longer time is needed.


|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.005.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.006.png)|
| :- | :- |
|Figure 2.1|Figure 2.2|

Figure 2.3 shows both VI and PI are able to achieve similar performance with close average award when delta is below 0.01. However when the delta is set to 0.1, VI shows a drop with zero average reward resulted. I think the threshold is too high (which is relative to the reward score) for VI (as a result of convergence then no more improvement) to conclude a policy, where the result from PI is from initial randomness and iterating improvement that can give a reasonably good performance. Figure 2.4 shows with a proper delta both VI and PI can generate the same output, but when delta is incorrectly used, PI could be a better choice.


|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.007.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.008.png)|
| :- | :- |
|Figure 2.3|Figure 2.4|

Table 1 summarizes and compares the best overall result. With some randomness VI and PI can have similar scores, where generally PI can do slightly better. PI takes few iterations but consumes more time. The best result can happen at different gamma and delta settings however.


||**VI**|**PI**|
| :- | :- | :- |
|**Score** |0\.76|0\.77|
|**Execution Time**|0\.004|0\.018|
|**Converging Iteration Number**|27|4|
|**Gamma** |0\.9|0\.9|
|**Delta**|0\.001|0\.0001|

### <a name="_6a5kg55cvpfn"></a>Reinforcement Learning 
In the following experiments Q-Learning will be used. Figure 3.1, 3.2, and 3.3 below show how different settings of learning rate (alpha) and discount factor can affect the result under  step-based decay. Also shown are the maximum, minimum, and average awards across different episodes. Apparently a low discount factor may likely result in a bad ending (with lower caring on the future value) and thus worse average performance. Moreover, a higher alpha value seems to help faster convergence in general, it is also able to achieve a better reward on average across the episodes.


|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.009.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.010.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.011.png)|
| :- | :- | :- |
|Figure 3.1|Figure 3.2|Figure 3.3|

Below the graphs further illustrate how different alpha and discount would behave under different decay strategies - 1) the step-based decay (with a decaying weight of 5/(1+episode) of random action from action space), 2) logarithmic (with a decaying weight of **0.9\*0.99^episode**), and 3) exponential (**episode^(-episode\*0.0005)**).

It also shows a low discount likely results in premature convergence and a bad performance. Nevertheless the performance is varying between the decay strategy and alpha. Particularly in this experiment, the step-based decay and 0.75 alpha in figure 3.4, and logarithmic and 0.55 alpha in figure 3.6 show outstanding results. While all of them are doing initial higher exploration followed by lower exploitation, a hyper parameter tuning is required to identify the best strategy.


|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.012.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.013.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.014.png)|
| :- | :- | :- |
|Figure 3.4|Figure 3.5|Figure 3.6|


|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.015.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.016.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.017.png)|
| :- | :- | :- |
|Figure 3.7|Figure 3.8|Figure 3.9|

Table 2 below lists some of the parameters giving the best performance of Q-Learning. Since episodes are used to learn over the previous episode until 3000 times is done, I can only tell from the graph if it looks like converging by flatting average value. Thus the average score and iterations required are averaged from 1000th episodes onward, and the time spent on converging is roughly calculated by (total time/3000\*500). This may not be 100% accurate as the initial 500 episodes has fewer number of iterations, but should be able to give us a rough idea how much time is required.


|**setting**|**Decay rate**|**Average Reward**|**Average iteration**|**Time spent to converge**|
| :- | :- | :- | :- | :- |
|(alpha=0.75, discount=0.95)|step-based|0\.73|42|2\.0885|
|(alpha=0.55, discount=0.95)|logarithmic|0\.72|41|2\.0806|
|(alpha=0.55, discount=0.95)|exponential|0\.70|42|2\.0252|

Compared with the previous section, Q-Learning requires a higher number of iterations and much longer time but the average reward score is lower. This is sensible as Q-Learning is model-free, has no prior knowledge on the existing state and reward. With purely random action and probability, the reinforcement algorithm is able to achieve an acceptable result.
## <a name="_k5v5e6vii0km"></a>Problem 2
Forest Management has been chosen as the second problem for the following experiments. The primary reason is, it could have a large number of states compared with the first problem; but on the other side it has basically 2 simple actions, wait and cut, for easier understanding. Also the reward value can be customized when the forest is in its oldest state and different action is performed. In the following experiments, different parameters of decay rates, epsilons, and minimum epsilons are used to demonstrate the result, and comparison will be made against the first problem which has much smaller size in terms of number of states.
### <a name="_37sfns9squ8c"></a>Value Iteration
Below figure 4.1 and 4.2 shows that a smaller number of states would need more iterations to converge, because at a higher number of problem size it is failing to find a better final value and take few iterations to obtain a converged point, which is likely a local maxima. 


|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.018.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.019.png)|
| - | - |
|Figure 4.1|Figure 4.2|

Figure 4.3 shows the higher the number of problem size is, the longer time it will take to converge in proportional to the problem space. Figure 4.4 verifies that a higher number of gamma (discount rate) would possibly find the optimal result value, which is consistent with the first experiment.


|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.020.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.021.png)|
| - | - |
|Figure 4.3|Figure 4.4|
### <a name="_x5ra27vzrvfz"></a>Policy Iteration
The experiment of policy iteration contrasts the value iteration that it requires a constant number of iterations to converge, as in figure 5.1. It is consistent with problem 1 that the initial random policy is improving through its internal value iteration, and the latter policy deduced from previous iteration is found to be identical to the previous one after improvement, is said to be converged. It is also consistent to problem 1 that the value iteration and policy iteration converge to the same policy.


|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.022.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.023.png)|
| - | - |
|Figure 5.1|Figure 5.2|

Figure 5.3 shows that the time spent on policy iteration could be less than value iteration, different from problem 1. This finding could be related to the problem nature and different setup of the experiments. Figure 5.4 shows the policy iteration has a different rate of v-value improvement from value iteration, with possible ending result though.

|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.024.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.025.png)|
| - | - |
|Figure 5.3|Figure 5.4|
### <a name="_jr2w8sgeugkc"></a>Reinforcement Learning
Similar to problem 1, Q-Learning is used to analyze the performance of reinforcement learning on the forest management problem with size = 100. Different exploration and exploitation strategies (by adjusting parameters) will be shown in the following experiments. The first experiment is to examine different decay rates. It is understandable a lower decay rate would result in a longer converging time, as shown in figure 6.1. As a lower decay rate means the epsilon will decrease exponentially slower. In particular experiment, it seems a high decay rate = 1 should be used for fast converging time and high mean reward.


|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.026.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.027.png)|
| - | - |
|Figure 6.1|Figure 6.2|

Next experiment is to examine the effect of the initial epsilon. A high epsilon means a random action would be taken so it is usually preferred initially for more exploration. Figure 6.2 shows that different settings of epsilons only shows a difference at the initial mean reward, and can later be converged to the same mean reward after some episodes.


|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.028.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.029.png)|
| - | - |
|Figure 6.3|Figure 6.4|

Below experiment shows the behavior at different minimum epsilon settings. When the minimum epsilon is high the Q-Learning is basically doing high if not pure exploration so it takes a long time to converge, and the mean reward is quite low. Only if the minimum epsilon is set to below 0.0001 it could result in optimal performance with sufficient exploitation. This experiment gives a different view of examining the behavior of epsilon setting from problem 1.


|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.030.png)|![](graph/cb777e5e-1f41-41f5-b5ec-c05ec9dd2847.031.png)|
| - | - |
|Figure 6.5|Figure 6.6|

From the experiment, it is found that Q-Learning takes a much longer time to converge and obtain the optimal result, compared with value iteration or policy iteration. With a proper setting its result is comparable, both are able to achieve a mean reward close to 1000. The finding is consistent with problem 1. 

Besides, it seems that the problem with a higher number of states requires fewer episodes to converge, compared with problem 1. The process takes long though, consumed by the calculation proportional to the number of states.

## <a name="_x932szno6yil"></a>Running under Anaconda
1. Run the following file to generate plots and analysis for the first problem, Frozen Lake
`python ./experiement_frozenlake.py`
1. Run the following file to generate plots and analysis for the first problem, Forest Management
`python ./experiement_forest.py`
