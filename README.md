# LunarLander_ddpg
# Overview 
# Actor Critic Network
![alt text](https://github.com/Tejan4422/LunarLander_ddpg/blob/main/lunarlanderpost.png)
* Actor-Critic combines two main methods in reinforcement learning:  policy based algorithms and value based algorithms. The result is a powerful method that is highly suited for environmentswith continuous action spaces.  The goal of the actor-critic method is to optimize both the policyfunction and the value function via function approximation:  neural networks.  The actor networkis given a state, and it calculates a probability distribution over a set of actions.  i.e;  it outputsthe best actions that can be taken.  This output is then fed into the Critic network that evaluatesthose actions.  The result is an interplay of two networks where the actor decides the best actionsand the critic estimates how good these actions are.
* One of the efficient alternatives in this case is to use actor critic algorithm along with deepfunction approximation.  With this method it is possible to learn high dimensional and continuousaction space
* In DDPG computation can be heavy which is why we learn in mini batches rather than learningentirely on the online environment.  In order to achieve this a Replay Buffer is used.  The oldsamples from replay buffer are deleted when it reaches to the maximum memory.  So, at each timecritic  updates  by  evaluating  at  each  timestamp.   Size  of  replay  buffer  is  fixed  rather  large  thanthe one used in Deep Q network as DDPG is an off-policy algorithm.  This increased size allowsalgorithm to learn from more transitions.

## Resources used : 
* Python 3.8
* packages : Gym, openAI, numpy, pandas, matplotlib

# Rsults/Case Studies : 
* Four different sets of hyper parameters were tested and following is the performance of the model
![alt text](https://github.com/Tejan4422/LunarLander_ddpg/blob/main/ddpg_set1.png "Model Performance")
![alt text](https://github.com/Tejan4422/LunarLander_ddpg/blob/main/ddpg_set2.png "Model Performance")
![alt text](https://github.com/Tejan4422/LunarLander_ddpg/blob/main/ddpg_set3.png "Model Performance")
![alt text](https://github.com/Tejan4422/LunarLander_ddpg/blob/main/ddpg_set4.png "Model Performance")

