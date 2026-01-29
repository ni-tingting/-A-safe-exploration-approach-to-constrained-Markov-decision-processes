""""
Implementation of gradient-descent ascent methods for CMDPs (in policy space)
"""
from einops import rearrange, repeat
import numpy as np
import numpy.random as rn
from scipy.special import softmax
from numpy import linalg
import numpy.random as rn
from scipy.special import softmax
from numpy import linalg
import math 


def secondgradient_logpi(env, policy, s):
    mat = np.zeros((env.m, env.n, env.m, env.n))
    for a in range(env.m):         
        temp = np.zeros((env.m))
        temp[:] = - policy[s, a] * policy[s, :]
        temp[a] += policy[s, a]
        mat[a, s, :, s]  = temp
    
    return rearrange(mat, 'a b c d -> (a b) (c d)')
    
def gradient_logpi(env, policy, s, a):
    temp = np.zeros((env.n, env.m))
    temp[s, :] = - policy[s, :]
    temp[s, a] += 1
    
    return rearrange(temp, 's a -> (s a)')

def rollout(env, sample_n: int, T: int, policy: np.ndarray):
    """
    Generate N trajectories of length T,
    following the given policy. Return trajectories and corresponding total discounted rewards.

    :param sample_n: Number of trajectories.
    :param T: Length of an episode.
    :param policy: Numpy array pi(a|s)_(s,a), shape (n, m).
    :param nu0: Different initial state distribution.
    :return: Trajectories, tot_rewards.
    """
    nu0 = env.nu0
    rewards_gradient = np.zeros((sample_n, env.n * env.m))
    M = np.zeros((sample_n, env.n * env.m, env.n * env.m))
    # sample trajectories
    for n in range(sample_n):
        logpi = np.zeros((env.n * env.m))
        logpi2 = np.zeros((env.n * env.m, env.n * env.m))
        s = rn.choice(env.n, p = nu0)
        for t in range(T):      
            # sample from policy
            a = rn.choice(env.m, p = policy[s, :])
            # sample from dynamics
            s_next = rn.choice(env.n, p = env.P[s, a, :]) 
            r = env.r[s, a]
            logpi += gradient_logpi(env, policy, s, a)
            logpi2 += secondgradient_logpi(env, policy, s)
            rewards_gradient[n, :] += env.gamma**t * r * logpi 
            temp = logpi2 + np.matmul(logpi.reshape((env.n * env.m,1)),logpi.reshape((1,env.n * env.m)))
            M[n, :, :] += env.gamma**t * r * temp

            s = s_next
            
    return  np.mean(rewards_gradient, axis = 0), linalg.norm(np.mean(M, axis = 0), axis = (0,1))
   

def softmaxpolicy(env, theta):
    """
    PG for softmax policy parametrization.
    """
    policy = np.zeros((env.n, env.m))
    theta = np.reshape(theta, (env.n, env.m))
    for s in range(env.n):
        policy[s, :] = softmax(theta[s, :])
    return policy

def LB_PG(env, theta: np.ndarray, T: int, thr: float, sample_n: int, stepsize = 5, max_iters=1e3, tol=1e-3, logging = False, check_steps = 10):
    """
    Running log barrier method for max_iters rounds
    """
    max_iters = int(max_iters)
    reward = np.zeros(max_iters)
    it = 0
    gradient = 0
    while it < max_iters :
        policy = softmaxpolicy(env, theta)
        occ = env.policy2stateactionocc(policy)
        reward[it] = np.sum(occ * env.r / (1-env.gamma))
        if it % check_steps == 0:
             if logging:     
                print(it, reward[it], linalg.norm(gradient))
                
        gradient, M = rollout(env, sample_n, T, policy)
        gradient = policy_gradient_softmax(env, theta)
                
        theta = theta + gradient / M       
        
        it += 1

    return reward, theta


def v_eval(env, policy: np.ndarray) -> np.ndarray:
    """
    Evaluate value function of policy by solving the linear equations.

    :policy: Policy, shape (n,m).
    :return: v: Values, shape (n,).
    """
    r = env.r 
    P_policy = np.einsum('s a t, s a -> s t', env.P, policy)
    r_policy = np.sum(policy * r, axis=1)
    return np.linalg.solve(np.eye(env.n) - env.gamma * P_policy, r_policy)

def eval_advantage_function(env, policy):
    """
    Compute the advantage function
    """
    v = v_eval(env, policy)
    return env.r + env.gamma * np.einsum('jki,i -> jk', env.P, v) - repeat(v, 's -> s a', a = env.m)

def policy_gradient_softmax(env, theta):
    """
    compute the gradient for saft max parameterization
    """
    policy = softmaxpolicy(env, theta)
    a = eval_advantage_function(env, policy)
    return 1 / (1 - env.gamma) * env.policy2stateactionocc(policy) * a

