""""
Implementation of gradient-descent ascent methods for CMDPs (in policy space)
"""
from einops import rearrange, repeat
import numpy as np
import numpy.random as rn
from scipy.special import softmax
from numpy import linalg
import math 

from scipy.optimize import minimize

def minimize_kl_with_constraints(p0, m):
    """
    Minimize the KL divergence between two distributions p and p0,
    subject to the constraint that p(i) >= m for all i and sum(p) = 1.

    Parameters:
    p0 (numpy array): The target distribution.
    m (float): The minimum value for each p(i).

    Returns:
    numpy array: The optimized distribution p.
    """

    def kl_divergence(p, p0):
        """KL divergence between two distributions p and p0."""
        return np.sum(p * np.log(p / p0))

    def constraint_sum(p):
        """Constraint that the sum of probabilities must equal 1."""
        return np.sum(p) - 1

    def constraint_min(p, m):
        """Constraint that each probability must be greater than or equal to m."""
        return p - m

    K = len(p0)  # Number of actions

    # Initial guess (uniform distribution)
    p_init = np.full(K, 1.0 / K)

    # Constraints: sum(p) = 1 and p >= m
    constraints = [
        {'type': 'eq', 'fun': constraint_sum},  # sum(p) = 1
        {'type': 'ineq', 'fun': lambda p: constraint_min(p, m)}  # p >= m
    ]

    # Bounds to enforce p(i) >= m and p(i) <= 1
    bounds = [(m, 1) for _ in range(K)]

    # Minimize the KL divergence
    result = minimize(kl_divergence, p_init, args=(p0,), constraints=constraints, bounds=bounds, method='SLSQP')

    # Return the optimized distribution
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed")


def gradient_logpi(env, policy, s, a):
    temp = np.zeros((env.n, env.m))
    temp[s, :] = - policy[s, :]
    temp[s, a] += 1
    
    return rearrange(temp, 's a -> (s a)')

def rollout_A(env, K: int, policy: np.ndarray, T = 20, W = 10):
    """
    Generate N trajectories of length T,
    following the given policy. Return trajectories and corresponding total discounted rewards.

    :param K: Number of steps for solving function approximation
    :param policy: Numpy array pi(a|s)_(s,a), shape (n, m).
    :param T: Length of an episode.
    :param W: w project back to l_2 ball of size W
    :return: w
    """
    w_tot = np.zeros((env.n * env.m, env.k + 1))
    w_up = np.zeros((env.n * env.m, env.k + 1))
    occ = rearrange(env.policy2stateactionocc(policy), 's a -> (s a)')
    # sample trajectories
    for n in range(K):
        sample_index = np.random.choice(len(occ), p = occ)
        (s_start , a_start) = np.unravel_index(sample_index, (env.n, env.m)) 
        wstep = 2 / (n + 1)
        for i in range(env.k + 1):
            w_tot[:, i] += (n + 1) * w_up[:, i] 
            adv = 0
            q = 0
            v = 0
            s = s_start
            a = a_start
            for t in range(T):
                if i == env.k:
                    q += env.gamma**t * env.r[s, a]
                else:
                    q += env.gamma**t * env.Psi[s, a, i]   
                s = rn.choice(env.n, p = env.P[s, a, :])
                a = rn.choice(env.m, p = policy[s, :])
            s = s_start 
            a = rn.choice(env.m, p = policy[s, :])
            for t in range(T):
                if i == env.k:
                    v += env.gamma**t * env.r[s, a]
                else:
                    v += env.gamma**t * env.Psi[s, a, i]   
                s = rn.choice(env.n, p = env.P[s, a, :])
                a = rn.choice(env.m, p = policy[s, :])
            adv = (q - v)
            w_up[:, i] = w_up[:, i] - 2 * wstep * gradient_logpi(env, policy, s_start, a_start)  * ( np.inner(w_up[:, i], gradient_logpi(env, policy, s_start, a_start)) - adv)
            if np.linalg.norm(w_up[:, i], ord = 2) > W:
                w_up[:, i] = W * w_up[:, i] / np.linalg.norm(w_up[:, i], ord = 2)
                    
    return 2 * w_tot / (K * (K + 1))

def rollout(env, sample_n: int, T: int, policy: np.ndarray):
    """
    Generate N trajectories of length T,
    following the given policy. Return trajectories and corresponding total discounted rewards.

    :param sample_n: Number of trajectories.
    :param T: Length of an episode.
    :param policy: Numpy array pi(a|s)_(s,a), shape (n, m).
    :param nu0: Different initial state distribution.
    :return: costs
    """
    nu0 = env.nu0
    tot_costs = np.zeros((sample_n, env.k))
    # sample trajectories
    for i in range(env.k):
        for n in range(sample_n):
            s = rn.choice(env.n, p = nu0)
            for t in range(T):      
                # sample from policy
                a = rn.choice(env.m, p = policy[s, :])
                # sample from dynamics
                s_next = rn.choice(env.n, p = env.P[s, a, :]) 
                c = env.Psi[s, a, i]
                tot_costs[n, i] += env.gamma**t * c      
                s = s_next
    
    return np.mean(tot_costs, axis = 0)
   
def policyupdate(env, A):
    """
    PG for softmax policy parametrization.
    """
    policy = np.zeros((env.n, env.m))
    for s in range(env.n):
        policy[s, :] = softmax(A[s, :])
    return policy
    
    
def softmaxpolicy(env, theta):
    """
    PG for softmax policy parametrization.
    """
    policy = np.zeros((env.n, env.m))
    theta = np.reshape(theta, (env.n, env.m))
    for s in range(env.n):
        policy[s, :] = softmax(theta[s, :])
    return policy

def RPG_PD_exact(env, theta: np.ndarray, epsilon = 0.1, tau = 0.1, max_iters= 4000):
    reward = np.zeros(max_iters)
    cost = np.zeros((max_iters, env.k))
    it = 0
    C = (1 + 1 / ((1 - env.gamma) * 0.2) + tau * np.log(env.m))/(1 - env.gamma - tau * np.log(epsilon/env.m))
    eta = np.min([1/C, epsilon * tau])
    lamda = np.zeros(env.k)
    while it < max_iters :
        occ = env.policy2stateactionocc(policy)
        reward[it] = np.sum(occ * env.r / (1-env.gamma))
        cost[it, :] = np.einsum('jki,jk->i', env.Psi, occ) / (1-env.gamma)   
        a = eval_q_function(env, policy) - np.einsum('ijk,i -> jk', np.array(c_eval_q_function(env, policy)), lamda) - tau * np.log(policy)
        a = policy * np.exp(eta * a)
        for s in range(env.n):
            policy[s, :] = a[s, :] / np.sum(a[s, :])
            policy[s, :] = minimize_kl_with_constraints(policy[s, :], epsilon / env.m)
        lamda = np.clip(lamda - eta * (env.b - 0.1 - cost[it, :]) - tau * eta * lamda, 0, None) 
        if it % 100 == 0:
            print(it, reward[it], cost[it, :])
        if  np.any(cost[it, :] > env.b):
            print(it, cost[it, :])
            break
        it += 1  
    return reward, cost, policy

def NPG_PD_stochastic(env, theta: np.ndarray, N = 500, T = 20, K = 500, W = 10, lamda_step = 0.005, theta_step = 0.005, max_iters= 4000, logging = False, check_steps = 10):
    """
    Running NPG_PD method for max_iters rounds
    """
    reward = np.zeros(max_iters)
    cost = np.zeros((max_iters, env.k))
    it = 0
    policy_gradient = np.zeros(env.n * env.m)
    lamda = np.zeros(env.k)
    while it < max_iters :
        policy = softmaxpolicy(env, theta)
        occ = env.policy2stateactionocc(policy)
        reward[it] = np.sum(occ * env.r / (1-env.gamma))
        cost[it, :] = np.einsum('jki,jk->i', env.Psi, occ) / (1-env.gamma) 
        w = rollout_A(env, K, policy, T , W)
        a =  w[:,2] - w[:, 0] * lamda[0] - w[:, 1] * lamda[1]
        theta = theta + theta_step * a / (1 - env.gamma)
        lamda = np.clip(lamda - lamda_step * (env.b - rollout(env, N, T, policy)), 0, None)    
        it += 1

    return reward, cost, theta

def NPG_PD_exact(env, theta: np.ndarray, lamda_step = 0.005, theta_step = 0.005, max_iters= 4000, logging = False, check_steps = 10):
    """
    Running NPG_PD method for max_iters rounds
    """
    reward = np.zeros(max_iters)
    cost = np.zeros((max_iters, env.k))
    it = 0
    policy_gradient = np.zeros(env.n * env.m)
    lamda = np.zeros(env.k)
    while it < max_iters :
        policy = softmaxpolicy(env, theta)
        occ = env.policy2stateactionocc(policy)
        reward[it] = np.sum(occ * env.r / (1-env.gamma))
        cost[it, :] = np.einsum('jki,jk->i', env.Psi, occ) / (1-env.gamma)   
        a =  eval_advantage_function(env, policy) - np.einsum('ijk,i -> jk', np.array(c_eval_advantage_function(env, policy)), lamda)
        theta = theta + theta_step * rearrange(a, 's a -> (s a)') / (1 - env.gamma)
        lamda = np.clip(lamda - lamda_step * (env.b - cost[it, :]), 0, None)    
        it += 1

    return reward, cost, theta

def CRPO_exact(env, theta: np.ndarray, N: int, T: int, epsilon = 0.01, theta_step = 0.001, max_iters= 4000, logging = False, check_steps = 10):
    """
    Running CRPO method for max_iters rounds
    """
    reward = np.zeros(max_iters)
    cost = np.zeros((max_iters, env.k))
    it = 0
    policy_gradient = np.zeros(env.n * env.m)
    while it < max_iters :
        policy = softmaxpolicy(env, theta)
        occ = env.policy2stateactionocc(policy)
        reward[it] = np.sum(occ * env.r / (1-env.gamma))
        cost[it, :] = np.einsum('jki,jk->i', env.Psi, occ) / (1-env.gamma) 
        if np.all(cost[it, :] < b + epsilon ):
            a =  eval_advantage_function(env, policy)    
            theta = theta + theta_step * rearrange(a, 's a -> (s a)') / (1 - env.gamma)
        else:
            i = np.where(cost[it, :] >= b + epsilon)[0][0]
            a = c_eval_advantage_function(env, policy)[i]
            theta = theta - theta_step * rearrange(a, 's a -> (s a)') / (1 - env.gamma)
        it += 1
        
    return reward, cost, theta

def v_c_eval(env, policy: np.ndarray) -> np.ndarray:
    """
    Evaluate value function of policy by solving the linear equations.

    :policy: Policy, shape (n,m).
    :return: v: Values, shape (n,).
    """
    c_v = []
    P_policy = np.einsum('s a t, s a -> s t', env.P, policy)
    for i in range(env.k):
        c = env.Psi[:,:,i]
        c_policy = np.sum(policy * c, axis=1)
        c_v.append(np.linalg.solve(np.eye(env.n) - env.gamma * P_policy, c_policy))
        
    return c_v

def c_eval_q_function(env, policy):
    """
    Compute the advantage function
    """
    v = v_c_eval(env, policy)
    q_c_v = []
    for i in range(env.k):
        q_c_v.append(env.Psi[:,:,i] + env.gamma * np.einsum('jki,i -> jk', env.P, v[i]))
     
    return np.array(q_c_v)

def c_eval_advantage_function(env, policy):
    """
    Compute the advantage function
    """
     
    return c_eval_q_function(env, policy) - np.repeat(v_c_eval(env, policy)[:, np.newaxis], env.m, axis=1) 

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
                     
def eval_q_function(env, policy):
    """
    Compute the advantage function
    """
    v = v_eval(env, policy)
    return env.r + env.gamma * np.einsum('jki,i -> jk', env.P, v)

def eval_advantage_function(env, policy):
    """
    Compute the advantage function
    """
    v = v_eval(env, policy)
    return eval_q_function(env, policy) - np.repeat(v_eval(env, policy)[:, np.newaxis], env.m, axis=1) 
                     
def policy_gradient_softmax(env, theta):
    """
    compute the gradient for saft max parameterization
    """
    policy = softmaxpolicy(env, theta)
    a = eval_advantage_function(env, policy)
    return 1 / (1 - env.gamma) * env.policy2stateactionocc(policy) * a

