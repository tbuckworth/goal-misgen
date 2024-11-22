# %%
import numpy as np
import torch
import einops

from helper_local import norm_funcs, dist_funcs
import matplotlib.pyplot as plt

# %%
n_states = 10
n_actions = 2
deterministic = True
sparse_reward = True
ascender = True
ascender_long = True

# %%
if deterministic:
    T = torch.zeros(n_states, n_actions, n_states)
    next_states = torch.randint(0, n_states, (n_states, n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            k = next_states[i, j]
            T[i, j, k] = 1
else:
    T = torch.randn(n_states, n_actions, n_states)
    T = T.softmax(dim=2)
    # T = T / T.sum(dim=2, keepdim=True)

assert T.sum(dim=2).allclose(torch.ones(n_states, n_actions))

true_R = torch.randn(n_states)
if sparse_reward:
    true_R = torch.zeros_like(true_R)
    true_R[np.random.randint(n_states)] = 1.

# true_R = true_R - true_R.min()

# true_h = torch.randn(n_states)
gamma = 0.9
# %%
if ascender:
    n_states = 6
    n_actions = 2
    T = torch.zeros(n_states, n_actions, n_states)
    for i in range(n_states - 2):
        T[i, 1, i + 1] = 1

    for i in range(1, n_states - 2):
        T[i, 0, i - 1] = 1

    T[4, :, 5] = 1
    T[5, :, 5] = 1
    T[0, 0, 0] = 1

assert (T.sum(-1) == 1).all(), T

true_R = torch.zeros(n_states)
true_R[4] = 10

gamma = 0.99

# %%
if ascender_long:
    n_states = 21
    n_actions = 2
    T = torch.zeros(n_states, n_actions, n_states)
    for i in range(n_states - 2):
        T[i+1, 1, i+2] = 1
        T[i+1, 0, i] = 1

true_R = torch.zeros(n_states)
true_R[-1] = 10
true_R[0] = -10

gamma = 0.99

# %%

def q_value_iteration(T, R, gamma, n_iterations=1000):
    Q = torch.zeros(n_states, n_actions)
    V = torch.zeros(n_states)

    for i in range(n_iterations):
        old_Q = Q
        # old_V = V
        V = Q.max(dim=1).values
        # Q = einops.repeat(R, 'states -> states actions', actions=n_actions)
        Q = einops.einsum(T, gamma * V, 'states actions next_states, next_states -> states actions') + R.unsqueeze(1)

        if (Q - old_Q).abs().max() < 1e-5:
            print(f'Q-value iteration converged in {i} iterations')
            break
    pi = torch.nn.functional.one_hot(Q.argmax(dim=1)).float()
    return Q, V, pi


Q, V, pi = q_value_iteration(T, true_R, gamma)


# %%
def soft_q_value_iteration(T, R, gamma, n_iterations=1000):
    Q = torch.zeros(n_states, n_actions)
    V = torch.zeros(n_states)
    # normal method

    for i in range(n_iterations):
        old_Q = Q
        V = Q.logsumexp(dim=-1)
        Q = einops.einsum(T, gamma * V, 'states actions next_states, next_states -> states actions') + R.unsqueeze(1)

        if (Q - old_Q).abs().max() < 1e-5:
            print(f'soft value iteration converged in {i} iterations')
            pi = Q.softmax(dim=1)
            return Q, V, pi
    print('soft value iteration did not converge after', n_iterations, 'iterations')


soft_Q, soft_V, soft_pi = soft_q_value_iteration(T, true_R, gamma, 10000)

# %%
state = T.argwhere().T[0]
next_state = T.argwhere().T[2]
R = true_R[next_state]
logp = soft_pi.log()[1:-1].reshape(-1)

# USE EITHER V or soft_V here:
adjustment = gamma * soft_V[next_state] - soft_V[state]
#This is because T.argwhere() has nothing for terminal states and then next_state doesn't exist
# full_adjustment = torch.concat((soft_V[:1],soft_V[:1], adjustment, soft_V[-1:]))
clp = logp + adjustment
cr = R + adjustment

nclp = norm_funcs["l2_norm"](clp)
ncr = norm_funcs["l2_norm"](cr)

print(f'Distance: {dist_funcs["l2_dist"](nclp, ncr).item():.4f}')

plt.scatter(logp, R)
plt.show()
plt.scatter(clp,cr)
plt.show()
plt.scatter(nclp,ncr)
plt.show()


# %%
def soft_q_value_iteration_with_pi(T, R, gamma, n_iterations=1000):
    Q = torch.zeros(n_states, n_actions)
    V = torch.zeros(n_states)
    # normal method

    for i in range(n_iterations):
        old_Q = Q
        V = Q.logsumexp(dim=-1)
        PI = Q.softmax(dim=-1)
        Q = einops.einsum(PI, T, gamma * V,
                          'states actions, states actions next_states, next_states -> states actions') + R.unsqueeze(1)

        if (Q - old_Q).abs().max() < 1e-10:
            print(f'soft value iteration with pi converged in {i} iterations')
            pi = Q.softmax(dim=1)
            return Q, V, pi
    print('soft value iteration did not converge after', n_iterations, 'iterations')


pi_Q, pi_V, pi_pi = soft_q_value_iteration_with_pi(T, true_R, gamma, 10000)

soft_A = soft_Q - soft_V.unsqueeze(-1)
pi_A = pi_Q - pi_V.unsqueeze(-1)
torch.allclose(soft_A, pi_A)
torch.corrcoef(torch.stack((pi_pi.flatten(), soft_pi.flatten())))

# %%
def soft_q_value_iteration_with_pi_no_v(T, R, gamma, n_iterations=1000):
    Q = torch.zeros(n_states, n_actions)
    # V = torch.zeros(n_states)
    # normal method

    for i in range(n_iterations):
        old_Q = Q
        PI = Q.softmax(dim=-1)
        Q = einops.einsum(PI, T, gamma * Q.logsumexp(dim=-1),
                          'states actions, states actions next_states, next_states -> states actions') + R.unsqueeze(1)

        if (Q - old_Q).abs().max() < 1e-10:
            print(f'soft value iteration with pi no v converged in {i} iterations')
            pi = Q.softmax(dim=1)
            return Q, Q.logsumexp(dim=-1), pi
    print('soft value iteration did not converge after', n_iterations, 'iterations')


pi_Q, pi_V, pi_pi = soft_q_value_iteration_with_pi_no_v(T, true_R, gamma, 10000)

soft_A = soft_Q - soft_V.unsqueeze(-1)
pi_A = pi_Q - pi_V.unsqueeze(-1)
torch.allclose(soft_A, pi_A)
torch.corrcoef(torch.stack((pi_pi.flatten(), soft_pi.flatten())))


# %%

# %%
def evaluate_policy(T, R, gamma, pi, n_iterations=1000):
    V = torch.zeros(n_states)
    for i in range(n_iterations):
        old_V = V.clone()
        V = einops.einsum(T, pi, gamma * V, 'states actions next_states, states actions, next_states -> states') + R
        if torch.abs(V - old_V).max() < 1e-5:
            print(f'Policy evaluation converged in {i + 1} iterations')
            return V
    return V


V_pi_opt = evaluate_policy(T, true_R, gamma, pi)

V_pi_soft = evaluate_policy(T, true_R, gamma, soft_pi)
# %%
torch.corrcoef(torch.stack((V_pi_opt, V_pi_soft)))

soft_A = soft_Q - soft_V.unsqueeze(1)
A = Q - V.unsqueeze(1)

torch.corrcoef(torch.stack((A.flatten(), soft_A.flatten())))


def inverse_reward_shaping(T, A, gamma, n_iterations=30000):
    R = torch.zeros(n_states, requires_grad=True)
    H = torch.randn(n_states, requires_grad=True)

    A.requires_grad = False
    optimizer = torch.optim.Adam([R, H], lr=1e-3)
    for i in range(n_iterations):
        # TODO: works in stochastic case -> generalize theorem C.1 of AIRL paper
        A_hat = R.unsqueeze(1) - H.unsqueeze(1) + gamma * einops.einsum(T, H,
                                                                        "states actions next_states, next_states -> states actions")
        loss = ((A_hat - A) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:
            print((R - R.min()).detach().cpu().numpy().round(decimals=2))
    # return R, H
    return R - R.min(), H


learned_R, learned_H = inverse_reward_shaping(T, soft_A, gamma)
torch.corrcoef(torch.stack((true_R, learned_R))).round(decimals=4)
#%%
cust_list = [[-1, 2],
             [0, 2],
             [0, 2],
             [0, 2],
             [0, 0],
             [0, 0],
             ]
# [0.06 4.01 0.   4.03 3.7  2.44]

cust_A = (1*torch.FloatTensor(cust_list)).log_softmax(dim=-1)
cust_R, cust_H = inverse_reward_shaping(T, cust_A, gamma)
torch.corrcoef(torch.stack((true_R, cust_R))).round(decimals=4)


# hard_A = torch.nn.functional.one_hot(soft_A.argmax(dim=-1)).float()
#
# learned_R_opt, _ = inverse_reward_shaping(T, hard_A, gamma)
# torch.corrcoef(torch.stack((true_R,learned_R_opt)))

# %%


# %%
import matplotlib.pyplot as plt

plt.scatter(true_R.cpu().numpy(), learned_R_opt.detach().cpu().numpy())
# plt.plot(learned_R_opt.detach().cpu().numpy(), label='Learned R')
plt.legend()
plt.show()


# plt.savefig('inverse_reward_shaping.png')

# %%
def implicit_policy_learning(T, R, gamma, n_iterations=30000):
    Q = torch.zeros(n_states, n_actions, requires_grad=True)
    V = torch.randn(n_states, requires_grad=True)
    R = R.unsqueeze(-1)
    R.requires_grad = False
    optimizer = torch.optim.Adam([Q, V], lr=1e-3)
    for i in range(n_iterations):
        old_Q = Q.detach().clone()
        PI = Q.softmax(dim=-1)
        A = PI.log()
        R_hat = A + V.unsqueeze(1) - gamma * einops.einsum(PI, T, V,
                                                           "states actions, states actions next_states, next_states -> states actions")
        loss = ((R_hat - R) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (Q - old_Q).abs().max() < 1e-5:
            print(f'implicit policy learning converged in {i} iterations')
            return A, V
    return A, V


learned_A, learned_V = implicit_policy_learning(T, true_R, gamma)

torch.corrcoef(torch.stack((learned_A.flatten(), soft_A.flatten())))

torch.corrcoef(torch.stack((learned_V, soft_V)))


# %%

def look_ahead_inverse_reward_shaping(T, A, gamma, n_iterations=10000):
    R = torch.zeros(n_states, requires_grad=True)
    V = torch.randn(n_states, requires_grad=True)
    VN = torch.randn(n_states, n_actions, requires_grad=True)

    A.requires_grad = False
    optimizer = torch.optim.Adam([R, V, VN], lr=1e-3)
    for i in range(n_iterations):
        # TODO: works in stochastic case -> generalize theorem C.1 of AIRL paper
        A_hat = R.unsqueeze(1) - V.unsqueeze(1) + gamma * VN
        true_VN = einops.einsum(T, V, "states actions next_states, next_states -> states actions")
        v_loss = ((VN - true_VN) ** 2).mean()
        # einops.einsum(T, H, "states actions next_states, next_states -> states actions"))
        loss = ((A_hat - A) ** 2).mean() + v_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return R, V


learned_R, _ = inverse_reward_shaping(T, soft_A, gamma)
torch.corrcoef(torch.stack((true_R, learned_R)))

plt.scatter(true_R.cpu().numpy(), learned_R.detach().cpu().numpy())
# plt.plot(learned_R_opt.detach().cpu().numpy(), label='Learned R')
plt.legend()
plt.show()

hard_A = torch.nn.functional.one_hot(soft_A.argmax(dim=-1)).float()

learned_R_opt, _ = inverse_reward_shaping(T, hard_A, gamma)
torch.corrcoef(torch.stack((true_R, learned_R_opt)))


# %%
def implicit_policy_learning2(T, R, gamma, alpha=0.1, n_iterations=10000):
    Q = torch.randn(n_states, n_actions, requires_grad=True)
    F = torch.randn(n_states, n_actions, requires_grad=True)
    R = R.unsqueeze(-1)
    R.requires_grad = False
    optimizer = torch.optim.Adam([Q, F], lr=1e-3)
    corr_list = []
    for i in range(n_iterations):
        PI = (Q / alpha).softmax(dim=-1)
        A = PI.log()
        R_hat = Q - gamma * F
        Qn = einops.einsum(PI, T, Q,
                           "states actions, states actions next_states, next_states actions -> states next_states actions")
        Vn = Qn.exp().sum(dim=-2).log()  # .unsqueeze(-1) # ?
        # R_hat = A + V.unsqueeze(1) - gamma *einops.einsum(T, V, "states actions next_states, next_states -> states actions")
        loss1 = ((R_hat - R) ** 2).mean()
        loss2 = ((Vn - F) ** 2).mean()
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        correlation = torch.corrcoef(torch.stack((A.flatten(), soft_A.flatten())))[0, 1].item()
        corr_list.append(correlation)
        if i % 1000 == 0:
            print(correlation)
            plt.scatter(A.exp().flatten().detach().cpu().numpy(), soft_A.exp().flatten().detach().cpu().numpy())
            plt.show()
    return A, corr_list

# learned_A, corr_list = implicit_policy_learning2(T, true_R, gamma, 0.1, 100000)

# torch.corrcoef(torch.stack((A.exp().flatten(), soft_A.exp().flatten())))[0, 1].item()

# plt.scatter(learned_A.exp().flatten().detach().cpu().numpy(), soft_A.exp().flatten().detach().cpu().numpy())
# plt.show()

# plt.plot(corr_list)
# plt.show()


# %%
