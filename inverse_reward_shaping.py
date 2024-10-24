#%%
import numpy as np
import torch
import einops

#%%
n_states = 10
n_actions = 5
deterministic = False
sparse_reward = False

#%%
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
#%%

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
    return Q,V, pi

Q, V, pi = q_value_iteration(T, true_R, gamma)



# %%
def soft_q_value_iteration(T, R, gamma, n_iterations=1000):
    Q = torch.zeros(n_states, n_actions)
    V = torch.zeros(n_states)
    #normal method

    for i in range(n_iterations):
        old_Q = Q
        V = Q.logsumexp(dim=-1)
        Q = einops.einsum(T, gamma * V, 'states actions next_states, next_states -> states actions') + R.unsqueeze(1)

        if (Q - old_Q).abs().max() < 1e-5:
            print(f'soft value iteration converged in {i} iterations')
            pi = Q.softmax(dim=1)
            return Q,V, pi
    print('soft value iteration did not converge after', n_iterations, 'iterations')
        
    
soft_Q, soft_V, soft_pi = soft_q_value_iteration(T, true_R, gamma)


#%%
def evaluate_policy(T, R, gamma, pi, n_iterations=1000):
    V = torch.zeros(n_states)
    for i in range(n_iterations):
        old_V = V.clone()
        V = einops.einsum(T, pi, gamma * V, 'states actions next_states, states actions, next_states -> states') + R
        if torch.abs(V - old_V).max() < 1e-5:
            print(f'Policy evaluation converged in {i+1} iterations')
            return V
    return V

V_pi_opt = evaluate_policy(T, true_R, gamma, pi)

V_pi_soft = evaluate_policy(T, true_R, gamma, soft_pi)
# %%
torch.corrcoef(torch.stack((V_pi_opt,V_pi_soft)))

soft_A = soft_Q - soft_V.unsqueeze(1)
A = Q - V.unsqueeze(1)

torch.corrcoef(torch.stack((A.flatten(), soft_A.flatten())))


def inverse_reward_shaping(T, A, gamma, n_iterations=10000):
    R = torch.zeros(n_states, requires_grad=True)
    H = torch.randn(n_states, requires_grad=True)

    A.requires_grad = False
    optimizer = torch.optim.Adam([R,H], lr=1e-3)
    for i in range(n_iterations):
        # TODO: works in stochastic case -> generalize theorem C.1 of AIRL paper
        A_hat = R.unsqueeze(1) - H.unsqueeze(1) + gamma *einops.einsum(T, H, "states actions next_states, next_states -> states actions")
        loss = ((A_hat - A)**2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return R, H
    # return R - R.min(), H





learned_R, learned_H = inverse_reward_shaping(T, soft_A, gamma)
torch.corrcoef(torch.stack((true_R,learned_R)))

hard_A = torch.nn.functional.one_hot(soft_A.argmax(dim=-1)).float()

learned_R_opt, _ = inverse_reward_shaping(T, hard_A, gamma)
torch.corrcoef(torch.stack((true_R,learned_R_opt)))



# %%
import matplotlib.pyplot as plt

plt.scatter(true_R.cpu().numpy(), learned_R_opt.detach().cpu().numpy())
# plt.plot(learned_R_opt.detach().cpu().numpy(), label='Learned R')
plt.legend()
plt.show()
# plt.savefig('inverse_reward_shaping.png')

# %%
def implicit_policy_learning(T, R, gamma, n_iterations=30000):
    PI = torch.zeros(n_states, n_actions, requires_grad=True)
    V = torch.randn(n_states, requires_grad=True)
    R = R.unsqueeze(-1)
    R.requires_grad = False
    optimizer = torch.optim.Adam([PI,V], lr=1e-3)
    for i in range(n_iterations):
        A = PI.softmax(dim=-1).log()
        R_hat = A + V.unsqueeze(1) - gamma *einops.einsum(T, V, "states actions next_states, next_states -> states actions")
        loss = ((R_hat - R)**2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
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
    optimizer = torch.optim.Adam([R,V,VN], lr=1e-3)
    for i in range(n_iterations):
        # TODO: works in stochastic case -> generalize theorem C.1 of AIRL paper
        A_hat = R.unsqueeze(1) - V.unsqueeze(1) + gamma * VN
        true_VN = einops.einsum(T, V, "states actions next_states, next_states -> states actions")
        v_loss = ((VN-true_VN)**2).mean()
                 # einops.einsum(T, H, "states actions next_states, next_states -> states actions"))
        loss = ((A_hat - A)**2).mean() + v_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return R, V

learned_R, _ = inverse_reward_shaping(T, soft_A, gamma)
torch.corrcoef(torch.stack((true_R,learned_R)))

plt.scatter(true_R.cpu().numpy(), learned_R.detach().cpu().numpy())
# plt.plot(learned_R_opt.detach().cpu().numpy(), label='Learned R')
plt.legend()
plt.show()


hard_A = torch.nn.functional.one_hot(soft_A.argmax(dim=-1)).float()

learned_R_opt, _ = inverse_reward_shaping(T, hard_A, gamma)
torch.corrcoef(torch.stack((true_R,learned_R_opt)))