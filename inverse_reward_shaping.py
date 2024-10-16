#%%
import torch
import einops

#%%
n_states = 10
n_actions = 5
deterministic = True

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
    T = T / T.sum(dim=2, keepdim=True)

assert T.sum(dim=2).allclose(torch.ones(n_states, n_actions))

true_R = torch.randn(n_states)

# true_h = torch.randn(n_states)
gamma = 0.9
#%%

def q_value_iteration(T, R, gamma, n_iterations=1000):
    Q = torch.zeros(n_states, n_actions)
    V = torch.zeros(n_states)

    for i in range(n_iterations):
        old_Q = Q
        old_V = V
        V = Q.max(dim=1).values
        Q = einops.repeat(R, 'states -> states actions', actions=n_actions) + gamma * einops.einsum(T, V, 'states actions states, states -> states actions')
        if (Q - old_Q).abs().max() < 1e-5:
            print(f'Q-value iteration converged in {i} iterations')
            break
    pi = Q.argmax(dim=1)
    return Q,V, pi

Q,V, pi = q_value_iteration(T, true_R, gamma)


# %%
def soft_q_value_iteration(T, R, gamma, n_iterations=1000):
    Q = torch.zeros(n_states, n_actions)
    V = torch.zeros(n_states)
    #normal method

    for i in range(n_iterations):
        old_Q = Q
        old_V = V
        V = Q.logsumexp(dim=-1)
        Q = einops.einsum(T, R+gamma*V, 'states actions states, states -> states actions')
        if (Q - old_Q).abs().max() < 1e-5:
            print(f'soft value iteration converged in {i} iterations')
            pi = Q.softmax(dim=1)
            return Q,V, pi
    print('soft value iteration did not converge after', n_iterations, 'iterations')
        
    
soft_Q, soft_V, soft_pi = soft_q_value_iteration(T, true_R, gamma)


#%%
def evaluate_policy(T, R, gamma, pi):
    V = torch.zeros(n_states)
    for i in range(n_iterations):
        old_V = V.clone()
        V = einops.einsum(T, pi, R + gamma * V, 'states actions states, states actions, states -> states')
        if torch.abs(V - old_V).max() < 1e-5:
            print(f'Policy evaluation converged in {i+1} iterations')
            return V
    return V

V = evaluate_policy(T, true_R, gamma, pi)

# %%
