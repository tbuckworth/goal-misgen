import einops
import numpy as np
import torch

from meg.meg_torch import state_action_occupancy, soft_value_iteration, soft_value_iteration_sa_rew


def uniform_policy_evaluation(T, R, gamma, n_iterations=1000, device='cpu'):
    n_states, n_actions = T.shape[:2]
    V = torch.zeros(n_states).to(device =device)
    # Uniform policy probability
    uniform_prob = 1 / n_actions

    for _ in range(n_iterations):
        old_V = V
        Q = R + gamma * einops.einsum(T, V, "s a ns, ns -> s a")
        V = (Q * uniform_prob).sum(dim=-1)
        if torch.allclose(V, old_V):
            return V
    return V

def canonicalise(T, R, gamma, policy_evaluation=uniform_policy_evaluation, n_iterations=1000, device='cpu'):
    V = policy_evaluation(T, R, gamma, n_iterations, device)
    CR = R + gamma * einops.einsum(T, V, "s a ns, ns -> s a") - V.unsqueeze(-1)
    return CR

def q_value_iteration(T, R, gamma, n_iterations=10000, device='cpu', slow = False):
    n_states, n_actions = T.shape[:2]
    Q = torch.zeros(n_states, n_actions).to(device=device)
    Qs = []
    alpha = 0.001
    for i in range(n_iterations):
        old_Q = Q
        # old_V = V
        V = Q.max(dim=1).values
        target_Q = R + gamma * einops.einsum(T, V, 'states actions next_states, next_states -> states actions')
        if slow:
            Q = Q + alpha * (target_Q - Q)
        else:
            Q = target_Q
        Qs.append(Q)
        if (Q - old_Q).abs().max() < 1e-5:
            print(f'Q-value iteration converged in {i} iterations')
            break
    pi = torch.nn.functional.one_hot(Q.argmax(dim=1)).float()
    return pi, Qs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_states = 2
    n_actions = 2
    gamma = 0.9
    CHOSEN_AXIS = 1
    # state 0: many cobras
    # state 1: few cobras
    # action 0: breed cobras
    # action 1: kill cobras

    T = torch.zeros((n_states, n_actions, n_states)).to(device=device)
    T[:, 0, 0] = 1  # breeding always takes you to many cobras
    T[:, 1, 1] = 1  # killing always takes you to few cobras
    assert torch.allclose(T.sum(dim=-1), torch.tensor(1.)), "T is not valid transition matrix"
    mu = torch.zeros((n_states,)).to(device=device)
    mu[0] = 1
    assert torch.allclose(mu.sum(), torch.tensor(1.)), "mu is not valid initialisation matrix"

    true_R = torch.zeros((n_states, n_actions)).to(device=device)
    true_R[1, 1] = 1.

    proxy_R = torch.zeros((n_states, n_actions)).to(device=device)
    proxy_R[0, 1] = 1.

    true_CR = canonicalise(T, true_R, gamma, device=device)
    proxy_CR = canonicalise(T, proxy_R, gamma, device=device)

    _, true_pi_soft_opt = soft_value_iteration_sa_rew(true_R, T, gamma=gamma, device="cuda")
    true_pi_hard_opt, true_Qs = q_value_iteration(T, true_R, gamma, device=device, slow=True)
    _, proxy_pi_soft_opt = soft_value_iteration_sa_rew(proxy_R, T, gamma=gamma, device="cuda")
    proxy_pi_hard_opt, proxy_Qs = q_value_iteration(T, proxy_R, gamma, device=device)

    breed = [1., 0.]
    kill = [0., 1.]

    pi_kill = torch.tensor([kill, kill]).to(device=device)
    pi_flip = torch.tensor([kill, breed]).to(device=device)
    pi_stick = torch.tensor([breed, kill]).to(device=device)
    pi_breed = torch.tensor([breed, breed]).to(device=device)

    pi_uniform = torch.ones((n_states, n_actions)).softmax(dim=-1).to(device=device)

    all_pis = [true_pi_soft_opt, proxy_pi_soft_opt, true_pi_hard_opt, proxy_pi_hard_opt,
               pi_uniform,
               pi_kill, pi_flip, pi_stick, pi_breed]

    r = torch.rand((1000, n_states, n_actions)).to(device=device)
    div = torch.ones((1000, 1, 1)).cumsum(dim=0).to(device=device)

    all_pis += list((r * div).softmax(dim=-1).to(device=device).unbind(dim=-0))
    all_pis += list(torch.rand((100, n_states, n_actions)).softmax(dim=-1).to(device=device).unbind(dim=0))



    ds = torch.stack([state_action_occupancy(pi, T, gamma, mu, device=device) for pi in all_pis])
    x, y = ds[..., CHOSEN_AXIS].detach().cpu().numpy().T

    true_x, true_y = generate_occupancy_trajectories(CHOSEN_AXIS, T, device, gamma, mu, true_Qs, true_pi_hard_opt)
    proxy_x, proxy_y = generate_occupancy_trajectories(CHOSEN_AXIS, T, device, gamma, mu, proxy_Qs, proxy_pi_hard_opt)

    px, py = true_CR[...,CHOSEN_AXIS].cpu().numpy()
    prx, pry = proxy_CR[...,CHOSEN_AXIS].cpu().numpy()
    # This is what it should be, but doesn't line up:
    px, py = true_CR[0, 0].cpu().numpy(), true_CR[1, 1].cpu().numpy()

    (true_x[-1] - true_x[0], true_y[-1] - true_y[0])
    (proxy_x[-1] - proxy_x[0], proxy_y[-1] - proxy_y[0])

    circle_size = 5
    tri_size = 50

    import matplotlib.pyplot as plt
    plt.scatter(x[4:], y[4:], s=circle_size)
    plt.arrow(x[4], y[4], px, py, head_width=0.05, head_length=0.05,
              fc='red', ec='red', linewidth=2, label='True Reward Direction')
    plt.arrow(x[4], y[4], prx, pry, head_width=0.05, head_length=0.05,
              fc='orange', ec='orange', linewidth=2, label='Proxy Reward Direction')
    plt.scatter(true_x, true_y, color='red', label='Hard $\pi$ on True Reward', alpha=0.7, s=circle_size)
    plt.scatter(proxy_x, proxy_y, color='orange', label='Hard $\pi$ on Proxy Reward', alpha=0.7, s=circle_size)
    plt.scatter(x[0], y[0], color='red', label="Soft $\pi*$ for True Reward", alpha=0.7, s=tri_size)
    plt.scatter(x[1], y[1], color='orange', label="Soft $\pi*$ for Proxy Reward", alpha=0.7, s=tri_size)
    # plt.scatter(x[2], y[2], color='red', label="Hard $\pi*$ for True Reward")
    # plt.scatter(x[3], y[3], color='orange', label="Hard $\pi*$ for Proxy Reward")

    plt.xlabel('(Many Cobras, Kill) Occupancy')
    plt.ylabel('(Few Cobras, Kill) Occupancy')
    plt.legend()
    plt.title('State-Action Occupancy vs Reward Functions\nCobra Breeding Environment')
    plt.savefig("cobra_state_action_occupancy.png")
    plt.show()

    print("done")


def generate_occupancy_trajectories(CHOSEN_AXIS, T, device, gamma, mu, true_Qs, true_pi_hard_opt):
    tp0 = torch.stack(true_Qs).softmax(dim=-1)
    true_pis = torch.stack([true_Qs[-1] * np.log((i + 5.44) / 2) for i in range(100)]).softmax(dim=-1)
    true_pis = torch.concat([tp0, true_pis, true_pi_hard_opt.unsqueeze(0)])
    true_ds = torch.stack([state_action_occupancy(pi, T, gamma, mu, device=device) for pi in true_pis])
    true_x, true_y = true_ds[..., CHOSEN_AXIS].detach().cpu().numpy().T
    return true_x, true_y


def project_reward(CHOSEN_AXIS, true_R):
    projected_R = (true_R - true_R.mean(dim=-1))[..., CHOSEN_AXIS]
    px, py = projected_R.detach().cpu().numpy()
    return px, py


def project_reward_to_kill_axes(CHOSEN_AXIS, R):
    other_axis = 0 if CHOSEN_AXIS else 1
    # R is shape (n_states=2, n_actions=2)
    px = (R[0, CHOSEN_AXIS] - R[0, other_axis]).item()  # state “many cobras”
    py = (R[1, CHOSEN_AXIS] - R[1, other_axis]).item()  # state “few cobras”
    return px, py

def projected_arrow(R, p0, p1, gamma=0.9):
    D  = 1 + gamma*(p0 - p1)
    v0 = torch.tensor([(1-gamma*p1)/D - gamma*p0*(1-gamma*p1)/D**2,
                       gamma*p1/D       - gamma**2*p0*p1/D**2])
    v1 = torch.tensor([-gamma*p0/D**2,
                       gamma*p0/D - gamma*p0/D**2])
    B  = torch.stack([v0, v1], dim=1)       # 2×2
    # canonicalised reward (true or proxy)
    Rcanon = R - R.mean(dim=1, keepdim=True)
    # return = R·η;  we only need the 4 numbers of R
    r_vec  = torch.tensor([*Rcanon.view(-1).tolist()])
    # least-squares coefficients
    coeff  = torch.linalg.lstsq(B, r_vec[:2]).solution   # first 2 coords suffice
    return coeff      # (Δx, Δy) for the arrow


if __name__ == "__main__":
    main()
