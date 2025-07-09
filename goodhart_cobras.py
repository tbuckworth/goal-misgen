import einops
import torch

from meg.meg_torch import state_action_occupancy, soft_value_iteration, soft_value_iteration_sa_rew

def q_value_iteration(T, R, gamma, n_iterations=1000):
    n_states, n_actions = T.shape[:2]
    Q = torch.zeros(n_states, n_actions)
    V = torch.zeros(n_states)

    for i in range(n_iterations):
        old_Q = Q
        # old_V = V
        V = Q.max(dim=1).values
        # Q = einops.repeat(R, 'states -> states actions', actions=n_actions)
        Q = einops.einsum(T, gamma * V, 'states actions next_states, next_states -> states actions') + R

        if (Q - old_Q).abs().max() < 1e-5:
            print(f'Q-value iteration converged in {i} iterations')
            break
    pi = torch.nn.functional.one_hot(Q.argmax(dim=1)).float()
    return pi

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
    T[:, 0, 0] = 1 # breeding always takes you to many cobras
    T[:, 1, 1] = 1 # killing always takes you to few cobras
    assert torch.allclose(T.sum(dim=-1),torch.tensor(1.)), "T is not valid transition matrix"
    mu = torch.zeros((n_states,)).to(device=device)
    mu[0] = 1
    assert torch.allclose(mu.sum(), torch.tensor(1.)), "mu is not valid initialisation matrix"

    true_R = torch.zeros((n_states, n_actions)).to(device=device)
    true_R[1, 1] = 1.

    proxy_R = torch.zeros((n_states, n_actions)).to(device=device)
    proxy_R[0, 1] = 1.

    _, true_pi_soft_opt = soft_value_iteration_sa_rew(true_R, T, gamma=gamma, device="cuda")
    true_pi_hard_opt = q_value_iteration(true_R, T, gamma)
    _, proxy_pi_soft_opt = soft_value_iteration_sa_rew(proxy_R, T, gamma=gamma, device="cuda")
    proxy_pi_hard_opt = q_value_iteration(proxy_R, T, gamma)


    breed = [1., 0.]
    kill = [0., 1.]

    pi_kill = torch.tensor([kill, kill]).to(device=device)
    pi_flip = torch.tensor([kill, breed]).to(device=device)
    pi_stick = torch.tensor([breed, kill]).to(device=device)
    pi_breed = torch.tensor([breed, breed]).to(device=device)

    all_pis = [true_pi_soft_opt, proxy_pi_soft_opt, true_pi_hard_opt, proxy_pi_hard_opt,
               pi_kill, pi_flip, pi_stick, pi_breed]

    r = torch.rand((1000, n_states, n_actions)).to(device=device)
    div = torch.ones((1000,1,1)).cumsum(dim=0).to(device=device)

    all_pis += list((r*div).softmax(dim=-1).to(device=device).unbind(dim=-0))
    all_pis += list(torch.rand((100, n_states, n_actions)).softmax(dim=-1).to(device=device).unbind(dim=0))

    ds = torch.stack([state_action_occupancy(pi, T, gamma, mu, device=device) for pi in all_pis])
    x, y = ds[...,CHOSEN_AXIS].detach().cpu().numpy().T

    px, py = project_reward(CHOSEN_AXIS, true_R)
    prx, pry = project_reward(CHOSEN_AXIS, proxy_R)


    import matplotlib.pyplot as plt
    plt.scatter(x, y)
    plt.arrow(x.mean(), y.mean(), px, py, head_width=0.05, head_length=0.05, fc='red', ec='red', linewidth=2, label='True Reward Direction')
    plt.arrow(x.mean(), y.mean(), prx, pry, head_width=0.05, head_length=0.05, fc='orange', ec='orange', linewidth=2, label='Proxy Reward Direction')
    plt.scatter(x[0], y[0], color='red', label="Soft Optimal Policy for True Reward", alpha=0.5)
    plt.scatter(x[1], y[1], color='orange', label="Soft Optimal Policy for Proxy Reward", alpha=0.5)
    plt.scatter(x[2], y[2], color='red', label="Hard Optimal Policy for True Reward")
    plt.scatter(x[3], y[3], color='orange', label="Hard Optimal Policy for Proxy Reward")

    plt.xlabel('(Many Cobras, Kill) Occupancy')
    plt.ylabel('(Few Cobras, Kill) Occupancy')
    plt.legend()
    plt.title('State-Action Occupancy vs Reward Functions\nCobra Breeding Environment')
    plt.savefig("cobra_state_action_occupancy.png")
    plt.show()


    print("done")


def project_reward(CHOSEN_AXIS, true_R):
    projected_R = (true_R - true_R.mean(dim=-1))[..., CHOSEN_AXIS]
    px, py = projected_R.detach().cpu().numpy()
    return px, py


if __name__ == "__main__":
    main()
