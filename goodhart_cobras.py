import einops
import numpy as np
import torch

from meg.meg_torch import state_action_occupancy, soft_value_iteration, soft_value_iteration_sa_rew


def uniform_policy_evaluation(T, R, gamma, n_iterations=1000, device='cpu'):
    n_states, n_actions = T.shape[:2]
    V = torch.zeros(n_states).to(device=device)
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


def q_value_iteration(T, R, gamma, n_iterations=1000, device='cpu', slow=False):
    n_states, n_actions = T.shape[:2]
    Q = torch.zeros(n_states, n_actions).to(device=device)
    Qs = []
    alpha = 0.01
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


def ppo_tabular(
        T, R, gamma,
        n_iterations: int = 1000,
        device: str = 'cpu',
        lr: float = 0.1,
        clip_eps: float = 0.2,
        eval_iters: int = 20,
):
    """
    Tabular PPO with a full-model critic.
    Matches the interface of `q_value_iteration`.
    Returns:
        pi  – final π(s,a) probabilities, shape [S, A] (analogous to one-hot arg-max in QVI)
        policies – list of π_t(s,a) tensors across updates (mirrors the role of Qs in QVI)
    """
    T, R = T.to(device), R.to(device)
    n_states, n_actions = T.shape[:2]

    # actor parameters (state-wise logits)
    logits = torch.zeros(n_states, n_actions, device=device, requires_grad=True)
    optimiser = torch.optim.SGD([logits], lr=lr)

    policies = []
    old_probs = torch.softmax(logits.detach(), dim=1)  # π₀

    for i in range(n_iterations):
        # ---- critic: evaluate V^π and Q^π with full model ----
        V = torch.zeros(n_states, device=device)
        for _ in range(eval_iters):  # iterative policy evaluation
            Q_pi = R + gamma * einops.einsum(
                T, V, 's a s2, s2 -> s a')
            V = (old_probs * Q_pi).sum(dim=1)

        if Q_pi.isnan().any():
            break

        advantages = Q_pi - V.unsqueeze(1)  # A(s,a)

        # ---- actor: clipped PPO update ----
        optimiser.zero_grad()
        new_probs = torch.softmax(logits, dim=1)
        ratio = new_probs / old_probs  # π_θ / π_old

        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        loss = -torch.min(unclipped, clipped).mean()  # maximise surrogate ⇒ minimise –surrogate
        loss.backward()
        optimiser.step()

        policies.append(new_probs.detach())  # track policy over time

        # ---- convergence check ----
        if (i > 100 and (new_probs - old_probs).abs().max() < 1e-8):
            print(f'PPO converged in {i} iterations')
            break

        old_probs = new_probs.detach()

    pi = old_probs  # final stochastic policy
    return pi, [p.log() for p in policies]


def ppo_fixed_adv_tabular(
        T, R, gamma,
        n_iterations: int = 1000,
        device: str = 'cpu',
        lr: float = 0.1,
        clip_eps: float = 0.2,
        eval_iters: int = 20,
):
    """
    Tabular PPO with a full-model critic.
    Matches the interface of `q_value_iteration`.
    Returns:
        pi  – final π(s,a) probabilities, shape [S, A] (analogous to one-hot arg-max in QVI)
        policies – list of π_t(s,a) tensors across updates (mirrors the role of Qs in QVI)
    """
    T, R = T.to(device), R.to(device)
    n_states, n_actions = T.shape[:2]

    advantages = canonicalise(T, R, gamma, device=device)

    # actor parameters (state-wise logits)
    logits = torch.zeros(n_states, n_actions, device=device, requires_grad=True)
    optimiser = torch.optim.SGD([logits], lr=lr)

    policies = []
    old_probs = torch.softmax(logits.detach(), dim=1)  # π₀

    for i in range(n_iterations):
        # ---- actor: clipped PPO update ----
        optimiser.zero_grad()
        new_probs = torch.softmax(logits, dim=1)
        ratio = new_probs / old_probs  # π_θ / π_old

        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        loss = -torch.min(unclipped, clipped).mean()  # maximise surrogate ⇒ minimise –surrogate
        loss.backward()
        optimiser.step()

        policies.append(new_probs.detach())  # track policy over time

        # ---- convergence check ----
        if (i > 100 and (new_probs - old_probs).abs().max() < 1e-8):
            print(f'PPO converged in {i} iterations')
            break

        old_probs = new_probs.detach()

    pi = old_probs  # final stochastic policy
    return pi, [p.log() for p in policies]


def cobras():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_states = 2
    n_actions = 2
    gamma = 0.9
    CHOSEN_AXIS = 1
    state0 = "Many Cobras"
    state1 = "Few Cobras"
    # state 1: few cobras
    # action 0: breed cobras
    # action 1: kill cobras
    act_name = "Kill" if CHOSEN_AXIS else "Breed"

    T = torch.zeros((n_states, n_actions, n_states)).to(device=device)
    T[:, 0, 0] = 1  # breeding always takes you to many cobras
    T[:, 1, 1] = 1  # killing always takes you to few cobras
    mu = torch.zeros((n_states,)).to(device=device)
    mu[0] = 1

    true_R = torch.zeros((n_states, n_actions)).to(device=device)
    true_R[1, 1] = 1.

    proxy_R = torch.zeros((n_states, n_actions)).to(device=device)
    proxy_R[0, 1] = 1.

    reward_dict = {
        "True Reward": true_R,
        "Proxy Reward": proxy_R,
    }

    plot_state_action_occupancies(
        "Cobras",
        n_states,
        n_actions,
        T,
        mu,
        gamma,
        reward_dict,
        CHOSEN_AXIS,
        device,
        state0,
        state1,
        act_name,
    )


def unit_circle_points(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)


def plot_state_action_occupancies(
        name,
        n_states,
        n_actions,
        T,
        mu,
        gamma,
        reward_list,
        CHOSEN_AXIS,
        device,
        state0="State 0",
        state1="State 1",
        act_name=None,
        add_random_policy=False,
        rl_algo=None,
):
    if rl_algo is None:
        rl_algo = {"PPO": ppo_tabular}
    if act_name is None:
        act_name = f"Action {CHOSEN_AXIS}"
    assert torch.allclose(T.sum(dim=-1), torch.tensor(1.)), "T is not valid transition matrix"
    assert torch.allclose(mu.sum(), torch.tensor(1.)), "mu is not valid initialisation matrix"
    # _, true_pi_soft_opt = soft_value_iteration_sa_rew(true_R, T, gamma=gamma, device="cuda")
    # _, proxy_pi_soft_opt = soft_value_iteration_sa_rew(proxy_R, T, gamma=gamma, device="cuda")

    # px, py, true_x, true_y = generate_reward_data(CHOSEN_AXIS, T, device, gamma, mu, true_R)
    #
    # prx, pry, proxy_x, proxy_y = generate_reward_data(CHOSEN_AXIS, T, device, gamma, mu, proxy_R)

    if isinstance(reward_list, dict):
        reward_names = list(reward_list.keys())
        reward_list = list(reward_list.values())
    else:
        reward_names = None

    reward_data = []

    colours = ["red", "orange", "green", "blue", "cyan", "magenta", "purple", "brown"]
    shapes = ['o','x','+','v','s']

    for i, R in enumerate(reward_list):
        name = reward_names[i] if reward_names else None
        c = colours[i] if reward_names else 'orange'
        for j, (algo_name, algo) in enumerate(rl_algo.items()):
            arrow_x, arrow_y, sao_x, sao_y = generate_reward_data(CHOSEN_AXIS, T, device, gamma, mu, R, algo)
            _, soft_pi = soft_value_iteration_sa_rew(R, T, gamma=gamma, device=device)
            soft_x, soft_y = state_action_occupancy(soft_pi, T, gamma, mu, device=device)[..., CHOSEN_AXIS].detach().cpu().numpy()
            reward_data.append(
                dict(name=name,
                     arrow_x=arrow_x,
                     arrow_y=arrow_y,
                     sao_x=sao_x,
                     sao_y=sao_y,
                     algo_name=algo_name,
                     colour=c,
                     shape=shapes[j],
                     soft_x=soft_x,
                     soft_y=soft_y,
                     )
            )
    breed = [1., 0.]
    kill = [0., 1.]

    pi_kill = torch.tensor([kill, kill]).to(device=device)
    pi_flip = torch.tensor([kill, breed]).to(device=device)
    pi_stick = torch.tensor([breed, kill]).to(device=device)
    pi_breed = torch.tensor([breed, breed]).to(device=device)

    pi_uniform = torch.ones((n_states, n_actions)).softmax(dim=-1).to(device=device)

    all_pis = [pi_uniform,  # true_pi_soft_opt, proxy_pi_soft_opt, true_pi_hard_opt, proxy_pi_hard_opt,
               pi_kill, pi_flip, pi_stick, pi_breed]

    r = torch.rand((1000, n_states, n_actions)).to(device=device)
    div = torch.ones((1000, 1, 1)).cumsum(dim=0).to(device=device)

    all_pis += list((r * div).softmax(dim=-1).to(device=device).unbind(dim=-0))
    all_pis += list(torch.rand((100, n_states, n_actions)).softmax(dim=-1).to(device=device).unbind(dim=0))

    ds = torch.stack([state_action_occupancy(pi, T, gamma, mu, device=device) for pi in all_pis])
    x, y = ds[..., CHOSEN_AXIS].detach().cpu().numpy().T

    circle_size = 5
    tri_size = 50

    if add_random_policy:
        pis = torch.stack(all_pis)
        e = -pis * pis.log()
        e[e.isnan()] = 0
        entropies = e.sum(dim=-1).mean(dim=-1)
        ni = (entropies == entropies[entropies > 0.0001].min()).argwhere()[0].item()

        new_R = pis[ni].log()
        new_CR = canonicalise(T, new_R, gamma, device=device)
        new_CR = new_CR / new_CR.max()
        nrx, nry = new_CR[..., CHOSEN_AXIS].cpu().numpy()
        npx, npy = ds[ni, ..., CHOSEN_AXIS].cpu().numpy()


    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))  # width=10 inches, height=6 inches
    plt.scatter(x[4:], y[4:], s=circle_size)

    for d in reward_data:
        params = dict(
            x=d["sao_x"],
            y=d["sao_y"],
            color=d["colour"],
            alpha=0.5,
            s=circle_size,
            marker=d['shape'],
            )
        if len(reward_data)<5:
            params['label'] = f"{d['algo_name']} {d['name']}"
        plt.scatter(**params)

    for d in reward_data:
        params = dict(
            x=x[0],
            y=y[0],
            dx=d["arrow_x"],
            dy=d["arrow_y"],
            width=0.05,
            head_width=0.1,
            head_length=0.1,
            fc=d["colour"],
            edgecolor='black',
            linewidth=0.5,
        )
        if len(reward_data)<5:
            params['label'] = d['name']
        plt.arrow(**params)

    for d in reward_data:
        params = dict(
            x=d['soft_x'],
            y=d['soft_y'],
            color=d["colour"],
            s=tri_size,
        )
        if len(reward_data)<5:
            params['label'] = f"Soft $\pi*$: {d['name']}"
        plt.scatter(**params)

    if add_random_policy:
        plt.scatter(npx, npy, color='green', s=tri_size)
        plt.arrow(x[0], y[0], nrx, nry, width=0.05, head_width=0.1, head_length=0.1,
                  fc='green', edgecolor='black', linewidth=.5, label='Policy Reward Direction')
    plt.xlabel(f'({state0}, {act_name}) Occupancy')
    plt.ylabel(f'({state1}, {act_name}) Occupancy')
    plt.legend()
    plt.title(f'State-Action Occupancy vs Reward Functions\n{name} Environment')
    plt.savefig(f"data/{name}_state_action_occupancy.png")
    plt.show()

    print("done")

    soft_pis = [soft_value_iteration_sa_rew(R, T, gamma=gamma, device=device) for R in reward_list]
    implied_R = [canonicalise(T, pi.log(), gamma, device=device) for _, pi in soft_pis]
    from titus_meg import norm_funcs, dist_funcs
    normalize = norm_funcs["l2_norm"]
    distance = dist_funcs["l2_dist"]
    pircs = [distance(normalize(R), normalize(IR)) for R, IR in zip(reward_list, implied_R)]
    # These are all close to zero, yay!
    # TODO: do the same for hard pis trained on PPO - check at various stages of training.




def generate_reward_data(CHOSEN_AXIS, T, device, gamma, mu, true_R, rl_algo=ppo_tabular):
    true_CR = canonicalise(T, true_R, gamma, device=device)
    _, true_Qs = rl_algo(T, true_R, gamma, device=device)  # , slow=True)

    true_x, true_y = generate_occupancy_trajectories(CHOSEN_AXIS, T, device, gamma, mu, true_Qs)
    px, py = true_CR[..., CHOSEN_AXIS].cpu().numpy()
    return px, py, true_x, true_y


def projection(CHOSEN_AXIS, ds, true_CR):
    z = (ds * true_CR.unsqueeze(0)).sum(dim=-1).sum(dim=-1)
    coeffs = ds[..., CHOSEN_AXIS]
    # Option A: pinv
    px, py = torch.linalg.pinv(coeffs) @ z  # shape (2,)
    # Option B: lstsq (PyTorch ≥2.0)
    # returns solution of shape (2,1)
    # sol_ls = torch.linalg.lstsq(coeffs, z.unsqueeze(1)).solution
    # px, py = sol_ls[0, 0], sol_ls[1, 0]
    return px.item(), py.item()


def generate_occupancy_trajectories(CHOSEN_AXIS, T, device, gamma, mu, true_Qs):
    true_pis = torch.stack(true_Qs).softmax(dim=-1)
    true_pis = true_pis[~true_pis.isnan().any(dim=-1).any(dim=-1)]
    # true_pis = torch.stack([true_Qs[-1] * np.log((i + 5.44) / 2) for i in range(100)]).softmax(dim=-1)
    # true_pis = torch.concat([tp0, true_pi_hard_opt.unsqueeze(0)])
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
    D = 1 + gamma * (p0 - p1)
    v0 = torch.tensor([(1 - gamma * p1) / D - gamma * p0 * (1 - gamma * p1) / D ** 2,
                       gamma * p1 / D - gamma ** 2 * p0 * p1 / D ** 2])
    v1 = torch.tensor([-gamma * p0 / D ** 2,
                       gamma * p0 / D - gamma * p0 / D ** 2])
    B = torch.stack([v0, v1], dim=1)  # 2×2
    # canonicalised reward (true or proxy)
    Rcanon = R - R.mean(dim=1, keepdim=True)
    # return = R·η;  we only need the 4 numbers of R
    r_vec = torch.tensor([*Rcanon.view(-1).tolist()])
    # least-squares coefficients
    coeff = torch.linalg.lstsq(B, r_vec[:2]).solution  # first 2 coords suffice
    return coeff  # (Δx, Δy) for the arrow


def random(temp=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_states = 2
    n_actions = 2
    gamma = 0.9
    CHOSEN_AXIS = 0
    # state 0: many cobras
    # state 1: few cobras
    # action 0: breed cobras
    # action 1: kill cobras

    T = torch.rand((n_states, n_actions, n_states)).mul(temp).softmax(dim=-1).to(device=device)
    mu = torch.rand((n_states,)).mul(temp).softmax(dim=-1).to(device=device)

    true_R = torch.rand((n_states, n_actions)).mul(temp).to(device=device)
    true_R = (true_R.exp() / true_R.exp().sum()) * max(10 - temp, 1)

    proxy_R = torch.rand((n_states, n_actions)).mul(temp).to(device=device)
    proxy_R = (proxy_R.exp() / proxy_R.exp().sum()) * max(10 - temp, 1)

    plot_state_action_occupancies(
        f"Random with temp {1 / temp:.2f}",
        n_states,
        n_actions,
        T,
        mu,
        gamma,
        true_R,
        proxy_R,
        CHOSEN_AXIS,
        device,
    )


def run_all_randoms():
    for i in range(1, 10):
        try:
            random(i)
        except RuntimeError:
            pass


def policy_to_reward(temp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_states = 2
    n_actions = 2
    gamma = 0.9
    CHOSEN_AXIS = 0
    # state 0: many cobras
    # state 1: few cobras
    # action 0: breed cobras
    # action 1: kill cobras

    T = torch.rand((n_states, n_actions, n_states)).mul(temp).softmax(dim=-1).to(device=device)
    mu = torch.rand((n_states,)).mul(temp).softmax(dim=-1).to(device=device)

    true_R = torch.rand((n_states, n_actions)).mul(temp).to(device=device)
    true_R = (true_R.exp() / true_R.exp().sum()) * max(10 - temp, 1)

    proxy_R = torch.rand((n_states, n_actions)).mul(temp).to(device=device)
    proxy_R = (proxy_R.exp() / proxy_R.exp().sum()) * max(10 - temp, 1)

    plot_state_action_occupancies(
        f"Random with temp {1 / temp:.2f}",
        n_states,
        n_actions,
        T,
        mu,
        gamma,
        true_R,
        proxy_R,
        CHOSEN_AXIS,
        device,
    )


def star(inv_temp=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_states = 2
    n_actions = 2
    gamma = 0.9
    CHOSEN_AXIS = 0

    T = torch.rand((n_states, n_actions, n_states)).mul(inv_temp).softmax(dim=-1).to(device=device)
    mu = torch.rand((n_states,)).mul(inv_temp).softmax(dim=-1).to(device=device)

    n_arrows = 32

    u = unit_circle_points(n_arrows)
    u = u.repeat(2).reshape(n_arrows, 2, 2)
    u[..., 1] *= -1
    reward_list = torch.tensor(u).to(device=device, dtype=torch.float32).unbind(0)

    plot_state_action_occupancies(
        f"Random with temp {1 / inv_temp:.2f}",
        n_states,
        n_actions,
        T,
        mu,
        gamma,
        reward_list,
        CHOSEN_AXIS,
        device,
    )


def fixed_adv(temp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_states = 2
    n_actions = 2
    gamma = 0.9
    CHOSEN_AXIS = 0

    T = torch.rand((n_states, n_actions, n_states)).mul(temp).softmax(dim=-1).to(device=device)
    mu = torch.rand((n_states,)).mul(temp).softmax(dim=-1).to(device=device)

    true_R = torch.rand((n_states, n_actions)).mul(temp).to(device=device)
    true_R = (true_R.exp() / true_R.exp().sum()) * max(10 - temp, 1)

    proxy_R = torch.rand((n_states, n_actions)).mul(temp).to(device=device)
    proxy_R = (proxy_R.exp() / proxy_R.exp().sum()) * max(10 - temp, 1)

    reward_dict = {
        "True Reward": true_R,
        "Proxy Reward": proxy_R,
    }

    plot_state_action_occupancies(
        f"Random with temp {1 / temp:.2f}",
        n_states,
        n_actions,
        T,
        mu,
        gamma,
        reward_dict,
        CHOSEN_AXIS,
        device,
        rl_algo={"PPO": ppo_tabular, "Fixed Adv": ppo_fixed_adv_tabular},
    )


if __name__ == "__main__":
    star(2)
