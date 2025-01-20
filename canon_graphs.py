import matplotlib.pyplot as plt
import torch

from helper_local import norm_funcs


def scatter(canon_logp, canon_true_r, straight=False, title=""):
    if straight:
        # plt.scatter(x=canon_logp.detach().cpu().numpy(),
        #             y=canon_logp.detach().cpu().numpy(),
        #             c='black',
        #             )
        plt.scatter(x=canon_true_r.detach().cpu().numpy(),
                    y=canon_true_r.detach().cpu().numpy(),
                    c='black',
                    )
    plt.scatter(x=canon_logp.detach().cpu().numpy(),
                y=canon_true_r.detach().cpu().numpy(),
                )
    plt.xlabel('logp')
    plt.ylabel('rew')
    plt.title(title)
    # plt.savefig("data/ascent_canon_norm_scatter_train.png")
    plt.show()

# distance(normalize(canon_logp), normalize(canon_true_r))

# scatter(canon_logp, canon_true_r, straight=True)

normalize = norm_funcs["l2_norm"]

def main(logp, rew, canon_logp, canon_true_r):
    env_name = "Coinrun"
    scatter(logp, rew, straight=True,
            title=f'{env_name} Evaluation environment')
    scatter(canon_logp, canon_true_r, straight=True,
            title=f'{env_name} Canonicalised, Evaluation environment')
    scatter(normalize(canon_logp), normalize(canon_true_r),straight=True,
            title=f'{env_name} Canonicalised, Normalized, Evaluation environment')

    ((normalize(canon_logp)-normalize(canon_true_r))**2).mean().sqrt()

    torch.corrcoef(torch.stack((logp, rew)))[0,1]
    torch.corrcoef(torch.stack((canon_logp, canon_true_r)))[0,1]
    torch.corrcoef(torch.stack((normalize(canon_logp), normalize(canon_true_r))))[0,1]

    flt = rew != 0
    torch.corrcoef(torch.stack((logp[flt], rew[flt])))[0,1]
    torch.corrcoef(torch.stack((canon_logp[flt], canon_true_r[flt])))[0,1]
    torch.corrcoef(torch.stack((normalize(canon_logp[flt]), normalize(canon_true_r[flt]))))[0,1]


