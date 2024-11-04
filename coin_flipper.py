import itertools
import numpy as np

def coin_flipper():
    prob_of_h = 2/3
    max_tosses = 10

    # np.zeros((2**max_tosses, max_tosses))
    btab = np.array(list(itertools.product([True,False], repeat=10)))

    p_tab = np.ones_like(btab).astype(float)
    p_tab[btab] = prob_of_h
    p_tab[~btab] = 1-prob_of_h

    cum_probs = np.cumprod(p_tab, axis=1)
    cum_tab = np.cumsum(btab, axis=1)

    


    for n_tosses in reversed(range(10)):




    print(btab)
        

if __name__ == "__main__":
    coin_flipper()
    # coin_test(1000000)
    
    print("done")
