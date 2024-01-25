import math
import random
import numba


@numba.njit(parallel=False)
def direct_gillespie(N, beta, mu):
    """_summary_

    Parameters
    ----------
    N : int
        Population size
    beta : float
        Rate of infection
    mu : float
        Rate of recovery
    """
    
    t = [0]  # initial time
    I = [1]  # initially one person is infected
    S = [N - 1]  # initially all but one is susceptible
    R = [0]  # initially nobody is recovered
    
    while I[-1] > 0:
        infection_rate = beta*I[-1]*S[-1]/N
        removal_rate = mu*I[-1]
        rate_sum = infection_rate + removal_rate
          
        wait_time = -math.log(random.random())/rate_sum
          
        if random.random() < infection_rate/rate_sum:  # infection
            S.append(S[-1] - 1)
            I.append(I[-1] + 1)
            R.append(R[-1])
        else:  # removal
            S.append(S[-1])
            I.append(I[-1] - 1)
            R.append(R[-1] + 1)
        
        t.append(t[-1] + wait_time)
    
    return t, I, R[-1]
          