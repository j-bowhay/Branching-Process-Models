import math
import random

from tqdm import tqdm
import numba
import numpy as np


@numba.njit(parallel=False)
def direct_gillespie_sir(N, beta, mu):
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


def run_ensemble_sir(N, beta, mu, number_of_sims):
    ts = []
    Is = []
    R_finals = []
    max_t = 0

    for _ in tqdm(range(number_of_sims)):
        t, I, R_final = direct_gillespie_sir(N, beta, mu)
        
        ts.append(t)
        Is.append(I)
        R_finals.append(R_final)
        
        if t[-1] > max_t:
            max_t = t[-1]
    
    return ts, Is, R_finals, max_t

@numba.njit(parallel=False)
def direct_gillespie_sir_partial_immunity(N, beta, k1, k2):
    
    t = [0]  # initial time
    I = [1]  # initially one person is infected
    S = [N - 1]  # initially all but one is susceptible
    R = [0]  # initially nobody is recovered
    infections = 0
    
    while I[-1] > 0:
        infection_rate = beta*I[-1]*S[-1]/N
        removal_rate = k1*I[-1]
        resusceptible_rate = k2*I[-1]
        rate_sum = infection_rate + removal_rate + resusceptible_rate
          
        wait_time = -math.log(random.random())/rate_sum

        choice = random.random()
          
        if choice < infection_rate/rate_sum:  # infection
            S.append(S[-1] - 1)
            I.append(I[-1] + 1)
            R.append(R[-1])
            infections += 1
        elif choice < (infection_rate+removal_rate)/rate_sum:  # removal
            S.append(S[-1])
            I.append(I[-1] - 1)
            R.append(R[-1] + 1)
        else: # transition back to being susceptible
            S.append(S[-1] + 1)
            I.append(I[-1] - 1)
            R.append(R[-1])
        
        t.append(t[-1] + wait_time)
    
    return t, I, infections


def run_ensemble_sir_partial_immunity(N, beta, k1, k2, number_of_sims):
    ts = []
    Is = []
    infections = []
    max_t = 0

    for _ in tqdm(range(number_of_sims)):
        t, I, infection = direct_gillespie_sir_partial_immunity(N, beta, k1, k2)
        
        ts.append(t)
        Is.append(I)
        infections.append(infection)
        
        if t[-1] > max_t:
            max_t = t[-1]
    
    return ts, Is, np.asarray(infections), max_t