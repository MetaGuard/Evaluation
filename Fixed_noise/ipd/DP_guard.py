# Source: https://diffprivlib.readthedocs.io/en/latest/modules/mechanisms.html?highlight=bounded#diffprivlib.mechanisms.LaplaceBoundedDomain
from diffprivlib.mechanisms.laplace import LaplaceBoundedDomain

def Laplace_Bounded_Mechanism(epsilon, sensitivity, lower, upper, deterministic_value):
    
    # Initialize differential privacy mechanism
    LapBoundMech = LaplaceBoundedDomain(epsilon=epsilon, sensitivity=sensitivity, lower=lower, upper=upper)

    # Produce differentially private output
    noisy_output = LapBoundMech.randomise(deterministic_value)
    return noisy_output

