import numpy as np
import math

def normal_fdp(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Calcula a função densidade de probabilidade (FDP) para uma distribuição normal.

    A fórmula da FDP da distribuição normal é dada por:
    FDP(x; μ, σ) = (1 / (σ * sqrt(2 * π))) * exp(-0.5 * ((x - μ) / σ) ** 2)

    Parâmetros
    ----------
    x : np.ndarray
        O conjunto de valores para os quais a FDP será calculada.
    mu : float
        A média da distribuição normal.
    sigma : float
        O desvio padrão da distribuição normal.

    Retorna
    -------
    np.ndarray
        O valor da FDP para cada valor no conjunto x dado.
    """
    coeficiente = 1.0 / (sigma * np.sqrt(2 * np.pi))
    expoente = -0.5 * ((x - mu) / sigma) ** 2
    return coeficiente * np.exp(expoente)

def exponencial_fdp(x: np.ndarray, lamb: float) -> np.ndarray:
    """
    Calcula a função densidade de probabilidade (FDP) para uma distribuição exponencial.

    A fórmula da FDP da distribuição exponencial é dada por:
    FDP(x; λ) = λ * exp(-λ * x)

    Parâmetros
    ----------
    x : np.ndarray
        O conjunto de valores para os quais a FDP será calculada.
    lamb : float
        O parâmetro de taxa da distribuição exponencial.

    Retorna
    -------
    np.ndarray
        O valor da FDP para cada valor no conjunto x dado.
    """
    return lamb * np.exp(-lamb * x)

def exponencial_fda(x: np.ndarray, lamb: float) -> np.ndarray:
    """
    Calcula a função de distribuição acumulada (FDA) para uma distribuição exponencial.

    A fórmula da FDA da distribuição exponencial é dada por:
    FDA(x; λ) = 1 - exp(-λ * x)

    Parâmetros
    ----------
    x : np.ndarray
        O conjunto de valores para os quais a FDA será calculada.
    lamb : float
        O parâmetro de taxa da distribuição exponencial.

    Retorna
    -------
    np.ndarray
        O valor da FDA para cada valor no conjunto x dado.
    """
    return 1 - np.exp(-lamb * x)

def binomial_fmp(n: np.ndarray, N: int, p: float) -> np.ndarray:
    """
    Calcula a função de massa de probabilidade (FMP) para uma distribuição binomial.

    A fórmula da FMP da distribuição binomial é dada por:
    P_N(n) = (N! / (n! * (N - n)!)) * p^n * (1 - p)^(N - n)

    Parâmetros
    ----------
    n : np.ndarray
        O conjunto de valores (número de sucessos) para os quais a FMP será calculada.
    N : int
        O número de experimentos.
    p : float
        A probabilidade de sucesso em cada experimento.

    Retorna
    -------
    np.ndarray
        O valor da FMP para cada valor no conjunto n dado.
    """
    q = 1 - p
    coef_binomial = np.array([math.comb(N, ni) for ni in n])
    return coef_binomial * (p ** n) * (q ** (N - n))

