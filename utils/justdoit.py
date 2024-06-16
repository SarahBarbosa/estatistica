from typing import List
import numpy as np
import math
import scipy.stats as stats

def fatorial(n: np.ndarray) -> np.ndarray:
    """
    Calcula o fatorial de um número ou de um array de números 
    usando aritmética exata de inteiros.
    
    Definição:
    O fatorial de um número inteiro não negativo 'n' é o produto de 
    todos os inteiros positivos menores ou iguais a 'n':
    
        n! = n * (n - 1) * (n - 2) * ... * 1

    Se 'n' é 0, então n! = 1 por definição.
    
    Parâmetros
    ----------
    n : int ou np.ndarray de ints
        Valores de entrada. Se 'n' < 0, o valor de retorno é 0 para 
        esse elemento.
    
    Retorna
    -------
    fatorial : int ou np.ndarray de ints
        Fatorial de 'n', como inteiro ou array de inteiros.
    """
    n = np.asarray(n)
    resultado = np.zeros_like(n, dtype=object)
    
    for i, valor in np.ndenumerate(n):
        if valor < 0:
            resultado[i] = 0
        else:
            resultado[i] = math.factorial(valor)

    if resultado.size == 1:
        return resultado.item()
    return resultado

def media_amostral(dados: List[float]) -> float:
    """
    Calcula a média amostral de cada variável aleatória.

    Parâmetros
    ----------
    dados : list
        Lista contendo os dados.

    Retorna
    -------
    float
        Média amostral.
    """
    return sum(dados) / len(dados)

def variancia_amostral(dados: List[float]) -> float:
    """
    Calcula a variância amostral de cada variável aleatória.

    Parâmetros
    ----------
    dados : list
        Lista contendo os dados.

    Retorna
    -------
    float
        Variância amostral.
    """
    media = media_amostral(dados)
    return sum((xi - media) ** 2 for xi in dados) / (len(dados) - 1)

def covariancia_amostral(dados1: List[float], dados2: List[float]) -> float:
    """
    Calcula a covariância entre duas variáveis aleatórias.

    Parâmetros
    ----------
    dados1 : list
        Lista contendo os dados da primeira variável aleatória.
    dados2 : list
        Lista contendo os dados da segunda variável aleatória.

    Retorna
    -------
    float
        Covariância entre as duas variáveis aleatórias.
    """
    media1 = media_amostral(dados1)
    media2 = media_amostral(dados2)
    return sum((dados1[i] - media1) * (dados2[i] - media2) for i in range(len(dados1))) / (len(dados1) - 1)

def correlacao_amostral(dados1: List[float], dados2: List[float]) -> float:
    """
    Calcula o coeficiente de correlação entre duas variáveis aleatórias.

    Parâmetros
    ----------
    dados1 : list
        Lista contendo os dados da primeira variável aleatória.
    dados2 : list
        Lista contendo os dados da segunda variável aleatória.

    Retorna
    -------
    float
        Coeficiente de correlação entre as duas variáveis aleatórias.
    """
    cov = covariancia_amostral(dados1, dados2)
    dp_x = variancia_amostral(dados1) ** 0.5
    dp_y = variancia_amostral(dados2) ** 0.5
    return cov / (dp_x * dp_y)

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

def poisson_fmp(n: np.ndarray, mu: float) -> np.ndarray:
    """
    Calcula a função de massa de probabilidade (FMP) para uma distribuição de Poisson.

    A fórmula da FMP da distribuição Poisson é dada por:
    P_N(n) = (μ^n / n!) * e^(-μ)

    Parâmetros
    ----------
    n : np.ndarray
        O conjunto de valores (número de sucessos) para os quais a FMP será calculada.
    mu : float
        A média da distribuição de Poisson.

    Retorna
    -------
    np.ndarray
        O valor da FMP para cada valor no conjunto n dado.    
    """
    if mu >= 12:  # Para mu >> 0, o código fica inviável (devido o fatorial)
        return stats.poisson.pmf(n, mu)

    return (mu ** n / fatorial(n)) * np.exp(-mu) 

def poisson_cdf(k: int, mu: float) -> float:
    """
    Calcula a função de distribuição acumulada (CDF) da distribuição de Poisson.

    Parameters
    ----------
    k : int
        Valor inteiro até onde a soma deve ser calculada.
    mu : float
        Média da distribuição de Poisson.

    Returns
    -------
    float
        Valor da CDF para k e mu dados.
    """
    p_total = 0.0
    for x in range(k + 1):
        p_total += poisson_fmp(x, mu)
    return p_total