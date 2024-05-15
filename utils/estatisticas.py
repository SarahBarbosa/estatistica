from typing import List, Union
import numpy as np

def media_amostral(dados: np.ndarray) -> np.ndarray:
    """
    Calcula a média amostral de cada variável aleatória.

    Parâmetros
    ----------
    dados : numpy.ndarray
        Array NumPy contendo os dados.

    Retorna
    -------
    numpy.ndarray
        Array contendo as médias amostrais.
    """
    return np.mean(dados, axis=0)

def variancia_amostral(dados: np.ndarray) -> np.ndarray:
    """
    Calcula a variância amostral de cada variável aleatória.

    Parâmetros
    ----------
    dados : numpy.ndarray
        Array NumPy contendo os dados.

    Retorna
    -------
    numpy.ndarray
        Array contendo as variâncias amostrais.
    """
    return np.var(dados, axis=0, ddof=1)

def covariancia_amostral(dados1: np.ndarray, dados2: np.ndarray) -> float:
    """
    Calcula a covariância entre duas variáveis aleatórias.

    Parâmetros
    ----------
    dados1 : numpy.ndarray
        Array NumPy contendo os dados da primeira variável aleatória.
    dados2 : numpy.ndarray
        Array NumPy contendo os dados da segunda variável aleatória.

    Retorna
    -------
    float
        Covariância entre as duas variáveis aleatórias.
    """
    return np.cov(dados1, dados2, rowvar=False)[0, 1]

def correlacao_amostral(dados1: np.ndarray, dados2: np.ndarray) -> float:
    """
    Calcula o coeficiente de correlação entre duas variáveis aleatórias.

    Parâmetros
    ----------
    dados1 : numpy.ndarray
        Array NumPy contendo os dados da primeira variável aleatória.
    dados2 : numpy.ndarray
        Array NumPy contendo os dados da segunda variável aleatória.

    Retorna
    -------
    float
        Coeficiente de correlação entre as duas variáveis aleatórias.
    """
    cov = covariancia_amostral(dados1, dados2)
    dp_x = np.std(dados1, ddof=1)
    dp_y = np.std(dados2, ddof=1)
    return cov / (dp_x * dp_y)
