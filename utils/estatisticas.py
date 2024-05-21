from typing import List

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
