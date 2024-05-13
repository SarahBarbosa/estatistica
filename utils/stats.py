import numpy as np

class Normal:
    @staticmethod
    def fdp(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
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


class Exponencial:
    @staticmethod
    def fdp(x: np.ndarray, lamb: float) -> np.ndarray:
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

    @staticmethod
    def fda(x: np.ndarray, lamb: float) -> np.ndarray:
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
    