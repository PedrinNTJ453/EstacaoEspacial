"""Transformadores customizados usados no pipeline (precisam estar num .py
para o pickle conseguir carregar o modelo depois, inclusive no Streamlit)."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ExtrairTitulo(BaseEstimator, TransformerMixin):
    """Cria a coluna 'titulo' (Cap., Ten., Eng., Dra., Passag.) a partir de
    'nome_tripulante' e descarta a coluna de nome."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'nome_tripulante' in X.columns:
            X['titulo'] = (
                X['nome_tripulante']
                .astype(str)
                .str.extract(r'^\s*([A-Za-zÀ-ÿ]+)\.')[0]
                .fillna('Outros')
            )
            X = X.drop(columns=['nome_tripulante'])
        return X
