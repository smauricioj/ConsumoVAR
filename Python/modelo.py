# -*- coding: utf-8 -*-
# Autor: Sergio P
# Data: 27/05/2022

from pandas import DataFrame, DatetimeIndex, Series
from pandas import read_excel, concat

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", ConvergenceWarning)

from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.ar_model import AutoReg

from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.tsa.stattools import grangercausalitytests

from sklearn.preprocessing import MinMaxScaler

from util import cvrsme


class Modelo():
    ''' Classe para implementação de modelos de auto-regressão '''

    def __init__(self, conf: dict):
        ''' Inicialização de instância da classe Modelo '''
        self._conf = conf

        # Séries temporais
        self._data = None
        self._trein_df = None
        self._valid_df = None
        self._prediction = None

        # Propriedades do modelo
        self._municipio = None
        self._variaveis = None
        self._p_start = None
        self._p_end = None

        # Configurações do modelo
        self._explicada = conf['Modelo']['Variaveis']['Explicada']
        self._explicativas = conf['Modelo']['Variaveis']['Explicativas']
        self._param = conf['Modelo']['Parametros']
        self._scale = True
        self._scaler = MinMaxScaler()

        # Métricas de comparação
        self._metrics = dict(
            aic = {
                'value':None,
                'sense':'min'
            },
            cv = {
                'value':None,
                'sense':'min'
            }
        )

        # Teste Granger
        self.__g_test = 'ssr_chi2test'
        self.__g_maxlag = 2

    # Séries temporais
    @property
    def prediction(self) -> DataFrame:
        '''
        ### Descrição
        Propriedade que representa a previsão realizada para o
        período de validação + horizonte
        '''
        return self._prediction

    @property
    def data(self) -> DataFrame:
        '''
        ### Descrição
        Propriedade que representa a todas as séries temporais
        das features para regressão
        '''
        return self._data

    # Propriedades do modelo

    @property
    def municipio(self) -> str:
        '''
        ### Descrição
        Propriedade que define um municipio para filtrar variáveis
        e calcular modelo de regressão.

        Limpa todas as variáveis já armazenadas
        '''
        return self._municipio

    @municipio.setter
    def municipio(self, municipio: str) -> None:
        print(f'Para o municipio: {municipio}')
        self._municipio = municipio
        self._data = DataFrame()  # Deleter de data?

    @property
    def variaveis(self) -> list:
        '''
        ### Descrição
        Propriedade que define um quais variáveis usar
        no modelo de regressão

        Verificadas em relação à explicativa e explicadas
        do arquivo de configuração

        Valor None representa que ainda não foi ajustado
        '''
        return self._variaveis

    @variaveis.setter
    def variaveis(self, mode: str | list) -> None:
        all_var = [v for v in self._explicativas]
        all_var.append(self._explicada)
        accepted_modes = dict(
            AR=[self._explicada],
            all=all_var
        )

        if isinstance(mode, str):
            if mode not in accepted_modes:
                raise ValueError('Modo de variáveis desconhecido.')
            else:
                var = accepted_modes[mode]
        else:
            if set(mode).issubset(set(all_var)):
                if self._explicada not in mode:
                    mode.append(self._explicada)
                var = mode
            else:
                raise ValueError('Variável desconhecida.')

        self._variaveis = var

    @variaveis.deleter
    def variaveis(self) -> None:
        ''' Limpa variáveis '''
        self._variaveis = None

    # Métricas de comparação
    @property
    def metric_values(self):
        # print(self._metrics)
        return {
            k: v['value']
            for k, v in self._metrics.items()
            if v['value'] != None
        }

    @property
    def metric_senses(self):
        return {
            k: v['sense']
            for k, v in self._metrics.items()
        }

    @property
    def all_metrics(self):
        return self._metrics.keys()

    # Métodos internos
    def _set_data(self) -> None:
        '''
        ### Descrição

        Método para instancializar as variáveis
        do modelo de regressão
        '''
        assert isinstance(self._municipio, str)

        # Limpeza de dados

        self._data = DataFrame()
        self._trein_df = DataFrame()
        self._valid_df = DataFrame()
        self._prediction = DataFrame()

        # Procura todas as tabelas, remove indesejadas

        bulk = list(self._conf['Dados']['dir'].glob('**/*.xlsx'))

        if isinstance(self._variaveis, list):
            bulk = [fp for fp in bulk if fp.stem in self._variaveis]
        else:
            bulk = [fp for fp in bulk if '_old' not in fp.stem]

        for fp in bulk:
            # Leitura dos dados da feature

            df = read_excel(
                fp,
                index_col=0,
                decimal=',',
                parse_dates=True
            )

            # Filtra para apenas um município

            feat = df.loc[:, (self._municipio)]
            feat.rename(fp.stem, inplace=True)
            feat.rename_axis(index='Ano', inplace=True)

            # Adiciona os dados na instância

            self._data = concat((self._data, feat), axis='columns')

        # Index temporal

        self._data.index = DatetimeIndex(
            self._data.index.values,
            freq='infer',
            yearfirst=True
        )

        # Separa treinamento e validação

        valid_size = self._param['reg_valid_size']
        self._trein_df = self._data[:-valid_size].copy(deep=True)
        self._valid_df = self._data[-valid_size:].copy(deep=True)

        # Escala normalizada

        if self._scale:
            self._scaler = self._scaler.fit(self._trein_df)
            self._trein_df = DataFrame(
                self._scaler.transform(self._trein_df),
                index=self._trein_df.index,
                columns=self._trein_df.columns
            )

        # Limites temporais da predição

        self._p_start = len(self._trein_df)
        self._p_end = (len(self._trein_df) +
                       self._param['reg_valid_size'] +
                       self._param['reg_horizonte'] - 1
                       )

    def _update_metrics(self, f_model, prediction) -> None:
        for metric in self.all_metrics:
            try:
                self._metrics[metric]['value'] = f_model.__getattribute__(metric)
            except AttributeError:
                self._metrics[metric]['value'] = None
        # Estimativa calculada fora
        self._prediction = prediction

        # CVRSME calculado em função à validação
        self._metrics['cv']['value'] = cvrsme(
            prediction[0:len(self._valid_df)],
            self._valid_df[self._explicada]
        )

        # LLF_r é LLF dividido pelo número de explicativas usadas
        if 'llf' in self._metrics:
            if self._metrics['llf'] != None:
                self._metrics['llf_r'] = self._metrics['llf'] / \
                    max((len(self._variaveis) - 1), 1)
            else:
                self._metrics['llf_r'] = None

    def _estacionario(self, data: DataFrame) -> (DataFrame, int):
        '''
        ### Descrição

        Método que diferencia um conjunto de séries temporais
        até que todas sejam estacionárias.
        O número de diferenciações é limitado em 2.

        Retorna as séries estacionárias e o número de diferenciações.
        '''

        # Funções de teste

        def kpss_feats(df) -> bool:
            '''
            Teste KPSS para hipótese de série temporal estacionária
            Para todas as colunas de df
            Testa e armazena o valor-p
            Se qualquer valor-p for < 0.05, alguma série é não-estacionária
            '''
            p_values = list()
            for feat in df.columns:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    _, p_value, lags, _ = kpss(
                        array(df[feat]),
                        nlags=3
                    )
                    p_values.append(p_value)
            return any([p_value < 0.05 for p_value in p_values])

        def adf_feats(df) -> bool:
            '''
            Teste ADF para hipótese de série temporal estacionária
            Para todas as colunas de df
            Testa e armazena o valor-p
            Se qualquer valor-p for > 0.05, alguma série é não-estacionária
            '''
            p_values = list()
            for feat in df.columns:
                _, p_value, lags, _, _, _ = adfuller(
                    array(df[feat]),
                    autolag='t-stat'
                )
                p_values.append(p_value)
            return any([p_value > 0.05 for p_value in p_values])

        # Diferencia até todas as colunas ficarem estacionárias

        count_diffs = 0

        while (adf_feats(data) or kpss_feats(data)) and count_diffs < 2:
            count_diffs += 1
            data = data.diff().dropna()

        # Retorna dados estacionários

        return (data, count_diffs)

    # Metodos
    def VAR(self, feats=None):
        # Insere variáveis na tabela de dados

        if feats == None:
            assert isinstance(self.variaveis, list)
            assert len(self.variaveis) >= 2
        else:
            self.variaveis = feats
        self._set_data()

        # Modelo VARMAX para estacionaridade

        model = VARMAX(
            self._trein_df,
            order=(self._param['VAR_lags'], 0),
            trend=self._param['VAR_trend'],
            enforce_stationarity=True
        )
        try:
            f_model = model.fit(
                disp=False,
                maxiter=self._param['VAR_fit_maxiter']
            )
        except Exception as exc:
            # Se não conseguir ajustar, retorna Falso
            print(exc)

            for metric in self.all_metrics:
                self._metrics[metric] = None
            return False
        else:
            # Se conseguir ajustar, atualiza métricas e retorna True

            predict = f_model.get_prediction(
                start=self._p_start,
                end=self._p_end
            )
            prediction = predict.predicted_mean
            if self._scale:
                pred_values = self._scaler.inverse_transform(prediction)
                prediction = DataFrame(
                    data=pred_values,
                    index=prediction.index,
                    columns=prediction.columns
                )
            self._update_metrics(f_model, prediction[self._explicada])
            return True

    def AR(self):
        # Insere variáveis na tabela de dados
        self.variaveis = 'AR'
        self._set_data()

        # Modelo de Auto Regressão
        model = AutoReg(
            self._trein_df,
            lags=self._param['AR_lags'],
            trend=self._param['AR_trend']
        )
        try:
            f_model = model.fit()
        except:
            # Se não conseguir ajustar, retorna Falso

            for metric in self.all_metrics:
                self._metrics[metric] = None
            return False
        else:
            # Se conseguir ajustar, atualiza métricas e retorna True

            predict = f_model.get_prediction(
                start=self._p_start,
                end=self._p_end
            )
            prediction = predict.predicted_mean
            if self._scale:
                pred_values = self._scaler.inverse_transform(
                    DataFrame(prediction)
                )
                prediction = Series(
                    data=[v[0] for v in pred_values],
                    index=prediction.index
                )
            self._update_metrics(f_model, prediction)
            return True

    def teste_granger(self) -> None:
        '''
        ### Descrição
        Método para testar todas as séries temporais nas suas
        relações de causalidade Granger.

        Os resultados são printados na tela.
        '''
        features = self._data.columns
        granger_df = DataFrame(
            zeros((len(features), len(features))),
            columns=features,
            index=features
        )
        for c in granger_df.columns:
            for r in granger_df.index:
                if c == r:
                    granger_df.loc[r, c] = 1.0
                else:
                    test_r = grangercausalitytests(
                        self._data[[r, c]],
                        maxlag=self.__g_maxlag,
                        verbose=False
                    )
                    p_values = [round(test_r[i+1][0][self.__g_test][1], 4)
                                for i in range(maxlag)]
                    min_p_value = min(p_values)
                    granger_df.loc[r, c] = min_p_value
                    if (min_p_value < 0.05):
                        texto = f'{c} é causa-G de {r} '
                        texto += f'com p_value {min_p_value}'
                        print(texto)

        granger_df.columns = [var + '_x' for var in features]
        granger_df.index = [var + '_y' for var in features]
        print(granger_df)
