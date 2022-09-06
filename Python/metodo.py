# -*- coding: utf-8 -*-
# Autor: Sergio P
# Data: 22/04/2022

from pandas import DataFrame, Series, ExcelWriter, read_excel
from pandas import concat, read_excel, to_numeric

from itertools import combinations

from gerador_graficos import GeradorGraficos
from util import progress_bar, cvrsme
from modelo import Modelo

import warnings
from re import sub


class Metodo():
    '''
    Classe para comparação entre diferentes modelos de auto-regressão.
    '''

    def __init__(self, conf: dict):
        '''
        Inicialização de instância da classe Metodo
        '''
        self.__conf = conf

        # Argumentos do modelo
        self.__municipio = None
        self.__explicada = conf['Modelo']['Variaveis']['Explicada']
        self.__explicativas = conf['Modelo']['Variaveis']['Explicativas']

        # Gráficos
        self.__gg = GeradorGraficos(conf)

        # Estatísticas
        tabelas_dir = conf['Resultados']['dir']/'Tabelas'
        tabelas_dir.mkdir(exist_ok=True)
        self._tab_xlsx = tabelas_dir/'default.xlsx'

    def set_municipio(self, municipio) -> None:
        '''
        ### Descrição
        Define um municipio para filtrar variáveis
        e calcular modelo de regressão
        '''
        self.__municipio = municipio

    def processa_regressao_modelo(self) -> None:

        # Dataframes de resultados
        predictions_df = DataFrame()  # Previsões
        statistics_df = DataFrame()  # Tabela de estatísticas

        # Função que adiciona uma nova predição no df de imagem
        def append_prediction(
            predictions: DataFrame,
            new: DataFrame | Series,
            label: str
        ) -> DataFrame:
            new.rename(label, inplace=True)
            predictions = concat([predictions, new], axis='columns')
            return predictions

        # Modelo
        modelo = Modelo(self.__conf)
        modelo.municipio = self.__municipio

        # Modelo AR
        completou = modelo.AR()

        if completou:

            # Adiciona predição no df de imagem final
            predictions_df = append_prediction(
                predictions_df,
                modelo.prediction,
                'AR'
            )

            # print(modelo.metric_values)

            # Dados estatísticos
            for metrica in modelo.metric_values:
                statistics_df.loc[f'AR', metrica] = modelo.metric_values[metrica]

        # Força bruta em modelos VAR
        testes = dict()
        count_testes = 1
        for l in range(1, len(self.__explicativas)+1):
            for subset in combinations(self.__explicativas, l):
                var_total = list(subset)
                testes[f'Teste{count_testes}'] = var_total
                count_testes += 1

        # Para testar cada variável individualmente
        testes = {
            f'Teste_{n}': [var]
            for n, var in enumerate(self.__explicativas)
        }

        # Dicionário de melhores modelos
        melhores = {
            metrica: (None, None) # Teste_n, valor
            for metrica in modelo.all_metrics
        }

        # Barra de progresso
        bar_progress = 0
        bar_total = len(testes)

        # Procura os melhores modelos dentre os testes
        for teste, features in testes.items():
            # Printa e avança barra
            progress_bar(bar_progress, bar_total)
            bar_progress += 1

            # Modelo VAR
            completou = modelo.VAR(features)

            # Dados estatísticos

            row = f'VAR({teste})'
            for metrica in modelo.metric_values:
                statistics_df.loc[row, metrica] = modelo.metric_values[metrica]
            for feat in self.__explicativas:
                statistics_df.loc[row, str(feat)] = str(
                    feat in modelo.variaveis)

            if completou:
                # Adiciona predição no df de imagem final
                predictions_df = append_prediction(
                    predictions_df,
                    modelo.prediction,
                    row
                )

                # Verifica se é o melhor para todas as métricas
                for metric, reg in melhores.items():
                    if metric in modelo.metric_values:
                        new_reg = (teste, modelo.metric_values[metric])
                        # Caso não tenha nenhum registro
                        if reg == (None, None):
                            melhores[metric] = new_reg
                        # Caso tenha
                        else:
                            if modelo.metric_senses[metric] == 'min':
                                if reg[1] > modelo.metric_values[metric]:
                                    melhores[metric] = new_reg
                            else:
                                if reg[1] < modelo.metric_values[metric]:
                                    melhores[metric] = new_reg

        # Barra 100%
        progress_bar(bar_progress, bar_total)

        # Mantém nas previões apenas os melhores modelos
        c_keep = ['AR']
        for metric, reg in melhores.items():
            new_c = f'$VAR_{{{metric.upper()}}}$'
            c_keep.append(new_c)
            predictions_df[new_c] = predictions_df[f'VAR({reg[0]})']

        predictions_df = predictions_df[c_keep]

        # Tratamento final de estatísticas
        path = self._tab_xlsx.with_stem(f'Estatísticas_{self.__explicada}')
        xlsx_kwargs = dict(
            path=path,
            engine='openpyxl'
        )
        if path.is_file():
            # Caso já exista o arquivo, append
            xlsx_kwargs['mode'] = 'a'
            xlsx_kwargs['if_sheet_exists'] = 'replace'
        else:
            # Caso não exista, cria
            xlsx_kwargs['mode'] = 'w'

        with ExcelWriter(**xlsx_kwargs) as writer:
            # Escreve os dados na aba do município
            statistics_df.to_excel(
                excel_writer=writer,
                sheet_name=self.__municipio,
                float_format='%.2f'
            )

        # Salva imagem
        self.__gg.line_predictions(
            concat(
                [modelo.data[self.__explicada], predictions_df],
                axis='columns'
            ),
            file_suffix=f'{self.__municipio}',
            folder=f'Estimativas {self.__explicada.replace("_"," ")}'
        )

    def estatisticas_testes(self):
        # Resumo da tabela de estatísticas
        data_dict = read_excel(
            self._tab_xlsx.with_stem(f'Estatísticas_{self.__explicada}'),
            sheet_name=None,
            index_col=0
        )
        df = concat(data_dict.values(), keys=data_dict.keys())

        # Parâmetros para salvar
        n_levels = df.index.nlevels
        n_metrics = len(Modelo(self.__conf).all_metrics)
        models = [x for x in df.index.unique(level=1) if x != 'AR']
        # models.append('cv')
        # models.append('aic')
        groups = [(True, 'Regiões'), (False, 'Municípios')]

        # Variáveis e funções do cálculo de erro
        def filter(df, level, value, model, regiao):
            # Não sei se tem jeito melhor... :D
            def contains(regiao, x):
                if regiao:
                    return 'Região' in x
                else:
                    return ('Região' not in x) and \
                           (x not in ['Municípios','Regiões'])
            df_filter = df[df.index.get_level_values(0).isin([
                    x
                    for x in list(df.index.get_level_values(0))
                    if contains(regiao, x)
                ])
            ]
            if model in ['cv', 'aic']:
                df_filter = df_filter.iloc[
                    df_filter.index.get_level_values(1) != 'AR'
                ]
                idx_to_keep = list()
                for name, df in df_filter[model].groupby(level=0):
                    idx_to_keep.append(to_numeric(df).idxmin())
                return df_filter[value].loc[idx_to_keep]
            else:        
                return df_filter[value].\
                       iloc[df_filter[value].
                       index.get_level_values(level) == model]

        for regiao, nome in groups:
            df_ar = filter(df, 1, 'cv', 'AR', regiao)
            for model in models:
                df_var = filter(df, 1, 'cv', model, regiao)
                diff = df_var.reset_index(drop=True)
                diff -= df_ar.reset_index(drop=True)
                # cell_text = f'{diff.mean():.2f} \u00B1 {diff.std():.2f}'
                df.loc[(nome, model), 'media'] = diff.mean()
                df.loc[(nome, model), 'desvio'] = diff.std()
                df.loc[(nome, model), 'erro padrão'] = diff.sem()

        # for idx, col in enumerate(df):
            # Contagem de escolha das variáveis
            # if idx >= n_metrics:
            #     df.loc[('Contagens', 'VERDADEIRO'), col] = len(
            #         df[df[col] == True])
            #     df.loc[('Contagens', 'FALSO'), col] = len(
            #         df[df[col] == False])
            # Erro do melhor VAR em relação ao AR

        # Gerar imagens dos resumos
        df_resume = df.loc[
            [x for (_,x) in groups]
        ][['media','erro padrão']].apply(to_numeric)
        for name, data in df_resume.groupby(level=0):
            self.__gg.resume(name, data.droplevel(level=0))

        # Salva em arquivo novo
        tab_file_name = f'Estatísticas_Resumo_{self.__explicada}'
        with ExcelWriter(
            path=self._tab_xlsx.with_stem(tab_file_name),
            mode='w',
            engine='xlsxwriter'
        ) as writer:
            df.to_excel(
                excel_writer=writer,
                sheet_name='Resumo Estatísticas',
                float_format='%.2f'
            )
            workbook = writer.book
            worksheet = writer.sheets['Resumo Estatísticas']

            # Formatações
            # Add a format. Light red fill with dark red text.
            format1 = workbook.add_format({
                'bg_color': '#FFC7CE',
                'font_color': '#9C0006'
            })

            # Add a format. Green fill with dark green text.
            format2 = workbook.add_format({
                'bg_color': '#C6EFCE',
                'font_color': '#006100'
            })

            # Booleanos
            def bool_format(value, format):
                worksheet.conditional_format(
                    0,
                    n_levels + n_metrics,
                    len(df),
                    n_levels + n_metrics + len(self.__explicativas) - 1,
                    dict(
                        type='cell',
                        criteria='=',
                        value=value,
                        format=format
                    )
                )

            bool_format(False, format1)
            bool_format(True, format2)

            # Largura de colunas
            for idx, col in enumerate(df):
                series = df[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # largest item
                    len(str(series.name)),  # column name/header
                    len('VERDADEIRO')+2  # hahaha
                ))
                worksheet.set_column(
                    idx + n_levels,
                    idx + n_levels,
                    max_len + 1  # lil bit extra
                )

            # Formatação manual dos index
            worksheet.set_column(0, 0, 20)  # Cidades
            worksheet.set_column(1, 1, 16)  # Testes

    def imagens(self):
        # Imagens de features
        all_features = self.__explicativas
        all_features.append(self.__explicada)
        self.__gg.features(['Vendas_Total_por_unidade'])
        # self.__gg.features_grids()
        # self.__gg.variacao('PIB')
