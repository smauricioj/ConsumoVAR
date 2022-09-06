# -*- coding: utf-8 -*-
# Autor: Sergio P
# Data: 25/04/2022

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from pandas import read_excel, DataFrame, concat
from pathlib import Path
from numpy import polyfit, poly1d


class GeradorGraficos():
    ''' Gera gráficos a partir de dataframes pandas '''
    file_preffix: str
    ''' Prefixo no nome dos arquivos '''
    file_suffix: str
    ''' Sufixo no nome dos arquivos '''
    directory: str
    ''' Pasta destino dos arquivos '''
    ext: str
    ''' Extensão usada para salvar arquivos '''
    unit_mapper: dict
    ''' Mapeamento entre variáveis e suas unidades '''

    def __init__(self, conf: dict):
        self._conf = conf
        self.directory = conf['Resultados']['dir']
        self.ext = conf['Resultados']['fig_ext']
        self.features_groups = conf['Resultados']['features_groups']
        self.file_preffix = ''
        self.file_suffix = ''
        self.unit_mapper = {
            # População
            'Populacao': 'habitantes',
            'Variação_de_Populacao': '% (2004)',
            # PIB
            'PIB_per_capita': 'mil R$ / habitante',
            'PIB_per_capita_Deflacionado': 'mil R$ (2004) / habitante',
            'PIB': 'mil R$',
            'PIB_Deflacionado': 'mil R$ (2004)',
            # Fota
            'Automoveis': 'veículos',
            'Motos': 'veículos',
            'Veiculos': 'veículos',
            'Índice_de_Motorização_Autos': 'veículos / 100 habitantes',
            'Índice_de_Motorização_Motos': 'veículos / 100 habitantes',
            'Índice_de_Motorização': 'veículos / 100 habitantes',
            'Auto_per_capita': 'veículos / habitante',
            'Moto_per_capita': 'veículos / habitante',
            'Variação_de_Automoveis': '% (2004)',
            'Variação_de_Motos': '% (2004)',
            # Preços
            'Precos_Gasolina': 'R$ / L',
            'Precos_Etanol': 'R$ / L',
            'Precos_Medio': 'R$ / L',
            'Precos_Gasolina_Deflacionado': 'R$ (2004) / L',
            'Precos_Etanol_Deflacionado': 'R$ (2004) / L',
            'Precos_Medio_Deflacionado': 'R$ (2004) / L',
            'Razão_dos_Preços': '%',
            # Consumo
            'Vendas_Total': 'ML',
            'Vendas_Gasolina': 'ML',
            'Vendas_Etanol': 'ML',
            'Índice_de_Consumo_Populacional': 'L / habitantes',
            'Vendas_Total_por_unidade': 'L / veículo',
            'Vendas_Gasolina_por_unidade': 'L / veículo',
            'Vendas_Etanol_por_unidade': 'L / veículo'
        }

    def _prep_and_save(func):
        def inner(self, *args, **kwargs):
            if 'file_suffix' in kwargs:
                self.file_suffix = kwargs['file_suffix']
            else:
                self.file_suffix = ''
            func(self, *args, **kwargs)
            plt.tight_layout()
            file_name = f'{self.file_preffix}_{self.file_suffix}{self.ext}'
            file = Path(self.directory/file_name)
            if file.parent.is_dir():
                pass
            else:
                file.parent.mkdir(parents=True)
            plt.savefig(file)
            plt.close()
            return func
        return inner

    @_prep_and_save
    def line_predictions(self, df, *, file_suffix: str, folder: str):
        self.file_preffix = f'{folder}/predictions'
        df.rename(
            columns=lambda x: 'Série Histórica' if 'Vendas' in x else x,
            inplace=True
        )
        # Gráfico de validação cruzada
        # df.rename(
        #     columns={'AR':'Estimação'}, inplace=True
        # )
        # plt.fill_between(
        #     df.index.values,
        #     df['Estimação'],
        #     df['Série Histórica'],
        #     facecolor='green',
        #     alpha=0.2,
        #     label='Erro'
        # )
        df.plot.line(
            grid=True,
            rot=30
        )

        plt.legend(loc='best')

        def arrow(text, pos_x, pos_y_offset, linestyle):
            plt.axvline(
                pos_x,
                color='black',
                linestyle=linestyle
            )

            # Calculando posição y relativa ao '0' do eixo vertical
            pos_y_diff = plt.gca().get_ylim()[1]
            pos_y_diff -= plt.gca().get_ylim()[0]
            pos_y = plt.gca().get_ylim()[0] + pos_y_diff*pos_y_offset

            # Plot da seta
            plt.text(
                pos_x, pos_y, text,
                horizontalalignment='right',
                verticalalignment='bottom',
                bbox=dict(
                    boxstyle='larrow',
                    fc='w',
                    ec='0.5',
                    alpha=0.9
                )
            )

        # Encontra a região de interseção
        s = df.copy().dropna()

        arrow('Treinamento', s.index[0], 0.01, linestyle=':')
        arrow('Validação', s.index[-1], 0.01, linestyle=':')
        plt.xlabel('Ano')
        plt.ylabel('Vendas (ML)')

    def features(self, variaveis=None):
        # Procura todas as features, remove indesejadas
        self.file_preffix = 'Features/feature'
        bulk = list(self._conf['Dados']['dir'].glob('**/*.xlsx'))
        if isinstance(variaveis, list):
            bulk = [fp for fp in bulk if fp.stem in variaveis]
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

            for group in self.features_groups:
                if fp.stem in self.unit_mapper:
                    y_label = self.unit_mapper[fp.stem]
                else:
                    y_label = 'default_unit'
                self._feature_group(
                    df[group['elementos']],
                    file_suffix='_'.join([fp.stem, group['nome']]),
                    title='',
                    y_label=y_label
                )

    @_prep_and_save
    def _feature_group(self, df,
                       *, file_suffix: str, title: str, y_label: str):
        # Características do plot
        df.rename(
            lambda x: x.replace('Região de ', ''),
            axis='columns',
            inplace=True
        )
        df.plot.line()
        # Plot da variação anual média
        d = polyfit(range(0,len(df)),df.mean(axis=1),1)
        f = poly1d(d)
        df.insert(len(df.columns),'Reg',f(range(0,len(df))))
        df['Reg'].plot.line(
            ls='-.',
            lw=2.2,
            color='black',
            legend=True,
            label=f'Variação anual +{d[0]:.2f}'
        )
        #
        plt.grid()
        plt.xlabel('Ano')
        plt.ylabel(y_label)

    def features_grids(self):
        # Procura todas as features, remove indesejadas
        bulk = list(self._conf['Dados']['dir'].glob('**/*.xlsx'))

        for feat_grid in self._conf['Resultados']['features_grids']:
            # Feature grids são conjuntos de gráficos para plotar em matriz
            # Procura todos e coloca em ordem alfabética
            var_fps = sorted(
                [fp
                 for fp in bulk
                 if fp.stem in feat_grid['elementos']],
                key=lambda fp: fp.stem
            )
            for group in self.features_groups:
                # Onde vai salvar
                self.file_preffix = f'Grids/{group["pasta"]}/feature'
                # Para cada região intermediária
                # Essa função existe pra salvar individualmente imagens
                self._feature_grid_group(
                    var_fps, group,
                    file_suffix='_'.join([feat_grid['nome'], group['nome']]),
                    feat_grid=feat_grid
                )

    @_prep_and_save
    def _feature_grid_group(self, var_fps, group,
                            *, file_suffix: str, feat_grid: dict):
        # Essa função existe pra salvar individualmente imagens
        # (prep and save)

        # Cria uma imagem com 2 colunas de subplots
        # grid_nrows = 2
        grid_ncols = 2
        grid_nrows = (len(var_fps) // grid_ncols)
        grid_nrows += (len(var_fps) % grid_ncols != 0)
        args = (grid_nrows, grid_ncols)
        kwargs = {
            'sharex': 'col',
            'dpi': 100
        }
        if 'figsize' in feat_grid:
            kwargs['figsize'] = [
                feat_grid['figsize'][0],
                feat_grid['figsize'][1]*grid_nrows
            ]
        else:
            kwargs['figsize'] = [6.4, 2.4*grid_nrows]
        fig, axes = plt.subplots(*args, **kwargs)

        # Para cada variável
        for var_count, fp in enumerate(var_fps):
            var_df = read_excel(
                fp,
                index_col=0,
                decimal=',',
                parse_dates=True
            )

            # Filtra apenas os municípios da região
            var_df = var_df[group['elementos']]
            var_df.rename(
                lambda x: x.replace('Região de ', ''),
                axis='columns',
                inplace=True
            )

            # Procura a unidade no mapper
            if fp.stem in self.unit_mapper:
                y_label = self.unit_mapper[fp.stem]
            else:
                y_label = 'default_unit'

            # Cada subplot tem um título próprio
            title = fp.stem.replace('_', ' ')
            if 'replacements' in feat_grid:
                for replacement in feat_grid['replacements'].values():
                    title = title.replace(*replacement)
            title = title.replace('Precos', 'Preços')

            # Calcula qual eixo dos subplots usar pela ordem das variáveis
            grid_row = var_count // grid_ncols
            grid_col = var_count % grid_ncols
            try:
                ax = axes[grid_row, grid_col]
            except IndexError:
                ax = axes[grid_col]

            # Plota
            var_df.plot(
                ax=ax,
                legend=False,
                ylabel=y_label
            )

            # Edição dos eixos
            ax.grid(
                axis='both',
                linestyle='--',
                linewidth=0.35
            )
            ax.set_title(
                title,
                fontdict={
                    'fontsize': 10
                }
            )
            ax.ticklabel_format(
                axis='y',
                style='sci',
                scilimits=[-2, 4],
                useMathText=True
            )

            if grid_row + 1 == grid_nrows:  # Ultima linha de gráficos
                ax.tick_params(axis='x', rotation=35)
                ax.set_xlabel('Ano')

            if grid_col + 1 == grid_ncols:  # Ultima coluna de gráficos
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")

        # Título da imagem inteira
        # plt.suptitle(feat_grid['nome'])

        # Legenda (essa parte foi chata, não mexer)
        try:
            handles, labels = axes[0, 0].get_legend_handles_labels()
        except IndexError:
            handles, labels = axes[0].get_legend_handles_labels()
        if 'bottom_space' in feat_grid:
            bottom_space = feat_grid['bottom_space']
        else:
            bottom_space = 0.34
        bottom_space -= (0.06*grid_nrows)  # deveria ser exponencial
        plt.legend(handles, labels,
                   loc='lower center',
                   mode='expand',
                   ncol=3,
                   bbox_to_anchor=(0, 0, 1, 1),
                   bbox_transform=plt.gcf().transFigure,
                   borderaxespad=0.3
                   )
        plt.subplots_adjust(wspace=0, bottom=bottom_space)

    @_prep_and_save
    def resume(self, name, data):
        self.file_preffix = f'resume_{name}'
        # print(data)
        data.rename(
            index={
                "VAR(Teste_0)": "IMA",
                "VAR(Teste_1)": "IMM",
                "VAR(Teste_2)": "PIB",
                "VAR(Teste_3)": "PMD",
                "cv": "\n $VAR_{{CV}}$",
                "aic": "\n $VAR_{{AIC}}$"
            },
            inplace=True
        )
        plt.errorbar(
            data.index,
            'media',
            yerr='erro padrão',
            fmt='o',
            mfc='black',
            capsize=10,
            data=data
        )
        plt.grid(which='major')
        for ax in plt.gcf().axes:
            ax.tick_params(axis='x', labelsize=10)
            ax.yaxis.set_minor_locator(tck.AutoMinorLocator(4))
            ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%d%%'))
        plt.grid(visible=True, which='minor', ls='--', lw=0.5)
        plt.xlabel(
            'Variável de interesse'
        )
        plt.ylabel(
            'Redução média do erro de estimação ($\\overline{\\Delta CV}$)'
        )
