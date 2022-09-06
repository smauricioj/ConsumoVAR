# -*- coding: utf-8 -*-
# Autor: Sergio P
# Data: 13/04/2022

from pandas import read_html, read_csv, read_excel
from pandas import DataFrame, ExcelWriter, DatetimeIndex
from pandas import to_numeric, concat

from urllib.request import urlopen
from urllib.parse import urlunparse

from pathlib import Path
from unidecode import unidecode
from json import load

from util import progress_bar


class Interfaces():
    '''
    Classe de interfaces com APIs, URLs ou outros pontos de acesso
    '''

    def __init__(self, conf: dict):
        '''
        Inicialização de instância da classe Interfaces
        '''
        self.__conf = conf

        # Período de pequisa
        self.begin = int(conf['Dados']['import_years']['begin'])
        self.end = int(conf['Dados']['import_years']['end'])

        # Placeholder pra atributos
        self.url_parts = None
        self.data_dir = None
        self.localidades = {}

        # Tabela locais e ids
        self.locals_df = read_csv(
            Path.cwd()/'conf'/'locals.csv',
            encoding='utf-8'
        ).astype({
            'Detran_SC': str,
            'IBGE': str
        })

        self.__import_groups = [
            {
                'nome': nome,
                'elementos': elementos
            } for nome, elementos in
            {
                name: list(group['nome'].values)
                for name, group in self.locals_df.groupby('grupo')
            }.items()
        ]

        # Mapper pra deixar tudo em unicode
        self.index_mapper = {
            unidecode(getattr(l, 'nome').capitalize()): getattr(l, 'nome')
            for l in self.locals_df.itertuples(index=False, name='l')
            if getattr(l, 'include')
        }

        for g in self.__import_groups:
            key = unidecode(g["nome"].capitalize())
            value = g["nome"]
            self.index_mapper[key] = value

        # Lista de interfaces implementadas
        self.interface_list = {
            'Detran_SC': self.getDataFromDetran,
            'IBGE': self.getDataFromIBGE,
            'ANP': self.getDataFromANP
        }

    def saveDataFrame(
        self,
        data: DataFrame | dict,
        agrupamentos=True,
        agrup_func='soma'
    ) -> bool:
        '''
        ### Descrição
        **Método interno**

        Método para salvar os dados em um arquivo .xlsx .

        **args**

        *df: DataFrame* -> dados a salvar

        *df: dict* -> dicionário de múltiplos dados a salvar

        *agrupamentos = True* -> se é para agrupar as cidades

        *agrup_func = 'soma'* -> qual função usar para agrupar
        dados das cidades. Aceita 'soma' e 'media'

        **return**

        *bool* -> True, se completou. False, se não
        '''

        # Sempre usa dicionário de dataframes.
        if isinstance(data, DataFrame):
            data = {'default': data}

        with ExcelWriter(self.data_path) as writer:
            # Para cada df, salva em sheet diferente
            for sheet, df in data.items():
                # Normaliza nomes das cidades
                df.index = df.index.str.capitalize()
                df.index = df.index.map(unidecode)
                df.rename(index=self.index_mapper, inplace=True)

                # Normaliza anos
                df.columns = df.columns.map(str)

                # Agrupamentos
                if agrupamentos:
                    for group in self.__import_groups:
                        n = group['nome']
                        e = group['elementos']
                        if agrup_func == 'soma':
                            df.loc[n] = df.loc[e].sum()
                        else:
                            df.loc[n] = df.loc[e].mean()

                # Ordena por anos e valores no ultimo ano
                df.sort_index(axis=0, inplace=True)
                df.sort_index(axis=1, inplace=True)

                # Inverte as tabelas pq eu fiz elas ao contrário :)
                df = df.stack().unstack(level=0)

                df.index = DatetimeIndex(
                    df.index.values,
                    freq='infer',
                    yearfirst=True
                )

                # Salva para excel
                df.to_excel(writer, sheet_name=sheet)

        return self.data_path.is_file()

    def setUp(self, dl_data: dict) -> None:
        '''
        ### Descrição
        ***Método interno***

        Método para ajustar atributos da classe
        durante pesquisa de acordo com a fonte de dados desejada

        **args**

        *dl_data: dict* -> diciónário de informações
        '''

        # Informações para urllib
        self.url_parts = None
        if dl_data['interface'] in self.__conf['Interfaces']:
            self.url_parts = self.__conf['Interfaces'][dl_data['interface']]

        # File paths
        self.data_dir = self.__conf['Dados']['dir']/dl_data['interface']
        self.data_path = self.data_dir/dl_data['file_name']
        if not self.data_dir.is_dir():
            self.data_dir.mkdir()
        if self.data_path.is_file():
            stem = self.data_path.stem+'_old'
            replace_name = self.data_path.with_stem(stem)
            self.data_path.replace(replace_name)

        # Dicionário com locais e ids
        self.localidades = {
            getattr(l, dl_data['interface']): getattr(l, 'nome')
            for l in self.locals_df.itertuples(index=False, name='l')
            if getattr(l, 'include')
        }

    def getData(self, dl_data: dict) -> bool:
        '''
        ### Descrição
        Método para buscar dados de acordo com
        o dicinário de informações.

        **args**

        *dl_data : dict* -> informações para download.

        **return**

        *bool* -> True, se completou. False, se não.
        '''

        self.setUp(dl_data)
        get_interface = None
        if dl_data['interface'] in self.interface_list:
            print(f"Processando dados na interface {dl_data['interface']}")
            return self.interface_list[dl_data['interface']](dl_data['opt'])
        else:
            return False

    def derivados(self):
        '''
        ### Descrição
        Método para cálculo de variáveis derivadas.
        '''
        derivados_path = self.__conf['Dados']['dir']/'Derivados'
        derivados_path.mkdir(exist_ok=True)
        dados_path = self.__conf['Dados']['dir']

        def prep_df(p, name) -> DataFrame:
            fp = list(p.glob(f'**/{name}.xlsx'))[0]
            df = read_excel(fp, index_col=0, decimal=',')
            return df

        # Vendas como soma de etanol e gasolina
        df_gasolina = prep_df(dados_path, 'Vendas_Gasolina')
        df_etanol = prep_df(dados_path, 'Vendas_Etanol')
        df = df_gasolina + df_etanol
        with ExcelWriter(derivados_path/'Vendas_Total.xlsx') as writer:
            df.to_excel(writer)

        # Métrica de interesse estranha by Werner
        df_autos = prep_df(dados_path, 'Automoveis')
        df_motos = prep_df(dados_path, 'Motos')
        df_veh = (df_autos + (0.3 * df_motos))
        for vendas_file in ['Vendas_Total', 'Vendas_Gasolina', 'Vendas_Etanol']:
            df_vendas = prep_df(dados_path, vendas_file)
            df_por_unidade = (1_000_000 * df_vendas) / df_veh
            with ExcelWriter(
                derivados_path/f'{vendas_file}_por_unidade.xlsx'
            ) as writer:
                df_por_unidade.to_excel(writer)

        # Média ponderada de preços
        df_preco_g = prep_df(dados_path, 'Precos_Gasolina')
        df_preco_e = prep_df(dados_path, 'Precos_Etanol')
        df_vendas_g = prep_df(dados_path, 'Vendas_Gasolina')
        df_vendas_e = prep_df(dados_path, 'Vendas_Etanol')
        df_gasolina = (df_preco_g * df_vendas_g)
        df_etanol = (df_preco_e * df_vendas_e)
        df_soma = (df_vendas_g + df_vendas_e)
        df_media_ponderada = (df_gasolina + df_etanol) / df_soma
        with ExcelWriter(
            derivados_path/f'Precos_Medio.xlsx'
        ) as writer:
            df_media_ponderada.to_excel(writer)

        # Lista de divisões
        l = list()
        l.append(('PIB', 'Populacao', 'PIB_per_capita', 1))
        l.append(('Automoveis', 'Populacao', 'Índice_de_Motorização_Autos', 100))
        l.append(('Motos', 'Populacao', 'Índice_de_Motorização_Motos', 100))
        l.append(('Precos_Etanol','Precos_Gasolina','Razão_dos_Preços', 100))
        l.append(('Vendas_Total','Populacao','Índice_de_Consumo_Populacional', 1_000_000))

        for over, under, result, ratio in l:
            df_over = prep_df(dados_path, over)
            df_under = prep_df(dados_path, under)
            df_over *= ratio
            df = df_over/df_under
            with ExcelWriter(derivados_path/f'{result}.xlsx') as writer:
                df.to_excel(writer)
                
        for frota in ['Automoveis','Motos','Populacao']:
            df_frota = prep_df(dados_path, frota)
            df_frota = 100*df_frota.div(df_frota.iloc[0])
            with ExcelWriter(derivados_path/f'Variação_de_{frota}.xlsx') as writer:
                df_frota.to_excel(writer)

    def deflacao(self):
        # Opções de leitura de csv
        col_types = {
            'ANO': int,
            'IPCA': float
        }
        csv_kwargs = {
            'sep': ';',
            'usecols': col_types.keys(),
            'dtype': col_types,
            'engine': 'c',
            'decimal': ',',
            'encoding': 'utf-8',
            'parse_dates': ['ANO']
        }
        df = read_csv(
            self.__conf['brutos_dir']/'inflacao_anual.csv',
            **csv_kwargs
        )

        # Índice de deflação
        df['IP'] = 1.0
        for i in range(1, len(df)):
            df.loc[i,'IP'] = df.loc[i-1,'IP']*(1 + (df.loc[i,'IPCA']/100.0))

        # Fator de deflacionamento
        df['FD'] = 1.0
        for i in range(0, len(df)):
            df.loc[i,'FD'] = df.loc[0,'IP']/df.loc[i,'IP']

        df.set_index('ANO', inplace=True)
        
        dados_path = self.__conf['Dados']['dir']
        var_valores = list(dados_path.glob('**/Precos*.xlsx'))
        var_valores += list(dados_path.glob('**/PIB*.xlsx'))
        var_valores = [
            v
            for v in var_valores
            if 'Deflacionado' not in v.stem
            and 'old' not in v.stem
        ]

        # print(*var_valores, sep='\n')

        for fp in var_valores:
            preco_df = read_excel(fp, index_col=0, decimal=',')
            for column in preco_df:
                preco_df[column] = preco_df[column]*df['FD']
            with ExcelWriter(
                fp.with_stem(fp.stem+'_Deflacionado')
            ) as writer:
                preco_df.to_excel(writer)


    def getDataFromDetran(self, opt: dict) -> bool:
        '''
        ### Descrição
        Busca informações a API do Detran SC e salva em arquivo

        **args**

        *opt : dict* -> dicinário com informações sobre 'download'

        **return**

        *bool* -> True, se completou. False, se não.

        ### Exemplo funcional:

            http://consultas.detrannet.sc.gov.br/
            Estatistica/Veiculos/winVeiculos.asp?
            lst_municipio=8105&
            lst_ano=2018&
            lst_mes=0

        Retorna os dados para Florianópolis, no ano de 2018.
        '''

        # Variáveis
        df_total = {n: DataFrame() for n in opt['grupos']}

        bar_total = len(self.localidades.items()) * \
            len(range(self.begin, self.end+1))

        bar_progress = 0

        for local_id, local in self.localidades.items():
            for ano in range(self.begin, self.end+1):
                # Barra de progresso
                progress_bar(bar_progress, bar_total)
                bar_progress += 1

                # Query part
                query_app = f'lst_municipio={local_id}&'
                query_app += f'lst_ano={ano}&'
                query_app += 'lst_mes=0'
                self.url_parts['query'] = query_app

                # Request e DataFrame
                url = urlunparse(self.url_parts.values())
                df = read_html(url, header=1, skiprows=0, index_col=0)[0]

                # Soma por grupo, aplica média no ano
                grupo_series = []
                for grupo_nome, grupo in opt['grupos'].items():
                    checked_grupo = [i for i in grupo if i in list(df.index)]
                    grupo_total = df.loc[checked_grupo].sum()
                    grupo_total.name = grupo_nome
                    grupo_series.append(grupo_total)
                mean = concat(grupo_series, axis=1)
                mean = mean[mean.values.sum(axis=1) != 0]
                mean = mean.mean().astype(int)
                for n, df_local in df_total.items():
                    df_local.at[local, str(ano)] = mean[n]
        progress_bar(bar_progress, bar_total)

        # Salva
        return self.saveDataFrame(df_total)

    def getDataFromIBGE(self, opt: dict) -> bool:
        '''
        ### Descrição
        Busca informações a API do IBGE (v3) e salva em arquivo

        **args**

        *opt : dict* -> dicinário com informações sobre download

        **return**

        *bool* -> True, se completou. False, se não.

        ### Exemplo funcional:

            https://servicodados.ibge.gov.br/
            api/v3/
            agregados/6579/
            periodos/2011|2012/
            variaveis/9324?
            localidades=N6[4200606]

        Retorna a população residente estimada para
        Águas Mornas no período entre 2011 e 2012
        '''

        # Variáveis
        periodos = '|'.join(
            [str(ano) for ano in range(self.begin, self.end+1)]
        )

        # Funções úteis
        def request_df() -> DataFrame:
            # Request
            with urlopen(urlunparse(self.url_parts.values())) as url_data:
                series = load(url_data)[0]["resultados"][0]["series"]

            # Descompacta series no dicionário
            data = {
                self.localidades[s_data["localidade"]["id"]]:s_data["serie"]
                for s_data in series
            }

            # Formato tratado
            return DataFrame(data=data).apply(to_numeric).transpose()

        # Path part
        path_app = f'/api/v3/agregados/{opt["agregados"]}/'
        path_app += f'periodos/{periodos}/'
        path_app += f'variaveis/{opt["variaveis"]}'
        self.url_parts['path'] = path_app

        # Query part
        query_app = f'localidades={opt["loc_nivel"]}'
        query_app += f'[{",".join(self.localidades)}]'
        self.url_parts['query'] = query_app

        df = request_df()

        # Caso seja população, é preciso verificar 2010 e 2007
        if opt["agregados"] == "6579":
            if 2010 in range(self.begin, self.end+1):
                # Path part
                # Censo de 2010, população municipal
                path_app = '/api/v3/agregados/1309/'
                path_app += 'periodos/2010/'
                path_app += 'variaveis/93'
                self.url_parts['path'] = path_app

                # Query part
                # em cada cidade, total em gênero, total em divisão regional
                query_app = f'localidades={opt["loc_nivel"]}'
                query_app += f'[{",".join(self.localidades)}]'
                query_app += '&classificacao=2[0]|11277[0]'
                self.url_parts['query'] = query_app

                censo_df = request_df()

                # Concatena em colunas
                df = concat(
                    [df, censo_df],
                    axis='columns',
                    ignore_index=False
                )

            if 2007 in range(self.begin, self.end+1):
                if all(x in df.columns for x in ['2006', '2008']):
                    df['2007'] = (
                        df[['2006', '2008']]
                        .mean(axis=1)
                        .apply(int)
                    )
                else:
                    raise ValueError(
                        'Populacao de 2007 é média, precisa de 2006 e 2008.'
                    )

        # Salva
        return self.saveDataFrame(df)

    def getDataFromANP(self, opt: dict) -> bool:
        '''
        ### Descrição
        Busca informações nos dados brutos da ANP
        e salva em arquivo

        **args**

        *opt : dict* -> dicinário com informações sobre 'download'

        **return**

        *bool* -> True, se completou. False, se não.
        '''

        # Variáveis
        brutos_dir = self.__conf['brutos_dir']
        df = DataFrame()

        if opt['fonte'] == 'precos':

            # Opções de leitura de csv
            col_types = {
                'Estado - Sigla': str,
                'Municipio': str,
                'Produto': str,
                'Valor de Venda': float
            }
            csv_kwargs = {
                'sep': ';',
                'usecols': col_types.keys(),
                'dtype': col_types,
                'engine': 'c',
                'decimal': ',',
                'encoding': 'utf-8'
            }

            # Iteração de arquivos
            df_anos = list()
            fl_paths = sorted(Path(brutos_dir).glob('ca-*-01.csv'))
            semesters = ['01', '02']

            # Variáveis da barra
            bar_progress = 0
            bar_total = len(sorted(Path(brutos_dir).glob('ca-*-01.csv'))) * \
                len(semesters)

            for p in fl_paths:
                # Para cada ano
                _, ano, _ = p.stem.split('-')
                if int(ano) in range(self.begin, self.end+1):
                    df_ano = DataFrame()
                    for sem in semesters:
                        progress_bar(bar_progress, bar_total)
                        bar_progress += 1
                        # Para cada semestre
                        csv_file_name = p.with_stem(f'ca-{ano}-{sem}')
                        df_sem = read_csv(csv_file_name, **csv_kwargs)

                        # Filtra cidades e produtos
                        df_sem = df_sem[df_sem['Municipio'].isin(
                            self.localidades)]
                        df_sem = df_sem[df_sem['Produto'].isin(
                            opt['produtos'])]

                        # Agrupa por municípios, média dos postos
                        df_sem = df_sem.groupby(['Municipio']).mean()
                        df_sem['Semestre'] = [
                            sem for _ in range(len(df_sem.index))]

                        # Concatena semestres, vertical
                        df_ano = concat([df_ano, df_sem])

                    # Agrupa por municípios, média dos semestres
                    df_ano = df_ano.groupby(['Municipio']).mean()

                    # Concatena anos, horizontal
                    rename_kws = {
                        'columns': {
                            'Valor de Venda': f'{ano}'
                        }
                    }
                    df_ano = df_ano.rename(**rename_kws)
                    df_anos.append(df_ano)
            progress_bar(bar_progress, bar_total)
            df = concat(df_anos, axis='columns', ignore_index=False)
            return self.saveDataFrame(df, agrup_func='mean')

        elif opt['fonte'] == 'vendas':
            # Opções de leitura de csv
            col_types = {
                'ANO': int,
                'MUNICÍPIO': str,
                'VENDAS': float
            }
            csv_kwargs = {
                'sep': ';',
                'usecols': col_types.keys(),
                'dtype': col_types,
                'engine': 'c',
                'decimal': ',',
                'encoding': 'utf-8'
            }

            # DataFrame com dados brutos
            df = read_csv(brutos_dir/opt['file'], **csv_kwargs)

            # Filtra dados indesejados
            df = df[df['MUNICÍPIO'].isin(self.localidades)]
            df = df[df['ANO'].isin(range(self.begin, self.end+1))]

            # Transforma litros em ~MEGALITROS~
            df = df.apply(lambda x: x/1_000_000 if x.name == 'VENDAS' else x)

            # Formato tratado
            df = df.pivot_table('VENDAS', 'MUNICÍPIO', 'ANO')
            return self.saveDataFrame(df)
        else:
            return True
