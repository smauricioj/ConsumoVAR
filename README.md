# ConsumoVAR

## Objetivos do projeto

Projeto que implementa a série de operações e modelagens para a análise do consumo de combustível nas maiores cidades do estado de Santa Catarina.

## O que está incluso?

### ETL de dados públicos

Diversas séries históricas são obtidas, investigadas e carregadas em planilhas com esse código. As fontes de dados utilizadas são:

+ IBGE
  - [Link para documentação](https://servicodados.ibge.gov.br/api/docs)
  + API de acesso gratuito
  * Dados estruturados em banco de dados não relacional (JSON)
  - Usado para obter dados sobre população estimada e PIB
+ Detran Santa Catarina
  - [Link para documentação](https://www.detran.sc.gov.br/transparencia/estatisticas/veiculos-2/)
  + Pra ser 100% sincero, o Detran mudou a interface durante o projeto :-(
  * Agora eles usam Power BI e não mais a GUI antiga
  - Ainda funcionam os requests que faço, mas talvez um dia eles deixem de funcionar
  + Usado para obter dados sobre a frota veicular (automoveis e motos)
+ ANP
  - [Link para dados](https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-estatisticos)
  + A agência não fornece nenhum tipo de acesso automatizado aos dados
  * Então eu baixei tudo no computador e trabalhei a partir daí
  - Usado para obter dados sobre a venda de combustíveis e seus preços

### Modelos de auto-regressão

Com auxílio da biblioteca [statsmodels](https://www.statsmodels.org/stable/index.html), modelos de auto-regressão são implementados para análise das séries históricas obtidas. Seus resultados são voláteis, mas os modelos são parametrizados e de fácil manipulação pelo código. Tais modelos são usados para realizar estimações do consumo de combustível nos anos futuros.

### Método de análise

As modelos de auto-regressão são comparados em uma varredura longitudinal ao longo dos municípios e das variáveis de suporte em busca da maior redução no erro de estimação. Os modelos são analisados com base em diversas métricas através de valização cruzada hold-out e seus resultados são armazenados em planilhas digitais.

### Suporte à inspeção visual

Todas as séries históricas são plotadas em gráficos de linhas para inspeção visual. Conjuntos de séries são agrupadas para facilitar a compreensão de relacionamentos entre variáveis. 

## Estrutura do projeto

```text
ConsumoVAR/
├── Python/
│   ├── conf/
│   │   ├── conf.json
│   │   └── locals.csv
│   ├── requirements.txt
│   ├── gerador_graficos.py
│   ├── interfaces.py
│   ├── metodo.py
│   ├── modelo.py
│   ├── util.py
│   └── main.py
├── README.md
├── LICENSE.md
└── .gitignore
```