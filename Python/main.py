# -*- coding: utf-8 -*-
# Autor: Sergio Prolo
# Data: 28/04/2022

from json import load
from pathlib import Path

from interfaces import Interfaces
from metodo import Metodo
from modelo import Modelo


def main():
    # Dicionário de configurações

    c_path = Path.cwd()/'conf'/'conf.json'
    if c_path.is_file():
        with open(c_path, encoding='utf-8') as c_file:
            conf = load(c_file)
    else:
        raise NameError

    # Diretórios

    main_dir = Path.cwd().parent
    conf['Resultados']['dir'] = main_dir/conf['Resultados']['dir']
    conf['Dados']['dir'] = main_dir/conf['Dados']['dir']
    conf['brutos_dir'] = main_dir/'Brutos'

    # Classes

    inter = Interfaces(conf)
    metodo = Metodo(conf)
    modelo = Modelo(conf)

    # Funções úteis
    
    def roda(locs=None):
        if locs is None:
            locs = inter.index_mapper.values()
        elif locs == 'Regiões':
            locs = [x for x in inter.index_mapper.values() if 'Região' in x]

        for loc in locs:
            metodo.set_municipio(loc)
            metodo.processa_regressao_modelo()

    # Operações

    # for dl_data in conf['Downloads']:
    #     inter.getData(dl_data)
    # inter.getData(conf['Downloads'][5])
    # inter.derivados()
    # inter.deflacao()
    # roda()
    # roda('Regiões')
    # roda(['Região de Joinville'])
    # metodo.estatisticas_testes()
    metodo.imagens()

if __name__ == '__main__':
    main()
