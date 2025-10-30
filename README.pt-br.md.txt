## Geospatial-Automation-Suite

Uma suíte de pipelines Python para inteligência territorial sob demanda. Automatiza análises ambientais complexas (CAR, Biomas, Solos, Hidrografia) para imóveis rurais e municípios. Integra processamento local (GeoPandas, Rasterio) e em nuvem (Google Earth Engine) para gerar relatórios em PDF, transformando dias de trabalho manual em minutos.

## Visão Geral: O Problema

Em muitas instituições ambientais e governamentais, a geração de relatórios técnicos e mapas situacionais é um gargalo operacional severo. Este processo é frequentemente manual, repetitivo e lento, exigindo que analistas passem dias recortando, processando e estilizando mapas para um único imóvel rural ou município. Esse fluxo de trabalho manual não é escalável e está sujeito a erros humanos.

## A Solução: Uma Suíte de Inteligência Sob Demanda

Este repositório contém uma suíte pronta para produção de pipelines de dados modulares que transformam todo esse fluxo de trabalho em um sistema automatizado, escalável e sob demanda. Ele é projetado como uma "fábrica de soluções" que ingere dados geoespaciais brutos (vetoriais e raster) e entrega dossiês profissionais em PDF, completos com mapas de alta qualidade e tabelas estatísticas.

## Principais Funcionalidades e Análises

Esta suíte é composta por vários pipelines independentes e configuráveis que podem ser executados para qualquer Área de Interesse (AOI), como um município ou um imóvel rural específico.

### 3.1. Análise CAR (Cadastro Ambiental Rural)

* **Dossiê Municipal:** Processa a base de dados completa do SICAR (atributos e geometrias) para um determinado município, gerando um dossiê completo em PDF com mapas e estatísticas para cada status do CAR (Ativo, Pendente, Cancelado).
* **Caracterização em Nível de Imóvel:** Realiza uma análise detalhada de um único imóvel rural, carregando todas as suas camadas declaradas (APP, Reserva Legal, Uso Consolidado, Vegetação Nativa).
* **Balanço Automático de Reserva Legal (RL):** Calcula automaticamente o déficit ou superávit de Reserva Legal (RL) cruzando a localização do imóvel com os limites oficiais de Biomas e da Amazônia Legal para determinar o percentual exigido correto (ex: 20%, 35%, 80%).

### 3.2. Análise Ambiental Temática (Processamento Local)

* **Solos:** Recorta o mapa nacional de solos do IBGE para a AOI e quantifica a área (ha) de cada classe de solo.
* **Geomorfologia:** Recorta e quantifica as unidades geomorfológicas com base nos dados do IBGE/CPRM.
* **Hidrografia:** Analisa a rede hidrográfica oficial para mapear todos os cursos d'água, identificar nascentes e calcular a densidade de drenagem (km/km²) para a AOI.
* **Biomas e Vegetação:** Identifica e quantifica os Biomas e Fitofisionomias (tipos de vegetação) presentes na AOI.
* **Clima:** Processa dados raster regionais para analisar a precipitação média anual e a Temperatura da Superfície Terrestre (LST).

### 3.3. Análise Multitemporal e em Nuvem (Integração GEE)

* **Mudança no Uso e Cobertura do Solo:** Realiza uma análise multitemporal (1985-2024) usando dados do MapBiomas para quantificar a evolução do uso do solo (ex: pastagem, agricultura, floresta) dentro da AOI.
* **Análise de Terreno (GEE):** Integra-se com a API do Google Earth Engine para calcular a declividade (em graus) a partir de DEMs NASADEM em tempo real, classificando o resultado em classes de relevo padrão.
* **Carbono Orgânico do Solo (GEE):** Quantifica o Carbono Orgânico do Solo (SOC) usando os assets do MapBiomas Soil, fornecendo métricas ambientais vitais sem armazenamento local de dados.
* **Classificação Florestal (GEE):** Classifica o status da floresta (ex: Primária, Secundária Jovem) usando assets globais da NASA/ORNL.
* **Socioambiental (GEE):** Analisa dados de "Tree Proximate People" (TPP) da FAO/GEE para avaliar a densidade populacional próxima à cobertura florestal.

## Tecnologias Utilizadas

Este projeto foi construído usando uma pilha robusta de bibliotecas geoespaciais de código aberto e serviços em nuvem.

* **Processamento Principal de Dados:**
    * Python 3.x
    * GeoPandas & Pandas: Para toda a manipulação de dados vetoriais e análise estatística.
    * Rasterio & Shapely: Para todo o processamento raster, recorte e operações de máscara.
    * NumPy: Para cálculos numéricos e manipulação de arrays.
* **Nuvem & Big Data:**
    * Google Earth Engine (GEE): API Python para processamento em nuvem de grandes conjuntos de dados (DEMs, Carbono do Solo).
    * Geemap: Para desenvolvimento interativo e exportação de dados do GEE.
* **Visualização de Dados e Relatórios:**
    * Matplotlib: O motor principal para criar todos os mapas estáticos e gráficos estatísticos.
    * Contextily: Para adicionar mapas base da web (ex: CartoDB Positron) aos mapas.
    * Matplotlib-Scalebar: Para adicionar barras de escala cartograficamente corretas.
    * PdfPages: Para compilar todas as saídas (mapas e tabelas) em um único dossiê PDF de várias páginas.

## Fontes de Dados

Os dados geoespaciais brutos (shapefiles, GeoTIFFs) necessários para rodar estes pipelines não estão incluídos neste repositório devido ao seu tamanho.

**Arquivos para testar os códigos:** [**Download dos Dados de Entrada (Google Drive)**](https://drive.google.com/drive/folders/1X7DmXw88nwcVNRUOHANM8g19bBM2alZI?usp=drive_link)

*Nota: Esta escolha de arquitetura (separar código de dados) mantém o repositório leve e rápido. Por favor, baixe os dados e atualize os caminhos dos arquivos nas classes de `Config` no topo de cada script de pipeline para corresponder à sua localização em sua máquina local.*

## Estrutura do Projeto

Este repositório é organizado como uma suíte de pipelines modulares. Cada script é autocontido e projetado para resolver uma tarefa de análise específica.
