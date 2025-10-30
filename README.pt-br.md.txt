# Suite de Automação Geoespacial

[](https://opensource.org/licenses/MIT)
[](https://www.python.org/)
[](https://geopandas.org/)
[](https://earthengine.google.com/)

Leia este documento em [Inglês (US)](README.md).

Uma suíte de pipelines em Python para inteligência territorial sob demanda. Automatiza análises ambientais complexas (CAR, Biomas, Solos, Hidro) para imóveis e municípios. Integra processamento local (GeoPandas, Rasterio) e em nuvem (Google Earth Engine) para gerar dossiês técnicos em PDF, transformando dias de trabalho manual em minutos.

## 1\. Visão Geral: O Problema

Em muitas instituições ambientais e governamentais, a geração de relatórios técnicos e mapas situacionais é um gargalo operacional severo. Esse processo é frequentemente manual, repetitivo e lento, exigindo que analistas passem dias recortando, processando e estilizando mapas para um único imóvel rural ou município. Este fluxo de trabalho manual não é escalável e está sujeito a erros humanos.

## 2\. A Solução: Uma Suíte de Inteligência Sob Demanda

Este repositório contém uma suíte de pipelines de dados modulares e prontos para produção que transformam todo esse fluxo de trabalho em um sistema automatizado, escalável e sob demanda.

Ele foi projetado como uma "fábrica de soluções" que ingere dados geoespaciais brutos (vetoriais e raster) e produz dossiês profissionais em PDF como saída, completos com mapas de alta qualidade e tabelas estatísticas.

## 3\. Principais Recursos e Análises

Esta suíte é composta por vários pipelines independentes e configuráveis que podem ser executados para qualquer Área de Interesse (AOI), seja um município ou um imóvel rural específico.

### 3.1. Análise do CAR (Cadastro Ambiental Rural)

  * [cite\_start]**Dossiê Municipal:** Processa a base de dados completa do SICAR (atributos e geometrias) [cite: 268, 269] [cite\_start]para um determinado município, gerando um dossiê completo em PDF com mapas e estatísticas para cada status do CAR (Ativo, Pendente, Cancelado)[cite: 272, 285].
  * [cite\_start]**Caracterização de Imóvel:** Realiza uma análise detalhada de um único imóvel rural, carregando todas as suas camadas declaradas (APP, Reserva Legal, Uso Consolidado, Vegetação Nativa)[cite: 347, 348, 417, 418].
  * [cite\_start]**Balanço Automático de Reserva Legal (RL):** Calcula automaticamente o déficit ou superávit de Reserva Legal (RL) cruzando a localização do imóvel com os limites oficiais de Biomas e da Amazônia Legal [cite: 375, 445] [cite\_start]para determinar o percentual exigido correto (ex: 20%, 35%, 80%)[cite: 376, 446].

### 3.2. Análises Temáticas (Processamento Local)

  * [cite\_start]**Solos:** Recorta o mapa nacional de solos do IBGE para a AOI e quantifica a área (ha) de cada classe de solo[cite: 10, 11].
  * [cite\_start]**Geomorfologia:** Recorta e quantifica as unidades geomorfológicas com base nos dados do IBGE/CPRM[cite: 41, 42].
  * [cite\_start]**Hidrografia:** Analisa a rede hidrográfica oficial para mapear cursos d'água, identificar nascentes [cite: 108] [cite\_start]e calcular a densidade de drenagem (km/km²) para a AOI[cite: 109, 110].
  * [cite\_start]**Biomas e Vegetação:** Identifica e quantifica os Biomas e Fitofisionomias (tipos de vegetação) presentes na AOI[cite: 73, 92, 93].
  * [cite\_start]**Clima:** Processa dados raster regionais para analisar a precipitação média anual [cite: 171, 172] [cite\_start]e a Temperatura da Superfície Terrestre (LST)[cite: 202, 203].

### 3.3. Análise Multitemporal e em Nuvem (Integração GEE)

  * [cite\_start]**Mudança no Uso e Cobertura do Solo:** Realiza uma análise multitemporal (1985-2024) usando dados do MapBiomas [cite: 130, 137] [cite\_start]para quantificar a evolução do uso do solo (ex: pastagem, agricultura, floresta) dentro da AOI[cite: 139].
  * [cite\_start]**Análise de Terreno (GEE):** Integra-se com a API do Google Earth Engine para calcular a declividade (graus) [cite: 233] [cite\_start]a partir de DEMs NASADEM em tempo real, classificando o resultado em classes de relevo padronizadas[cite: 229, 238].
  * [cite\_start]**Carbono do Solo (GEE):** Quantifica o Carbono Orgânico do Solo (SOC) usando os *assets* do MapBiomas Soil[cite: 238], fornecendo métricas ambientais vitais sem armazenamento local de dados.
  * [cite\_start]**Classificação Florestal (GEE):** Classifica o status da floresta (ex: Primária, Secundária Jovem) [cite: 385] [cite\_start]usando *assets* globais da NASA/ORNL[cite: 383, 384].
  * [cite\_start]**Socioambiental (GEE):** Analisa dados "Tree Proximate People" (TPP) da FAO/GEE [cite: 303, 309] [cite\_start]para avaliar a densidade populacional próxima à cobertura florestal[cite: 311, 312, 313, 314].

## 4\. Tecnologias Utilizadas (Stack)

Este projeto foi construído usando uma pilha robusta de bibliotecas geoespaciais de código aberto e serviços em nuvem.

  * **Processamento Central de Dados:**

      * `Python 3.x`
      * [cite\_start]`GeoPandas` & `Pandas`: Para toda a manipulação de dados vetoriais e análises estatísticas[cite: 1, 33, 66, 100, 130, 158, 193, 227, 263, 302, 342, 345, 382, 415, 452].
      * [cite\_start]`Rasterio` & `Shapely`: Para todo processamento raster, recorte (clip) e operações de máscara[cite: 130, 158, 193, 227, 302, 342, 382].
      * [cite\_start]`NumPy`: Para cálculos numéricos e manipulação de arrays raster[cite: 1, 33, 66, 100, 130, 158, 193, 227, 263, 302, 342, 345, 382, 415, 452].

  * **Nuvem & Big Data:**

      * [cite\_start]`Google Earth Engine (GEE)`: API Python para processamento em nuvem de grandes conjuntos de dados (DEMs, Carbono do Solo)[cite: 230, 305, 386].
      * [cite\_start]`Geemap`: Para desenvolvimento interativo e exportação de dados do GEE[cite: 227, 302, 382].

  * **Visualização de Dados e Relatórios:**

      * [cite\_start]`Matplotlib`: O motor principal para a criação de todos os mapas estáticos e gráficos estatísticos[cite: 1, 33, 66, 100, 130, 158, 193, 227, 263, 302, 342, 345, 382, 415, 452].
      * [cite\_start]`Contextily`: Para adicionar *basemaps* de mapas web (ex: CartoDB Positron) aos mapas[cite: 1, 33, 66, 130, 158, 193, 227, 302, 345, 382, 415, 452].
      * [cite\_start]`Matplotlib-Scalebar`: Para adicionar barras de escala cartograficamente corretas[cite: 1, 33, 66, 100, 130, 158, 193, 227, 263, 302, 345, 382, 415, 452].
      * [cite\_start]`PdfPages`: Para compilar todas as saídas (mapas e tabelas) em um único dossiê PDF de múltiplas páginas[cite: 1, 33, 66, 100, 130, 158, 193, 227, 263, 302, 345, 382, 415, 452].

## 5\. Estrutura do Projeto

Este repositório está organizado como uma suíte de pipelines modulares. Cada script é autônomo e projetado para resolver uma tarefa de análise específica.

  * **`/pipelines`**: Contém os scripts Python principais, cada um representando uma análise completa (ex: `pipeline_analise_solos.py`, `pipeline_caracterizacao_car.py`).
  * [cite\_start]**`/gee_scripts`**: Contém quaisquer scripts independentes do Earth Engine usados para prototipagem ou exploração de dados (ex: `gee_indices_cbers.js` [cite: 343, 344][cite\_start], `gee_modis_temperatura.js` [cite: 457, 460]).
  * **`/exemplos`**: Contém exemplos de saídas em PDF geradas pelos pipelines, exibindo o produto final.

## 6\. Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

