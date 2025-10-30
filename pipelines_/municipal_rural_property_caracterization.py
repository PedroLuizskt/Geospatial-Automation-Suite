# Caracterização CAR por Município MG - Autor Pedro Luiz

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.backends.backend_pdf import PdfPages
import contextily as ctx
from dataclasses import dataclass
from unidecode import unidecode
import textwrap
from datetime import datetime

@dataclass
class CarMunicipioConfig:
    NOME_MUNICIPIO_ALVO: str = 'Inhaúma'
    ESTADO_SIGLA: str = 'MG'
    PATH_CSV_ATRIBUTOS: str = r"C:\Users\pedro\Downloads\python_gis\projeto_CAR\MG\base_processada\CAR_MG_Atributos.csv"
    PATH_GPKG_GEOMETRIAS: str = r"C:\Users\pedro\Downloads\python_gis\projeto_CAR\MG\base_processada\CAR_MG_Geometrias.gpkg"
    PATH_MUNICIPIOS_IBGE: str = r"C:\Users\pedro\Downloads\python_gis\projeto_CAR\BR_Municipios_2024\BR_Municipios_2024.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\python_gis\projeto_CAR\MG\re_2_"
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    CRS_METRICO_PADRAO: str = 'EPSG:5880' 
    CRS_WEB_MERCATOR: str = 'EPSG:3857'
    CORES_STATUS_GERAL: dict = None

    def __post_init__(self):
        self.CORES_STATUS_GERAL = {
            'Ativo': '#2ca25f',
            'Pendente': '#fdae61',
            'Cancelado': '#bdbdbd',
            'Suspenso': '#d73027',
            'Não Classificado': '#808080'
        }

    @property
    def NOME_ARQUIVO_SAIDA(self) -> str:
        nome_base = unidecode(self.NOME_MUNICIPIO_ALVO).replace(' ', '_').lower()
        return f"Dossie_CAR_{nome_base}.pdf"

class CarMunicipioPipeline:
    def __init__(self, config: CarMunicipioConfig):
        self.config = config
        self.db_atributos = self._carregar_banco_de_dados_csv()
        self.gdf_geometrias_base = self._carregar_banco_de_geometrias()

    def _carregar_banco_de_dados_csv(self) -> pd.DataFrame:
        print(f"Carregando banco de dados de atributos...")
        dtype_mapping = {'cod_imovel': 'str', 'municipio': 'str', 'ind_status': 'str'}
        try:
            return pd.read_csv(
                self.config.PATH_CSV_ATRIBUTOS,
                sep=';',
                encoding='utf-8',
                low_memory=False,
                dtype=dtype_mapping
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo CSV não encontrado em: {self.config.PATH_CSV_ATRIBUTOS}")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar CSV: {e}")

    def _carregar_banco_de_geometrias(self) -> gpd.GeoDataFrame:
        print(f"Carregando banco de dados de geometrias...")
        try:
            gdf = gpd.read_file(self.config.PATH_GPKG_GEOMETRIAS, layer='car_geometrias')
            if gdf.crs is None:
                 print(f"[AVISO] CRS do GeoPackage não definido. Assumindo {self.config.CRS_GEOGRAFICO}.")
                 gdf.set_crs(self.config.CRS_GEOGRAFICO, inplace=True)
            elif gdf.crs.to_string() != self.config.CRS_GEOGRAFICO:
                 gdf = gdf.to_crs(self.config.CRS_GEOGRAFICO)

            if gdf.duplicated(subset='cod_imovel').any():
                print("  Removendo geometrias duplicadas por 'cod_imovel'...")
                gdf.drop_duplicates(subset='cod_imovel', keep='first', inplace=True)
            return gdf
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo GeoPackage não encontrado em: {self.config.PATH_GPKG_GEOMETRIAS}")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar GeoPackage: {e}")

    def _obter_limite_municipal(self) -> gpd.GeoDataFrame:
        cfg = self.config
        print(f"Carregando limite oficial para '{cfg.NOME_MUNICIPIO_ALVO}'...")
        try:
            municipios_br = gpd.read_file(cfg.PATH_MUNICIPIOS_IBGE, encoding='latin-1')
            municipios_br = municipios_br.to_crs(cfg.CRS_GEOGRAFICO)
        except Exception as e:
             raise RuntimeError(f"Erro ao carregar shapefile de municípios: {e}")

        nome_alvo_norm = unidecode(cfg.NOME_MUNICIPIO_ALVO.strip().upper())
        municipios_br['NM_MUN_NORM'] = municipios_br['NM_MUN'].astype(str).apply(lambda x: unidecode(x.strip().upper()))
        limite_gdf = municipios_br[(municipios_br['NM_MUN_NORM'] == nome_alvo_norm) & (municipios_br['SIGLA_UF'] == cfg.ESTADO_SIGLA)].copy()

        if limite_gdf.empty:
            raise ValueError(f"Limite para '{cfg.NOME_MUNICIPIO_ALVO}' ({cfg.ESTADO_SIGLA}) não encontrado.")
        return limite_gdf.dissolve().reset_index() 

    def _estimate_utm_crs(self, gdf_geo: gpd.GeoDataFrame) -> str:
        try:
            centroid = gdf_geo.union_all().centroid
            utm_zone = int((centroid.x + 180) / 6) + 1
            is_south = centroid.y < 0
            epsg = 32700 + utm_zone if is_south else 32600 + utm_zone
            return f"EPSG:{epsg}"
        except Exception:
            return self.config.CRS_METRICO_PADRAO

    def _processar_dados_municipio(self) -> gpd.GeoDataFrame:
        cfg = self.config
        print(f"\n--- Processando dados para {cfg.NOME_MUNICIPIO_ALVO} ---")

        nome_alvo_norm = unidecode(cfg.NOME_MUNICIPIO_ALVO.strip().upper())
        if 'municipio_norm' not in self.db_atributos.columns:
            self.db_atributos['municipio_norm'] = self.db_atributos['municipio'].astype(str).apply(lambda x: unidecode(x.strip().upper()))

        df_municipio = self.db_atributos[self.db_atributos['municipio_norm'] == nome_alvo_norm].copy()

        if df_municipio.empty:
            raise ValueError(f"Nenhum imóvel encontrado para '{cfg.NOME_MUNICIPIO_ALVO}' no arquivo de atributos.")

        status_map = { 'AT': 'Ativo', 'PE': 'Pendente', 'CA': 'Cancelado', 'SU': 'Suspenso' }
        df_municipio['status_geral'] = df_municipio['ind_status'].map(status_map).fillna('Não Classificado')
        print(f"  Coluna 'status_geral' criada com sucesso ({len(df_municipio)} imóveis).")

        cod_imoveis_municipio = df_municipio['cod_imovel'].unique()
        print(f"  Filtrando geometrias para {len(cod_imoveis_municipio)} imóveis...")
        gdf_geom_municipio = self.gdf_geometrias_base[self.gdf_geometrias_base['cod_imovel'].isin(cod_imoveis_municipio)].copy()

        print(f"  Juntando atributos ({len(df_municipio)}) com geometrias ({len(gdf_geom_municipio)})...")
        gdf_final = gdf_geom_municipio.merge(df_municipio, on='cod_imovel', how='inner')

        if len(gdf_final) != len(df_municipio):
             print(f"[ALERTA] A junção resultou em {len(gdf_final)} registros, diferente dos {len(df_municipio)} atributos. Verifique possíveis inconsistências.")
             

        print(f"  Validando e corrigindo geometrias...")
        gdf_final['geometry'] = gdf_final.geometry.buffer(0)
        invalidas_apos = len(gdf_final) - gdf_final.geometry.is_valid.sum()
        if invalidas_apos == 0:
            print("  [OK] Geometrias válidas.")
        else:
            print(f"  [ATENÇÃO] {invalidas_apos} geometrias ainda inválidas após buffer(0).")

        gdf_final['num_area'] = pd.to_numeric(gdf_final['num_area'], errors='coerce').fillna(0)
        gdf_final = gdf_final.reset_index(drop=True)

        print(f"  Análise concluída: {len(gdf_final)} imóveis processados para {cfg.NOME_MUNICIPIO_ALVO}.")
        return gdf_final

    def _gerar_tabela_detalhada(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        cfg = self.config
        if gdf.empty:
            return pd.DataFrame()

        tabela_gdf = gdf.copy()
        crs_metrico = self._estimate_utm_crs(tabela_gdf)
        tabela_gdf = tabela_gdf.to_crs(crs_metrico)

        try:
             tabela_gdf['centroid_geo'] = tabela_gdf.geometry.centroid.to_crs(cfg.CRS_GEOGRAFICO)
             tabela_gdf['coordenada'] = tabela_gdf['centroid_geo'].apply(lambda p: f'{p.y:.4f}, {p.x:.4f}' if p else 'N/A')
        except Exception as e:
             print(f"[AVISO] Erro ao calcular centróides: {e}. Coordenadas serão 'N/A'.")
             tabela_gdf['coordenada'] = 'N/A'

        tabela_final = tabela_gdf[['cod_imovel', 'status_geral', 'des_condic', 'coordenada', 'num_area']].rename(columns={
            'cod_imovel': 'Código do Imóvel',
            'status_geral': 'Status CAR',
            'des_condic': 'Condição da Análise',
            'coordenada': 'Centróide (Lat, Lon)',
            'num_area': 'Área (ha)'
        })

        return tabela_final.sort_values(by='Área (ha)', ascending=False)

    def _add_grid_inteligente(self, ax, gdf_geografico):
        cfg = self.config; minx, miny, maxx, maxy = gdf_geografico.total_bounds; num_ticks = 5
        x_ticks = np.linspace(minx, maxx, num_ticks); y_ticks = np.linspace(miny, maxy, num_ticks)
        lon_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_ticks, [miny] * num_ticks), crs=cfg.CRS_GEOGRAFICO); lat_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([minx] * num_ticks, y_ticks), crs=cfg.CRS_GEOGRAFICO)
        lon_ticks_proj = lon_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.x; lat_ticks_proj = lat_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.y
        ax.set_xticks(lon_ticks_proj); ax.set_yticks(lat_ticks_proj); ax.set_xticklabels([f'{x:.2f}°' for x in x_ticks], rotation=45, ha='right'); ax.set_yticklabels([f'{y:.2f}°' for y in y_ticks])
        ax.grid(True, linestyle='--', alpha=0.6, color='gray'); ax.tick_params(axis='both', labelsize=10)

    def _plotar_mapa_por_status(self, gdf_status: gpd.GeoDataFrame, limite_municipal: gpd.GeoDataFrame, status: str) -> plt.Figure:
        cfg = self.config; fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
        ax.set_facecolor('#f0f0f0')
        plot_gdf = gdf_status.to_crs(cfg.CRS_WEB_MERCATOR); limite_plot = limite_municipal.to_crs(cfg.CRS_WEB_MERCATOR)
        cor_plot = cfg.CORES_STATUS_GERAL.get(status, '#808080')

        limite_plot.plot(ax=ax, facecolor='#EFEFEF', edgecolor='black', linewidth=0.5, zorder=2)
        if not plot_gdf.empty: plot_gdf.plot(ax=ax, color=cor_plot, edgecolor='darkgrey', linewidth=0.2, alpha=0.7, zorder=3)

        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zorder=1)
        self._add_grid_inteligente(ax, limite_municipal.to_crs(cfg.CRS_GEOGRAFICO))
        ax.set_title(f"Imóveis Rurais com Status CAR: {status}\n{cfg.NOME_MUNICIPIO_ALVO} ({len(gdf_status)} imóveis)", fontsize=16, pad=20)

        ax.add_artist(ScaleBar(1, 'm', location='upper right', box_alpha=0.8))
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=16, xycoords='axes fraction')

        info_text = f"Fonte: SICAR ({datetime.now().year}), IBGE (2024)\nDatum: SIRGAS 2000\nProjeção: Web Mercator \nAutor: Pedro Luiz"
        fig.text(0.82, 0.75, info_text, ha='left', va='bottom', fontsize=8, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=0.5))

        ax.set_xlabel(None); ax.set_ylabel(None)
        plt.tight_layout(rect=[0, 0.03, 0.97, 0.93])
        return fig

    def _exportar_dossie_pdf(self, sumario_geral: pd.DataFrame, resultados_detalhados: dict, nome_area: str):
        cfg = self.config
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, cfg.NOME_ARQUIVO_SAIDA)
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        print(f"\nExportando dossiê completo para: {caminho_completo}...")

        with PdfPages(caminho_completo) as pdf:
            fig_capa = plt.figure(figsize=(8.27, 11.69))
            fig_capa.suptitle(f"Dossiê de Situação Cadastral (CAR)", fontsize=22, fontweight='bold', y=0.95)
            fig_capa.text(0.5, 0.90, f"Município de {nome_area} - {cfg.ESTADO_SIGLA}", fontsize=18, ha='center')
            fig_capa.text(0.5, 0.87, f"Relatório gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", fontsize=10, ha='center', style='italic', color='gray')

            ax_pie = fig_capa.add_axes([0.1, 0.52, 0.8, 0.3])
            cores = [cfg.CORES_STATUS_GERAL.get(status, '#808080') for status in sumario_geral.index]
            def autopct_generator(pct): return ('%1.1f%%' % pct) if pct > 3 else ''
            wedges, texts, autotexts = ax_pie.pie(
                sumario_geral['Contagem'], autopct=autopct_generator,
                startangle=90, colors=cores, pctdistance=0.8,
                wedgeprops={'edgecolor':'white'},
                explode=[0.02] * len(sumario_geral)
            )
            plt.setp(autotexts, size=10, weight="bold", color="black")
            ax_pie.set_title("Distribuição de Imóveis por Status CAR", fontsize=14, pad=20)
            ax_pie.legend(wedges, sumario_geral.index, title="Status CAR", loc="center left", bbox_to_anchor=(0.95, 0.5), fontsize='small')

            ax_table = fig_capa.add_axes([0.1, 0.15, 0.8, 0.3])
            ax_table.axis('off')
            sumario_geral_display = sumario_geral.copy()
            sumario_geral_display['Área Total (ha)'] = sumario_geral_display['Área Total (ha)'].map('{:,.2f}'.format)
            sumario_geral_display.reset_index(inplace=True)
            tab = ax_table.table(
                cellText=sumario_geral_display.values, colLabels=sumario_geral_display.columns,
                cellLoc='left', loc='center', colWidths=[0.3, 0.2, 0.3]
            )
            tab.auto_set_font_size(False); tab.set_fontsize(12); tab.scale(1, 2)
            for (i, j), cell in tab.get_celld().items():
                if i == 0: cell.set_text_props(weight='bold')

            pdf.savefig(fig_capa, bbox_inches='tight')
            plt.close(fig_capa)

            for status, dados in resultados_detalhados.items():
                if 'figura' in dados:
                     pdf.savefig(dados['figura'], bbox_inches='tight')
                     plt.close(dados['figura'])
                else:
                     print(f"[AVISO] Figura para o status '{status}' não encontrada.")

                tabela = dados.get('tabela')
                if tabela is None or tabela.empty: continue

                linhas_por_pagina = 20
                num_paginas = int(np.ceil(len(tabela) / linhas_por_pagina))

                for i in range(num_paginas):
                    fig_tabela, ax_tabela = plt.subplots(figsize=(11.69, 8.27)) # Landscape A4
                    ax_tabela.axis('off')
                    ax_tabela.set_title(f"Tabela de Imóveis - Status: {status}\n{nome_area} (Pág. {i+1}/{num_paginas})", fontsize=10, pad=25)

                    df_pagina = tabela.iloc[i*linhas_por_pagina : (i+1)*linhas_por_pagina].copy()
                    df_pagina['Área (ha)'] = pd.to_numeric(df_pagina['Área (ha)'], errors='coerce').map('{:,.2f}'.format)
                    df_pagina['Condição da Análise'] = df_pagina['Condição da Análise'].apply(lambda x: '\n'.join(textwrap.wrap(str(x), width=45)))

                    col_widths = [0.40, 0.10, 0.35, 0.15, 0.10] # Ajustado para 5 colunas
                    tab = ax_tabela.table(
                        cellText=df_pagina.values, colLabels=df_pagina.columns,
                        cellLoc='left', loc='center',
                        colWidths=col_widths
                    )
                    tab.auto_set_font_size(False); tab.set_fontsize(8)
                    tab.scale(1, 1.8) # Ajustar escala vertical se necessário
                    pdf.savefig(fig_tabela, bbox_inches='tight')
                    plt.close(fig_tabela)

        print("Dossiê exportado com sucesso!")

    def run(self):
        plt.ioff()
        try:
            limite_gdf = self._obter_limite_municipal()
            gdf_completo = self._processar_dados_municipio()

            if gdf_completo.empty:
                print(f"\nNenhum imóvel encontrado ou processado para {self.config.NOME_MUNICIPIO_ALVO}.")
                return

            print("\n" + "="*80)
            print(f"       INICIANDO GERAÇÃO DE RELATÓRIOS POR STATUS CAR")
            print("="*80)

            sumario_geral_df = gdf_completo.groupby('status_geral').agg(
                Contagem=('cod_imovel', 'nunique'), Area_Total_ha=('num_area', 'sum')
            ).rename(columns={'Area_Total_ha': 'Área Total (ha)'}).sort_values(by='Contagem', ascending=False)

            resultados_finais = {}
            status_presentes = sorted(gdf_completo['status_geral'].unique())

            for status in status_presentes:
                print(f"\nProcessando status: '{status}'...")
                gdf_filtrado = gdf_completo[gdf_completo['status_geral'] == status].copy()
                if gdf_filtrado.empty:
                     print("  Nenhum imóvel encontrado para este status.")
                     resultados_finais[status] = {'figura': None, 'tabela': pd.DataFrame()}
                     continue

                figura = self._plotar_mapa_por_status(gdf_filtrado, limite_gdf, status)
                tabela = self._gerar_tabela_detalhada(gdf_filtrado)
                resultados_finais[status] = {'figura': figura, 'tabela': tabela}

            self._exportar_dossie_pdf(sumario_geral_df, resultados_finais, self.config.NOME_MUNICIPIO_ALVO)

            print("\nAnálise e exportação concluídas.")

        except Exception as e:
            print(f"\n[ERRO CRÍTICO NO PIPELINE] {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.ion()

if __name__ == "__main__":
    config = CarMunicipioConfig(NOME_MUNICIPIO_ALVO = 'Buritizeiro')
    pipeline = CarMunicipioPipeline(config)
    pipeline.run()