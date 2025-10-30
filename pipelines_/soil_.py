## Análise de Solos - Autor = Pedro Luiz

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings
import contextily as ctx
import unicodedata
import re
from dataclasses import dataclass
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from IPython.display import display, HTML

@dataclass
class Config:
    PATH_SOLOS_BRASIL: str = r"C:\Users\pedro\Downloads\python_gis\script_solos\Solos_5000mil\Solos_5000.shp"
    PATH_MUNICIPIOS_BRASIL: str = r"C:\Users\pedro\Downloads\python_gis\script_solos\BR_Municipios_2024\BR_Municipios_2024.shp"
    PATH_IMOVEL_ALVO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\MG-3121605-0C5FDE61120E4BB7B3CD1728A97638E2\Area_do_Imovel\Area_do_Imovel.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\estagio_ie\resultados_"
    COLUNA_SOLOS: str = "DSC_COMPO1"
    COLUNA_MUNICIPIO: str = 'NM_MUN'
    COLUNA_NOME_IMOVEL: str = 'recibo'
    DPI_SAIDA: int = 300
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    CRS_WEB_MERCATOR: str = 'EPSG:3857'
    TIPO_DE_AREA: str = 'imovel'
    NOME_MUNICIPIO_ALVO: str = 'Água Boa'
    REGRAS_DE_CORES: dict = None
    COR_PADRAO: str = '#E1E1E1'

    def __post_init__(self):
        if self.REGRAS_DE_CORES is None:
            self.REGRAS_DE_CORES = {
                'vermelho': '#9E452C', 'amarelo': '#E09752', 'cinzento': '#BDB76B',
                'glei': '#A9A9A9', 'plinto': '#B0171F', 'húmico': '#654321',
                'melânico': '#3D2B1F', 'êbanico': '#292421', 'latossolo': '#BC4935',
                'argissolo': '#D2945C', 'cambissolo': '#8B5A2B', 'neossolo': '#CDAF95',
                'afloramentos': '#708090'
            }

class SoilAnalysisPipeline:
    def __init__(self, config: Config):
        self.config = config
        self._setup_environment()

    def _setup_environment(self):
        warnings.filterwarnings('ignore', 'The Shapely GEOS version used')
        warnings.filterwarnings('ignore', 'DataFrame is highly fragmented')
        plt.style.use('seaborn-v0_8-whitegrid')
        pd.set_option('display.max_rows', 20)

    def _carregar_dados_vetoriais(self, path: str, crs_alvo: str, **kwargs) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path, **kwargs)
        if gdf.crs is None:
            gdf.set_crs(self.config.CRS_GEOGRAFICO, inplace=True)
        return gdf.to_crs(crs_alvo)

    def _sanitizar_nome_arquivo(self, nome: str) -> str:
        nome = unicodedata.normalize('NFKD', nome).encode('ascii', 'ignore').decode('utf-8')
        nome = re.sub(r'[^\w\s-]', '', nome).strip().lower()
        nome = re.sub(r'[-\s]+', '-', nome)
        return nome[:100]

    def _obter_area_de_interesse(self) -> gpd.GeoDataFrame:
        cfg = self.config
        if cfg.TIPO_DE_AREA == 'municipio':
            municipios_gdf = self._carregar_dados_vetoriais(cfg.PATH_MUNICIPIOS_BRASIL, cfg.CRS_GEOGRAFICO)
            area_gdf_filtrada = municipios_gdf[municipios_gdf[cfg.COLUNA_MUNICIPIO].str.contains(cfg.NOME_MUNICIPIO_ALVO, case=False, na=False)]
            if area_gdf_filtrada.empty:
                raise ValueError(f"Município '{cfg.NOME_MUNICIPIO_ALVO}' não encontrado.")
            area_gdf = area_gdf_filtrada.dissolve().reset_index()
        elif cfg.TIPO_DE_AREA == 'imovel':
            area_gdf = self._carregar_dados_vetoriais(cfg.PATH_IMOVEL_ALVO, cfg.CRS_GEOGRAFICO)
        else:
            raise ValueError("Tipo de área inválido. Escolha 'municipio' ou 'imovel'.")
        if area_gdf.empty:
            raise ValueError("A Área de Interesse (AOI) não pôde ser carregada ou está vazia.")
        return area_gdf
    
    def _estimate_utm_crs(self, gdf: gpd.GeoDataFrame) -> str:
        try:
            centroid = gdf.union_all().centroid
            utm_zone = int((centroid.x + 180) / 6) + 1
            south = centroid.y < 0
            epsg = 32700 + utm_zone if south else 32600 + utm_zone
            return f"EPSG:{epsg}"
        except Exception:
            return 'EPSG:5880'

    def _analisar_solos(self, solos_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame):
        cfg = self.config
        solos_recortado_gdf = gpd.clip(solos_gdf, aoi_gdf)
        if solos_recortado_gdf.empty:
            return None, None

        crs_metrico_local = self._estimate_utm_crs(solos_recortado_gdf)
        solos_metric_gdf = solos_recortado_gdf.to_crs(crs_metrico_local)
        solos_metric_gdf['area_ha'] = solos_metric_gdf.geometry.area / 10000
        
        estatisticas = (solos_metric_gdf.groupby(cfg.COLUNA_SOLOS)[['area_ha']]
                                      .sum()
                                      .sort_values(by='area_ha', ascending=False)
                                      .reset_index())
        return solos_recortado_gdf, estatisticas

    def _encontrar_cor_para_solo(self, descricao: str) -> str:
        for palavra_chave, cor_hex in self.config.REGRAS_DE_CORES.items():
            if palavra_chave in str(descricao).lower():
                return cor_hex
        return self.config.COR_PADRAO

    def _add_grid_inteligente(self, ax, gdf_geografico):
        cfg = self.config
        minx, miny, maxx, maxy = gdf_geografico.total_bounds
        x_interval, y_interval = (maxx - minx) / 5, (maxy - miny) / 5
        x_ticks, y_ticks = np.arange(minx, maxx, x_interval), np.arange(miny, maxy, y_interval)
        
        lon_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_ticks, np.full(len(x_ticks), miny)), crs=cfg.CRS_GEOGRAFICO)
        lat_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.full(len(y_ticks), minx), y_ticks), crs=cfg.CRS_GEOGRAFICO)
        
        lon_ticks_proj = lon_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.x
        lat_ticks_proj = lat_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.y
        
        ax.set_xticks(lon_ticks_proj)
        ax.set_yticks(lat_ticks_proj)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{x_ticks[pos]:.2f}°" if pos < len(x_ticks) else ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{y_ticks[pos]:.2f}°" if pos < len(y_ticks) else ""))
        
        ax.grid(True, linestyle='--', alpha=0.5, color='gray')
        ax.tick_params(axis='both', which='major', labelsize=10, labelrotation=45)

    def _gerar_figura_mapa(self, solos_gdf, aoi_gdf, nome_area):
        cfg = self.config
        solos_plot_gdf = solos_gdf.to_crs(cfg.CRS_WEB_MERCATOR)
        aoi_plot_gdf = aoi_gdf.to_crs(cfg.CRS_WEB_MERCATOR)
        
        fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27), facecolor='white')
        ax.set_facecolor('#f0f0f0')
        
        mapa_de_cores = {desc: self._encontrar_cor_para_solo(desc) for desc in solos_plot_gdf[cfg.COLUNA_SOLOS].unique()}
        
        solos_plot_gdf.plot(ax=ax, color=solos_plot_gdf[cfg.COLUNA_SOLOS].map(mapa_de_cores), edgecolor='black', linewidth=0.2)
        aoi_plot_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2.0, linestyle='--')
        
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto')
        self._add_grid_inteligente(ax, aoi_gdf)
        
        ax.add_artist(ScaleBar(1, loc='upper left', bbox_to_anchor=(0.02, 0.98), bbox_transform=ax.transAxes, frameon=True, box_alpha=0.7, pad=0.5, color='black', font_properties={'size': 10}))
        
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=cor, edgecolor='k', label=desc) for desc, cor in sorted(mapa_de_cores.items())]
        ax.legend(handles=legend_elements, title='Classes de Solo', loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=True, facecolor='white', edgecolor='darkgray')
        
        x_norte, y_norte, arrow_length = 0.98, 0.98, 0.08
        ax.annotate('N', xy=(x_norte, y_norte), xytext=(x_norte, y_norte - arrow_length), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
        
        texto_creditos = f"Fonte: IBGE (2007)\nProjeção: Web Mercator (EPSG:3857)\nDatum Geográfico: SIRGAS 2000\nAutor: Pedro Luiz"
        ax.text(1.03, 0, texto_creditos, transform=ax.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='left', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8, ec='darkgray', lw=1))
        
        ax.set_title(f'Mapa de Classes de Solo - {nome_area}', fontsize=18, fontweight='bold', pad=20)
        plt.subplots_adjust(left=0.1, right=0.75, top=0.92, bottom=0.1)
        return fig

    def _exportar_relatorio_pdf(self, figura_mapa, df_estatisticas, nome_area, nome_arquivo_saida):
        cfg = self.config
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo_saida)
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        
        with PdfPages(caminho_completo) as pdf:
            pdf.savefig(figura_mapa, dpi=cfg.DPI_SAIDA)
            plt.close(figura_mapa)
            
            fig_tabela, ax_tabela = plt.subplots(figsize=(11.69, 8.27), facecolor='white')
            ax_tabela.axis('off')
            ax_tabela.set_title(f"Análise Quantitativa de Solos - {nome_area}", fontsize=18, fontweight='bold', pad=20)
            
            df_display = df_estatisticas.copy()
            df_display['Área (ha)'] = df_display['area_ha'].map('{:,.2f}'.format)
            df_display.rename(columns={cfg.COLUNA_SOLOS: 'Classe de Solo'}, inplace=True)
            
            tabela = ax_tabela.table(cellText=df_display[['Classe de Solo', 'Área (ha)']].values,
                                     colLabels=['Classe de Solo', 'Área (ha)'],
                                     cellLoc='left', 
                                     loc='upper center',
                                     colWidths=[0.7, 0.2])
            
            tabela.auto_set_font_size(False)
            tabela.set_fontsize(12)
            tabela.scale(1, 2)
            
            for (i, j), cell in tabela.get_celld().items():
                if i == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#40466e')
                cell.set_edgecolor('lightgray')
                cell.set_height(0.1)
                
            pdf.savefig(fig_tabela, bbox_inches='tight', dpi=cfg.DPI_SAIDA)
            plt.close(fig_tabela)
            
        return caminho_completo

    def run(self):
        try:
            print("="*80); print("       INICIANDO PIPELINE DE ANÁLISE DE SOLOS "); print("="*80)

            aoi_gdf = self._obter_area_de_interesse()
            
            solos_brasil_gdf = self._carregar_dados_vetoriais(
                self.config.PATH_SOLOS_BRASIL, 
                crs_alvo=self.config.CRS_GEOGRAFICO, 
                encoding='latin1'
            )
            
            solos_recortado_gdf, estatisticas_df = self._analisar_solos(solos_brasil_gdf, aoi_gdf)
            
            if solos_recortado_gdf is None or estatisticas_df.empty:
                print(f"\nAVISO: Nenhum polígono de solo encontrado na área de interesse. Relatório não gerado.")
            else:
                if self.config.TIPO_DE_AREA == 'municipio':
                    nome_area = self.config.NOME_MUNICIPIO_ALVO
                else: 
                    nome_area = aoi_gdf[self.config.COLUNA_NOME_IMOVEL].iloc[0]

                nome_base_sanitizado = self._sanitizar_nome_arquivo(nome_area)
                nome_arquivo_final = f"Relatorio_Solos_{nome_base_sanitizado}.pdf"

                display(HTML(f"<h3>Relatório para: '{nome_area}'</h3>"))
                figura_mapa_display = self._gerar_figura_mapa(solos_recortado_gdf, aoi_gdf, nome_area)
                display(figura_mapa_display)
                plt.close(figura_mapa_display)

                figura_mapa_pdf = self._gerar_figura_mapa(solos_recortado_gdf, aoi_gdf, nome_area)
                caminho_salvo = self._exportar_relatorio_pdf(figura_mapa_pdf, estatisticas_df, nome_area, nome_arquivo_final)
                
                print("\n" + "="*80); print("               RELATÓRIO DE ANÁLISE CONCLUÍDO"); print("="*80)
                print(f"\nO relatório final foi salvo em: {caminho_salvo}")
        
        except Exception as e:
            print(f"\n[ERRO CRÍTICO] Ocorreu um erro no pipeline: {e}")
            print("Verifique os caminhos dos arquivos e os parâmetros na classe de Configuração.")

if __name__ == "__main__":
    configuracao_analise_municipio = Config(
        TIPO_DE_AREA='imovel',
        NOME_MUNICIPIO_ALVO='Poços de Caldas'
    )
    
    pipeline_municipio = SoilAnalysisPipeline(configuracao_analise_municipio)
    pipeline_municipio.run()