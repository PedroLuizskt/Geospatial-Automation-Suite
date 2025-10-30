## Análise de Temperatura da Média Diurna da Superfície LST - Autor = Pedro Luiz

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
import contextily as ctx
import unicodedata
import re
import numpy as np
import pandas as pd
import os
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
from dataclasses import dataclass, field
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FuncFormatter
from IPython.display import display, HTML


@dataclass
class TemperatureConfig:
    """Encapsula todas as configurações do pipeline de análise de temperatura."""
    PATH_TEMPERATURA_BRASIL: str = r"C:\Users\pedro\Downloads\python_gis\script_temperatura\temperatura_media_anual_brasil_2015-2024.tif"
    PATH_MUNICIPIOS_BRASIL: str = r"C:\Users\pedro\Downloads\python_gis\script_temperatura\BR_Municipios_2024\BR_Municipios_2024.shp"
    PATH_IMOVEL_ALVO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\MG-3121605-0C5FDE61120E4BB7B3CD1728A97638E2\Area_do_Imovel\Area_do_Imovel.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\estagio_ie\resultados_"
    COLUNA_NOME_MUNICIPIO: str = 'NM_MUN'
    COLUNA_NOME_IMOVEL: str = 'recibo'
    ENCODING_VETORES: str = 'utf-8'
    DPI_SAIDA: int = 300
    TIPO_DE_AREA: str = 'imovel' 
    NOME_MUNICIPIO_ALVO: str = 'Urupema'
    CRS_GEOGRAFICO: str = 'EPSG:4674'     
    CRS_WEB_MERCATOR: str = 'EPSG:3857'   
    
    
    CORES_TEMPERATURA: list = field(default_factory=lambda: ['#0000ff', '#00ffff', '#ffff00', '#ff0000', '#800000'])
    MIN_TEMP: float = 15.0 
    MAX_TEMP: float = 35.0 


class TemperaturePipeline:
    """Orquestra o fluxo completo da análise de temperatura da superfície."""

    def __init__(self, config: TemperatureConfig):
        self.config = config

    def _sanitizar_nome_arquivo(self, nome: str) -> str:
        """Normaliza e limpa uma string para ser usada como um nome de arquivo seguro."""
        nome = unicodedata.normalize('NFKD', nome).encode('ascii', 'ignore').decode('utf-8')
        nome = re.sub(r'[^\w\s-]', '', nome).strip().lower()
        nome = re.sub(r'[-\s]+', '-', nome)
        return nome[:100]

    def _carregar_dados_vetoriais(self, path: str, crs_alvo: str, **kwargs) -> gpd.GeoDataFrame:
        """Carrega dados vetoriais, valida o CRS e converte para o CRS alvo."""
        print(f"Carregando dados de: {os.path.basename(path)}...")
        gdf = gpd.read_file(path, **kwargs)
        if gdf.crs is None:
            print(f"  AVISO: CRS não definido. Assumindo {self.config.CRS_GEOGRAFICO}.")
            gdf.set_crs(self.config.CRS_GEOGRAFICO, inplace=True)
        return gdf.to_crs(crs_alvo)

    def _obter_area_de_interesse(self) -> gpd.GeoDataFrame:
        """Filtra, carrega e unifica a geometria da AOI, garantindo o CRS padrão."""
        cfg = self.config
        print(f"\nObtendo Área de Interesse (AOI): '{cfg.TIPO_DE_AREA}'...")
        if cfg.TIPO_DE_AREA == 'municipio':
            gdf = self._carregar_dados_vetoriais(cfg.PATH_MUNICIPIOS_BRASIL, cfg.CRS_GEOGRAFICO, encoding=cfg.ENCODING_VETORES)
            area_filtrada_gdf = gdf[gdf[cfg.COLUNA_NOME_MUNICIPIO] == cfg.NOME_MUNICIPIO_ALVO]
            if area_filtrada_gdf.empty: raise ValueError(f"Município '{cfg.NOME_MUNICIPIO_ALVO}' não encontrado.")
            print(f"  Unificando geometria para '{cfg.NOME_MUNICIPIO_ALVO}' (dissolve)...")
            area_gdf = area_filtrada_gdf.dissolve().reset_index()

        elif cfg.TIPO_DE_AREA == 'imovel':
            area_gdf = self._carregar_dados_vetoriais(cfg.PATH_IMOVEL_ALVO, cfg.CRS_GEOGRAFICO, encoding=cfg.ENCODING_VETORES)
        else:
            raise ValueError("Tipo de área inválido. Escolha 'municipio' ou 'imovel'.")
        if area_gdf.empty: raise ValueError("A Área de Interesse (AOI) está vazia.")
        return area_gdf

    def _processar_temperatura(self, path_raster: str, aoi_gdf: gpd.GeoDataFrame):
        """Recorta, analisa e vetoriza os dados de temperatura para a AOI."""
        print("Processando dados de Temperatura da Superfície (LST)...")
        with rasterio.open(path_raster) as src:
            aoi_reprojetado = aoi_gdf.to_crs(src.crs)
            raster_array, transform = mask(src, aoi_reprojetado.geometry, crop=True, all_touched=True, nodata=np.nan)
            raster_array = raster_array[0]
            
            dados_validos = raster_array[~np.isnan(raster_array)]
            if dados_validos.size == 0:
                return None, None
            
            estatisticas = {
                "LST Mínima (°C)": np.min(dados_validos), 
                "LST Máxima (°C)": np.max(dados_validos), 
                "LST Média (°C)": np.mean(dados_validos)
            }
            estatisticas_df = pd.DataFrame.from_dict(estatisticas, orient='index', columns=['Valor'])
            
            print("Vetorizando dados para o mapa...")
            resultados = ({'properties': {'valor': v}, 'geometry': shape(s)} 
                          for i, (s, v) in enumerate(shapes(raster_array.astype('float32'), mask=~np.isnan(raster_array), transform=transform)) if v > 0)
            temperatura_gdf = gpd.GeoDataFrame.from_features(list(resultados), crs=src.crs)
            
            return gpd.clip(temperatura_gdf, aoi_reprojetado), estatisticas_df

    def _add_grid_inteligente(self, ax, gdf_geografico):
        """Adiciona uma grade de coordenadas geográficas formatadas ao mapa."""
        cfg = self.config
        minx, miny, maxx, maxy = gdf_geografico.total_bounds
        x_interval, y_interval = (maxx - minx) / 5, (maxy - miny) / 5
        x_ticks = np.arange(minx, maxx, x_interval); y_ticks = np.arange(miny, maxy, y_interval)
        
        lon_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_ticks, [miny] * len(x_ticks)), crs=cfg.CRS_GEOGRAFICO)
        lat_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([minx] * len(y_ticks), y_ticks), crs=cfg.CRS_GEOGRAFICO)
        
        lon_ticks_proj = lon_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.x
        lat_ticks_proj = lat_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.y
        
        ax.set_xticks(lon_ticks_proj); ax.set_yticks(lat_ticks_proj)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{x_ticks[pos]:.2f}°" if pos < len(x_ticks) else ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{y_ticks[pos]:.2f}°" if pos < len(y_ticks) else ""))
        
        ax.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax.tick_params(axis='x', rotation=45, labelsize=10); ax.tick_params(axis='y', labelsize=10)

    def _plotar_mapa(self, vetor_temperatura: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame, nome_area: str) -> plt.Figure:
        """Cria e formata a figura do mapa de temperatura."""
        cfg = self.config
        print("Gerando o mapa com formatação profissional...")
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
        ax.set_facecolor('#f0f0f0')
        
        temp_plot = vetor_temperatura.to_crs(cfg.CRS_WEB_MERCATOR)
        aoi_plot = aoi_gdf.to_crs(cfg.CRS_WEB_MERCATOR)
        
        cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_temp", cfg.CORES_TEMPERATURA)
        
        temp_plot.plot(column='valor', ax=ax, cmap=cmap_custom, vmin=cfg.MIN_TEMP, vmax=cfg.MAX_TEMP, linewidth=0, alpha=0.75)
        aoi_plot.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, linestyle='--')
        
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto')
        self._add_grid_inteligente(ax, aoi_gdf)
        
        ax.set_title(f'Temperatura Média da Superfície (LST) (2015-2024) - {nome_area}', fontsize=18, fontweight='bold')
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction')
        
        norm = mcolors.Normalize(vmin=cfg.MIN_TEMP, vmax=cfg.MAX_TEMP)
        sm = plt.cm.ScalarMappable(cmap=cmap_custom, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
        cbar.set_label('LST (°C)', size=12)
        
        ax.add_artist(ScaleBar(1, 'm', location='lower right', box_alpha=0.8, pad=0.5))
        fig.text(0.82, 0.2, f"Fonte: Google Earth Engine\nAutor: Pedro Luiz\nDatum: SIRGAS 2000", ha='left', va='bottom', fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1))
        
        plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
        return fig

    def _exportar_relatorio_pdf(self, figura_mapa: plt.Figure, df_estatisticas: pd.DataFrame, nome_area: str, nome_arquivo_saida: str):
        """Exporta o relatório final em PDF."""
        cfg = self.config
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo_saida)
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        print(f"\nExportando relatório PDF para: {caminho_completo}...")
        
        with PdfPages(caminho_completo) as pdf:
            pdf.savefig(figura_mapa, bbox_inches='tight', dpi=cfg.DPI_SAIDA)
            plt.close(figura_mapa)
            
            fig_tabela, ax_tabela = plt.subplots(figsize=(8.27, 11.69))
            ax_tabela.axis('off')
            ax_tabela.set_title(f"Análise Estatística de LST\n{nome_area}", fontsize=16, fontweight='bold', pad=20)
            
            df_display = df_estatisticas.copy()
            df_display['Valor'] = [f"{v:_.2f}".replace('.', ',').replace('_', '.') for v in df_display['Valor']]
            
            tabela = ax_tabela.table(cellText=df_display.values, rowLabels=df_display.index, 
                                     colLabels=df_display.columns, cellLoc='center', 
                                     loc='upper center', rowLoc='left', colWidths=[0.4])
            tabela.auto_set_font_size(False); tabela.set_fontsize(14); tabela.scale(1.5, 2.5)
            
            pdf.savefig(fig_tabela, bbox_inches='tight', dpi=cfg.DPI_SAIDA)
            plt.close(fig_tabela)
            
        print("Relatório exportado com sucesso!")
        return caminho_completo

    def run(self):
        """Orquestra a execução completa do pipeline."""
        try:
            print("="*80); print("       INICIANDO PIPELINE DE ANÁLISE DE TEMPERATURA "); print("="*80)
            
            aoi_gdf = self._obter_area_de_interesse()
            
            if self.config.TIPO_DE_AREA == 'municipio':
                nome_area = self.config.NOME_MUNICIPIO_ALVO
            else: # 'imovel'
                if self.config.COLUNA_NOME_IMOVEL in aoi_gdf.columns:
                    nome_area = aoi_gdf[self.config.COLUNA_NOME_IMOVEL].iloc[0]
                else:
                    print(f"AVISO: Coluna '{self.config.COLUNA_NOME_IMOVEL}' não encontrada. Usando nome genérico.")
                    nome_area = "Imovel Rural Sem Nome"

            nome_base_sanitizado = self._sanitizar_nome_arquivo(nome_area)
            nome_arquivo_final = f"Relatorio_Temperatura_{nome_base_sanitizado}.pdf"

            temperatura_gdf, estatisticas_df = self._processar_temperatura(self.config.PATH_TEMPERATURA_BRASIL, aoi_gdf)
            
            if temperatura_gdf is None or temperatura_gdf.empty:
                print("\nAVISO: Nenhum dado de temperatura encontrado para a área de interesse.")
            else:
                display(HTML(f"<h2>Análise de Temperatura para: {nome_area}</h2>"))
                figura_para_display = self._plotar_mapa(temperatura_gdf, aoi_gdf, nome_area)
                display(figura_para_display)
                plt.close(figura_para_display)
                
                print("\n" + "="*60); print(f"             ANÁLISE ESTATÍSTICA PARA: {nome_area.upper()}"); print("="*60)
                stats_console = estatisticas_df.copy()
                stats_console['Valor'] = [f"{v:_.2f}".replace('.', ',').replace('_', '.') for v in stats_console['Valor']]
                print(stats_console.to_string()); print("="*60)
                
                figura_para_pdf = self._plotar_mapa(temperatura_gdf, aoi_gdf, nome_area)
                caminho_salvo = self._exportar_relatorio_pdf(figura_para_pdf, estatisticas_df, nome_area, nome_arquivo_final)
                print(f"\nAnálise de temperatura para '{nome_area}' concluída! Relatório salvo em:\n{caminho_salvo}")
        
        except (ValueError, FileNotFoundError) as e:
            print(f"\n[ERRO CONTROLADO] {e}")
        except Exception as e:
            print(f"\n[ERRO CRÍTICO INESPERADO] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    config = TemperatureConfig(
        TIPO_DE_AREA='imovel',
        NOME_MUNICIPIO_ALVO='Telêmaco Borba'
    )
    pipeline = TemperaturePipeline(config)
    pipeline.run()