## Análise Carbono Orgânico do Solo Asset 2 - Autor = Pedro Luiz

import ee
import geemap
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot
from rasterio.mask import mask  
import contextily as ctx
import unicodedata
import re
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FuncFormatter
from IPython.display import display, HTML

@dataclass
class SoilCarbonConfig:
    PATH_IMOVEL_ALVO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\MT-5101902-F96B956F1B80430580432988F1C9E039\Area_do_Imovel\Area_do_Imovel.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\python_gis\quant_C\resultados_carbono"
    DPI_SAIDA: int = 300
    COLUNA_NOME_IMOVEL: str = 'recibo'
    ASSET_ID: str = 'projects/mapbiomas-public/assets/brazil/soil/collection2_1/mapbiomas_brazil_collection21_soil_carbon_v2'
    ESCALA_METROS: int = 30
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    CRS_WEB_MERCATOR: str = 'EPSG:3857'
    ENCODING_IMOVEL: str = 'utf-8'
    COLORMAP: str = 'viridis'

class SoilCarbonPipeline:
    def __init__(self, config: SoilCarbonConfig):
        self.config = config
        self._initialize_gee()

    def _initialize_gee(self):
        try:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
            print("Serviço Google Earth Engine inicializado com sucesso.")
        except Exception:
            print("Autenticação GEE necessária. Siga as instruções no navegador.")
            ee.Authenticate()
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
            print("Serviço GEE inicializado com sucesso após autenticação.")

    def _sanitizar_nome_arquivo(self, nome: str) -> str:
        nome = unicodedata.normalize('NFKD', str(nome)).encode('ascii', 'ignore').decode('utf-8')
        nome = re.sub(r'[^\w\s-]', '', nome).strip().lower()
        return re.sub(r'[-\s]+', '-', nome)[:100]

    def _get_data_from_gee(self, aoi_gdf: gpd.GeoDataFrame, nome_sanitizado: str) -> str:
        cfg = self.config
        print("\nIniciando processamento em nuvem no Google Earth Engine...")
        aoi_ee = geemap.geopandas_to_ee(aoi_gdf)
        median_image = ee.Image(cfg.ASSET_ID).reduce(ee.Reducer.median()).rename('soc_median')
        clipped_image = median_image.clip(aoi_ee)
        caminho_temporario = os.path.join(cfg.PATH_EXPORTACAO, f'temp_soc_{nome_sanitizado}.tif')
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        print(f"  -> Realizando download do raster processado para: {os.path.basename(caminho_temporario)}")
        geemap.ee_export_image(
            clipped_image, filename=caminho_temporario, scale=cfg.ESCALA_METROS,
            region=aoi_ee.geometry(), crs=cfg.CRS_WEB_MERCATOR
        )
        if not os.path.exists(caminho_temporario):
            raise FileNotFoundError("Falha no download da imagem do GEE.")
        print("  -> Download concluído com sucesso.")
        return caminho_temporario

    def _analyze_local_raster(self, raster_path: str, aoi_gdf: gpd.GeoDataFrame):
        print("Analisando o raster localmente...")
        with rasterio.open(raster_path) as src:
            aoi_reprojetado = aoi_gdf.to_crs(self.config.CRS_WEB_MERCATOR)
            raster_array, _ = mask(src, aoi_reprojetado.geometry, crop=True, nodata=np.nan)
            
            raster_array = raster_array[0]
            dados_validos_brutos = raster_array[~np.isnan(raster_array)]
            
            if dados_validos_brutos.size == 0:
                print("AVISO: Nenhum pixel com dados válidos encontrado no imóvel.")
                return None

            dados_validos_limpos = np.where(dados_validos_brutos < 0, 0, dados_validos_brutos)
            pixels_corrigidos = np.sum(dados_validos_brutos < 0)
            if pixels_corrigidos > 0:
                print(f"  -> {pixels_corrigidos} pixels com valores negativos foram corrigidos para 0.")

            stats = {
                "Mínimo (t/ha)": np.min(dados_validos_limpos),
                "Máximo (t/ha)": np.max(dados_validos_limpos),
                "Média (t/ha)": np.mean(dados_validos_limpos),
                "Mediana (t/ha)": np.median(dados_validos_limpos),
                "Desvio Padrão (t/ha)": np.std(dados_validos_limpos)
            }
            stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Valor'])
            print("  -> Estatísticas calculadas a partir dos dados limpos.")
            return stats_df

    def _add_grid_inteligente(self, ax, gdf_geografico):
        cfg = self.config
        minx, miny, maxx, maxy = gdf_geografico.total_bounds
        x_interval, y_interval = (maxx - minx) / 5, (maxy - miny) / 5
        x_ticks, y_ticks = np.arange(minx, maxx, x_interval), np.arange(miny, maxy, y_interval)
        lon_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_ticks, [miny] * len(x_ticks)), crs=cfg.CRS_GEOGRAFICO)
        lat_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([minx] * len(y_ticks), y_ticks), crs=cfg.CRS_GEOGRAFICO)
        lon_ticks_proj, lat_ticks_proj = lon_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.x, lat_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.y
        ax.set_xticks(lon_ticks_proj); ax.set_yticks(lat_ticks_proj)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{x_ticks[pos]:.3f}°" if pos < len(x_ticks) else ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{y_ticks[pos]:.3f}°" if pos < len(y_ticks) else ""))
        ax.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax.tick_params(axis='x', rotation=45, labelsize=9); ax.tick_params(axis='y', labelsize=9)

    def _plot_map(self, raster_path: str, aoi_gdf: gpd.GeoDataFrame, nome_area: str) -> plt.Figure:
        cfg = self.config
        print("Gerando mapa profissional...")
        fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27), facecolor='white')
        ax.set_facecolor('#f0f0f0')

        with rasterio.open(raster_path) as src:
            aoi_plot = aoi_gdf.to_crs(cfg.CRS_WEB_MERCATOR)
            data, transform = mask(src, aoi_plot.geometry, crop=True, nodata=np.nan)
            im = rasterio.plot.show(data, ax=ax, transform=transform, cmap=cfg.COLORMAP, alpha=0.85, zorder=2)
            cbar = fig.colorbar(im.get_images()[0], ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
            cbar.set_label('Carbono Orgânico do Solo (toneladas/hectare)', size=12)
            aoi_plot.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, linestyle='--', zorder=3)
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zorder=1)
        
        self._add_grid_inteligente(ax, aoi_gdf)
        
        ax.set_title(f'Carbono Orgânico do Solo (0-30cm) - Mediana da Série Temporal(1985-2023)\nImóvel: {nome_area}', fontsize=16, fontweight='bold')
        ax.add_artist(ScaleBar(1, 'm', location='upper right', box_alpha=0.8, pad=0.5))
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction')
        fig.text(0.75, 0.09, f"Fonte: MapBiomas Soil Collection 2.1 (GEE)\nAutor: Pedro Luiz\nDatum Geográfico: SIRGAS 2000",
                 ha='left', va='bottom', fontsize=9, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1))
        plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.95])
        print("  -> Mapa gerado com sucesso.")
        return fig

    def _export_report(self, figure: plt.Figure, stats_df: pd.DataFrame, nome_area: str):
        cfg = self.config
        nome_sanitizado = self._sanitizar_nome_arquivo(nome_area)
        nome_arquivo = f"Relatorio_Carbono_Solo_{nome_sanitizado}.pdf"
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo)
        print(f"\nExportando relatório PDF para: {caminho_completo}")

        with PdfPages(caminho_completo) as pdf:
            pdf.savefig(figure, dpi=cfg.DPI_SAIDA)
            plt.close(figure)
            if stats_df is not None:
                fig_tabela, ax_tabela = plt.subplots(figsize=(8.27, 11.69))
                ax_tabela.axis('off')
                ax_tabela.set_title(f"Estatísticas de Carbono Orgânico do Solo\n{nome_area}", fontsize=16, fontweight='bold', pad=20)
                df_display = stats_df.copy()
                df_display['Valor'] = df_display['Valor'].map('{:,.2f}'.format)
                tabela = ax_tabela.table(cellText=df_display.values, rowLabels=df_display.index,
                                         colLabels=df_display.columns, cellLoc='center', loc='upper center',
                                         rowLoc='left', colWidths=[0.3])
                tabela.auto_set_font_size(False); tabela.set_fontsize(14); tabela.scale(1.5, 2.5)
                pdf.savefig(fig_tabela, bbox_inches='tight', dpi=cfg.DPI_SAIDA)
                plt.close(fig_tabela)
        print("  -> Relatório exportado com sucesso!")
        return caminho_completo

    def run(self):
        caminho_temp = None
        try:
            print("="*80); print("      INICIANDO PIPELINE ANÁLISE DE CARBONO DO SOLO "); print("="*80)
            
            aoi_gdf = gpd.read_file(self.config.PATH_IMOVEL_ALVO, encoding=self.config.ENCODING_IMOVEL).to_crs(self.config.CRS_GEOGRAFICO)
            nome_imovel = aoi_gdf[self.config.COLUNA_NOME_IMOVEL].iloc[0] if self.config.COLUNA_NOME_IMOVEL in aoi_gdf.columns else "Imovel Sem Nome"
            nome_sanitizado = self._sanitizar_nome_arquivo(nome_imovel)
            
            caminho_temp = self._get_data_from_gee(aoi_gdf, nome_sanitizado)
            stats_df = self._analyze_local_raster(caminho_temp, aoi_gdf)
            
            if stats_df is not None:
                display(HTML(f"<h2>Análise de Carbono do Solo para: {nome_imovel}</h2>"))
                display(stats_df.style.format({'Valor': '{:,.2f}'}))
                figura_mapa = self._plot_map(caminho_temp, aoi_gdf, nome_imovel)
                display(figura_mapa)
                figura_pdf = self._plot_map(caminho_temp, aoi_gdf, nome_imovel)
                self._export_report(figura_pdf, stats_df, nome_imovel)
            else:
                print(f"Análise para '{nome_imovel}' não gerou resultados, o relatório não será criado.")

            print("\n" + "="*80); print("      PIPELINE CONCLUÍDO COM SUCESSO"); print("="*80)
        except Exception as e:
            print(f"\n[ERRO CRÍTICO NO PIPELINE] Ocorreu um erro: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if caminho_temp and os.path.exists(caminho_temp):
                os.remove(caminho_temp)
                print(f"\nArquivo temporário '{os.path.basename(caminho_temp)}' removido.")

if __name__ == "__main__":
    configuracao = SoilCarbonConfig()
    pipeline = SoilCarbonPipeline(configuracao)
    pipeline.run()