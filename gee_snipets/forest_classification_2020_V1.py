# Classes de floresta VIA GEE - Autor = Pedro Luiz

import ee
import geemap
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import contextily as ctx
import unicodedata
import re
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass, field
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar
from IPython.display import display, HTML


@dataclass
class ForestClassConfig:
    PATH_IMOVEL_ALVO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\PA-1501006-618F8CF3DE6B427DA662E8005BFBFE0A\Area_do_Imovel\Area_do_Imovel.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\python_gis\classes_floresta\re_"
    COLUNA_NOME_IMOVEL: str = 'recibo'
    ENCODING_IMOVEL: str = 'utf-8'
    DPI_SAIDA: int = 300
    ASSET_ID: str = 'NASA/ORNL/global_forest_classification_2020/V1'
    ASSET_BAND: str = 'classification'
    ESCALA_METROS: int = 30
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    CRS_WEB_MERCATOR: str = 'EPSG:3857'
    CLASSES_FLORESTAIS: dict = field(default_factory=dict)

    def __post_init__(self):
        self.CLASSES_FLORESTAIS = {
            1: {'nome': 'Floresta Primária', 'cor': '#00ff00'},
            2: {'nome': 'Floresta Secundária Jovem (<=20 anos)', 'cor': '#ff0000'},
            3: {'nome': 'Floresta Secundária Antiga (>20 anos)', 'cor': '#6666ff'}
        }


class ForestClassificationPipeline:
    def __init__(self, config: ForestClassConfig):
        self.config = config
        self._initialize_gee()

    def _initialize_gee(self):
        try:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
            print("Serviço Google Earth Engine inicializado com sucesso.")
        except Exception:
            print("Autenticação GEE necessária...")
            ee.Authenticate()
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
            print("Serviço GEE inicializado com sucesso.")

    def _sanitizar_nome_arquivo(self, nome: str) -> str:
        nome = unicodedata.normalize('NFKD', str(nome)).encode('ascii', 'ignore').decode('utf-8')
        nome = re.sub(r'[^\w\s-]', '', nome).strip().lower()
        return re.sub(r'[-\s]+', '-', nome)[:100]

    def _load_aoi(self) -> gpd.GeoDataFrame:
        cfg = self.config
        print(f"\nCarregando a Área de Interesse (AOI) de: {cfg.PATH_IMOVEL_ALVO}")
        try:
            aoi_gdf = gpd.read_file(cfg.PATH_IMOVEL_ALVO, encoding=cfg.ENCODING_IMOVEL)
            aoi_gdf_geo = aoi_gdf.to_crs(cfg.CRS_GEOGRAFICO)
            
            nome_imovel = None
            if cfg.COLUNA_NOME_IMOVEL in aoi_gdf_geo.columns:
                nome_imovel = aoi_gdf_geo[cfg.COLUNA_NOME_IMOVEL].iloc[0]
            
            aoi_dissolved = aoi_gdf_geo.dissolve()
            
            if nome_imovel:
                aoi_dissolved[cfg.COLUNA_NOME_IMOVEL] = nome_imovel

            print("  -> Geometria da AOI carregada e unificada com sucesso.")
            return aoi_dissolved
        except Exception as e:
            raise FileNotFoundError(f"Não foi possível ler o arquivo do imóvel. Erro: {e}")

    def _get_data_from_gee(self, aoi_gdf: gpd.GeoDataFrame, nome_sanitizado: str) -> str:
        cfg = self.config
        print("\nIniciando processamento em nuvem no Google Earth Engine...")
        aoi_ee = geemap.geopandas_to_ee(aoi_gdf)
        
        image = ee.ImageCollection(cfg.ASSET_ID).select(cfg.ASSET_BAND).mosaic()
        clipped_image = image.clip(aoi_ee)
        
        caminho_temporario = os.path.join(cfg.PATH_EXPORTACAO, f'temp_forest_class_{nome_sanitizado}.tif')
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        
        print(f"  -> Realizando download do raster recortado para a AOI...")
        geemap.ee_export_image(
            clipped_image,
            filename=caminho_temporario, 
            scale=cfg.ESCALA_METROS,
            region=aoi_ee.geometry(), 
            crs=cfg.CRS_WEB_MERCATOR
        )
        if not os.path.exists(caminho_temporario):
            raise FileNotFoundError("Falha no download da imagem do GEE.")
        print("  -> Download concluído com sucesso.")
        return caminho_temporario

    def _calculate_statistics(self, raster_path: str, aoi_gdf_mercator: gpd.GeoDataFrame) -> pd.DataFrame:
        cfg = self.config
        print("Analisando o raster localmente para quantificar as áreas...")
        with rasterio.open(raster_path) as src:
            raster_array, _ = mask(src, aoi_gdf_mercator.geometry, crop=True, nodata=0)
            raster_array = raster_array[0]
            dados_validos = raster_array[raster_array != 0]
            if dados_validos.size == 0:
                print("  -> AVISO: Nenhuma classe de floresta do asset foi encontrada na área do imóvel.")
                return pd.DataFrame(columns=['Classe', 'Área (ha)'])
            
            unique_classes, counts = np.unique(dados_validos, return_counts=True)
            pixel_count_dict = dict(zip(unique_classes, counts))
            pixel_area_m2 = cfg.ESCALA_METROS ** 2
            resultados = []
            for classe_id, info in cfg.CLASSES_FLORESTAIS.items():
                if classe_id in pixel_count_dict:
                    num_pixels = pixel_count_dict[classe_id]
                    area_ha = (num_pixels * pixel_area_m2) / 10000
                    resultados.append({'Classe': info['nome'], 'Área (ha)': area_ha})
            stats_df = pd.DataFrame(resultados)
            print("  -> Análise de área concluída.")
            return stats_df.sort_values(by='Área (ha)', ascending=False)

    def _add_grid_inteligente(self, ax, gdf_geografico):
        cfg = self.config
        minx, miny, maxx, maxy = gdf_geografico.total_bounds
        x_interval, y_interval = (maxx - minx) / 5, (maxy - miny) / 5
        x_ticks = np.arange(minx, maxx + x_interval, x_interval)
        y_ticks = np.arange(miny, maxy + y_interval, y_interval)
        
        lon_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_ticks, [miny] * len(x_ticks)), crs=cfg.CRS_GEOGRAFICO)
        lat_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([minx] * len(y_ticks), y_ticks), crs=cfg.CRS_GEOGRAFICO)
        
        lon_ticks_proj = lon_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.x
        lat_ticks_proj = lat_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.y
        
        ax.set_xticks(lon_ticks_proj)
        ax.set_yticks(lat_ticks_proj)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{x_ticks[pos]:.2f}°" if pos < len(x_ticks) else ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{y_ticks[pos]:.2f}°" if pos < len(y_ticks) else ""))
        
        ax.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    def _plot_map(self, raster_path: str, aoi_gdf_geo: gpd.GeoDataFrame, aoi_gdf_mercator: gpd.GeoDataFrame, nome_area: str) -> plt.Figure:
        cfg = self.config
        print("Gerando o mapa de classificação florestal com formatação padronizada...")
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
        ax.set_facecolor('#f0f0f0')

        with rasterio.open(raster_path) as src:
            data, transform = mask(src, aoi_gdf_mercator.geometry, crop=True, nodata=0)
            
            plot_data = data.astype(float)
            plot_data[plot_data == 0] = np.nan
            
            classes_presentes = [c for c in np.unique(plot_data[~np.isnan(plot_data)]) if c in cfg.CLASSES_FLORESTAIS]
            
            if not classes_presentes:
                ax.text(0.5, 0.5, 'Nenhuma classe florestal encontrada na área', transform=ax.transAxes, ha='center', fontsize=12, color='black')
            else:
                cores = [cfg.CLASSES_FLORESTAIS[c]['cor'] for c in sorted(classes_presentes)]
                cmap = ListedColormap(cores)
                bounds = sorted([c - 0.5 for c in classes_presentes]) + [max(classes_presentes) + 0.5]
                norm = BoundaryNorm(bounds, cmap.N)
                show(plot_data, ax=ax, transform=transform, cmap=cmap, norm=norm, alpha=0.8, zorder=2)
        
        aoi_gdf_mercator.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, linestyle='--', zorder=3)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto', zorder=1)
        
        self._add_grid_inteligente(ax, aoi_gdf_geo)

        ax.set_title(f'Classificação Florestal (IPCC Tier 1) - Imóvel: {nome_area}', fontsize=18, fontweight='bold')
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction', color='black')
        
        legend_elements = [Patch(facecolor=info['cor'], edgecolor='black', label=info['nome']) for c_id, info in cfg.CLASSES_FLORESTAIS.items() if c_id in classes_presentes]
        ax.legend(handles=legend_elements, title='Classes na Área', loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True)

        ax.add_artist(ScaleBar(1, 'm', location='upper right', box_alpha=0.8, pad=0.5))
        fig.text(0.65, 0.2, f"Fonte: NASA ORNL DAAC (2020)\nProcessamento: GEE\nAutor: Pedro Luiz", ha='left', va='bottom', fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1))
        
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        plt.tight_layout(rect=[0, 0.05, 0.85, 0.95])
        print("  -> Mapa gerado.")
        return fig

    def _export_report(self, figure: plt.Figure, stats_df: pd.DataFrame, nome_area: str):
        cfg = self.config
        nome_sanitizado = self._sanitizar_nome_arquivo(nome_area)
        nome_arquivo = f"Relatorio_Florestal_{nome_sanitizado}.pdf"
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo)
        print(f"\nExportando relatório PDF para: {caminho_completo}")

        with PdfPages(caminho_completo) as pdf:
            pdf.savefig(figure, dpi=cfg.DPI_SAIDA, bbox_inches='tight', pad_inches=0.1)
            plt.close(figure)

            if not stats_df.empty:
                fig_tabela, ax_tabela = plt.subplots(figsize=(8.27, 11.69))
                ax_tabela.axis('off')
                ax_tabela.set_title(f"Quantificação de Área por Classe Florestal\nImóvel: {nome_area}", fontsize=16, fontweight='bold', pad=20)
                df_display = stats_df.copy()
                df_display['Área (ha)'] = df_display['Área (ha)'].map('{:,.2f}'.format)
                tabela = ax_tabela.table(cellText=df_display.values, colLabels=df_display.columns, cellLoc='center', loc='upper center', colWidths=[0.5, 0.2])
                tabela.auto_set_font_size(False); tabela.set_fontsize(12); tabela.scale(1.5, 2.5)
                pdf.savefig(fig_tabela, bbox_inches='tight', dpi=cfg.DPI_SAIDA)
                plt.close(fig_tabela)
        
        print(f"  -> Relatório exportado com sucesso!")
        return caminho_completo

    def run(self):
        caminho_temp = None
        try:
            print("="*80); print("      INICIANDO PIPELINE DE CLASSIFICAÇÃO FLORESTAL  "); print("="*80)
            aoi_gdf_geo = self._load_aoi()
            aoi_gdf_mercator = aoi_gdf_geo.to_crs(self.config.CRS_WEB_MERCATOR)
            nome_imovel = aoi_gdf_geo[self.config.COLUNA_NOME_IMOVEL].iloc[0] if self.config.COLUNA_NOME_IMOVEL in aoi_gdf_geo.columns else "Imovel_Sem_Nome"
            nome_sanitizado = self._sanitizar_nome_arquivo(nome_imovel)
            caminho_temp = self._get_data_from_gee(aoi_gdf_geo, nome_sanitizado)
            stats_df = self._calculate_statistics(caminho_temp, aoi_gdf_mercator)
            
            if not stats_df.empty:
                display(HTML(f"<h3>Análise Quantitativa para: {nome_imovel}</h3>"))
                display(stats_df)
                
                print("\n" + "="*80)
                display(HTML(f"<h3>Mapa de Classificação Florestal para: {nome_imovel}</h3>"))
                figura_para_display = self._plot_map(caminho_temp, aoi_gdf_geo, aoi_gdf_mercator, nome_imovel)
                display(figura_para_display)
                plt.close(figura_para_display)

                figura_para_pdf = self._plot_map(caminho_temp, aoi_gdf_geo, aoi_gdf_mercator, nome_imovel)
                self._export_report(figura_para_pdf, stats_df, nome_imovel)
            else:
                print(f"\nAnálise para '{nome_imovel}' não gerou resultados. O relatório não será criado.")
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
    configuracao = ForestClassConfig()
    pipeline = ForestClassificationPipeline(configuracao)
    pipeline.run()