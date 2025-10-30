## Pipeline de Análise Socioambiental: Tree Proximate People (FAO/GEE) por Município - Autor = Pedro Luiz

import ee
import geemap
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar
from IPython.display import display, HTML

@dataclass
class TPPConfig:
    NOME_MUNICIPIO_ALVO: str = 'Diamantina'
    SIGLA_UF: str = 'MG'
    PATH_MUNICIPIOS: str = r"C:\Users\pedro\Downloads\python_gis\tree_proximate\BR_Municipios_2024\BR_Municipios_2024.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\python_gis\tree_proximate\re_"
    COLUNA_NOME_MUNICIPIO: str = 'NM_MUN'
    COLUNA_SIGLA_UF: str = 'SIGLA_UF'
    ENCODING_MUNICIPIOS: str = 'utf-8'
    DPI_SAIDA: int = 300
    ASSET_ID: str = 'FAO/SOFO/1/TPP'
    BANDAS: list = field(default_factory=lambda: ['TPP_1km', 'TPP_1km_cropland', 'TPP_500m', 'TPP_500m_cropland'])
    ESCALA_METROS: int = 1000
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    CRS_WEB_MERCATOR: str = 'EPSG:3857'

class TPPPipeline:
    def __init__(self, config: TPPConfig):
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

    def _normalize_string(self, text: str) -> str:
        if not isinstance(text, str): return ""
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower().strip()

    def _load_aoi(self) -> gpd.GeoDataFrame:
        cfg = self.config
        print(f"\nCarregando malha municipal e buscando por '{cfg.NOME_MUNICIPIO_ALVO}' em {cfg.SIGLA_UF}...")
        try:
            municipios_gdf = gpd.read_file(cfg.PATH_MUNICIPIOS, encoding=cfg.ENCODING_MUNICIPIOS)
            municipios_gdf_uf = municipios_gdf[municipios_gdf[cfg.COLUNA_SIGLA_UF] == cfg.SIGLA_UF.upper()].copy()
            
            if municipios_gdf_uf.empty:
                raise ValueError(f"Nenhum município encontrado para a UF '{cfg.SIGLA_UF}'. Verifique a sigla e o shapefile.")
            
            municipios_gdf_uf['NM_MUN_NORM'] = municipios_gdf_uf[cfg.COLUNA_NOME_MUNICIPIO].apply(self._normalize_string)
            nome_alvo_norm = self._normalize_string(cfg.NOME_MUNICIPIO_ALVO)
            
            aoi_gdf = municipios_gdf_uf[municipios_gdf_uf['NM_MUN_NORM'] == nome_alvo_norm].to_crs(cfg.CRS_GEOGRAFICO)
            
            if aoi_gdf.empty:
                raise ValueError(f"Município '{cfg.NOME_MUNICIPIO_ALVO}' não encontrado em {cfg.SIGLA_UF}.")
                
            print("  -> Município encontrado. Geometria da AOI definida.")
            return aoi_gdf.dissolve().reset_index()
        except Exception as e:
            raise FileNotFoundError(f"Não foi possível processar o shapefile de municípios. Erro: {e}")

    def _get_data_from_gee(self, aoi_gdf: gpd.GeoDataFrame, nome_sanitizado: str) -> str:
        cfg = self.config
        print("\nIniciando processamento em nuvem no Google Earth Engine...")
        aoi_ee = geemap.geopandas_to_ee(aoi_gdf)
        image = ee.ImageCollection(cfg.ASSET_ID).mosaic().select(cfg.BANDAS)
        clipped_image = image.clip(aoi_ee)
        caminho_temporario = os.path.join(cfg.PATH_EXPORTACAO, f'temp_tpp_{nome_sanitizado}.tif')
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        print(f"  -> Realizando download do raster multi-banda para a AOI...")
        geemap.ee_export_image(clipped_image, filename=caminho_temporario, scale=cfg.ESCALA_METROS, region=aoi_ee.geometry(), crs=cfg.CRS_WEB_MERCATOR)
        if not os.path.exists(caminho_temporario):
            raise FileNotFoundError("Falha no download da imagem do GEE.")
        print("  -> Download concluído com sucesso.")
        return caminho_temporario

    def _analyze_local_raster(self, raster_path: str, aoi_gdf_mercator: gpd.GeoDataFrame) -> pd.DataFrame:
        cfg = self.config
        print("Analisando o raster localmente para extrair estatísticas...")
        resultados = []
        with rasterio.open(raster_path) as src:
            raster_bands, _ = mask(src, aoi_gdf_mercator.geometry, crop=True, nodata=np.nan)
            pixel_area_ha = (cfg.ESCALA_METROS ** 2) / 10000
            for i, band_name in enumerate(cfg.BANDAS):
                dados_validos = raster_bands[i][~np.isnan(raster_bands[i])]
                
                if dados_validos.size == 0:
                    pop_total, dens_media, dens_min, dens_max = 0, 0, 0, 0
                else:
                    pop_total = np.sum(dados_validos * pixel_area_ha)
                    dens_media = np.mean(dados_validos)
                    dens_min = np.min(dados_validos)
                    dens_max = np.max(dados_validos)
                
                resultados.append({
                    'Banda': band_name,
                    'População Total (estimada)': pop_total,
                    'Densidade Média (pessoas/ha)': dens_media,
                    'Densidade Mínima (pessoas/ha)': dens_min,
                    'Densidade Máxima (pessoas/ha)': dens_max
                })
        stats_df = pd.DataFrame(resultados)
        print("  -> Análise estatística concluída.")
        return stats_df

    def _add_grid_inteligente(self, ax, gdf_geografico):
        cfg = self.config
        minx, miny, maxx, maxy = gdf_geografico.total_bounds
        x_interval, y_interval = (maxx - minx) / 5, (maxy - miny) / 5
        x_ticks = np.arange(minx, maxx + x_interval, x_interval)
        y_ticks = np.arange(miny, maxy + y_interval, y_interval)
        lon_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_ticks, [miny] * len(x_ticks)), crs=cfg.CRS_GEOGRAFICO)
        lat_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([minx] * len(y_ticks), y_ticks), crs=cfg.CRS_GEOGRAFICO)
        lon_ticks_proj, lat_ticks_proj = lon_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.x, lat_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.y
        ax.set_xticks(lon_ticks_proj); ax.set_yticks(lat_ticks_proj)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{x_ticks[pos]:.2f}°" if pos < len(x_ticks) else ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{y_ticks[pos]:.2f}°" if pos < len(y_ticks) else ""))
        ax.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax.tick_params(axis='x', rotation=45, labelsize=10); ax.tick_params(axis='y', labelsize=10)

    def _plot_map(self, raster_path: str, band_index: int, band_name: str, aoi_gdf_geo: gpd.GeoDataFrame, aoi_gdf_mercator: gpd.GeoDataFrame) -> plt.Figure:
        cfg = self.config
        print(f"Gerando mapa para a banda: {band_name}...")
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
        ax.set_facecolor('#f0f0f0')

        with rasterio.open(raster_path) as src:
            data, transform = mask(src, aoi_gdf_mercator.geometry, crop=True, nodata=np.nan)
            plot_data = data[band_index-1].astype(float)
            plot_data[plot_data <= 0] = np.nan
            im = show(plot_data, ax=ax, transform=transform, cmap='inferno', alpha=0.8, zorder=2)
        
        path = Path.make_compound_path(*[Path(np.asarray(geom.exterior.coords)[:,:2]) for geom in aoi_gdf_mercator.geometry])
        clip_patch = PathPatch(path, transform=ax.transData, facecolor='none', edgecolor='none')
        ax.add_patch(clip_patch)
        for collection in ax.collections:
            collection.set_clip_path(clip_patch)
        if im.get_images():
            im.get_images()[0].set_clip_path(clip_patch)

        aoi_gdf_mercator.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, linestyle='--', zorder=3)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto', zorder=1)
        self._add_grid_inteligente(ax, aoi_gdf_geo)
        
        minx, miny, maxx, maxy = aoi_gdf_mercator.total_bounds
        ax.set_xlim(minx - (maxx-minx)*0.05, maxx + (maxx-minx)*0.05)
        ax.set_ylim(miny - (maxy-miny)*0.05, maxy + (maxy-miny)*0.05)

        ax.set_title(f'População Próxima a Árvores (TPP) - {cfg.NOME_MUNICIPIO_ALVO}\nBanda: {band_name}', fontsize=16, fontweight='bold')
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction')
        cbar = fig.colorbar(im.get_images()[0], ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
        cbar.set_label('Densidade Populacional (pessoas / ha)', size=12)
        ax.add_artist(ScaleBar(1, 'm', location='upper right', box_alpha=0.8, pad=0.5))
        fig.text(0.75, 0.13, f"Fonte: FAO/SOFO (2022)\nProcessamento: GEE\nAutor: Pedro Luiz", ha='left', va='bottom', fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1))
        ax.set_xlabel(None); ax.set_ylabel(None)
        plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
        print("  -> Mapa gerado.")
        return fig

    def _plot_dominant_band_map(self, raster_path: str, aoi_gdf_geo: gpd.GeoDataFrame, aoi_gdf_mercator: gpd.GeoDataFrame) -> plt.Figure:
        cfg = self.config
        print("Gerando o mapa de predominância de bandas...")
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
        ax.set_facecolor('#f0f0f0')

        with rasterio.open(raster_path) as src:
            all_bands, transform = mask(src, aoi_gdf_mercator.geometry, crop=True, nodata=0)
            
            dominant_band_idx = np.argmax(all_bands, axis=0) + 1
            valid_mask = all_bands.sum(axis=0) > 0
            dominant_band_data = dominant_band_idx.astype(float)
            dominant_band_data[~valid_mask] = np.nan
            
            cores = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
            cmap = ListedColormap(cores)
            bounds = np.arange(0.5, len(cfg.BANDAS) + 1.5)
            norm = BoundaryNorm(bounds, cmap.N)
            
            im = show(dominant_band_data, ax=ax, transform=transform, cmap=cmap, norm=norm, alpha=0.8, zorder=2)

        path = Path.make_compound_path(*[Path(np.asarray(geom.exterior.coords)[:,:2]) for geom in aoi_gdf_mercator.geometry])
        clip_patch = PathPatch(path, transform=ax.transData, facecolor='none', edgecolor='none')
        ax.add_patch(clip_patch)
        for collection in ax.collections:
            collection.set_clip_path(clip_patch)
        if im.get_images():
            im.get_images()[0].set_clip_path(clip_patch)

        aoi_gdf_mercator.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, linestyle='--', zorder=3)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto', zorder=1)
        self._add_grid_inteligente(ax, aoi_gdf_geo)
        
        minx, miny, maxx, maxy = aoi_gdf_mercator.total_bounds
        ax.set_xlim(minx - (maxx-minx)*0.05, maxx + (maxx-minx)*0.05)
        ax.set_ylim(miny - (maxy-miny)*0.05, maxy + (maxy-miny)*0.05)
        
        ax.set_title(f'Mapa de Predominância de Bandas TPP - {cfg.NOME_MUNICIPIO_ALVO}', fontsize=16, fontweight='bold')
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction')
        
        legend_elements = [Patch(facecolor=cores[i], edgecolor='black', label=band_name) for i, band_name in enumerate(cfg.BANDAS)]
        ax.legend(handles=legend_elements, title='Banda Predominante', loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True)

        ax.add_artist(ScaleBar(1, 'm', location='upper right', box_alpha=0.8, pad=0.5))
        fig.text(0.75, 0.18, f"Fonte: FAO/SOFO (2022)\nProcessamento: GEE\nAutor: Pedro Luiz", ha='left', va='bottom', fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1))
        ax.set_xlabel(None); ax.set_ylabel(None)
        plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
        print("  -> Mapa de predominância gerado.")
        return fig

    def _export_report(self, figures: list, stats_df: pd.DataFrame, nome_municipio: str):
        cfg = self.config
        nome_sanitizado = self._normalize_string(nome_municipio)
        nome_arquivo = f"Relatorio_TPP_{nome_sanitizado}.pdf"
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo)
        print(f"\nExportando relatório PDF para: {caminho_completo}")

        with PdfPages(caminho_completo) as pdf:
            fig_tabela, ax_tabela = plt.subplots(figsize=(11.69, 8.27))
            ax_tabela.axis('off')
            df_display = stats_df.copy()
            df_display['População Total (estimada)'] = df_display['População Total (estimada)'].map('{:,.0f}'.format)
            for col in ['Densidade Média (pessoas/ha)', 'Densidade Mínima (pessoas/ha)', 'Densidade Máxima (pessoas/ha)']:
                df_display[col] = df_display[col].map('{:,.2f}'.format)
            col_widths = [0.25, 0.18, 0.18, 0.18, 0.18]
            tabela = ax_tabela.table(cellText=df_display.values, colLabels=df_display.columns, cellLoc='center', loc='center', colWidths=col_widths)
            tabela.auto_set_font_size(False); tabela.set_fontsize(10); tabela.scale(1.5, 3)
            plt.suptitle(f"Análise 'Tree Proximate People' (TPP)\nMunicípio de {nome_municipio}", fontsize=16, fontweight='bold', y=0.85)
            pdf.savefig(fig_tabela, bbox_inches='tight', dpi=cfg.DPI_SAIDA)
            plt.close(fig_tabela)

            for fig in figures:
                pdf.savefig(fig, dpi=cfg.DPI_SAIDA, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
        
        print(f"  -> Relatório exportado com sucesso!")

    def run(self):
        caminho_temp = None
        try:
            print("="*80); print("      INICIANDO PIPELINE DE ANÁLISE 'TREE PROXIMATE PEOPLE' (TPP) "); print("="*80)
            aoi_gdf_geo = self._load_aoi()
            aoi_gdf_mercator = aoi_gdf_geo.to_crs(self.config.CRS_WEB_MERCATOR)
            nome_municipio = aoi_gdf_geo[self.config.COLUNA_NOME_MUNICIPIO].iloc[0]
            nome_sanitizado = self._normalize_string(nome_municipio)
            caminho_temp = self._get_data_from_gee(aoi_gdf_geo, nome_sanitizado)
            stats_df = self._analyze_local_raster(caminho_temp, aoi_gdf_mercator)
            
            display(HTML(f"<h3>Análise Quantitativa para: {nome_municipio}</h3>"))
            display(stats_df.style.format({
                'População Total (estimada)': '{:,.0f}',
                'Densidade Média (pessoas/ha)': '{:,.2f}',
                'Densidade Mínima (pessoas/ha)': '{:,.2f}',
                'Densidade Máxima (pessoas/ha)': '{:,.2f}'
            }))
            
            map_figures_for_pdf = []
            
            for i, band_name in enumerate(self.config.BANDAS, 1):
                print("\n" + "="*80)
                display(HTML(f"<h3>Mapa para a banda: {band_name}</h3>"))
                fig = self._plot_map(caminho_temp, i, band_name, aoi_gdf_geo, aoi_gdf_mercator)
                display(fig)
                map_figures_for_pdf.append(fig)

            print("\n" + "="*80)
            display(HTML(f"<h3>Mapa de Predominância de Bandas para: {nome_municipio}</h3>"))
            dominant_fig = self._plot_dominant_band_map(caminho_temp, aoi_gdf_geo, aoi_gdf_mercator)
            display(dominant_fig)
            map_figures_for_pdf.append(dominant_fig)

            self._export_report(map_figures_for_pdf, stats_df, nome_municipio)

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
    configuracao = TPPConfig(NOME_MUNICIPIO_ALVO="São Desidério", SIGLA_UF="BA")
    pipeline = TPPPipeline(configuracao)
    pipeline.run()