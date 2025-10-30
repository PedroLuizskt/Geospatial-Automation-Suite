## Análise de Declividade via GEE - Autor = Pedro Luiz

import ee
import geemap
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot
from rasterio.mask import mask
import numpy as np
import pandas as pd
import contextily as ctx
import unicodedata
import re
from dataclasses import dataclass, field
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar
from IPython.display import display, HTML

@dataclass
class SlopeConfig:
    PATH_MUNICIPIOS: str = r"C:\Users\pedro\Downloads\python_gis\script_precipitacao\BR_Municipios_2024\BR_Municipios_2024.shp"
    PATH_IMOVEL_ALVO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\MT-5106422-2960697C669941729C7EF7C2930CBA5A\Area_do_Imovel\Area_do_Imovel.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\estagio_ie\resultados_"
    COLUNA_NOME_MUNICIPIO: str = 'NM_MUN'
    COLUNA_NOME_IMOVEL: str = 'recibo'
    ENCODING_VETORES: str = 'utf-8'
    DPI_SAIDA: int = 300
    TIPO_DE_AREA: str = 'municipio'
    NOME_MUNICIPIO_ALVO: str = 'Caraguatatuba'
    ESCALA_DOWNLOAD_METROS: int = 90
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    CRS_WEB_MERCATOR: str = 'EPSG:3857'
    CLASSES_RELEVO: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.CLASSES_RELEVO = {
            'Plano (0-3%)': (0, 1.72), 
            'Suave Ondulado (3-8%)': (1.72, 4.57), 
            'Ondulado (8-20%)': (4.57, 11.31), 
            'Forte Ondulado (20-45%)': (11.31, 24.23), 
            'Montanhoso/Escarpado (>45%)': (24.23, 90)
        }

class SlopePipeline:
    def __init__(self, config: SlopeConfig):
        self.config = config
        self._initialize_gee()

    def _initialize_gee(self):
        try: 
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        except Exception: 
            ee.Authenticate()
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    def _normalize_string(self, text: str) -> str:
        if not isinstance(text, str): return ""
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower().strip()

    def _sanitizar_nome_arquivo(self, nome: str) -> str:
        nome_norm = self._normalize_string(nome)
        nome_sanitizado = re.sub(r'[^\w\s-]', '', nome_norm)
        return re.sub(r'[-\s]+', '-', nome_sanitizado)[:100]

    def _carregar_dados_vetoriais(self, path: str, crs_alvo: str, **kwargs) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path, **kwargs)
        if gdf.crs is None:
            gdf.set_crs(self.config.CRS_GEOGRAFICO, inplace=True)
        return gdf.to_crs(crs_alvo)

    def _obter_area_de_interesse(self) -> gpd.GeoDataFrame:
        cfg = self.config
        if cfg.TIPO_DE_AREA == 'municipio':
            gdf = self._carregar_dados_vetoriais(cfg.PATH_MUNICIPIOS, cfg.CRS_GEOGRAFICO, encoding=cfg.ENCODING_VETORES)
            area_filtrada_gdf = gdf[gdf[cfg.COLUNA_NOME_MUNICIPIO] == cfg.NOME_MUNICIPIO_ALVO]
            if area_filtrada_gdf.empty: raise ValueError(f"Município '{cfg.NOME_MUNICIPIO_ALVO}' não encontrado.")
            area_gdf = area_filtrada_gdf.dissolve().reset_index()
        elif cfg.TIPO_DE_AREA == 'imovel':
            area_gdf = self._carregar_dados_vetoriais(cfg.PATH_IMOVEL_ALVO, cfg.CRS_GEOGRAFICO, encoding=cfg.ENCODING_VETORES)
        else:
            raise ValueError("Tipo de área inválido.")
        if area_gdf.empty: raise ValueError("A Área de Interesse (AOI) está vazia.")
        return area_gdf

    def _obter_declividade_gee(self, aoi_local_gdf: gpd.GeoDataFrame, nome_area_sanitizado: str) -> str:
        cfg = self.config
        aoi_ee = geemap.geopandas_to_ee(aoi_local_gdf)
        dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
        declividade_ee = ee.Terrain.slope(dem)
        caminho_temporario = os.path.join(cfg.PATH_EXPORTACAO, f'temp_slope_{nome_area_sanitizado}.tif')
        geemap.ee_export_image(declividade_ee, filename=caminho_temporario, scale=cfg.ESCALA_DOWNLOAD_METROS, region=aoi_ee.geometry(), crs=cfg.CRS_WEB_MERCATOR)
        if not os.path.exists(caminho_temporario):
            raise FileNotFoundError("Falha no download da imagem do GEE.")
        return caminho_temporario

    def _estimate_utm_crs(self, gdf_geo: gpd.GeoDataFrame) -> str:
        try:
            centroid = gdf_geo.union_all().centroid
            utm_zone = int((centroid.x + 180) / 6) + 1
            south = centroid.y < 0
            epsg = 32700 + utm_zone if south else 32600 + utm_zone
            return f"EPSG:{epsg}"
        except Exception:
            return 'EPSG:5880'

    def _calcular_estatisticas(self, path_raster_local: str, aoi_gdf_projetado: gpd.GeoDataFrame):
        cfg = self.config
        with rasterio.open(path_raster_local) as src:
            raster_array, transform = mask(src, aoi_gdf_projetado.geometry, crop=True, nodata=np.nan)
            raster_array = raster_array[0]
            num_pixels_validos = (~np.isnan(raster_array)).sum()
            if num_pixels_validos == 0: 
                return None, None
            bounds = rasterio.transform.array_bounds(raster_array.shape[0], raster_array.shape[1], transform)
            temp_gdf_mercator = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=cfg.CRS_WEB_MERCATOR)
            temp_gdf_geo = temp_gdf_mercator.to_crs(cfg.CRS_GEOGRAFICO)
            crs_metrico_local = self._estimate_utm_crs(temp_gdf_geo)
            area_total_m2 = temp_gdf_geo.to_crs(crs_metrico_local).area.iloc[0]
            pixel_area_m2 = area_total_m2 / raster_array.size if raster_array.size > 0 else 0
        dados_validos = raster_array[~np.isnan(raster_array)].flatten()
        desc_stats = {"Mínima (°)": np.min(dados_validos), "Máxima (°)": np.max(dados_validos), "Média (°)": np.mean(dados_validos), "Desvio Padrão (°)": np.std(dados_validos)}
        desc_stats_df = pd.DataFrame.from_dict(desc_stats, orient='index', columns=['Valor'])
        class_stats = [{'Classe': n, 'Área (ha)': (np.sum((dados_validos >= v[0]) & (dados_validos < v[1])) * pixel_area_m2) / 10000} for n, v in cfg.CLASSES_RELEVO.items()]
        return desc_stats_df, pd.DataFrame(class_stats)

    def _plotar_mapa(self, path_raster_local: str, aoi_gdf_geografico: gpd.GeoDataFrame, aoi_gdf_projetado: gpd.GeoDataFrame, nome_area: str) -> plt.Figure:
        cfg = self.config
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
        ax.set_facecolor('#f0f0f0')
        with rasterio.open(path_raster_local) as src:
            data, transform = mask(src, aoi_gdf_projetado.geometry, crop=True, nodata=np.nan)
            im = rasterio.plot.show(data, ax=ax, cmap='RdYlGn_r', vmin=0, vmax=45, transform=transform, alpha=0.8, zorder=2)
        aoi_gdf_projetado.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=3)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto', zorder=1)
        self._add_grid_inteligente(ax, aoi_gdf_geografico)
        ax.set_title(f'Mapa de Declividade - {nome_area}', fontsize=18, fontweight='bold')
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction')
        cbar = fig.colorbar(im.get_images()[0], ax=ax, orientation='vertical', shrink=0.8, pad=0.03)
        cbar.set_label('Declividade (Graus)', size=12)
        ax.add_artist(ScaleBar(1, 'm', location='upper right', box_alpha=0.8, pad=0.5))
        fig.text(0.83, 0.2, f"Fonte: GEE (NASA/NASADEM)\nAutor: Pedro Luiz\nDatum: SIRGAS 2000", ha='left', va='bottom', fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1))
        plt.tight_layout(rect=[0, 0.05, 0.9, 0.95]); return fig
        
    def _add_grid_inteligente(self, ax, gdf_geografico):
        cfg = self.config
        minx, miny, maxx, maxy = gdf_geografico.total_bounds
        x_interval, y_interval = (maxx - minx) / 5, (maxy - miny) / 5
        x_ticks, y_ticks = np.arange(minx, maxx, x_interval), np.arange(miny, maxy, y_interval)
        lon_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_ticks, [miny] * len(x_ticks)), crs=cfg.CRS_GEOGRAFICO)
        lat_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([minx] * len(y_ticks), y_ticks), crs=cfg.CRS_GEOGRAFICO)
        lon_ticks_proj = lon_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.x
        lat_ticks_proj = lat_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.y
        ax.set_xticks(lon_ticks_proj); ax.set_yticks(lat_ticks_proj)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{x_ticks[pos]:.2f}°" if pos < len(x_ticks) else ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{y_ticks[pos]:.2f}°" if pos < len(y_ticks) else ""))
        ax.grid(True, linestyle='--', alpha=0.6, color='gray'); ax.tick_params(axis='x', rotation=45, labelsize=10); ax.tick_params(axis='y', labelsize=10)

    def _exportar_relatorio_pdf(self, figura_mapa: plt.Figure, desc_df: pd.DataFrame, class_df: pd.DataFrame, nome_area: str, nome_arquivo_saida: str):
        cfg = self.config; caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo_saida); os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        with PdfPages(caminho_completo) as pdf:
            pdf.savefig(figura_mapa, bbox_inches='tight', dpi=cfg.DPI_SAIDA); plt.close(figura_mapa)
            fig_desc, ax_desc = plt.subplots(figsize=(8.27, 11.69)); ax_desc.axis('off'); ax_desc.set_title(f"Estatísticas Descritivas\n{nome_area}", fontsize=16, fontweight='bold', pad=20)
            df_display_desc = desc_df.copy(); df_display_desc['Valor'] = [f"{v:_.2f}".replace('.', ',').replace('_', '.') for v in df_display_desc['Valor']]
            tabela_desc = ax_desc.table(cellText=df_display_desc.values, rowLabels=df_display_desc.index, colLabels=df_display_desc.columns, cellLoc='center', loc='upper center', rowLoc='left')
            tabela_desc.auto_set_font_size(False); tabela_desc.set_fontsize(14); tabela_desc.scale(1.2, 2.5); pdf.savefig(fig_desc, bbox_inches='tight', dpi=cfg.DPI_SAIDA); plt.close(fig_desc)
            fig_class, ax_class = plt.subplots(figsize=(8.27, 11.69)); ax_class.axis('off'); ax_class.set_title(f"Área por Classe de Relevo\n{nome_area}", fontsize=16, fontweight='bold', pad=20)
            df_display_class = class_df.copy(); df_display_class['Área (ha)'] = [f"{v:_.2f}".replace('.', ',').replace('_', '.') for v in df_display_class['Área (ha)']]
            tabela_class = ax_class.table(cellText=df_display_class.values, colLabels=df_display_class.columns, cellLoc='center', loc='upper center', colWidths=[0.5, 0.2])
            tabela_class.auto_set_font_size(False); tabela_class.set_fontsize(12); tabela_class.scale(1.2, 2.5); pdf.savefig(fig_class, bbox_inches='tight', dpi=cfg.DPI_SAIDA); plt.close(fig_class)
    
    def run(self):
        caminho_temp = None
        try:
            print("="*80); print("       INICIANDO PIPELINE DE ANÁLISE DE DECLIVIDADE VIA GEE "); print("="*80)
            aoi_gdf_geografico = self._obter_area_de_interesse()
            aoi_gdf_projetado = aoi_gdf_geografico.to_crs(self.config.CRS_WEB_MERCATOR)
            if self.config.TIPO_DE_AREA == 'municipio':
                nome_area = self.config.NOME_MUNICIPIO_ALVO
            else:
                nome_area = aoi_gdf_geografico[self.config.COLUNA_NOME_IMOVEL].iloc[0] if self.config.COLUNA_NOME_IMOVEL in aoi_gdf_geografico.columns else "Imovel Rural Sem Nome"
            nome_base_sanitizado = self._sanitizar_nome_arquivo(nome_area)
            nome_arquivo_final = f"Relatorio_Declividade_{nome_base_sanitizado}.pdf"
            caminho_temp = self._obter_declividade_gee(aoi_gdf_geografico, nome_base_sanitizado)
            desc_stats_df, class_stats_df = self._calcular_estatisticas(caminho_temp, aoi_gdf_projetado)
            if desc_stats_df is None:
                print(f"\nAVISO: Nenhum dado válido encontrado para '{nome_area}'."); return
            
            display(HTML(f"<h2>Análise de Declividade para: {nome_area}</h2>"))
            figura_para_display = self._plotar_mapa(caminho_temp, aoi_gdf_geografico, aoi_gdf_projetado, nome_area)
            display(figura_para_display); plt.close(figura_para_display)
            print("\n" + "="*60); print(f"             ANÁLISE ESTATÍSTICA PARA: {nome_area.upper()}"); print("="*60)
            desc_console = desc_stats_df.copy(); desc_console['Valor'] = [f"{v:_.2f}".replace('.', ',').replace('_', '.') for v in desc_console['Valor']]; print(desc_console.to_string())
            print("-" * 60); print("Área por Classe de Relevo:")
            class_console = class_stats_df.copy()
            class_console['Área (ha)'] = [f"{v:_.2f}".replace('.', ',').replace('_', '.') for v in class_console['Área (ha)']]
            print(class_console.to_string(index=False)); print("="*60)
            figura_para_pdf = self._plotar_mapa(caminho_temp, aoi_gdf_geografico, aoi_gdf_projetado, nome_area)
            self._exportar_relatorio_pdf(figura_para_pdf, desc_stats_df, class_stats_df, nome_area, nome_arquivo_final)
            print(f"\nAnálise de declividade para '{nome_area}' concluída!")

        except Exception as e:
            print(f"\n[ERRO] {e}"); import traceback; traceback.print_exc()
        finally:
            if caminho_temp and os.path.exists(caminho_temp):
                os.remove(caminho_temp)

if __name__ == "__main__":
    config = SlopeConfig(TIPO_DE_AREA='imovel', NOME_MUNICIPIO_ALVO='Santos')
    pipeline = SlopePipeline(config)
    pipeline.run()