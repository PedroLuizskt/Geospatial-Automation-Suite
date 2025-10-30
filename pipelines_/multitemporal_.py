## Análise Multitemporal de Uso e Cobertura do Solo (1985-2024) - Autor = Pedro Luiz

import os
import sys
try:
    import pyproj
    proj_data_dir = pyproj.datadir.get_data_dir()
    os.environ['PROJ_LIB'] = proj_data_dir
except Exception:
    pass

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio
import rasterio.plot
from rasterio.warp import reproject, Resampling
import contextily as ctx
import unicodedata
import re
import numpy as np
import pandas as pd
from rasterio.mask import mask
from dataclasses import dataclass, field
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FuncFormatter
from IPython.display import display

@dataclass
class RasterAnalysisConfig:
    PATH_RASTER_1985: str = r"C:\Users\pedro\Downloads\python_gis\script_usoecob\uso_solo_1985.tif"
    PATH_RASTER_2024: str = r"C:\Users\pedro\Downloads\python_gis\script_usoecob\uso_solo_2024.tif"
    PATH_MUNICIPIOS_BRASIL: str = r"C:\Users\pedro\Downloads\python_gis\script_usoecob\BR_Municipios_2024\BR_Municipios_2024.shp"
    PATH_IMOVEL_ALVO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\MG-3121605-0C5FDE61120E4BB7B3CD1728A97638E2\Area_do_Imovel\Area_do_Imovel.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\estagio_ie\resultados_"
    
    COLUNA_NOME_MUNICIPIO: str = 'NM_MUN'
    COLUNA_NOME_IMOVEL: str = 'recibo'
    ENCODING_MUNICIPIOS: str = 'utf-8'
    ENCODING_IMOVEL: str = 'utf-8'
    DPI_SAIDA: int = 300
    TIPO_DE_AREA: str = 'imovel'
    NOME_MUNICIPIO_ALVO: str = 'Sorriso'
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    CRS_WEB_MERCATOR: str = 'EPSG:3857'

    MAPBIOMAS_LEGEND: dict = field(default_factory=dict)
    MAPBIOMAS_CORES: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.MAPBIOMAS_LEGEND = {3: "Formação Florestal", 4: "Formação Savânica", 5: "Mangue", 6: "Floresta Alagável", 49: "Restinga Arbórea", 11: "Campo Alagado e Área Pantanosa", 12: "Formação Campestre", 32: "Apicum", 29: "Afloramento Rochoso", 50: "Restinga Herbácea", 15: "Pastagem", 19: "Lavoura Temporária", 39: "Soja", 20: "Cana", 40: "Arroz", 62: "Algodão", 41: "Outras Lavouras Temporárias", 36: "Lavoura Perene", 46: "Café", 47: "Citrus", 35: "Dendê", 48: "Outras Lavouras Perenes", 9: "Silvicultura", 21: "Mosaico de Usos", 23: "Praia, Duna e Areal", 24: "Área Urbanizada", 30: "Mineração", 25: "Outras Áreas não Vegetadas", 33: "Rio, Lago e Oceano", 31: "Aquicultura", 27: "Não observado"}
        self.MAPBIOMAS_CORES = {3: "#1f8d49", 4: "#7dc975", 5: "#04381d", 6: "#007785", 49: "#02d659", 11: "#519799", 12: "#d6bc74", 32: "#fc8114", 29: "#ffaa5f", 50: "#ad5100", 15: "#edde8e", 19: "#C27BA0", 39: "#f5b3c8", 20: "#db7093", 40: "#c71585", 62: "#ff69b4", 41: "#f54ca9", 36: "#d082de", 46: "#d68fe2", 47: "#9932cc", 35: "#9065d0", 48: "#e6ccff", 9: "#7a5900", 21: "#ffefc3", 23: "#ffa07a", 24: "#d4271e", 30: "#9c0027", 25: "#db4d4f", 33: "#2532e4", 31: "#091077", 27: "#ffffff"}

class MultitemporalRasterPipeline:
    def __init__(self, config: RasterAnalysisConfig):
        self.config = config

    def _carregar_dados_vetoriais(self, path: str, **kwargs) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path, **kwargs)
        if gdf.crs is None:
            gdf.set_crs(self.config.CRS_GEOGRAFICO, inplace=True)
        return gdf.to_crs(self.config.CRS_GEOGRAFICO)

    def _obter_area_de_interesse(self) -> gpd.GeoDataFrame:
        cfg = self.config
        if cfg.TIPO_DE_AREA == 'municipio':
            gdf = self._carregar_dados_vetoriais(cfg.PATH_MUNICIPIOS_BRASIL, encoding=cfg.ENCODING_MUNICIPIOS)
            nome_norm = unicodedata.normalize('NFKD', cfg.NOME_MUNICIPIO_ALVO).encode('ascii', 'ignore').decode('utf-8').lower()
            area_filtrada = gdf[gdf[cfg.COLUNA_NOME_MUNICIPIO].str.lower().apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8')) == nome_norm]
            if area_filtrada.empty: raise ValueError(f"Município '{cfg.NOME_MUNICIPIO_ALVO}' não encontrado.")
            area_gdf = area_filtrada.dissolve().reset_index()
            area_gdf[cfg.COLUNA_NOME_MUNICIPIO] = area_filtrada[cfg.COLUNA_NOME_MUNICIPIO].iloc[0]
        elif cfg.TIPO_DE_AREA == 'imovel':
            gdf = self._carregar_dados_vetoriais(cfg.PATH_IMOVEL_ALVO, encoding=cfg.ENCODING_IMOVEL)
            if gdf.empty: raise ValueError("Arquivo do imóvel está vazio.")
            area_gdf = gdf.dissolve().reset_index()
        else: raise ValueError("TIPO_DE_AREA deve ser 'municipio' ou 'imovel'.")
        return area_gdf

    def _calculate_area_stats(self, raster_path: str, aoi_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        cfg = self.config; PIXEL_AREA_HA = (30 * 30) / 10000
        with rasterio.open(raster_path) as src:
            aoi_reproj = aoi_gdf.to_crs(src.crs)
            masked_array, _ = mask(src, aoi_reproj.geometry, crop=True, nodata=0)
            masked_array = masked_array[0]
            class_codes, counts = np.unique(masked_array[masked_array != 0], return_counts=True)
            if class_codes.size == 0: return pd.DataFrame(columns=['Classe', 'Área (ha)'])
            stats_df = pd.DataFrame({'Código': class_codes, 'Contagem': counts})
            stats_df['Classe'] = stats_df['Código'].map(cfg.MAPBIOMAS_LEGEND)
            stats_df['Área (ha)'] = stats_df['Contagem'] * PIXEL_AREA_HA
            return stats_df[['Classe', 'Área (ha)']].dropna(subset=['Classe'])

    def _combine_stats(self, stats_1985: pd.DataFrame, stats_2024: pd.DataFrame) -> pd.DataFrame:
        df1 = stats_1985.set_index('Classe'); df2 = stats_2024.set_index('Classe')
        combined_df = df1.join(df2, how='outer', lsuffix='_1985', rsuffix='_2024').fillna(0)
        combined_df.columns = ['Área 1985 (ha)', 'Área 2024 (ha)']
        combined_df = combined_df.loc[(combined_df.sum(axis=1) > 0)]
        return combined_df.reset_index()

    def _add_coordinate_grid(self, ax, aoi_gdf: gpd.GeoDataFrame):
        aoi_geo = aoi_gdf.to_crs(self.config.CRS_GEOGRAFICO)
        minx, miny, maxx, maxy = aoi_geo.total_bounds
        
        x_ticks = np.linspace(minx, maxx, 5)
        y_ticks = np.linspace(miny, maxy, 5)

        lon_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_ticks, [miny] * len(x_ticks)), crs=self.config.CRS_GEOGRAFICO)
        lat_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([minx] * len(y_ticks), y_ticks), crs=self.config.CRS_GEOGRAFICO)

        lon_ticks_proj = lon_gdf.to_crs(self.config.CRS_WEB_MERCATOR).geometry.x
        lat_ticks_proj = lat_gdf.to_crs(self.config.CRS_WEB_MERCATOR).geometry.y

        ax.set_xticks(lon_ticks_proj)
        ax.set_yticks(lat_ticks_proj)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{x_ticks[pos]:.2f}°" if pos < len(x_ticks) else ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{y_ticks[pos]:.2f}°" if pos < len(y_ticks) else ""))
        
        ax.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    def _plotar_mapa(self, raster_path: str, aoi_gdf: gpd.GeoDataFrame, ano: str, nome_area: str) -> plt.Figure:
        cfg = self.config
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.2], wspace=0.3)
        ax_mapa = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[0, 1])
        ax_mapa.set_facecolor('#f0f0f0'); ax_info.set_axis_off()

        with rasterio.open(raster_path) as src:
            aoi_for_masking = aoi_gdf.to_crs(src.crs)
            out_image, out_transform = mask(src, aoi_for_masking.geometry, crop=True, nodata=0)
            reproj_image, reproj_transform = reproject(source=out_image, src_transform=out_transform, src_crs=src.crs, dst_crs=cfg.CRS_WEB_MERCATOR, resampling=Resampling.nearest)

        plot_data = reproj_image[0].astype(float); plot_data[plot_data == 0] = np.nan
        valores_existentes = np.unique(plot_data[~np.isnan(plot_data)])
        
        if valores_existentes.size > 0:
            cores = [cfg.MAPBIOMAS_CORES.get(int(c)) for c in sorted(valores_existentes) if int(c) in cfg.MAPBIOMAS_CORES]
            classes_norm = sorted([int(c) - 0.5 for c in valores_existentes]) + [max(valores_existentes) + 0.5]
            cmap = ListedColormap(cores); norm = BoundaryNorm(classes_norm, cmap.N)
            rasterio.plot.show(plot_data, ax=ax_mapa, transform=reproj_transform, cmap=cmap, norm=norm, interpolation='nearest', zorder=2)
            
            elementos_legenda = [Patch(facecolor=cfg.MAPBIOMAS_CORES.get(int(c)), label=f"{cfg.MAPBIOMAS_LEGEND.get(int(c))}") for c in sorted(valores_existentes) if int(c) in cfg.MAPBIOMAS_LEGEND]
            ax_info.legend(handles=elementos_legenda, title='Classes de Uso', loc='upper left', bbox_to_anchor=(0.01, 1.0), fontsize='medium', title_fontsize='large')
        
        aoi_plot = aoi_gdf.to_crs(cfg.CRS_WEB_MERCATOR)
        aoi_plot.plot(ax=ax_mapa, facecolor='none', edgecolor='red', linewidth=1.5, linestyle='--', zorder=3)
        
        ctx.add_basemap(ax_mapa, crs=cfg.CRS_WEB_MERCATOR, source=ctx.providers.CartoDB.Positron, zorder=1)
        
        self._add_coordinate_grid(ax_mapa, aoi_gdf)
        
        ax_mapa.set_xlabel(None); ax_mapa.set_ylabel(None)
        ax_mapa.add_artist(ScaleBar(1, "m", location="lower right", box_alpha=0.8, pad=0.5))
        ax_mapa.annotate('N', xy=(0.05, 0.98), xytext=(0.05, 0.93), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction')
        ax_info.text(0.05, 0.05, "Fonte: MapBiomas Collection 8 (2024)\nAutor: Pedro Luiz\nDatum: SIRGAS 2000", ha='left', va='bottom', fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1), transform=ax_info.transAxes)
        
        fig.suptitle(f'Uso e Cobertura do Solo - {nome_area} ({ano})', fontsize=18, fontweight='bold')
        return fig

    def _exportar_relatorio_pdf(self, figuras: dict, df_estatisticas: pd.DataFrame, nome_area: str, nome_arquivo_saida: str):
        cfg = self.config
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo_saida)
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        
        with PdfPages(caminho_completo) as pdf:
            for ano in sorted(figuras.keys()):
                pdf.savefig(figuras[ano], bbox_inches='tight', dpi=cfg.DPI_SAIDA)
                plt.close(figuras[ano])
            
            if not df_estatisticas.empty:
                fig_tabela, ax_tabela = plt.subplots(figsize=(8.27, 11.69))
                ax_tabela.axis('off')
                ax_tabela.set_title(f"Quadro de Áreas - {nome_area}", fontsize=16, fontweight='bold', pad=20)
                
                df_display = df_estatisticas.copy()
                for col in [c for c in df_display.columns if 'Área' in c]:
                    df_display[col] = df_display[col].map('{:,.2f}'.format)

                tabela = ax_tabela.table(cellText=df_display.values, colLabels=df_display.columns, cellLoc='center', loc='upper center', colWidths=[0.4, 0.2, 0.2])
                tabela.auto_set_font_size(False); tabela.set_fontsize(10); tabela.scale(1, 1.8)
                
                pdf.savefig(fig_tabela, bbox_inches='tight', dpi=cfg.DPI_SAIDA)
                plt.close(fig_tabela)

    def run(self):
        try:
            aoi_gdf = self._obter_area_de_interesse()
            nome_area = aoi_gdf[self.config.COLUNA_NOME_IMOVEL].iloc[0] if self.config.TIPO_DE_AREA == 'imovel' and self.config.COLUNA_NOME_IMOVEL in aoi_gdf.columns else self.config.NOME_MUNICIPIO_ALVO
            
            stats_1985 = self._calculate_area_stats(self.config.PATH_RASTER_1985, aoi_gdf)
            stats_2024 = self._calculate_area_stats(self.config.PATH_RASTER_2024, aoi_gdf)
            tabela_final = self._combine_stats(stats_1985, stats_2024)

            if not tabela_final.empty:
                print("\n" + "="*90)
                print(f"  QUADRO DE ÁREAS (1985-2024) PARA: {nome_area.upper()}")
                print("="*90)
                print(tabela_final.to_string(index=False))
                print("="*90)
            
            figura_1985 = self._plotar_mapa(self.config.PATH_RASTER_1985, aoi_gdf, "1985", nome_area)
            display(figura_1985)
            plt.close(figura_1985)
            
            figura_2024 = self._plotar_mapa(self.config.PATH_RASTER_2024, aoi_gdf, "2024", nome_area)
            display(figura_2024)
            plt.close(figura_2024)
            
            nome_sanitizado = unicodedata.normalize('NFKD', nome_area).encode('ascii', 'ignore').decode('utf-8').replace(' ', '_').lower()
            nome_arquivo_final = f"Relatorio_UsoSolo_{nome_sanitizado}_1985_2024.pdf"
            
            figuras_para_exportar = {
                "1985": self._plotar_mapa(self.config.PATH_RASTER_1985, aoi_gdf, "1985", nome_area),
                "2024": self._plotar_mapa(self.config.PATH_RASTER_2024, aoi_gdf, "2024", nome_area)
            }
            self._exportar_relatorio_pdf(figuras_para_exportar, tabela_final, nome_area, nome_arquivo_final)
            
        except Exception as e:
            print(f"\n[ERRO CRÍTICO] Ocorreu um erro no pipeline: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    config = RasterAnalysisConfig(TIPO_DE_AREA='imovel')
    pipeline = MultitemporalRasterPipeline(config)
    pipeline.run()