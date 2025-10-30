import ee
import geemap
import geopandas as gpd
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from IPython.display import display, HTML

@dataclass
class ImageAvailabilityConfig:
    """Configuração para o script de diagnóstico de disponibilidade de imagens."""
    PATH_IMOVEL_ALVO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\MT-5106422-2960697C669941729C7EF7C2930CBA5A\Area_do_Imovel\Area_do_Imovel.shp"
    TARGET_YEAR: int = 2024
    ASSET_ID: str = "LANDSAT/LC09/C02/T1_TOA"
    CLOUD_COVER_THRESHOLD: int = 15
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    ENCODING_IMOVEL: str = 'utf-8'

class AvailabilityScout:
    """Pipeline para verificar a disponibilidade de imagens de satélite."""
    def __init__(self, config: ImageAvailabilityConfig):
        self.config = config
        self._initialize_gee()

    def _initialize_gee(self):
        try:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        except Exception:
            ee.Authenticate()
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    def _load_and_prepare_aoi(self) -> ee.FeatureCollection:
        try:
            gdf = gpd.read_file(self.config.PATH_IMOVEL_ALVO, encoding=self.config.ENCODING_IMOVEL)
            gdf_geo = gdf.to_crs(self.config.CRS_GEOGRAFICO)
            aoi_dissolved = gdf_geo.dissolve()
            return geemap.geopandas_to_ee(aoi_dissolved)
        except Exception as e:
            raise FileNotFoundError(f"Erro ao carregar ou processar a AOI: {e}")

    def run(self):
        """Executa a verificação e imprime o relatório de disponibilidade."""
        cfg = self.config
        print("="*80)
        print(f"  INICIANDO DIAGNÓSTICO DE DISPONIBILIDADE DE IMAGENS PARA O ANO {cfg.TARGET_YEAR}")
        print("="*80)

        try:
            aoi_ee = self._load_and_prepare_aoi()
            
            monthly_availability = []
            
            for month in range(1, 13):
                start_date = f"{cfg.TARGET_YEAR}-{month:02d}-01"
                end_date_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(day=28) + pd.DateOffset(days=4)
                end_date = (end_date_dt - pd.DateOffset(days=end_date_dt.day - 1)).strftime('%Y-%m-%d')

                collection = ee.ImageCollection(cfg.ASSET_ID) \
                               .filterBounds(aoi_ee) \
                               .filterDate(start_date, end_date) \
                               .filter(ee.Filter.lt('CLOUD_COVER', cfg.CLOUD_COVER_THRESHOLD))
                
                image_count = collection.size().getInfo()
                
                if image_count > 0:
                    dates = collection.aggregate_array('DATE_ACQUIRED').getInfo()
                    dates_formatted = ', '.join(sorted(list(set(dates))))
                else:
                    dates_formatted = "Nenhuma imagem disponível"
                
                month_name = datetime.strptime(str(month), "%m").strftime("%B")
                monthly_availability.append({
                    "Mês": month_name,
                    "Nº de Imagens Limpas": image_count,
                    "Datas Disponíveis": dates_formatted
                })
                print(f"  -> Mês de {month_name} analisado. Encontradas {image_count} imagens.")

            report_df = pd.DataFrame(monthly_availability)
            
            print("\n" + "="*80)
            print("  RELATÓRIO DE DISPONIBILIDADE CONCLUÍDO")
            print("="*80)
            
            display(HTML(f"<h3>Disponibilidade de Imagens Landsat 9 (Nuvens < {cfg.CLOUD_COVER_THRESHOLD}%) para o ano de {cfg.TARGET_YEAR}</h3>"))
            display(report_df)
            
            print("\n" + "="*80)
            print("FLUXO DE TRABALHO 'SCOUT' ENCERRADO. Use uma data da tabela acima para rodar o pipeline de análise.")
            print("="*80)

        except Exception as e:
            print(f"\\n[ERRO CRÍTICO] Ocorreu um erro: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    scout_config = ImageAvailabilityConfig(TARGET_YEAR=2024)
    scout = AvailabilityScout(scout_config)
    scout.run()

## Análise de índice de vegetação por diferença normalizada por imóvel - Autor = Pedro Luiz

import ee
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import contextily as ctx
import os
import unicodedata
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FuncFormatter
from IPython.display import display, HTML

@dataclass
class NDVIConfig:
    PATH_IMOVEL_ALVO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\MT-5106422-2960697C669941729C7EF7C2930CBA5A\Area_do_Imovel\Area_do_Imovel.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\python_gis\ndvi_\resultado_"
    TARGET_DATE: str = '2024-09-01'
    DATE_WINDOW_DAYS: int = 5
    ASSET_ID: str = "LANDSAT/LC09/C02/T1_TOA"
    NIR_BAND: str = 'B5'
    RED_BAND: str = 'B4'
    QA_BAND: str = 'QA_PIXEL'
    CLOUD_COVER_THRESHOLD: int = 20
    ESCALA_METROS: int = 30
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    CRS_WEB_MERCATOR: str = 'EPSG:3857'
    DPI_SAIDA: int = 300
    ENCODING_IMOVEL: str = 'utf-8'

class NDVIPipeline:
    def __init__(self, config: NDVIConfig):
        self.config = config
        self._initialize_gee()

    def _initialize_gee(self):
        try:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        except Exception:
            ee.Authenticate()
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    def _sanitizar_nome_arquivo(self, nome: str) -> str:
        nome_sanitizado = unicodedata.normalize('NFKD', str(nome)).encode('ascii', 'ignore').decode('utf-8')
        nome_sanitizado = re.sub(r'[^\w\s-]', '', nome_sanitizado).strip().lower()
        return re.sub(r'[-\s]+', '-', nome_sanitizado)[:100]

    def _load_and_prepare_aoi(self) -> gpd.GeoDataFrame:
        try:
            gdf = gpd.read_file(self.config.PATH_IMOVEL_ALVO, encoding=self.config.ENCODING_IMOVEL)
            gdf_geo = gdf.to_crs(self.config.CRS_GEOGRAFICO)
            aoi_dissolved = gdf_geo.dissolve().reset_index()
            if 'recibo' in gdf.columns:
                aoi_dissolved['recibo'] = gdf['recibo'].iloc[0]
            return aoi_dissolved
        except Exception as e:
            raise FileNotFoundError(f"Erro ao carregar ou processar a AOI: {e}")

    def _get_gee_collection(self, start_date: str, end_date: str, aoi_ee: ee.FeatureCollection):
        cfg = self.config
        
        def mask_clouds(image):
            qa = image.select(cfg.QA_BAND)
            cloud_shadow_bit_mask = (1 << 4)
            clouds_bit_mask = (1 << 3)
            dilated_cloud_bit_mask = (1 << 1)
            mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0) \
                     .And(qa.bitwiseAnd(clouds_bit_mask).eq(0)) \
                     .And(qa.bitwiseAnd(dilated_cloud_bit_mask).eq(0))
            return image.updateMask(mask)

        collection = ee.ImageCollection(cfg.ASSET_ID) \
                       .filterBounds(aoi_ee) \
                       .filterDate(start_date, end_date) \
                       .filter(ee.Filter.lt('CLOUD_COVER', cfg.CLOUD_COVER_THRESHOLD)) \
                       .map(mask_clouds)
        return collection

    def _get_ndvi_composite_from_gee(self, aoi_gdf: gpd.GeoDataFrame, nome_sanitizado: str) -> (str, str, str):
        cfg = self.config
        aoi_ee = geemap.geopandas_to_ee(aoi_gdf)
        target_datetime = datetime.strptime(cfg.TARGET_DATE, '%Y-%m-%d')
        
        start_date = (target_datetime.replace(day=1)).strftime('%Y-%m-%d')
        end_date = ((target_datetime + pd.DateOffset(months=1)).replace(day=1) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        
        print(f"Buscando imagens no mês de {target_datetime.strftime('%B de %Y')}...")

        collection = self._get_gee_collection(start_date, end_date, aoi_ee)
        
        image_count = collection.size().getInfo()
        if image_count == 0:
            return None, start_date, end_date

        print(f"  -> {image_count} imagens encontradas. Criando composite de NDVI...")
        
        ndvi_composite = collection.map(lambda img: img.normalizedDifference([cfg.NIR_BAND, cfg.RED_BAND]).rename('ndvi')) \
                                   .median() \
                                   .clip(aoi_ee)
        
        caminho_temporario = os.path.join(cfg.PATH_EXPORTACAO, f'temp_ndvi_{nome_sanitizado}.tif')
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)

        geemap.ee_export_image(
            ndvi_composite, filename=caminho_temporario, scale=cfg.ESCALA_METROS,
            region=aoi_ee.geometry(), crs=cfg.CRS_WEB_MERCATOR
        )
        if not os.path.exists(caminho_temporario):
            raise FileNotFoundError("Falha no download da imagem NDVI do GEE.")
        return caminho_temporario, start_date, end_date

    def _get_monthly_ndvi_series_from_gee(self, aoi_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        cfg = self.config
        aoi_ee = geemap.geopandas_to_ee(aoi_gdf)
        target_datetime = datetime.strptime(cfg.TARGET_DATE, '%Y-%m-%d')
        start_of_month = target_datetime.replace(day=1)
        end_of_month = (start_of_month + timedelta(days=32)).replace(day=1) - timedelta(days=1)

        collection = self._get_gee_collection(start_of_month.strftime('%Y-%m-%d'), end_of_month.strftime('%Y-%m-%d'), aoi_ee)
        
        if collection.size().getInfo() == 0:
            return pd.DataFrame(columns=['date', 'ndvi'])

        def calculate_mean_ndvi(image):
            mean_ndvi = image.normalizedDifference([cfg.NIR_BAND, cfg.RED_BAND]).reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi_ee.geometry(),
                scale=cfg.ESCALA_METROS, maxPixels=1e9
            )
            return ee.Feature(None, {'date': image.date().format('YYYY-MM-dd'), 'ndvi': mean_ndvi.get('nd')})

        ndvi_series = collection.map(calculate_mean_ndvi).getInfo()
        
        df_list = [item['properties'] for item in ndvi_series['features'] if item['properties']['ndvi'] is not None]
        if not df_list:
            return pd.DataFrame(columns=['date', 'ndvi'])

        df = pd.DataFrame(df_list)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values(by='date').reset_index(drop=True)

    def _calculate_statistics_from_raster(self, raster_path: str, aoi_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        with rasterio.open(raster_path) as src:
            aoi_reprojetado = aoi_gdf.to_crs(self.config.CRS_WEB_MERCATOR)
            
            raster_array, _ = mask(src, aoi_reprojetado.geometry, crop=True, nodata=np.nan)
            dados_validos = raster_array[0][~np.isnan(raster_array[0])]
            
            if dados_validos.size == 0: return pd.DataFrame()

            stats = {
                "NDVI Mínimo": np.min(dados_validos), "NDVI Máximo": np.max(dados_validos),
                "NDVI Médio": np.mean(dados_validos), "NDVI Mediana": np.median(dados_validos),
                "Desvio Padrão": np.std(dados_validos)
            }
            return pd.DataFrame.from_dict(stats, orient='index', columns=['Valor'])

    def _add_grid_inteligente(self, ax, gdf_geografico):
        minx, miny, maxx, maxy = gdf_geografico.total_bounds
        x_ticks, y_ticks = np.linspace(minx, maxx, 5), np.linspace(miny, maxy, 5)
        lon_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_ticks, [miny] * len(x_ticks)), crs=self.config.CRS_GEOGRAFICO)
        lat_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([minx] * len(y_ticks), y_ticks), crs=self.config.CRS_GEOGRAFICO)
        lon_ticks_proj, lat_ticks_proj = lon_gdf.to_crs(self.config.CRS_WEB_MERCATOR).geometry.x, lat_gdf.to_crs(self.config.CRS_WEB_MERCATOR).geometry.y
        ax.set_xticks(lon_ticks_proj)
        ax.set_yticks(lat_ticks_proj)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{x_ticks[pos]:.4f}°" if pos < len(x_ticks) else ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{y_ticks[pos]:.4f}°" if pos < len(y_ticks) else ""))
        ax.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

    def _plot_ndvi_map(self, raster_path: str, aoi_gdf: gpd.GeoDataFrame, nome_area: str, date_range: tuple) -> plt.Figure:
        cfg = self.config
        fig, ax = plt.subplots(1, 1, figsize=(16, 14), facecolor='white')
        ax.set_facecolor('#f0f0f0')

        with rasterio.open(raster_path) as src:
            aoi_plot = aoi_gdf.to_crs(cfg.CRS_WEB_MERCATOR)
            data, transform = mask(src, aoi_plot.geometry, crop=True, nodata=np.nan)
            im = show(data, ax=ax, transform=transform, cmap='RdYlGn', vmin=-0.2, vmax=1.0, alpha=0.85, zorder=2)
            cbar = fig.colorbar(im.get_images()[0], ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
            cbar.set_label('Índice de Vegetação por Diferença Normalizada (NDVI)', size=12)
            aoi_plot.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, linestyle='--', zorder=3)
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zorder=1)
        
        self._add_grid_inteligente(ax, aoi_gdf)
        
        month_name = datetime.strptime(cfg.TARGET_DATE, '%Y-%m-%d').strftime("%B de %Y")
        title = f'Análise de NDVI (Landsat 9) - {nome_area}\nMediana de {month_name.capitalize()}'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.add_artist(ScaleBar(1, 'm', location='upper right', box_alpha=0.8, pad=0.5))
        ax.annotate('N', xy=(0.05, 0.98), xytext=(0.05, 0.92), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction')
        
        info_text = "Fonte: USGS/Google (Landsat 9)\n\nProcessamento: GEE\n\nAutor: Pedro Luiz - SIRGAS 2000"
        ax.text(0.75, 0.15, info_text, transform=ax.transAxes, fontsize=9, style='italic',
                ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1))
        
        plt.tight_layout(rect=[0.05, 0.05, 0.85, 0.95])
        return fig

    def _plot_monthly_graph(self, monthly_df: pd.DataFrame, nome_area: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
        ax.set_facecolor('#f0f0f0')
        if monthly_df.empty:
            ax.text(0.5, 0.5, 'Não há dados de imagem sem nuvens para gerar o gráfico mensal.',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return fig

        ax.plot(monthly_df['date'], monthly_df['ndvi'], marker='o', linestyle='-', color='#2ca25f')
        month_name = datetime.strptime(self.config.TARGET_DATE, '%Y-%m-%d').strftime("%B de %Y")
        title = f'Variação Diária do NDVI Médio - {nome_area}\n{month_name.capitalize()}'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=40)
        ax.set_ylabel('NDVI Médio', fontsize=12)
        ax.set_xlabel('Data de Aquisição', fontsize=12)
        ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        fig.autofmt_xdate(rotation=45)
        ax.set_ylim(0, 1)
        plt.tight_layout(pad=3.0)
        return fig

    def _export_to_pdf(self, map_fig: plt.Figure, stats_df: pd.DataFrame, graph_fig: plt.Figure, nome_area: str, date_range: tuple):
        cfg = self.config
        nome_sanitizado = self._sanitizar_nome_arquivo(nome_area)
        date_sanitizado = datetime.strptime(cfg.TARGET_DATE, '%Y-%m-%d').strftime('%Y%m')
        nome_arquivo = f"Relatorio_NDVI_{nome_sanitizado}_{date_sanitizado}.pdf"
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo)

        with PdfPages(caminho_completo) as pdf:
            pdf.savefig(map_fig, dpi=cfg.DPI_SAIDA, bbox_inches='tight')
            pdf.savefig(graph_fig, dpi=cfg.DPI_SAIDA, bbox_inches='tight')
            
            fig_table, ax_table = plt.subplots(figsize=(11.69, 8.27))
            ax_table.axis('off')
            month_name = datetime.strptime(cfg.TARGET_DATE, '%Y-%m-%d').strftime("%B de %Y")
            title = f"Estatísticas Descritivas do NDVI - Mediana Mensal\n{nome_area} ({month_name.capitalize()})"
            ax_table.set_title(title, fontsize=14, fontweight='bold', pad=40)
            
            if stats_df is not None and not stats_df.empty:
                df_display = stats_df.copy()
                df_display['Valor'] = df_display['Valor'].map('{:,.4f}'.format)
                tabela = ax_table.table(cellText=df_display.values, rowLabels=df_display.index,
                                        colLabels=df_display.columns, cellLoc='center', loc='upper center',
                                        rowLoc='left', colWidths=[0.3])
                tabela.auto_set_font_size(False); tabela.set_fontsize(12); tabela.scale(1.5, 2.0)
            else:
                ax_table.text(0.5, 0.5, 'Não há dados para exibir.', ha='center')
            
            pdf.savefig(fig_table, dpi=cfg.DPI_SAIDA, bbox_inches='tight')

        plt.close('all')
        return caminho_completo

    def run(self):
        caminho_temp_raster = None
        try:
            print("="*80); print("INICIANDO PIPELINE HÍBRIDO DE ANÁLISE DE NDVI (GEE + Python)"); print("="*80)
            
            aoi_gdf = self._load_and_prepare_aoi()
            nome_imovel = aoi_gdf.get('recibo', pd.Series(["ImovelSemNome"]))[0]
            nome_sanitizado = self._sanitizar_nome_arquivo(nome_imovel)
            
            print(f"\n[ETAPA 1/5] Processando composite de NDVI para o mês de referência de {self.config.TARGET_DATE}...")
            caminho_temp_raster, start_date, end_date = self._get_ndvi_composite_from_gee(aoi_gdf, nome_sanitizado)
            
            if not caminho_temp_raster:
                print(f"\n[ALERTA] Nenhuma imagem limpa encontrada no período de {start_date} a {end_date}. O pipeline será encerrado.")
                return

            print(f"\n[ETAPA 2/5] Calculando estatísticas do raster local...")
            stats_df = self._calculate_statistics_from_raster(caminho_temp_raster, aoi_gdf)
            display(HTML(f"<h3>Estatísticas do NDVI para: {nome_imovel} (Mês de Referência: {datetime.strptime(self.config.TARGET_DATE, '%Y-%m-%d').strftime('%B de %Y')})</h3>"))
            display(stats_df.style.format({'Valor': '{:,.4f}'}))

            print(f"\n[ETAPA 3/5] Gerando série temporal de NDVI para o mês...")
            monthly_df = self._get_monthly_ndvi_series_from_gee(aoi_gdf)

            print(f"\n[ETAPA 4/5] Gerando visualizações (Mapa e Gráfico)...")
            mapa_fig = self._plot_ndvi_map(caminho_temp_raster, aoi_gdf, nome_imovel, (start_date, end_date))
            display(HTML(f"<h3>Mapa NDVI para: {nome_imovel}</h3>"))
            display(mapa_fig)

            graph_fig = self._plot_monthly_graph(monthly_df, nome_imovel)
            display(HTML(f"<h3>Gráfico de Variação Mensal do NDVI para: {nome_imovel}</h3>"))
            display(graph_fig)

            print(f"\n[ETAPA 5/5] Exportando relatório consolidado em PDF...")
            caminho_salvo = self._export_to_pdf(mapa_fig, stats_df, graph_fig, nome_imovel, (start_date, end_date))
            print(f"\nRelatório final foi salvo em: {caminho_salvo}")
            
            print("\n" + "="*80); print("PIPELINE CONCLUÍDO COM SUCESSO"); print("="*80)

        except Exception as e:
            print(f"\n[ERRO CRÍTICO NO PIPELINE] Ocorreu um erro: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if caminho_temp_raster and os.path.exists(caminho_temp_raster):
                os.remove(caminho_temp_raster)
                print(f"\nArquivo temporário '{os.path.basename(caminho_temp_raster)}' removido.")

if __name__ == '__main__':
    config = NDVIConfig()
    pipeline = NDVIPipeline(config)
    pipeline.run()