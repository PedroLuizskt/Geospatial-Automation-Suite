## Análise de Feições Geomorfológicas - Autor = Pedro Luiz

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
import contextily as ctx
import unicodedata
import re
import numpy as np
from dataclasses import dataclass, field
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FuncFormatter
from IPython.display import display, HTML

@dataclass
class GeomorphologyConfig:
    PATH_GEOMORFOLOGIA: str = r"C:\Users\pedro\Downloads\python_gis\script_geomorfo\geom_area\geom_area_corrigido.shp"
    PATH_MUNICIPIOS: str = r"C:\Users\pedro\Downloads\python_gis\script_geomorfo\BR_Municipios_2024\BR_Municipios_2024.shp"
    PATH_IMOVEL_ALVO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\MT-5106422-2960697C669941729C7EF7C2930CBA5A\Area_do_Imovel\Area_do_Imovel.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\estagio_ie\resultados_"
    COLUNA_GEOMORFOLOGIA: str = "nm_unidade"
    COLUNA_NOME_MUNICIPIO: str = 'NM_MUN'
    COLUNA_NOME_IMOVEL: str = 'recibo'
    ENCODING_MUNICIPIOS: str = 'utf-8'
    ENCODING_GEOMORFOLOGIA: str = 'cp1252'
    ENCODING_IMOVEL: str = 'utf-8'
    DPI_SAIDA: int = 300
    TIPO_DE_AREA: str = 'municipio'
    NOME_MUNICIPIO_ALVO: str = 'Dianópolis'
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    CRS_WEB_MERCATOR: str = 'EPSG:3857'
    REGRAS_DE_CORES: dict = field(default_factory=dict)
    COR_PADRAO: str = '#E0E0E0'
    
    def __post_init__(self):
        self.REGRAS_DE_CORES = {
            'planalto': '#D2B48C', 'chapada': '#F4A460', 'chapadão': '#F4A460',
            'depressão': '#FFD700', 'pediplano': '#FFD700', 'vão': '#FEE090',
            'planície': '#90EE90', 'baixada': '#91cf60', 'pantanal': '#66BD63',
            'delta': '#1A9850', 'leque aluvial': '#006837', 'tabuleiro': '#d9ef8b',
            'serra': '#A50026', 'serrania': '#A50026', 'maciço': '#A50026',
            'crista': '#A9A9A9', 'patamar': '#A0522D'
        }

class GeomorphologyPipeline:
    def __init__(self, config: GeomorphologyConfig):
        self.config = config

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
            gdf = self._carregar_dados_vetoriais(cfg.PATH_MUNICIPIOS, cfg.CRS_GEOGRAFICO, encoding=cfg.ENCODING_MUNICIPIOS)
            target_norm = self._normalize_string(cfg.NOME_MUNICIPIO_ALVO)
            gdf['search_col'] = gdf[cfg.COLUNA_NOME_MUNICIPIO].apply(self._normalize_string)
            area_filtrada_gdf = gdf[gdf['search_col'] == target_norm]
            if area_filtrada_gdf.empty: raise ValueError(f"Município '{cfg.NOME_MUNICIPIO_ALVO}' não encontrado.")
            area_gdf = area_filtrada_gdf.dissolve().reset_index()
        elif cfg.TIPO_DE_AREA == 'imovel':
            area_gdf = self._carregar_dados_vetoriais(cfg.PATH_IMOVEL_ALVO, cfg.CRS_GEOGRAFICO, encoding=cfg.ENCODING_IMOVEL)
        else:
            raise ValueError("Tipo de área inválido.")
        if area_gdf.empty: raise ValueError("A Área de Interesse (AOI) está vazia.")
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

    def _analisar_geomorfologia(self, geomorfologia_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame):
        cfg = self.config
        recortado_gdf = gpd.clip(geomorfologia_gdf, aoi_gdf)
        if recortado_gdf.empty: return None, None
        
        crs_metrico_local = self._estimate_utm_crs(recortado_gdf)
        geom_metric_gdf = recortado_gdf.to_crs(crs_metrico_local)
        geom_metric_gdf['area_ha'] = geom_metric_gdf.geometry.area / 10000
        
        estatisticas = (geom_metric_gdf.groupby(cfg.COLUNA_GEOMORFOLOGIA)[['area_ha']]
                                     .sum()
                                     .sort_values(by='area_ha', ascending=False)
                                     .reset_index())
        return recortado_gdf, estatisticas

    def _encontrar_cor_para_unidade(self, descricao: str) -> str:
        desc_lower = str(descricao).lower()
        return next((cor for palavra, cor in self.config.REGRAS_DE_CORES.items() if palavra in desc_lower), self.config.COR_PADRAO)

    def _add_grid_inteligente(self, ax, gdf_geografico):
        cfg = self.config
        minx, miny, maxx, maxy = gdf_geografico.total_bounds
        x_interval, y_interval = (maxx - minx) / 5, (maxy - miny) / 5
        x_ticks, y_ticks = np.arange(minx, maxx, x_interval), np.arange(miny, maxy, y_interval)
        
        lon_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_ticks, [miny] * len(x_ticks)), crs=cfg.CRS_GEOGRAFICO)
        lat_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([minx] * len(y_ticks), y_ticks), crs=cfg.CRS_GEOGRAFICO)
        
        lon_ticks_proj, lat_ticks_proj = lon_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.x, lat_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.y
        
        ax.set_xticks(lon_ticks_proj)
        ax.set_yticks(lat_ticks_proj)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{x_ticks[pos]:.2f}°" if pos < len(x_ticks) else ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{y_ticks[pos]:.2f}°" if pos < len(y_ticks) else ""))
        
        ax.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    def _plotar_mapa(self, geom_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame, nome_area: str) -> plt.Figure:
        cfg = self.config
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
        ax.set_facecolor('#f0f0f0')
        
        geom_plot = geom_gdf.to_crs(cfg.CRS_WEB_MERCATOR)
        aoi_plot = aoi_gdf.to_crs(cfg.CRS_WEB_MERCATOR)
        
        mapa_de_cores = {unidade: self._encontrar_cor_para_unidade(unidade) for unidade in geom_plot[cfg.COLUNA_GEOMORFOLOGIA].unique()}
        
        geom_plot.plot(ax=ax, color=geom_plot[cfg.COLUNA_GEOMORFOLOGIA].map(mapa_de_cores), edgecolor='black', linewidth=0.2, alpha=0.8)
        aoi_plot.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, linestyle='--')
        
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto')
        self._add_grid_inteligente(ax, aoi_gdf)
        
        ax.set_title(f'Feições Geomorfológicas - {nome_area}', fontsize=18, fontweight='bold')
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction')
        
        elementos_legenda = [Patch(facecolor=cor, edgecolor='black', label=unidade) for unidade, cor in sorted(mapa_de_cores.items())]
        ax.legend(handles=elementos_legenda, title='Unidades Geomorfológicas', loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True)
        
        ax.add_artist(ScaleBar(1, 'm', location='lower right', box_alpha=0.8, pad=0.5))
        
        fig.text(0.72, 0.2, f"Fonte: IBGE/CPRM (2006)\nAutor: Pedro Luiz\nDatum Geográfico: SIRGAS 2000", ha='left', va='bottom', fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1))
        
        plt.tight_layout(rect=[0, 0.05, 0.85, 0.95])
        return fig

    def _exportar_relatorio_pdf(self, figura_mapa: plt.Figure, df_estatisticas: pd.DataFrame, nome_area: str, nome_arquivo_saida: str):
        cfg = self.config
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo_saida)
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        
        with PdfPages(caminho_completo) as pdf:
            pdf.savefig(figura_mapa, bbox_inches='tight', dpi=cfg.DPI_SAIDA); plt.close(figura_mapa)
            
            fig_tabela, ax_tabela = plt.subplots(figsize=(8.27, 11.69))
            ax_tabela.axis('off')
            ax_tabela.set_title(f"Análise Quantitativa de Geomorfologia\n{nome_area}", fontsize=16, fontweight='bold', pad=20)
            
            df_display = df_estatisticas.copy()
            df_display.rename(columns={cfg.COLUNA_GEOMORFOLOGIA: 'Unidade', 'area_ha': 'Área (ha)'}, inplace=True)
            df_display['Área (ha)'] = [f"{v:_.2f}".replace('.', ',').replace('_', '.') for v in df_display['Área (ha)']]
            
            tabela = ax_tabela.table(cellText=df_display.values, colLabels=df_display.columns, cellLoc='left', loc='upper center', colWidths=[0.6, 0.2])
            tabela.auto_set_font_size(False); tabela.set_fontsize(12); tabela.scale(1.2, 2.5)
            
            pdf.savefig(fig_tabela, bbox_inches='tight', dpi=cfg.DPI_SAIDA); plt.close(fig_tabela)
            
        return caminho_completo

    def run(self):
        try:
            print("="*80); print("       INICIANDO PIPELINE DE ANÁLISE DE GEOMORFOLOGIA "); print("="*80)
            
            aoi_gdf = self._obter_area_de_interesse()
            
            geomorfologia_gdf = self._carregar_dados_vetoriais(
                self.config.PATH_GEOMORFOLOGIA,
                crs_alvo=self.config.CRS_GEOGRAFICO,
                bbox=tuple(aoi_gdf.total_bounds),
                encoding=self.config.ENCODING_GEOMORFOLOGIA
            )
            
            if self.config.TIPO_DE_AREA == 'municipio':
                nome_area = self.config.NOME_MUNICIPIO_ALVO
            else:
                if self.config.COLUNA_NOME_IMOVEL in aoi_gdf.columns:
                    nome_area = aoi_gdf[self.config.COLUNA_NOME_IMOVEL].iloc[0]
                else:
                    nome_area = "Imovel Rural Sem Nome"
            
            nome_base_sanitizado = self._sanitizar_nome_arquivo(nome_area)
            nome_arquivo_final = f"Relatorio_Geomorfologia_{nome_base_sanitizado}.pdf"
            
            recortado_gdf, estatisticas_df = self._analisar_geomorfologia(geomorfologia_gdf, aoi_gdf)
            
            if recortado_gdf is None or recortado_gdf.empty:
                print(f"\nAVISO: Nenhuma feição geomorfológica encontrada para '{nome_area}'.")
            else:
                display(HTML(f"<h2>Análise Geomorfológica para: {nome_area}</h2>"))
                figura_para_display = self._plotar_mapa(recortado_gdf, aoi_gdf, nome_area)
                display(figura_para_display); plt.close(figura_para_display)
                
                print("\n" + "="*70); print(f"            ANÁLISE QUANTITATIVA PARA: {nome_area.upper()}"); print("="*70)
                stats_console = estatisticas_df.rename(columns={self.config.COLUNA_GEOMORFOLOGIA: 'Unidade', 'area_ha': 'Área (ha)'}).copy()
                stats_console['Área (ha)'] = [f"{v:_.2f}".replace('.', ',').replace('_', '.') for v in stats_console['Área (ha)']]
                print(stats_console.to_string(index=False)); print("="*70)
                
                figura_para_pdf = self._plotar_mapa(recortado_gdf, aoi_gdf, nome_area)
                caminho_salvo = self._exportar_relatorio_pdf(figura_para_pdf, estatisticas_df, nome_area, nome_arquivo_final)
                print(f"\nAnálise de geomorfologia para '{nome_area}' concluída! Relatório salvo em:\n{caminho_salvo}")
        
        except (ValueError, FileNotFoundError) as e:
            print(f"\n[ERRO CONTROLADO] {e}")
        except Exception as e:
            print(f"\n[ERRO CRÍTICO INESPERADO] {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__":
    config = GeomorphologyConfig(
        TIPO_DE_AREA='imovel',
        NOME_MUNICIPIO_ALVO='Manaus'
    )
    pipeline = GeomorphologyPipeline(config)
    pipeline.run()