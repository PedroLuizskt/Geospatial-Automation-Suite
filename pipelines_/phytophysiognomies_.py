## Análise de Biomas e Fitofisionomias - Autor = Pedro Luiz

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
class EcologyConfig:
    PATH_BIOMAS: str = r"C:\Users\pedro\Downloads\python_gis\script_bioma_fisio\biomas_\Biomas_Brasil_IBGE.shp"
    PATH_VEGETACAO: str = r"C:\Users\pedro\Downloads\python_gis\script_bioma_fisio\vege_area\vege_area_corrigido.shp"
    PATH_MUNICIPIOS: str = r"C:\Users\pedro\Downloads\python_gis\script_bioma_fisio\BR_Municipios_2024\BR_Municipios_2024.shp"
    PATH_IMOVEL_ALVO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\MG-3121605-0C5FDE61120E4BB7B3CD1728A97638E2\Area_do_Imovel\Area_do_Imovel.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\estagio_ie\resultados_"
    COLUNA_NOME_BIOMA: str = 'Bioma'
    COLUNA_FITOFISIONOMIA: str = 'legenda_1'
    COLUNA_NOME_MUNICIPIO: str = 'NM_MUN'
    COLUNA_NOME_IMOVEL: str = 'recibo'
    ENCODING_MUNICIPIOS: str = 'utf-8'
    ENCODING_BIOMAS: str = 'cp1252'
    ENCODING_VEGETACAO: str = 'cp1252'
    ENCODING_IMOVEL: str = 'utf-8'
    DPI_SAIDA: int = 300
    TIPO_DE_AREA: str = 'municipio'
    NOME_MUNICIPIO_ALVO: str = 'São Félix do Xingu'
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    CRS_WEB_MERCATOR: str = 'EPSG:3857'
    CORES_BIOMAS: dict = field(default_factory=dict)
    REGRAS_DE_CORES_FISIO: dict = field(default_factory=dict)
    COR_PADRAO_FISIO: str = '#C0C0C0'
    
    def __post_init__(self):
        self.CORES_BIOMAS = {'amazonia': '#006400', 'cerrado': '#7dc975', 'mata atlantica': '#1f8d49', 'caatinga': '#d6bc74', 'pampa': '#edde8e', 'pantanal': '#519799'}
        self.REGRAS_DE_CORES_FISIO = {'floresta': '#1f8d49', 'savana': '#7dc975', 'campestre': '#d6bc74', 'campo': '#d6bc74', 'mangue': '#04381d', 'restinga': '#ad5100', 'não vegetada': '#db4d4f', 'água': '#2532e4', 'afloramento': '#a9a9a9', 'secundária': '#bDB76B', 'antrópica': '#E97451', 'contato': '#A0522D'}

class EcologyPipeline:
    def __init__(self, config: EcologyConfig):
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

    def _analisar_dados_vetoriais(self, base_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame, coluna_analise: str, nome_analise: str):
        cfg = self.config
        recortado_gdf = gpd.clip(base_gdf, aoi_gdf)
        if recortado_gdf.empty:
            return None, None
        
        crs_metrico_local = self._estimate_utm_crs(recortado_gdf)
        metric_gdf = recortado_gdf.to_crs(crs_metrico_local)
        metric_gdf['area_ha'] = metric_gdf.geometry.area / 10000
        estatisticas = metric_gdf.groupby(coluna_analise)[['area_ha']].sum().sort_values(by='area_ha', ascending=False).reset_index()
        return recortado_gdf, estatisticas

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

    def _plotar_mapa_biomas(self, biomas_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame, nome_area: str) -> plt.Figure:
        cfg = self.config
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
        ax.set_facecolor('#f0f0f0')
        
        biomas_plot, aoi_plot = biomas_gdf.to_crs(cfg.CRS_WEB_MERCATOR), aoi_gdf.to_crs(cfg.CRS_WEB_MERCATOR)
        
        color_series = biomas_plot[cfg.COLUNA_NOME_BIOMA].apply(self._normalize_string).map(cfg.CORES_BIOMAS)
        biomas_plot.plot(ax=ax, color=color_series, edgecolor='black', linewidth=0.2, alpha=0.8)
        aoi_plot.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, linestyle='--')
        
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        self._add_grid_inteligente(ax, aoi_gdf)
        
        ax.set_title(f'Biomas - {nome_area}', fontsize=18, fontweight='bold')
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction')
        
        handles = [Patch(facecolor=cfg.CORES_BIOMAS.get(self._normalize_string(b), '#C0C0C0'), edgecolor='black', label=b) for b in biomas_gdf[cfg.COLUNA_NOME_BIOMA].unique()]
        ax.legend(handles=handles, title='Biomas', loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True)
        
        ax.add_artist(ScaleBar(1, 'm', location='lower right', box_alpha=0.8, pad=0.5))
        fig.text(0.68, 0.2, f"Fonte: IBGE (2019)\nAutor: Pedro Luiz\nDatum Geográfico: SIRGAS 2000", ha='left', va='bottom', fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1))
        
        plt.tight_layout(rect=[0, 0.05, 0.85, 0.95])
        return fig

    def _encontrar_cor_para_fisio(self, d: str) -> str:
        return next((c for k, c in self.config.REGRAS_DE_CORES_FISIO.items() if k in str(d).lower()), self.config.COR_PADRAO_FISIO)

    def _plotar_mapa_fitofisionomias(self, fisio_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame, nome_area: str) -> plt.Figure:
        cfg = self.config
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
        ax.set_facecolor('#f0f0f0')
        
        fisio_plot, aoi_plot = fisio_gdf.to_crs(cfg.CRS_WEB_MERCATOR), aoi_gdf.to_crs(cfg.CRS_WEB_MERCATOR)
        
        mapa_cores = {d: self._encontrar_cor_para_fisio(d) for d in fisio_plot[cfg.COLUNA_FITOFISIONOMIA].unique()}
        fisio_plot.plot(ax=ax, color=fisio_plot[cfg.COLUNA_FITOFISIONOMIA].map(mapa_cores), edgecolor='black', linewidth=0.2, alpha=0.8)
        aoi_plot.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, linestyle='--')
        
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        self._add_grid_inteligente(ax, aoi_gdf)
        
        ax.set_title(f'Fitofisionomias - {nome_area}', fontsize=18, fontweight='bold')
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction')
        
        handles = [Patch(facecolor=c, edgecolor='black', label=d) for d, c in sorted(mapa_cores.items())]
        ax.legend(handles=handles, title='Fitofisionomias', loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize='small')
        
        ax.add_artist(ScaleBar(1, 'm', location='lower right', box_alpha=0.8, pad=0.5))
        fig.text(0.68, 0.2, f"Fonte: IBGE (2021)\nAutor: Pedro Luiz\nDatum Geográfico: SIRGAS 2000", ha='left', va='bottom', fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1))
        
        plt.tight_layout(rect=[0, 0.05, 0.85, 0.95])
        return fig
        
    def _exportar_relatorio_pdf(self, resultados: dict, nome_area: str, nome_arquivo_saida: str):
        cfg = self.config
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo_saida)
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        
        with PdfPages(caminho_completo) as pdf:
            for analise, dados in resultados.items():
                if dados.get('figura'):
                    pdf.savefig(dados['figura'], bbox_inches='tight', dpi=cfg.DPI_SAIDA)
                    plt.close(dados['figura'])
                
                if dados.get('stats') is not None:
                    fig_tabela, ax_tabela = plt.subplots(figsize=(8.27, 11.69))
                    ax_tabela.axis('off')
                    ax_tabela.set_title(f"Análise Quantitativa de {analise}\n{nome_area}", fontsize=16, fontweight='bold', pad=20)
                    
                    df_display = dados['stats'].copy()
                    df_display.rename(columns={dados['coluna']: analise, 'area_ha': 'Área (ha)'}, inplace=True)
                    df_display['Área (ha)'] = [f"{v:_.2f}".replace('.', ',').replace('_', '.') for v in df_display['Área (ha)']]
                    
                    tabela = ax_tabela.table(cellText=df_display.values, colLabels=df_display.columns, cellLoc='left', loc='upper center', colWidths=[0.6, 0.2])
                    tabela.auto_set_font_size(False); tabela.set_fontsize(12); tabela.scale(1.2, 2.5)
                    
                    pdf.savefig(fig_tabela, bbox_inches='tight', dpi=cfg.DPI_SAIDA)
                    plt.close(fig_tabela)
                    
        return caminho_completo

    def run(self):
        try:
            print("="*80); print("     INICIANDO PIPELINE INTEGRADO: BIOMAS E FITOFISIONOMIAS "); print("="*80)
            
            aoi_gdf = self._obter_area_de_interesse()

            biomas_gdf = self._carregar_dados_vetoriais(self.config.PATH_BIOMAS, self.config.CRS_GEOGRAFICO, bbox=tuple(aoi_gdf.total_bounds), encoding=self.config.ENCODING_BIOMAS)
            fisio_gdf = self._carregar_dados_vetoriais(self.config.PATH_VEGETACAO, self.config.CRS_GEOGRAFICO, bbox=tuple(aoi_gdf.total_bounds), encoding=self.config.ENCODING_VEGETACAO)
            
            if self.config.TIPO_DE_AREA == 'municipio':
                nome_area = self.config.NOME_MUNICIPIO_ALVO
            else:
                nome_area = aoi_gdf[self.config.COLUNA_NOME_IMOVEL].iloc[0] if self.config.COLUNA_NOME_IMOVEL in aoi_gdf.columns else "Imovel Rural Sem Nome"

            nome_base_sanitizado = self._sanitizar_nome_arquivo(nome_area)
            nome_arquivo_final = f"Relatorio_Bioma_Fisio_{nome_base_sanitizado}.pdf"
            
            resultados_analise = {}
            
            recorte_biomas, stats_biomas = self._analisar_dados_vetoriais(biomas_gdf, aoi_gdf, self.config.COLUNA_NOME_BIOMA, "Biomas")
            if recorte_biomas is not None:
                resultados_analise['Biomas'] = {'gdf': recorte_biomas, 'stats': stats_biomas, 'coluna': self.config.COLUNA_NOME_BIOMA}

            recorte_fisio, stats_fisio = self._analisar_dados_vetoriais(fisio_gdf, aoi_gdf, self.config.COLUNA_FITOFISIONOMIA, "Fitofisionomias")
            if recorte_fisio is not None:
                resultados_analise['Fitofisionomias'] = {'gdf': recorte_fisio, 'stats': stats_fisio, 'coluna': self.config.COLUNA_FITOFISIONOMIA}

            if not resultados_analise:
                print("\nNenhuma análise produziu resultados."); return
            
            for nome_analise, dados in resultados_analise.items():
                plot_func = self._plotar_mapa_biomas if nome_analise == 'Biomas' else self._plotar_mapa_fitofisionomias
                figura = plot_func(dados['gdf'], aoi_gdf, nome_area)
                
                display(HTML(f"<h2>Análise de {nome_analise} para: {nome_area}</h2>"))
                display(figura)
                plt.close(figura)
                
                print("\n" + "="*70); print(f"        ANÁLISE QUANTITATIVA DE {nome_analise.upper()}"); print("="*70)
                stats_console = dados['stats'].rename(columns={dados['coluna']: nome_analise, 'area_ha': 'Área (ha)'}).copy()
                stats_console['Área (ha)'] = [f"{v:_.2f}".replace('.', ',').replace('_', '.') for v in stats_console['Área (ha)']]
                print(stats_console.to_string(index=False)); print("="*70)

            if 'Biomas' in resultados_analise:
                resultados_analise['Biomas']['figura'] = self._plotar_mapa_biomas(resultados_analise['Biomas']['gdf'], aoi_gdf, nome_area)
            if 'Fitofisionomias' in resultados_analise:
                resultados_analise['Fitofisionomias']['figura'] = self._plotar_mapa_fitofisionomias(resultados_analise['Fitofisionomias']['gdf'], aoi_gdf, nome_area)

            caminho_salvo = self._exportar_relatorio_pdf(resultados_analise, nome_area, nome_arquivo_final)
            print(f"\nAnálise integrada para '{nome_area}' concluída! Relatório salvo em:\n{caminho_salvo}")
        
        except (ValueError, FileNotFoundError) as e:
            print(f"\n[ERRO CONTROLADO] {e}")
        except Exception as e:
            print(f"\n[ERRO CRÍTICO INESPERADO] {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__":
    config = EcologyConfig(
        TIPO_DE_AREA='imovel',
        NOME_MUNICIPIO_ALVO='Querência'
    )
    pipeline = EcologyPipeline(config)
    pipeline.run()