## Caracterização CAR - Autor: Pedro Luiz 

import geopandas as gpd
import pandas as pd
import os
import unicodedata
import re
import numpy as np
import textwrap
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import contextily as ctx
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FuncFormatter
from IPython.display import display, HTML

@dataclass
class CARConfig:
    BASE_PATH: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\SC-4204558-9329B4217DB340D195B79197FF534CD3"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\result_2_"
    PATH_AMAZONIA_LEGAL: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\Limites_Amazonia_Legal_2024_shp\Limites_Amazonia_Legal_2024.shp"
    PATH_BIOMAS: str = r"C:\Users\pedro\Downloads\python_gis\script_bioma_fisio\biomas_\Biomas_Brasil_IBGE.shp"
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    CRS_WEB_MERCATOR: str = 'EPSG:3857'
    ENCODING_CAR: str = 'latin-1'
    ENCODING_AMAZONIA: str = 'utf-8'
    ENCODING_BIOMAS: str = 'cp1252'
    DPI_SAIDA: int = 300
    COLUNA_TEMA: str = 'tema'
    COLUNA_NOME_IMOVEL: str = 'recibo'
    COLUNA_BIOMA: str = 'Bioma'
    TEMA_IMOVEL_TOTAL: str = 'Área do Imovel'
    TEMA_APP_TOTAL: str = 'APP Total'
    TEMA_RL_TOTAL: str = 'Área de Reserva Legal Total'
    TEMA_VEG_NATIVA: str = 'Remanescente de Vegetação Nativa'
    TEMA_AREA_CONSOLIDADA: str = 'Área Consolidada'
    STYLE_MAP: dict = field(default_factory=dict)
    PATH_IMOVEL: str = field(init=False)
    PATH_APP: str = field(init=False)
    PATH_RL: str = field(init=False)
    PATH_COBERTURA: str = field(init=False)
    PATH_SERVIDAO: str = field(init=False)

    def __post_init__(self):
        self.PATH_IMOVEL = os.path.join(self.BASE_PATH, r"Area_do_Imovel\Area_do_Imovel.shp")
        self.PATH_APP = os.path.join(self.BASE_PATH, r"Area_de_Preservacao_Permanente\Area_de_Preservacao_Permanente.shp")
        self.PATH_RL = os.path.join(self.BASE_PATH, r"Reserva_Legal\Reserva_Legal.shp")
        self.PATH_COBERTURA = os.path.join(self.BASE_PATH, r"Cobertura_do_Solo\Cobertura_do_Solo.shp")
        self.PATH_SERVIDAO = os.path.join(self.BASE_PATH, r"Servidao_Administrativa\Servidao_Administrativa.shp")
        self.STYLE_MAP = {
            'VEGETACAO_NATIVA': {'facecolor': '#2ca25f', 'alpha': 0.7, 'edgecolor': '#2ca25f', 'linewidth': 0.5},
            'AREA_CONSOLIDADA': {'facecolor': '#fdae6b', 'alpha': 0.7, 'edgecolor': '#fdae6b', 'linewidth': 0.5},
            'APP': {'facecolor': 'none', 'hatch': '///', 'edgecolor': 'blue', 'linewidth': 1.0},
            'RL': {'facecolor': 'none', 'hatch': '\\\\\\', 'edgecolor': 'green', 'linewidth': 1.0},
            'SERVIDAO': {'facecolor': 'none', 'hatch': 'xxx', 'edgecolor': 'red', 'linewidth': 1.0},
            'IMOVEL': {'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 2.0}
        }

class CARCharacterizationPipeline:
    def __init__(self, config: CARConfig):
        self.config = config
        self.cache_camadas = {}

    def _get_camada(self, nome_camada: str, path: str, encoding: str, filter_dict: dict = None, dissolve: bool = False) -> gpd.GeoDataFrame:
        cache_key = f"{nome_camada}_{filter_dict}_{dissolve}"
        if cache_key in self.cache_camadas:
            return self.cache_camadas[cache_key]

        print(f"Carregando e processando camada: '{nome_camada}'...")
        if not os.path.exists(path):
            if nome_camada == 'SERVIDAO':
                print(f"[AVISO] Camada '{nome_camada}' não encontrada. A área será considerada como zero.")
                return gpd.GeoDataFrame(geometry=[], crs=self.config.CRS_GEOGRAFICO)
            else:
                raise FileNotFoundError(f"Arquivo essencial não encontrado: {path}")

        gdf = gpd.read_file(path, encoding=encoding).to_crs(self.config.CRS_GEOGRAFICO)
        if filter_dict:
            gdf = gdf[gdf[self.config.COLUNA_TEMA] == filter_dict[self.config.COLUNA_TEMA]]
        if dissolve:
            gdf = gdf.dissolve().reset_index()
        
        gdf = gdf.reset_index(drop=True)
        self.cache_camadas[cache_key] = gdf
        return gdf

    def _estimate_utm_crs(self, gdf_geo: gpd.GeoDataFrame) -> str:
        try:
            
            centroid = gdf_geo.union_all().centroid
            utm_zone = int((centroid.x + 180) / 6) + 1
            is_south = centroid.y < 0
            epsg = 32700 + utm_zone if is_south else 32600 + utm_zone
            return f"EPSG:{epsg}"
        except Exception: return 'EPSG:5880'

    def _get_area_ha(self, gdf: gpd.GeoDataFrame) -> float:
        if gdf.empty or gdf.geometry.is_empty.all(): return 0.0
        crs_metrico = self._estimate_utm_crs(gdf)
        return gdf.to_crs(crs_metrico).area.sum() / 10000

    def _determinar_percentual_rl(self, imovel_gdf: gpd.GeoDataFrame) -> tuple[float, str, str]:
        print("Determinando percentual de Reserva Legal aplicável...")
        cfg = self.config
        
        amazonia_legal_gdf = self._get_camada('AMAZONIA_LEGAL', cfg.PATH_AMAZONIA_LEGAL, cfg.ENCODING_AMAZONIA)
        biomas_gdf = self._get_camada('BIOMAS', cfg.PATH_BIOMAS, cfg.ENCODING_BIOMAS)

        is_in_amazonia = not gpd.sjoin(imovel_gdf, amazonia_legal_gdf, how="inner", predicate="intersects").empty
        
        intersecao_bioma = gpd.sjoin(imovel_gdf, biomas_gdf, how="inner", predicate="intersects")
        if intersecao_bioma.empty:
            print("[AVISO] Não foi possível determinar o bioma do imóvel. Usando regra geral (20%).")
            return 0.20, "Indeterminado", "Fora da Amazônia Legal"

        bioma_imovel = intersecao_bioma[self.config.COLUNA_BIOMA].iloc[0]
        localizacao = "Amazônia Legal" if is_in_amazonia else "Fora da Amazônia Legal"

        if is_in_amazonia:
            if bioma_imovel == 'Amazônia':
                print(f"  -> Imóvel na Amazônia Legal, bioma '{bioma_imovel}'. RL = 80%")
                return 0.80, bioma_imovel, localizacao
            elif bioma_imovel == 'Cerrado':
                print(f"  -> Imóvel na Amazônia Legal, bioma '{bioma_imovel}'. RL = 35%")
                return 0.35, bioma_imovel, localizacao
            else:
                print(f"  -> Imóvel na Amazônia Legal, bioma '{bioma_imovel}'. RL = 20%")
                return 0.20, bioma_imovel, localizacao
        else:
            print(f"  -> Imóvel fora da Amazônia Legal, bioma '{bioma_imovel}'. RL = 20%")
            return 0.20, bioma_imovel, localizacao
            
    def _add_grid_inteligente(self, ax, gdf_geografico):
        cfg = self.config; minx, miny, maxx, maxy = gdf_geografico.total_bounds
        x_interval = (maxx - minx) / 5; y_interval = (maxy - miny) / 5
        x_ticks, y_ticks = np.arange(minx, maxx, x_interval), np.arange(miny, maxy, y_interval)
        lon_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_ticks, [miny] * len(x_ticks)), crs=cfg.CRS_GEOGRAFICO)
        lat_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([minx] * len(y_ticks), y_ticks), crs=cfg.CRS_GEOGRAFICO)
        lon_ticks_proj = lon_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.x; lat_ticks_proj = lat_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.y
        ax.set_xticks(lon_ticks_proj); ax.set_yticks(lat_ticks_proj)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{x_ticks[pos]:.2f}°" if pos < len(x_ticks) else "")); ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{y_ticks[pos]:.2f}°" if pos < len(y_ticks) else ""))
        ax.grid(True, linestyle='--', alpha=0.6, color='gray'); ax.tick_params(axis='x', rotation=45, labelsize=10); ax.tick_params(axis='y', labelsize=10)

    def _sanitizar_nome_arquivo(self, nome: str) -> str:
        if not isinstance(nome, str): return ""
        nome_norm = unicodedata.normalize('NFKD', nome).encode('ascii', 'ignore').decode('utf-8').lower().strip()
        nome_sanitizado = re.sub(r'[^\w\s-]', '', nome_norm)
        return re.sub(r'[-\s]+', '-', nome_sanitizado)[:100]

    def _gerar_mapa(self, data_layers: dict, nome_imovel: str) -> plt.Figure:
        cfg = self.config
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
        ax.set_facecolor('#f0f0f0')
        for layer_name, gdf in data_layers.items():
            if not gdf.empty:
                plot_gdf = gdf.to_crs(cfg.CRS_WEB_MERCATOR)
                plot_gdf.plot(ax=ax, **cfg.STYLE_MAP[layer_name])
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto')
        self._add_grid_inteligente(ax, data_layers['IMOVEL'])
        ax.set_title(f'Caracterização Completa do Imóvel Rural\n{nome_imovel}', fontsize=18, fontweight='bold')
        ax.add_artist(ScaleBar(1, 'm', location='upper right', box_alpha=0.8, pad=0.5))
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction')
        legend_elements = [
            Patch(**cfg.STYLE_MAP['VEGETACAO_NATIVA'], label='Remanescente de Vegetação Nativa'),
            Patch(**cfg.STYLE_MAP['AREA_CONSOLIDADA'], label='Área de Uso Consolidado'),
            Patch(**cfg.STYLE_MAP['APP'], label='APP (designação)'),
            Patch(**cfg.STYLE_MAP['RL'], label='Reserva Legal (designação)'),
            Patch(**cfg.STYLE_MAP['SERVIDAO'], label='Servidão Administrativa'),
            Line2D([0], [0], color='black', lw=2, label='Limite do Imóvel')
        ]
        ax.legend(handles=legend_elements, title='LEGENDA', loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True)
        fig.text(0.60, 0.12, f"Fonte: SICAR / IBGE\nAutor: Pedro Luiz\nDatum: SIRGAS 2000", ha='left', va='bottom', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1))
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        return fig

    def _exportar_relatorio_pdf(self, figura_mapa: plt.Figure, stats: dict, nome_imovel: str):
        cfg = self.config
        nome_base_sanitizado = self._sanitizar_nome_arquivo(nome_imovel)
        nome_arquivo_final = f"Relatorio_Caracterizacao_{nome_base_sanitizado}.pdf"
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo_final)
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        with PdfPages(caminho_completo) as pdf:
            pdf.savefig(figura_mapa, bbox_inches='tight', dpi=cfg.DPI_SAIDA); plt.close(figura_mapa)
            def plot_table(df, title, col_widths, ax):
                ax.axis('off'); ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
                tabela = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='left', loc='upper center', colWidths=col_widths)
                tabela.auto_set_font_size(False); tabela.set_fontsize(10); tabela.scale(1, 1.8)
                for (i, j), cell in tabela.get_celld().items():
                    if i == 0: cell.set_text_props(weight='bold')
                return ax.get_figure()

            fig_quadro, ax_quadro = plt.subplots(figsize=(8.27, 11.69))
            df_quadro_display = stats['quadro_geral'].copy()
            df_quadro_display['Área (ha)'] = df_quadro_display['Área (ha)'].map('{:,.4f}'.format); df_quadro_display['% da ATI'] = df_quadro_display['% da ATI'].map('{:,.2f}%'.format)
            pdf.savefig(plot_table(df_quadro_display, f"Quadro Geral de Áreas\n{nome_imovel}", [0.6, 0.2, 0.2], ax_quadro), bbox_inches='tight'); plt.close(fig_quadro)
            
            if not stats['detalhe_app'].empty:
                fig_app, ax_app = plt.subplots(figsize=(10.27, 11.69))
                df_app_display = stats['detalhe_app'].copy()
                df_app_display[cfg.COLUNA_TEMA] = df_app_display[cfg.COLUNA_TEMA].apply(lambda x: '\n'.join(textwrap.wrap(x, 80)))
                df_app_display['area_ha'] = df_app_display['area_ha'].map('{:,.4f}'.format)
                df_app_display.rename(columns={'tema': 'Tipo de APP', 'area_ha': 'Área (ha)'}, inplace=True)
                pdf.savefig(plot_table(df_app_display, f"Detalhamento de APP\n{nome_imovel}", [1.0, 0.2], ax_app), bbox_inches='tight'); plt.close(fig_app)
            
            
            fig_rl, ax_rl = plt.subplots(figsize=(8.27, 11.69))
            df_rl_display = stats['balanco_rl'].copy()
            for i in range(1, len(df_rl_display)):
                df_rl_display.iloc[i, 1] = f"{df_rl_display.iloc[i, 1]:,.4f}"
            
            pdf.savefig(plot_table(df_rl_display, f"Balanço de Reserva Legal (RL)\n{nome_imovel}", [0.6, 0.6], ax_rl), bbox_inches='tight'); plt.close(fig_rl)
        print(f"Relatório PDF exportado com sucesso para: {caminho_completo}")

    def run(self):
        print("="*80); print("INICIANDO CARACTERIZAÇÃO COMPLETA DE IMÓVEL RURAL (CAR) "); print("="*80)
        cfg = self.config
        
        imovel_gdf = self._get_camada('IMOVEL', cfg.PATH_IMOVEL, cfg.ENCODING_CAR, {cfg.COLUNA_TEMA: cfg.TEMA_IMOVEL_TOTAL}, dissolve=True)
        app_full = self._get_camada('APP_FULL', cfg.PATH_APP, cfg.ENCODING_CAR)
        app_total_gdf = self._get_camada('APP_TOTAL', cfg.PATH_APP, cfg.ENCODING_CAR, {cfg.COLUNA_TEMA: cfg.TEMA_APP_TOTAL}, dissolve=True)
        app_types_gdf = app_full[~app_full[cfg.COLUNA_TEMA].isin([cfg.TEMA_APP_TOTAL, "Curso d'água natural de 10 a 50 metros"])]
        rl_gdf = self._get_camada('RL', cfg.PATH_RL, cfg.ENCODING_CAR, {cfg.COLUNA_TEMA: cfg.TEMA_RL_TOTAL}, dissolve=True)
        cobertura_gdf = self._get_camada('COBERTURA', cfg.PATH_COBERTURA, cfg.ENCODING_CAR)
        servidao_gdf = self._get_camada('SERVIDAO', cfg.PATH_SERVIDAO, cfg.ENCODING_CAR, dissolve=True)
        
        veg_nativa_gdf = cobertura_gdf[cobertura_gdf[cfg.COLUNA_TEMA] == cfg.TEMA_VEG_NATIVA]
        area_consolidada_gdf = cobertura_gdf[cobertura_gdf[cfg.COLUNA_TEMA] == cfg.TEMA_AREA_CONSOLIDADA]

        area_total_ha = self._get_area_ha(imovel_gdf)
        area_app_ha = self._get_area_ha(app_total_gdf)
        area_rl_ha = self._get_area_ha(rl_gdf)
        area_veg_nativa_ha = self._get_area_ha(veg_nativa_gdf)
        area_consolidada_ha = self._get_area_ha(area_consolidada_gdf)
        area_servidao_ha = self._get_area_ha(servidao_gdf)
        ativos_ha = area_app_ha + area_rl_ha
        percent_ativos = (ativos_ha / area_total_ha) * 100 if area_total_ha > 0 else 0

        percent_rl_exigido, bioma, localizacao = self._determinar_percentual_rl(imovel_gdf)
        area_rl_exigida_ha = area_total_ha * percent_rl_exigido
        balanco_rl_ha = area_rl_ha - area_rl_exigida_ha
        
        df_quadro_areas = pd.DataFrame([
            {"Componente": "Área Total do Imóvel (ATI)", "Área (ha)": area_total_ha, "% da ATI": 100.0},
            {"Componente": "  - Remanescente de Vegetação Nativa", "Área (ha)": area_veg_nativa_ha, "% da ATI": (area_veg_nativa_ha/area_total_ha)*100 if area_total_ha > 0 else 0},
            {"Componente": "  - Área de Uso Consolidado", "Área (ha)": area_consolidada_ha, "% da ATI": (area_consolidada_ha/area_total_ha)*100 if area_total_ha > 0 else 0},
            {"Componente": "Área de Preservação Permanente (APP)", "Área (ha)": area_app_ha, "% da ATI": (area_app_ha/area_total_ha)*100 if area_total_ha > 0 else 0},
            {"Componente": "Área de Reserva Legal (RL)", "Área (ha)": area_rl_ha, "% da ATI": (area_rl_ha/area_total_ha)*100 if area_total_ha > 0 else 0},
            {"Componente": "Total de Ativos Ambientais (APP+RL)", "Área (ha)": ativos_ha, "% da ATI": percent_ativos},
            {"Componente": "Área de Servidão Administrativa", "Área (ha)": area_servidao_ha, "% da ATI": (area_servidao_ha/area_total_ha)*100 if area_total_ha > 0 else 0},
        ])
        
        df_app_detalhe = app_types_gdf.copy()
        crs_metrico = self._estimate_utm_crs(df_app_detalhe)
        if not df_app_detalhe.empty:
            df_app_detalhe['area_ha'] = df_app_detalhe.to_crs(crs_metrico).geometry.area / 10000
        df_app_stats = df_app_detalhe.groupby(cfg.COLUNA_TEMA)[['area_ha']].sum().sort_values('area_ha', ascending=False).reset_index()

        df_balanco_rl = pd.DataFrame([
            {"Balanço de Reserva Legal": f"Localização / Bioma", "Área (ha)": f"{localizacao} / {bioma}"},
            {"Balanço de Reserva Legal": f"RL Exigida ({percent_rl_exigido:.0%})", "Área (ha)": area_rl_exigida_ha},
            {"Balanço de Reserva Legal": "RL Declarada", "Área (ha)": area_rl_ha},
            {"Balanço de Reserva Legal": "Balanço (Déficit / Superávit)", "Área (ha)": balanco_rl_ha},
        ])

        nome_imovel = imovel_gdf[cfg.COLUNA_NOME_IMOVEL].iloc[0]
        display(HTML(f"<h2>Caracterização Ambiental do Imóvel: {nome_imovel}</h2>"))
        print("\n--- QUADRO GERAL DE ÁREAS ---")
        display(df_quadro_areas.style.format({'Área (ha)': '{:,.4f}', '% da ATI': '{:,.2f}%'}).hide(axis="index"))
        if not df_app_stats.empty:
            print("\n--- DETALHAMENTO DAS ÁREAS DE PRESERVAÇÃO PERMANENTE (APP) ---")
            display(df_app_stats.style.format({'area_ha': '{:,.4f}'}).hide(axis="index"))
        print("\n--- BALANÇO DE RESERVA LEGAL (RL) ---")
        
        
        df_balanco_display = df_balanco_rl.copy()
        for i in range(1, len(df_balanco_display)):
            df_balanco_display.iloc[i, 1] = f"{df_balanco_display.iloc[i, 1]:,.4f}"
        display(df_balanco_display.style.hide(axis="index"))

        data_layers = {
            'IMOVEL': imovel_gdf, 'VEGETACAO_NATIVA': veg_nativa_gdf,
            'AREA_CONSOLIDADA': area_consolidada_gdf, 'APP': app_total_gdf,
            'RL': rl_gdf, 'SERVIDAO': servidao_gdf
        }
        
        mapa_para_display = self._gerar_mapa(data_layers, nome_imovel)
        display(mapa_para_display)
        
        mapa_para_pdf = self._gerar_mapa(data_layers, nome_imovel)
        stats_dict = {'quadro_geral': df_quadro_areas, 'detalhe_app': df_app_stats, 'balanco_rl': df_balanco_rl}
        self._exportar_relatorio_pdf(mapa_para_pdf, stats_dict, nome_imovel)
        plt.close('all')

if __name__ == "__main__":
    config = CARConfig()
    pipeline = CARCharacterizationPipeline(config)
    pipeline.run()