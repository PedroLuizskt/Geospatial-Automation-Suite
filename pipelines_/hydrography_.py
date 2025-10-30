## Análise de Rede Hidrográfica - Autor = Pedro Luiz

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
import unicodedata
import re
import numpy as np
from dataclasses import dataclass
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_scalebar.scalebar import ScaleBar
from IPython.display import display, HTML

@dataclass
class HydrographyConfig:
    PATH_MUNICIPIOS: str = r"C:\Users\pedro\Downloads\python_gis\script_hidrografia\BR_Municipios_2024\BR_Municipios_2024.shp"
    PATH_RIOS: str = r"C:\Users\pedro\Downloads\python_gis\script_hidrografia\GEOFT_BHO_REF_RIO\GEOFT_BHO_REF_RIO.shp"
    PATH_PONTOS_DRENAGEM: str = r"C:\Users\pedro\Downloads\python_gis\script_hidrografia\GEOFT_BHO_REF_PONTO_DRENAGEM\GEOFT_BHO_REF_PONTO_DRENAGEM.shp"
    PATH_IMOVEL_ALVO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\MT-5106422-2960697C669941729C7EF7C2930CBA5A\Area_do_Imovel\Area_do_Imovel.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\estagio_ie\resultados_"
    COLUNA_NOME_MUNICIPIO: str = 'NM_MUN'
    COLUNA_NOME_IMOVEL: str = 'recibo'
    COLUNA_NOME_RIO: str = 'NORIOCOMP'
    ENCODING_MUNICIPIOS: str = 'utf-8'
    ENCODING_HIDROGRAFIA: str = 'cp1252'
    ENCODING_IMOVEL: str = 'utf-8'
    DPI_SAIDA: int = 300
    TIPO_DE_AREA: str = 'municipio'
    NOME_MUNICIPIO_ALVO: str = 'Curitiba'
    NUMERO_RIOS_DESTAQUE: int = 10
    CRS_GEOGRAFICO: str = 'EPSG:4674'

class HydrographyPipeline:
    def __init__(self, config: HydrographyConfig):
        self.config = config
        warnings.filterwarnings('ignore', 'The Shapely GEOS version used')

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
            if area_filtrada_gdf.empty:
                raise ValueError(f"Município '{cfg.NOME_MUNICIPIO_ALVO}' não encontrado.")
            area_gdf = area_filtrada_gdf.dissolve().reset_index()
            area_gdf[cfg.COLUNA_NOME_MUNICIPIO] = area_filtrada_gdf[cfg.COLUNA_NOME_MUNICIPIO].iloc[0]
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

    def _analisar_drenagem(self, drenagem_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame):
        cfg = self.config
        recortado_gdf = gpd.clip(drenagem_gdf, aoi_gdf).dropna(subset=[cfg.COLUNA_NOME_RIO])
        if recortado_gdf.empty:
            return recortado_gdf, []
            
        crs_metrico_local = self._estimate_utm_crs(recortado_gdf)
        metric_gdf = recortado_gdf.to_crs(crs_metrico_local)
        metric_gdf['comprimento_m'] = metric_gdf.geometry.length
        
        rios_principais = metric_gdf.groupby(cfg.COLUNA_NOME_RIO)['comprimento_m'].sum().nlargest(cfg.NUMERO_RIOS_DESTAQUE).index.tolist()
        return recortado_gdf, rios_principais

    def _analisar_nascentes(self, nascentes_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame):
        nascentes_filtradas_gdf = nascentes_gdf[nascentes_gdf['DSPONTO'] == 'Início do Curso D´água'].copy()
        recortado_gdf = gpd.clip(nascentes_filtradas_gdf, aoi_gdf)
        return recortado_gdf

    def _calcular_estatisticas(self, drenagem_gdf: gpd.GeoDataFrame, nascentes_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        crs_metrico_local = self._estimate_utm_crs(aoi_gdf)
        
        area_km2 = aoi_gdf.to_crs(crs_metrico_local).geometry.area.iloc[0] / 1_000_000
        num_nascentes = len(nascentes_gdf) if not nascentes_gdf.empty else 0
        extensao_km = drenagem_gdf.to_crs(crs_metrico_local).geometry.length.sum() / 1000 if not drenagem_gdf.empty else 0
        
        densidade_nascentes = num_nascentes / area_km2 if area_km2 > 0 else 0
        densidade_drenagem = extensao_km / area_km2 if area_km2 > 0 else 0
        
        return pd.DataFrame({
            'Métrica': ['Área Total (km²)', 'Nº de Nascentes', 'Densidade (nascentes/km²)', "Extensão dos Rios (km)", 'Densidade de Drenagem (km/km²)'],
            'Valor': [area_km2, num_nascentes, densidade_nascentes, extensao_km, densidade_drenagem]
        })

    def _plotar_mapa(self, drenagem_gdf, rios_principais, nascentes_gdf, aoi_gdf, nome_area) -> plt.Figure:
        cfg = self.config
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
        ax.set_facecolor('#f0f0f0')

        utm_crs = aoi_gdf.estimate_utm_crs()
        aoi_plot = aoi_gdf.to_crs(utm_crs)
        drenagem_plot = drenagem_gdf.to_crs(utm_crs)
        nascentes_plot = nascentes_gdf.to_crs(utm_crs)
        
        aoi_plot.plot(ax=ax, facecolor='#E5E5E5', edgecolor='black', linewidth=1.5, zorder=1)
        
        if not drenagem_plot.empty:
            drenagem_plot[~drenagem_plot[cfg.COLUNA_NOME_RIO].isin(rios_principais)].plot(ax=ax, color='cornflowerblue', linewidth=1.0, zorder=2)
            mapa_cores = {rio: plt.cm.viridis(i / len(rios_principais)) for i, rio in enumerate(rios_principais)} if rios_principais else {}
            for rio, cor in mapa_cores.items():
                drenagem_plot[drenagem_plot[cfg.COLUNA_NOME_RIO] == rio].plot(ax=ax, color=cor, linewidth=1.8, label=rio, zorder=3)
        
        if not nascentes_plot.empty:
            nascentes_plot.plot(ax=ax, marker='o', color='blue', markersize=25, edgecolor='white', zorder=4)
        
        ax.set_title(f'Rede Hidrográfica - {nome_area}', fontsize=18, fontweight='bold')
        ax.set_xlabel('Coordenada Leste (m)'); ax.set_ylabel('Coordenada Norte (m)')
        ax.tick_params(axis='x', rotation=45)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'.replace(',', '.')))
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y):,}'.replace(',', '.')))
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.add_artist(ScaleBar(1, 'm', location='lower right', box_alpha=0.8, pad=0.5))
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction')
        
        legenda_limite = 'Limite Municipal' if cfg.TIPO_DE_AREA == 'municipio' else 'Limite do Imóvel'
        handles = [
            Line2D([0], [0], color='black', lw=1.5, label=legenda_limite),
            Line2D([0], [0], marker='o', color='blue', markeredgecolor='white', markersize=8, ls='None', label='Nascentes'),
            Line2D([0], [0], color='cornflowerblue', lw=1.5, label='Outros Cursos d\'Água')
        ]
        if rios_principais:
            handles.append(Line2D([0], [0], color='white', lw=0, label='--- Rios Principais ---'))
            for rio, cor in mapa_cores.items():
                handles.append(Line2D([0], [0], color=cor, lw=1.8, label=rio))
        
        legenda = ax.legend(handles=handles, title='LEGENDA', loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.setp(legenda.get_title(), fontsize='14', fontweight='bold')
        
        fig.text(0.77, 0.2, f"Fonte: BHO (2023)\nAutor: Pedro Luiz\nDatum: SIRGAS 2000", ha='left', va='bottom', fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1))
        
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        return fig

    def _exportar_relatorio_pdf(self, figura_mapa: plt.Figure, df_estatisticas: pd.DataFrame, nome_area: str, nome_arquivo_saida: str):
        cfg = self.config
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo_saida)
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        
        with PdfPages(caminho_completo) as pdf:
            pdf.savefig(figura_mapa, bbox_inches='tight', dpi=cfg.DPI_SAIDA)
            plt.close(figura_mapa)
            
            fig_tabela, ax_tabela = plt.subplots(figsize=(8.27, 11.69))
            ax_tabela.axis('off')
            ax_tabela.set_title(f"Análise Quantitativa de Hidrografia\n{nome_area}", fontsize=16, fontweight='bold', pad=20)
            
            df_display = df_estatisticas.copy()
            df_display['Valor'] = [f"{v:_.2f}".replace('.', ',').replace('_', '.') if i != 1 else int(v) for i, v in enumerate(df_display['Valor'])]
            
            tabela = ax_tabela.table(cellText=df_display.values, colLabels=['Métrica', 'Valor'], cellLoc='left', loc='upper center', colWidths=[0.5, 0.2])
            tabela.auto_set_font_size(False); tabela.set_fontsize(12); tabela.scale(1.2, 2.5)
            
            pdf.savefig(fig_tabela, bbox_inches='tight', dpi=cfg.DPI_SAIDA)
            plt.close(fig_tabela)
            
        return caminho_completo

    def run(self):
        try:
            print("="*80); print("       INICIANDO PIPELINE DE ANÁLISE HIDROGRÁFICA "); print("="*80)
            
            aoi_gdf = self._obter_area_de_interesse()

            drenagem_base_gdf = self._carregar_dados_vetoriais(self.config.PATH_RIOS, self.config.CRS_GEOGRAFICO, bbox=tuple(aoi_gdf.total_bounds), encoding=self.config.ENCODING_HIDROGRAFIA)
            nascentes_base_gdf = self._carregar_dados_vetoriais(self.config.PATH_PONTOS_DRENAGEM, self.config.CRS_GEOGRAFICO, bbox=tuple(aoi_gdf.total_bounds), encoding=self.config.ENCODING_HIDROGRAFIA)

            if self.config.TIPO_DE_AREA == 'municipio':
                nome_area = self.config.NOME_MUNICIPIO_ALVO
            else:
                nome_area = aoi_gdf[self.config.COLUNA_NOME_IMOVEL].iloc[0] if self.config.COLUNA_NOME_IMOVEL in aoi_gdf.columns else "Imovel Rural Sem Nome"
            
            nome_base_sanitizado = self._sanitizar_nome_arquivo(nome_area)
            nome_arquivo_final = f"Relatorio_Hidrografia_{nome_base_sanitizado}.pdf"
            
            drenagem_gdf, rios_principais = self._analisar_drenagem(drenagem_base_gdf, aoi_gdf)
            nascentes_gdf = self._analisar_nascentes(nascentes_base_gdf, aoi_gdf)
            estatisticas_df = self._calcular_estatisticas(drenagem_gdf, nascentes_gdf, aoi_gdf)
            
            display(HTML(f"<h2>Análise Hidrográfica para: {nome_area}</h2>"))
            figura_para_display = self._plotar_mapa(drenagem_gdf, rios_principais, nascentes_gdf, aoi_gdf, nome_area)
            display(figura_para_display)
            plt.close(figura_para_display)
            
            print("\n" + "="*70); print(f"   ANÁLISE QUANTITATIVA DE HIDROGRAFIA PARA: {nome_area.upper()}"); print("="*70)
            stats_console = estatisticas_df.copy()
            stats_console['Valor'] = [f"{v:_.2f}".replace('.', ',').replace('_', '.') if i != 1 else int(v) for i, v in enumerate(stats_console['Valor'])]
            print(stats_console.to_string(index=False)); print("="*70)
            
            figura_para_pdf = self._plotar_mapa(drenagem_gdf, rios_principais, nascentes_gdf, aoi_gdf, nome_area)
            caminho_salvo = self._exportar_relatorio_pdf(figura_para_pdf, estatisticas_df, nome_area, nome_arquivo_final)
            print(f"\nAnálise hidrográfica para '{nome_area}' concluída! Relatório salvo em:\n{caminho_salvo}")
        
        except (ValueError, FileNotFoundError) as e:
            print(f"\n[ERRO CONTROLADO] {e}")
        except Exception as e:
            print(f"\n[ERRO CRÍTICO INESPERADO] {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__":
    config = HydrographyConfig(
        TIPO_DE_AREA='imovel',
        NOME_MUNICIPIO_ALVO='Maringá'
    )
    pipeline = HydrographyPipeline(config)
    pipeline.run()