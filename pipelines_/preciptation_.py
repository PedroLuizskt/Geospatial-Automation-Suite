## Análise de Precipitação Média Anual - Autor = Pedro Luiz

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import contextily as ctx
import unicodedata
import re
import numpy as np
import pandas as pd
import os
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
from dataclasses import dataclass, field
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FuncFormatter
from IPython.display import display, HTML


@dataclass
class PrecipitationConfig:
    """Encapsula todas as configurações do pipeline de análise de precipitação."""
    PASTA_RASTERS_REGIONAIS: str = r"C:\Users\pedro\Downloads\python_gis\script_precipitacao\raster_precipitacao"
    PATH_ESTADOS_BRASIL: str = r"C:\Users\pedro\Downloads\estagio_ie\Script_UsoeOC\BR_UF_2024\BR_UF_2024.shp"
    PATH_MUNICIPIOS_BRASIL: str = r"C:\Users\pedro\Downloads\python_gis\script_precipitacao\BR_Municipios_2024\BR_Municipios_2024.shp"
    PATH_IMOVEL_ALVO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\MT-5106422-2960697C669941729C7EF7C2930CBA5A\Area_do_Imovel\Area_do_Imovel.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\estagio_ie\resultados_"
    COLUNA_SIGLA_ESTADO: str = 'SIGLA_UF'
    COLUNA_NOME_MUNICIPIO: str = 'NM_MUN'
    COLUNA_NOME_IMOVEL: str = 'recibo'
    ENCODING_VETORES: str = 'utf-8'
    DPI_SAIDA: int = 300
    TIPO_DE_AREA: str = 'imovel' 
    NOME_MUNICIPIO_ALVO: str = 'Açailândia'
    SIGLA_ESTADO_ALVO: str = 'AM'
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    CRS_WEB_MERCATOR: str = 'EPSG:3857'
    
    # --- Estilo e Simbologia ---
    CORES_PRECIPITACAO: list = field(default_factory=lambda: ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695'])
    CLASSES_PRECIPITACAO: list = field(default_factory=lambda: [0, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 3000])
    
    # --- Mapeamentos Lógicos ---
    REGIOES_ESTADOS: dict = field(default_factory=dict)
    SIGLA_PARA_NOME: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.REGIOES_ESTADOS = {'Norte': ['Acre', 'Amapa', 'Amazonas', 'Para', 'Rondonia', 'Roraima', 'Tocantins'], 'Nordeste': ['Alagoas', 'Bahia', 'Ceara', 'Maranhao', 'Paraiba', 'Pernambuco', 'Piaui', 'Rio Grande Do Norte', 'Sergipe'], 'Centro-Oeste': ['Distrito Federal', 'Goias', 'Mato Grosso', 'Mato Grosso Do Sul'], 'Sudeste': ['Espirito Santo', 'Minas Gerais', 'Rio De Janeiro', 'Sao Paulo'], 'Sul': ['Parana', 'Rio Grande Do Sul', 'Santa Catarina']}
        self.SIGLA_PARA_NOME = {'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapa', 'AM': 'Amazonas', 'BA': 'Bahia', 'CE': 'Ceara', 'DF': 'Distrito Federal', 'ES': 'Espirito Santo', 'GO': 'Goias', 'MA': 'Maranhao', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso Do Sul', 'MG': 'Minas Gerais', 'PA': 'Para', 'PB': 'Paraiba', 'PR': 'Parana', 'PE': 'Pernambuco', 'PI': 'Piaui', 'RJ': 'Rio De Janeiro', 'RN': 'Rio Grande Do Norte', 'RS': 'Rio Grande Do Sul', 'RO': 'Rondonia', 'RR': 'Roraima', 'SC': 'Santa Catarina', 'SP': 'Sao Paulo', 'SE': 'Sergipe', 'TO': 'Tocantins'}


class PrecipitationPipeline:
    def __init__(self, config: PrecipitationConfig):
        self.config = config

    def _sanitizar_nome_arquivo(self, nome: str) -> str:
        nome = unicodedata.normalize('NFKD', nome).encode('ascii', 'ignore').decode('utf-8')
        nome = re.sub(r'[^\w\s-]', '', nome).strip().lower()
        nome = re.sub(r'[-\s]+', '-', nome)
        return nome[:100]

    def _carregar_dados_vetoriais(self, path: str, crs_alvo: str, **kwargs) -> gpd.GeoDataFrame:
        print(f"Carregando dados de: {os.path.basename(path)}...")
        gdf = gpd.read_file(path, **kwargs)
        if gdf.crs is None:
            print(f"  AVISO: CRS não definido. Assumindo {self.config.CRS_GEOGRAFICO}.")
            gdf.set_crs(self.config.CRS_GEOGRAFICO, inplace=True)
        return gdf.to_crs(crs_alvo)

    def _obter_area_de_interesse(self) -> gpd.GeoDataFrame:
        cfg = self.config
        print(f"\nObtendo Área de Interesse (AOI): '{cfg.TIPO_DE_AREA}'...")
        
        if cfg.TIPO_DE_AREA == 'estado':
            gdf = self._carregar_dados_vetoriais(cfg.PATH_ESTADOS_BRASIL, cfg.CRS_GEOGRAFICO, encoding=cfg.ENCODING_VETORES)
            area_filtrada_gdf = gdf[gdf[cfg.COLUNA_SIGLA_ESTADO] == cfg.SIGLA_ESTADO_ALVO]
            if area_filtrada_gdf.empty: raise ValueError(f"Estado '{cfg.SIGLA_ESTADO_ALVO}' não encontrado.")
            area_gdf = area_filtrada_gdf.dissolve().reset_index()

        elif cfg.TIPO_DE_AREA == 'municipio':
            gdf = self._carregar_dados_vetoriais(cfg.PATH_MUNICIPIOS_BRASIL, cfg.CRS_GEOGRAFICO, encoding=cfg.ENCODING_VETORES)
            area_filtrada_gdf = gdf[gdf[cfg.COLUNA_NOME_MUNICIPIO] == cfg.NOME_MUNICIPIO_ALVO]
            if area_filtrada_gdf.empty: raise ValueError(f"Município '{cfg.NOME_MUNICIPIO_ALVO}' não encontrado.")
            area_gdf = area_filtrada_gdf.dissolve().reset_index()

        elif cfg.TIPO_DE_AREA == 'imovel':
            area_gdf = self._carregar_dados_vetoriais(cfg.PATH_IMOVEL_ALVO, cfg.CRS_GEOGRAFICO, encoding=cfg.ENCODING_VETORES)
        else:
            raise ValueError("Tipo de área inválido.")
            
        if area_gdf.empty: raise ValueError("A Área de Interesse (AOI) está vazia.")
        return area_gdf

    def _encontrar_raster_regional(self, aoi_gdf: gpd.GeoDataFrame, estados_gdf: gpd.GeoDataFrame) -> str:
        cfg = self.config
        print("Identificando o raster regional correto...")
        
        intersecao = gpd.sjoin(aoi_gdf.head(1), estados_gdf, how="inner", predicate="intersects")
        if intersecao.empty:
            raise RuntimeError("Não foi possível determinar a qual estado a AOI pertence.")
        coluna_sufixo = cfg.COLUNA_SIGLA_ESTADO + '_right'
        coluna_alvo = coluna_sufixo if coluna_sufixo in intersecao.columns else cfg.COLUNA_SIGLA_ESTADO
        sigla_estado = intersecao[coluna_alvo].iloc[0]
        
        nome_estado_da_aoi = cfg.SIGLA_PARA_NOME.get(sigla_estado)
        
        for regiao, estados in cfg.REGIOES_ESTADOS.items():
            if nome_estado_da_aoi in estados:
                nome_arquivo = f"precipitacao_media_anual_Regiao_{regiao}_2009-2023.tif"
                caminho_raster = os.path.join(cfg.PASTA_RASTERS_REGIONAIS, nome_arquivo)
                if os.path.exists(caminho_raster):
                    print(f"  Área pertence à Região {regiao}. Usando: {nome_arquivo}")
                    return caminho_raster
        raise FileNotFoundError(f"Raster para a região do estado '{nome_estado_da_aoi}' não encontrado.")
    
    def _processar_precipitacao(self, path_raster: str, aoi_gdf: gpd.GeoDataFrame):
        print("Processando dados de precipitação...")
        with rasterio.open(path_raster) as src:
            aoi_reprojetado = aoi_gdf.to_crs(src.crs)
            raster_array, transform = mask(src, aoi_reprojetado.geometry, crop=True, all_touched=True, nodata=np.nan)
            raster_array = raster_array[0]
            
            dados_validos = raster_array[~np.isnan(raster_array)]
            if dados_validos.size == 0: return None, None
            
            estatisticas = {"Mínima (mm/ano)": np.min(dados_validos), "Máxima (mm/ano)": np.max(dados_validos), "Média (mm/ano)": np.mean(dados_validos)}
            estatisticas_df = pd.DataFrame.from_dict(estatisticas, orient='index', columns=['Valor'])
            
            print("Vetorizando dados para o mapa...")
            resultados = ({'properties': {'valor': v}, 'geometry': shape(s)} for i, (s, v) in enumerate(shapes(raster_array.astype('float32'), mask=~np.isnan(raster_array), transform=transform)) if v > 0)
            precipitacao_gdf = gpd.GeoDataFrame.from_features(list(resultados), crs=src.crs)
            
            return gpd.clip(precipitacao_gdf, aoi_reprojetado), estatisticas_df
    
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
        
        ax.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    def _plotar_mapa(self, vetor_precipitacao: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame, nome_area: str) -> plt.Figure:
        cfg = self.config
        print("Gerando o mapa com formatação profissional...")
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
        ax.set_facecolor('#f0f0f0')
        
        precip_plot = vetor_precipitacao.to_crs(cfg.CRS_WEB_MERCATOR)
        aoi_plot = aoi_gdf.to_crs(cfg.CRS_WEB_MERCATOR)
        
        cmap = ListedColormap(cfg.CORES_PRECIPITACAO)
        norm = BoundaryNorm(cfg.CLASSES_PRECIPITACAO, cmap.N)
        precip_plot.plot(column='valor', ax=ax, cmap=cmap, norm=norm, edgecolor='gray', linewidth=0.1, alpha=0.7)
        aoi_plot.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, linestyle='--')
        
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto')
        self._add_grid_inteligente(ax, aoi_gdf)
        
        ax.set_title(f'Precipitação Média Anual (2009-2023) - {nome_area}', fontsize=18, fontweight='bold')
        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords='axes fraction')
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
        cbar.set_label('Precipitação Média Anual (mm)', size=12)
        
        ax.add_artist(ScaleBar(1, 'm', location='lower right', box_alpha=0.8, pad=0.5))
        fig.text(0.80, 0.2, f"Fonte: WorldClim (2024)\nAutor: Pedro Luiz\nDatum: SIRGAS 2000", ha='left', va='bottom', fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1))
        plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
        return fig

    def _exportar_relatorio_pdf(self, figura_mapa: plt.Figure, df_estatisticas: pd.DataFrame, nome_area: str, nome_arquivo_saida: str):
        cfg = self.config
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo_saida)
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        print(f"\nExportando relatório PDF para: {caminho_completo}...")
        
        with PdfPages(caminho_completo) as pdf:
            pdf.savefig(figura_mapa, bbox_inches='tight', dpi=cfg.DPI_SAIDA)
            plt.close(figura_mapa)
            
            fig_tabela, ax_tabela = plt.subplots(figsize=(8.27, 11.69))
            ax_tabela.axis('off')
            ax_tabela.set_title(f"Análise Estatística de Precipitação\n{nome_area}", fontsize=16, fontweight='bold', pad=20)
            
            df_display = df_estatisticas.copy()
            formatted_values = [f"{v:_.2f}".replace('.', ',').replace('_', '.') for v in df_display['Valor']]
            df_display['Valor'] = formatted_values
            
            tabela = ax_tabela.table(cellText=df_display.values, rowLabels=df_display.index, colLabels=df_display.columns, cellLoc='center', loc='upper center', rowLoc='left', colWidths=[0.3])
            tabela.auto_set_font_size(False); tabela.set_fontsize(14); tabela.scale(1.5, 2.5)
            
            pdf.savefig(fig_tabela, bbox_inches='tight', dpi=cfg.DPI_SAIDA)
            plt.close(fig_tabela)
            
        print("Relatório exportado com sucesso!")
        return caminho_completo

    def run(self):
        try:
            print("="*80); print("       INICIANDO PIPELINE DE ANÁLISE DE PRECIPITAÇÃO "); print("="*80)
            
            aoi_gdf = self._obter_area_de_interesse()
            
            if self.config.TIPO_DE_AREA == 'municipio':
                nome_area = self.config.NOME_MUNICIPIO_ALVO
            elif self.config.TIPO_DE_AREA == 'estado':
                sigla = self.config.SIGLA_ESTADO_ALVO
                nome_area = f"Estado de {self.config.SIGLA_PARA_NOME.get(sigla, sigla)}"
            else:
                nome_area = aoi_gdf[self.config.COLUNA_NOME_IMOVEL].iloc[0] if self.config.COLUNA_NOME_IMOVEL in aoi_gdf.columns else "Imovel Rural Sem Nome"
            
            nome_base_sanitizado = self._sanitizar_nome_arquivo(nome_area)
            nome_arquivo_final = f"Relatorio_Precipitacao_{nome_base_sanitizado}.pdf"
            
            estados_gdf = self._carregar_dados_vetoriais(self.config.PATH_ESTADOS_BRASIL, self.config.CRS_GEOGRAFICO, encoding=self.config.ENCODING_VETORES)
            
            path_raster = self._encontrar_raster_regional(aoi_gdf, estados_gdf)
            precipitacao_gdf, estatisticas_df = self._processar_precipitacao(path_raster, aoi_gdf)
            
            if precipitacao_gdf is None or precipitacao_gdf.empty:
                print("\nAVISO: Nenhum dado de precipitação encontrado para a área de interesse.")
            else:
                display(HTML(f"<h2>Análise de Precipitação para: {nome_area}</h2>"))
                figura_para_display = self._plotar_mapa(precipitacao_gdf, aoi_gdf, nome_area)
                display(figura_para_display)
                plt.close(figura_para_display)
                
                print("\n" + "="*60); print(f"          ANÁLISE ESTATÍSTICA PARA: {nome_area.upper()}"); print("="*60)
                print(estatisticas_df.to_string()); print("="*60)
                
                figura_para_pdf = self._plotar_mapa(precipitacao_gdf, aoi_gdf, nome_area)
                caminho_salvo = self._exportar_relatorio_pdf(figura_para_pdf, estatisticas_df, nome_area, nome_arquivo_final)
                print(f"\nAnálise para '{nome_area}' concluída! Relatório salvo em:\n{caminho_salvo}")
        
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"\n[ERRO CONTROLADO] {e}")
        except Exception as e:
            print(f"\n[ERRO CRÍTICO INESPERADO] {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__":
    config = PrecipitationConfig(
        TIPO_DE_AREA='imovel',
    )
    pipeline = PrecipitationPipeline(config)
    pipeline.run()