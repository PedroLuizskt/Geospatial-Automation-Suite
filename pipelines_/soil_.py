## Análise de Solos - Autor = Pedro Luiz

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings
import contextily as ctx
import unicodedata
import re
from dataclasses import dataclass
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from IPython.display import display, HTML

@dataclass
class Config:
    PATH_SOLOS_BRASIL: str = r"C:\Users\pedro\Downloads\python_gis\script_solos\Solos_5000mil\Solos_5000.shp"
    PATH_MUNICIPIOS_BRASIL: str = r"C:\Users\pedro\Downloads\python_gis\script_solos\BR_Municipios_2024\BR_Municipios_2024.shp"
    PATH_IMOVEL_ALVO: str = r"C:\Users\pedro\Downloads\python_gis\script_estágio_VA\script_imoveis\MG-3121605-0C5FDE61120E4BB7B3CD1728A97638E2\Area_do_Imovel\Area_do_Imovel.shp"
    PATH_EXPORTACAO: str = r"C:\Users\pedro\Downloads\estagio_ie\resultados_"
    COLUNA_SOLOS: str = "DSC_COMPO1"
    COLUNA_MUNICIPIO: str = 'NM_MUN'
    COLUNA_NOME_IMOVEL: str = 'recibo'
    DPI_SAIDA: int = 300
    CRS_GEOGRAFICO: str = 'EPSG:4674'
    CRS_WEB_MERCATOR: str = 'EPSG:3857'
    TIPO_DE_AREA: str = 'imovel'
    NOME_MUNICIPIO_ALVO: str = 'Água Boa'
    REGRAS_DE_CORES: dict = None
    COR_PADRAO: str = '#E1E1E1'

    def __post_init__(self):
        if self.REGRAS_DE_CORES is None:
            self.REGRAS_DE_CORES = {
                'vermelho': '#9E452C', 'amarelo': '#E09752', 'cinzento': '#BDB76B',
                'glei': '#A9A9A9', 'plinto': '#B0171F', 'húmico': '#654321',
                'melânico': '#3D2B1F', 'êbanico': '#292421', 'latossolo': '#BC4935',
                'argissolo': '#D2945C', 'cambissolo': '#8B5A2B', 'neossolo': '#CDAF95',
                'afloramentos': '#708090'
            }

class SoilAnalysisPipeline:
    def __init__(self, config: Config):
        self.config = config
        self._setup_environment()

    def _setup_environment(self):
        warnings.filterwarnings('ignore', 'The Shapely GEOS version used')
        warnings.filterwarnings('ignore', 'DataFrame is highly fragmented')
        plt.style.use('seaborn-v0_8-whitegrid')
        pd.set_option('display.max_rows', 20)

    def _carregar_dados_vetoriais(self, path: str, crs_alvo: str, **kwargs) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path, **kwargs)
        if gdf.crs is None:
            gdf.set_crs(self.config.CRS_GEOGRAFICO, inplace=True)
        return gdf.to_crs(crs_alvo)

    def _sanitizar_nome_arquivo(self, nome: str) -> str:
        nome = unicodedata.normalize('NFKD', nome).encode('ascii', 'ignore').decode('utf-8')
        nome = re.sub(r'[^\w\s-]', '', nome).strip().lower()
        nome = re.sub(r'[-\s]+', '-', nome)
        return nome[:100]

    def _obter_area_de_interesse(self) -> gpd.GeoDataFrame:
        cfg = self.config
        if cfg.TIPO_DE_AREA == 'municipio':
            municipios_gdf = self._carregar_dados_vetoriais(cfg.PATH_MUNICIPIOS_BRASIL, cfg.CRS_GEOGRAFICO)
            area_gdf_filtrada = municipios_gdf[municipios_gdf[cfg.COLUNA_MUNICIPIO].str.contains(cfg.NOME_MUNICIPIO_ALVO, case=False, na=False)]
            if area_gdf_filtrada.empty:
                raise ValueError(f"Município '{cfg.NOME_MUNICIPIO_ALVO}' não encontrado.")
            area_gdf = area_gdf_filtrada.dissolve().reset_index()
        elif cfg.TIPO_DE_AREA == 'imovel':
            area_gdf = self._carregar_dados_vetoriais(cfg.PATH_IMOVEL_ALVO, cfg.CRS_GEOGRAFICO)
        else:
            raise ValueError("Tipo de área inválido. Escolha 'municipio' ou 'imovel'.")
        if area_gdf.empty:
            raise ValueError("A Área de Interesse (AOI) não pôde ser carregada ou está vazia.")
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

    def _analisar_solos(self, solos_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame):
        cfg = self.config
        solos_recortado_gdf = gpd.clip(solos_gdf, aoi_gdf)
        if solos_recortado_gdf.empty:
            return None, None

        crs_metrico_local = self._estimate_utm_crs(solos_recortado_gdf)
        solos_metric_gdf = solos_recortado_gdf.to_crs(crs_metrico_local)
        solos_metric_gdf['area_ha'] = solos_metric_gdf.geometry.area / 10000
        
        estatisticas = (solos_metric_gdf.groupby(cfg.COLUNA_SOLOS)[['area_ha']]
                                      .sum()
                                      .sort_values(by='area_ha', ascending=False)
                                      .reset_index())
        return solos_recortado_gdf, estatisticas

    def _encontrar_cor_para_solo(self, descricao: str) -> str:
        for palavra_chave, cor_hex in self.config.REGRAS_DE_CORES.items():
            if palavra_chave in str(descricao).lower():
                return cor_hex
        return self.config.COR_PADRAO

    def _add_grid_inteligente(self, ax, gdf_geografico):
        cfg = self.config
        minx, miny, maxx, maxy = gdf_geografico.total_bounds
        x_interval, y_interval = (maxx - minx) / 5, (maxy - miny) / 5
        x_ticks, y_ticks = np.arange(minx, maxx, x_interval), np.arange(miny, maxy, y_interval)
        
        lon_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_ticks, np.full(len(x_ticks), miny)), crs=cfg.CRS_GEOGRAFICO)
        lat_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.full(len(y_ticks), minx), y_ticks), crs=cfg.CRS_GEOGRAFICO)
        
        lon_ticks_proj = lon_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.x
        lat_ticks_proj = lat_gdf.to_crs(cfg.CRS_WEB_MERCATOR).geometry.y
        
        ax.set_xticks(lon_ticks_proj)
        ax.set_yticks(lat_ticks_proj)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{x_ticks[pos]:.2f}°" if pos < len(x_ticks) else ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{y_ticks[pos]:.2f}°" if pos < len(y_ticks) else ""))
        
        ax.grid(True, linestyle='--', alpha=0.5, color='gray')
        ax.tick_params(axis='both', which='major', labelsize=10, labelrotation=45)

    def _gerar_figura_mapa(self, solos_gdf, aoi_gdf, nome_area):
        cfg = self.config
        solos_plot_gdf = solos_gdf.to_crs(cfg.CRS_WEB_MERCATOR)
        aoi_plot_gdf = aoi_gdf.to_crs(cfg.CRS_WEB_MERCATOR)
        
        fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27), facecolor='white')
        ax.set_facecolor('#f0f0f0')
        
        mapa_de_cores = {desc: self._encontrar_cor_para_solo(desc) for desc in solos_plot_gdf[cfg.COLUNA_SOLOS].unique()}
        
        solos_plot_gdf.plot(ax=ax, color=solos_plot_gdf[cfg.COLUNA_SOLOS].map(mapa_de_cores), edgecolor='black', linewidth=0.2)
        aoi_plot_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2.0, linestyle='--')
        
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto')
        self._add_grid_inteligente(ax, aoi_gdf)
        
        ax.add_artist(ScaleBar(1, loc='upper left', bbox_to_anchor=(0.02, 0.98), bbox_transform=ax.transAxes, frameon=True, box_alpha=0.7, pad=0.5, color='black', font_properties={'size': 10}))
        
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=cor, edgecolor='k', label=desc) for desc, cor in sorted(mapa_de_cores.items())]
        ax.legend(handles=legend_elements, title='Classes de Solo', loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=True, facecolor='white', edgecolor='darkgray')
        
        x_norte, y_norte, arrow_length = 0.98, 0.98, 0.08
        ax.annotate('N', xy=(x_norte, y_norte), xytext=(x_norte, y_norte - arrow_length), arrowprops=dict(facecolor='black', width=4, headwidth=12), ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
        
        texto_creditos = f"Fonte: IBGE (2007)\nProjeção: Web Mercator (EPSG:3857)\nDatum Geográfico: SIRGAS 2000\nAutor: Pedro Luiz"
        ax.text(1.03, 0, texto_creditos, transform=ax.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='left', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8, ec='darkgray', lw=1))
        
        ax.set_title(f'Mapa de Classes de Solo - {nome_area}', fontsize=18, fontweight='bold', pad=20)
        plt.subplots_adjust(left=0.1, right=0.75, top=0.92, bottom=0.1)
        return fig

    def _exportar_relatorio_pdf(self, figura_mapa, df_estatisticas, nome_area, nome_arquivo_saida):
        cfg = self.config
        caminho_completo = os.path.join(cfg.PATH_EXPORTACAO, nome_arquivo_saida)
        os.makedirs(cfg.PATH_EXPORTACAO, exist_ok=True)
        
        with PdfPages(caminho_completo) as pdf:
            pdf.savefig(figura_mapa, dpi=cfg.DPI_SAIDA)
            plt.close(figura_mapa)
            
            fig_tabela, ax_tabela = plt.subplots(figsize=(11.69, 8.27), facecolor='white')
            ax_tabela.axis('off')
            ax_tabela.set_title(f"Análise Quantitativa de Solos - {nome_area}", fontsize=18, fontweight='bold', pad=20)
            
            df_display = df_estatisticas.copy()
            df_display['Área (ha)'] = df_display['area_ha'].map('{:,.2f}'.format)
            df_display.rename(columns={cfg.COLUNA_SOLOS: 'Classe de Solo'}, inplace=True)
            
            tabela = ax_tabela.table(cellText=df_display[['Classe de Solo', 'Área (ha)']].values,
                                     colLabels=['Classe de Solo', 'Área (ha)'],
                                     cellLoc='left', 
                                     loc='upper center',
                                     colWidths=[0.7, 0.2])
            
            tabela.auto_set_font_size(False)
            tabela.set_fontsize(12)
            tabela.scale(1, 2)
            
            for (i, j), cell in tabela.get_celld().items():
                if i == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#40466e')
                cell.set_edgecolor('lightgray')
                cell.set_height(0.1)
                
            pdf.savefig(fig_tabela, bbox_inches='tight', dpi=cfg.DPI_SAIDA)
            plt.close(fig_tabela)
            
        return caminho_completo

    def run(self):
        try:
            print("="*80); print("       INICIANDO PIPELINE DE ANÁLISE DE SOLOS "); print("="*80)

            aoi_gdf = self._obter_area_de_interesse()
            
            solos_brasil_gdf = self._carregar_dados_vetoriais(
                self.config.PATH_SOLOS_BRASIL, 
                crs_alvo=self.config.CRS_GEOGRAFICO, 
                encoding='latin1'
            )
            
            solos_recortado_gdf, estatisticas_df = self._analisar_solos(solos_brasil_gdf, aoi_gdf)
            
            if solos_recortado_gdf is None or estatisticas_df.empty:
                print(f"\nAVISO: Nenhum polígono de solo encontrado na área de interesse. Relatório não gerado.")
            else:
                if self.config.TIPO_DE_AREA == 'municipio':
                    nome_area = self.config.NOME_MUNICIPIO_ALVO
                else: 
                    nome_area = aoi_gdf[self.config.COLUNA_NOME_IMOVEL].iloc[0]

                nome_base_sanitizado = self._sanitizar_nome_arquivo(nome_area)
                nome_arquivo_final = f"Relatorio_Solos_{nome_base_sanitizado}.pdf"

                display(HTML(f"<h3>Relatório para: '{nome_area}'</h3>"))
                figura_mapa_display = self._gerar_figura_mapa(solos_recortado_gdf, aoi_gdf, nome_area)
                display(figura_mapa_display)
                plt.close(figura_mapa_display)

                figura_mapa_pdf = self._gerar_figura_mapa(solos_recortado_gdf, aoi_gdf, nome_area)
                caminho_salvo = self._exportar_relatorio_pdf(figura_mapa_pdf, estatisticas_df, nome_area, nome_arquivo_final)
                
                print("\n" + "="*80); print("               RELATÓRIO DE ANÁLISE CONCLUÍDO"); print("="*80)
                print(f"\nO relatório final foi salvo em: {caminho_salvo}")
        
        except Exception as e:
            print(f"\n[ERRO CRÍTICO] Ocorreu um erro no pipeline: {e}")
            print("Verifique os caminhos dos arquivos e os parâmetros na classe de Configuração.")

if __name__ == "__main__":
    configuracao_analise_municipio = Config(
        TIPO_DE_AREA='imovel',
        NOME_MUNICIPIO_ALVO='Poços de Caldas'
    )
    
    pipeline_municipio = SoilAnalysisPipeline(configuracao_analise_municipio)
    pipeline_municipio.run()



#####

import os
import sys
from pathlib import Path
import gc
import warnings
import unicodedata
import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --- 1. SANITIZAÇÃO DE AMBIENTE (WINDOWS/GDAL) ---
def sanitize_environment():
    original_path = os.environ.get('PATH', '')
    clean_path_list = [p for p in original_path.split(';') if 'PostgreSQL' not in p and 'PostGIS' not in p]
    os.environ['PATH'] = ';'.join(clean_path_list)
    
    venv_base = Path(sys.prefix)
    possible_paths = [
        venv_base / "Lib" / "site-packages" / "pyproj" / "proj_dir" / "share",
        venv_base / "Lib" / "site-packages" / "rasterio" / "proj_data",
        venv_base / "share" / "proj",
        venv_base / "Library" / "share" / "proj"
    ]
    
    for p in possible_paths:
        if (p / "proj.db").exists():
            os.environ['PROJ_LIB'] = str(p)
            break

sanitize_environment()

# --- IMPORTS ---
import ee
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FuncFormatter
import rasterio
from rasterio.mask import mask

# --- CONFIGURAÇÃO VISUAL ---
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.titlesize": 16
})

# --- 2. CONFIGURAÇÃO DO PROJETO ---
@dataclass
class SoilConfig:
    # --- MODO ---
    ANALYSIS_MODE: str = 'MUNICIPIO'
    
    # --- CAMINHOS ---
    PATH_SHP_MUN: Path = Path(r"C:\Users\pedro\Downloads\python_gis\be_diagre\BR_Municipios_2024\BR_Municipios_2024.shp")
    PATH_SHP_IMOVEL: Path = Path(r"C:\Users\pedro\Downloads\python_gis\script_solos\MT-5101902-F96B956F1B80430580432988F1C9E039\Area_do_Imovel\Area_do_Imovel.shp")
    PATH_SOLOS_IBGE: Path = Path(r"C:\Users\pedro\Downloads\python_gis\script_solos\Solos_5000mil\Solos_5000.shp")
    PATH_OUTPUT_ROOT: Path = Path(r"C:\Users\pedro\Downloads\python_gis\gee_indices\results")
    
    # --- PARÂMETROS ---
    TARGET_NAME: str = 'Três Corações'
    TARGET_UF: str = 'MG'
    
    # --- GEE ---
    ASSET_ID: str = "OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02"
    SCALE_METERS: int = 250
    
    # --- CRS ARCHITECTURE (AQUI ESTÁ A CORREÇÃO) ---
    CRS_PROJ: str = 'EPSG:3857'  # Apenas para Visualização (Mapas)
    CRS_GEO: str = 'EPSG:4674'   # Lat/Lon (SIRGAS 2000)
    CRS_CALC: str = 'EPSG:5880'  # SIRGAS 2000 / Brazil Polyconic (Para Áreas Corretas)
    
    # ... (Mantenha os dicionários de cores iguais) ...
    USDA_COLORS: Dict[int, str] = field(default_factory=lambda: {
        1: "#d5c36b", 2: "#b96947", 3: "#9d3706", 4: "#ae868f",
        5: "#f86714", 6: "#46d143", 7: "#368f20", 8: "#3e5a14",
        9: "#ffd557", 10: "#fff72e", 11: "#ff5a9d", 12: "#ff005b"
    })
    
    USDA_NAMES: Dict[int, str] = field(default_factory=lambda: {
        1: "Clay (Cl)", 2: "Silty Clay (SiCl)", 3: "Sandy Clay (SaCl)",
        4: "Clay Loam (ClLo)", 5: "Silty Clay Loam (SiClLo)", 6: "Sandy Clay Loam (SaClLo)",
        7: "Loam (Lo)", 8: "Silty Loam (SiLo)", 9: "Sandy Loam (SaLo)",
        10: "Silt (Si)", 11: "Loamy Sand (LoSa)", 12: "Sand (Sa)"
    })

    IBGE_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'vermelho': '#9E452C', 'amarelo': '#E09752', 'cinzento': '#BDB76B',
        'glei': '#A9A9A9', 'plinto': '#B0171F', 'húmico': '#654321',
        'melânico': '#3D2B1F', 'latossolo': '#BC4935', 'argissolo': '#D2945C', 
        'cambissolo': '#8B5A2B', 'neossolo': '#CDAF95', 'afloramento': '#708090',
        'espodossolo': '#A6A6A6', 'planossolo': '#8DA399'
    })

    BANDS: List[str] = field(default_factory=lambda: ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'])

# --- 3. UTILITÁRIOS ---
class GeoUtils:
    @staticmethod
    def normalize_string(text: str) -> str:
        if not isinstance(text, str): return ""
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower().strip()

    @staticmethod
    def sanitize_filename(nome: str) -> str:
        return re.sub(r'[^\w\s-]', '', GeoUtils.normalize_string(nome)).strip().replace(' ', '_')
        
    @staticmethod
    def get_ibge_color(descricao: str, color_map: dict, default='#E1E1E1') -> str:
        """Busca fuzzy da cor baseada no nome do solo"""
        desc_norm = str(descricao).lower()
        for key, color in color_map.items():
            if key in desc_norm:
                return color
        return default

# --- 4. SERVIÇOS (CLOUD & LOCAL) ---
class SoilService: # GEE Backend
    def __init__(self, config: SoilConfig):
        self.cfg = config
        self._auth()

    def _auth(self):
        try:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        except:
            ee.Authenticate()
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    def download_bands(self, roi_gdf, temp_dir: Path) -> Dict[str, Path]:
        files_map = {}
        roi_ee = geemap.geopandas_to_ee(roi_gdf.to_crs(self.cfg.CRS_GEO))
        image = ee.Image(self.cfg.ASSET_ID)
        
        for band in self.cfg.BANDS:
            out_path = temp_dir / f"temp_{band}.tif"
            files_map[band] = out_path
            if out_path.exists():
                try: os.remove(out_path)
                except: pass
            print(f"   [Cloud] Baixando raster: {band} (cm)...")
            geemap.download_ee_image(image.select(band), filename=str(out_path),
                region=roi_ee.geometry(), scale=self.cfg.SCALE_METERS,
                crs=self.cfg.CRS_PROJ, overwrite=True, num_threads=4)
        return files_map

class LocalGeoService: # Vector Backend
    def __init__(self, config: SoilConfig):
        self.cfg = config

    def get_roi(self) -> gpd.GeoDataFrame:
        if self.cfg.ANALYSIS_MODE == 'MUNICIPIO':
            print(f"   [Local] Carregando municípios: {self.cfg.PATH_SHP_MUN.name}")
            full_gdf = gpd.read_file(self.cfg.PATH_SHP_MUN)
            full_gdf['norm_nm'] = full_gdf['NM_MUN'].apply(GeoUtils.normalize_string)
            aoi = full_gdf[(full_gdf['norm_nm'] == GeoUtils.normalize_string(self.cfg.TARGET_NAME)) & 
                           (full_gdf['SIGLA_UF'] == self.cfg.TARGET_UF.upper())]
            if aoi.empty: raise ValueError("Município não encontrado.")
            # Dissolve para garantir polígono único e evitar duplicidade de área
            return aoi.dissolve()
        else:
            print(f"   [Local] Carregando imóvel: {self.cfg.PATH_SHP_IMOVEL.name}")
            return gpd.read_file(self.cfg.PATH_SHP_IMOVEL).to_crs(self.cfg.CRS_GEO)

    def process_ibge_soils(self, roi_gdf) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        print(f"   [Local] Diagnosticando Solos IBGE: {self.cfg.PATH_SOLOS_IBGE.name}...")
        
        try:
            # 1. Leitura Robusta (Sem bbox)
            # Como o shapefile não tem arquivo .prj, o filtro espacial nativo (bbox) falha.
            # Lemos tudo (é leve) e resolvemos na memória.
            solos = gpd.read_file(self.cfg.PATH_SOLOS_IBGE)
            
            if solos.empty:
                print("   AVISO: O shapefile base de solos está vazio.")
                return None, None
            
            # 2. Injeção de CRS (Forçando SIRGAS 2000)
            if solos.crs is None:
                print("   [Correção] Shapefile sem CRS (.prj). Injetando EPSG:4674 (SIRGAS 2000)...")
                # allow_override garante que não teremos avisos chatos no terminal
                solos.set_crs(self.cfg.CRS_GEO, inplace=True, allow_override=True)
            
            # 3. Clip Espacial Preciso
            print("   [Local] Realizando recorte fino (Clip)...")
            # Garante que ambos estão no mesmo CRS geográfico antes do clip
            roi_geo = roi_gdf.to_crs(self.cfg.CRS_GEO)
            solos_clip = gpd.clip(solos.to_crs(roi_geo.crs), roi_geo)
            
            if solos_clip.empty:
                print("   AVISO: Nenhum solo IBGE intercepta a Área de Interesse.")
                return None, None
            
            # 4. Cálculo de Área Correto (Metros - Polyconic EPSG:5880)
            print("   [Local] Calculando áreas (EPSG:5880 Brazil Polyconic)...")
            solos_calc = solos_clip.to_crs(self.cfg.CRS_CALC)
            solos_clip['area_ha'] = solos_calc.geometry.area / 10000
            
            # 5. Saneamento de Dados (Tratando os 441 nulos revelados no Raio-X)
            solos_clip['DSC_COMPO1'] = solos_clip['DSC_COMPO1'].fillna('Não Mapeado / Água')
            
            # 6. Estatísticas
            stats = solos_clip.groupby('DSC_COMPO1')['area_ha'].sum().reset_index()
            stats = stats.sort_values('area_ha', ascending=False)
            
            total_calc = stats['area_ha'].sum()
            print(f"   [Check] Área Total Calculada no IBGE: {total_calc:.2f} ha")
            
            return solos_clip, stats
            
        except Exception as e:
            print(f"   ERRO NO PROCESSAMENTO VETORIAL: {e}")
            import traceback; traceback.print_exc()
            return None, None

# --- 5. RENDERIZADOR HÍBRIDO (CORRIGIDO V4) ---
class HybridRenderer:
    def __init__(self, config: SoilConfig):
        self.cfg = config
        self.cmap_usda = ListedColormap([self.cfg.USDA_COLORS[i] for i in range(1, 13)])
        self.norm_usda = BoundaryNorm(range(1, 14), 12) 

    def _setup_grid(self, ax, crs_source, bounds=None):
        if bounds is None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            bounds = (xlim[0], ylim[0], xlim[1], ylim[1])
        
        minx, miny, maxx, maxy = bounds
        x_target = np.linspace(minx, maxx, 4)
        y_target = np.linspace(miny, maxy, 4)
        
        # Converte para LatLon para os labels
        pts_metric = gpd.points_from_xy(x_target, y_target, crs=crs_source)
        pts_latlon = pts_metric.to_crs(self.cfg.CRS_GEO)
        
        ax.set_xticks(x_target)
        ax.set_yticks(y_target)
        
        x_lbl = [f"{p.x:.2f}°" for p in pts_latlon]
        y_lbl = [f"{p.y:.2f}°" for p in pts_latlon]
        
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: x_lbl[p] if p < len(x_lbl) else ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: y_lbl[p] if p < len(y_lbl) else ""))
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.3, color='black', linewidth=0.5)

    def _draw_styled_box(self, ax, title):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
        ax.text(0.5, 1.0, title, transform=ax.transAxes, ha='center', va='center', 
                weight='bold', fontsize=9, 
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))
        ax.set_facecolor('none')

    # --- PÁGINA 1: MAPA VETORIAL IBGE (CORRIGIDO PARA METROS/KM) ---
    def generate_vector_page(self, gdf, stats_df):
        fig = plt.figure(figsize=(11.69, 8.27))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.15)
        
        # Mapa
        ax_map = fig.add_subplot(gs[0])
        
        # --- CORREÇÃO CRÍTICA DE ESCALA ---
        # 1. Reprojeta para Web Mercator (Metros) ANTES de plotar
        gdf_visual = gdf.to_crs(self.cfg.CRS_PROJ)
        
        # Define Cores Dinamicamente
        unique_soils = gdf_visual['DSC_COMPO1'].unique()
        color_dict = {soil: GeoUtils.get_ibge_color(soil, self.cfg.IBGE_COLORS) for soil in unique_soils}
        
        gdf_visual.plot(column='DSC_COMPO1', ax=ax_map, 
                       color=[color_dict[x] for x in gdf_visual['DSC_COMPO1']], 
                       edgecolor='black', linewidth=0.3)
        
        # Setup Visual
        # Passamos o CRS Projetado para que o grid calcule Lat/Lon corretamente
        self._setup_grid(ax_map, gdf_visual.crs) 
        
        # ScaleBar agora entende que os eixos estão em Metros -> Vai gerar km automaticamente
        ax_map.add_artist(ScaleBar(1, units='m', location='lower left', box_alpha=0.7))
        
        ax_map.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.88), 
                       arrowprops=dict(facecolor='black', width=4, headwidth=10),
                       ha='center', va='center', xycoords='axes fraction')
        ax_map.set_title(f"MAPEAMENTO PEDOLÓGICO (IBGE)\n{self.cfg.TARGET_NAME}", weight='bold', pad=15)

# Coluna Lateral
        gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[0.4, 0.6], hspace=0.15)
        
        # Legenda Dinâmica
        ax_leg = fig.add_subplot(gs_right[0])
        self._draw_styled_box(ax_leg, "LEGENDA IBGE")
        
        # --- CORREÇÃO ARQUITETURAL DA LEGENDA ---
        # 1. Usamos a lista do stats_df para garantir que a legenda siga a ordem decrescente de área
        classes_legenda = stats_df['DSC_COMPO1'].tolist()
        
        # 2. Desacoplamos Handles (Cores) e Labels (Textos) para garantir compatibilidade universal
        handles_lista = []
        labels_lista = []
        
        for solo in classes_legenda:
            # Pega a cor segura do dicionário (se der falha, usa cinza padrão)
            cor_segura = color_dict.get(solo, '#E1E1E1')
            
            # Tratamento de string rigoroso
            nome_str = str(solo)
            nome_formatado = nome_str[:28] + "..." if len(nome_str) > 28 else nome_str
            
            # Alimenta as listas independentes
            handles_lista.append(mpatches.Patch(color=cor_segura))
            labels_lista.append(nome_formatado)
            
        # 3. Forçamos o Matplotlib a ler ambas as listas explicitamente
        ax_leg.legend(
            handles=handles_lista, 
            labels=labels_lista, 
            loc='center', 
            fontsize=6, 
            frameon=False, 
            title="Classes Encontradas (Por Área)"
        )
        # Stats e Pizza
        ax_stats_frame = fig.add_subplot(gs_right[1])
        self._draw_styled_box(ax_stats_frame, "ESTATÍSTICAS (Área)")
        
        gs_inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_right[1], height_ratios=[0.4, 0.6])
        
        # Texto
        ax_txt = fig.add_subplot(gs_inner[0]); ax_txt.axis('off')
        total_ha = stats_df['area_ha'].sum()
        txt = f"Área Total: {total_ha:,.2f} ha\n\n"
        for _, row in stats_df.head(5).iterrows():
            pct = (row['area_ha']/total_ha)*100
            label = row['DSC_COMPO1']
            txt += f"• {label[:15]}: {pct:.1f}% ({row['area_ha']:.1f} ha)\n"
        ax_txt.text(0.1, 0.9, txt, va='top', fontsize=7, linespacing=1.6)
        
        # Pizza
        ax_pie = fig.add_subplot(gs_inner[1])
        colors_pie = [color_dict[n] for n in stats_df['DSC_COMPO1']]
        def my_autopct(pct): return f'{pct:.1f}%' if pct > 5 else ''
        ax_pie.pie(stats_df['area_ha'], colors=colors_pie, startangle=90, autopct=my_autopct, 
                  wedgeprops={'edgecolor':'white'}, textprops={'fontsize': 7})

        return fig

    # --- PÁGINA 2+: RASTER USDA ---
    def generate_raster_page(self, raster_path, band_name, roi_gdf):
        fig = plt.figure(figsize=(11.69, 8.27))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.15)
        
        ax_map = fig.add_subplot(gs[0])
        with rasterio.open(raster_path) as src:
            # O Raster já vem em EPSG:3857 do GEE (configurado na classe Config)
            roi_proj = roi_gdf.to_crs(src.crs)
            out_image, _ = mask(src, roi_proj.geometry, crop=True)
            data = out_image[0].astype(float)
            data[(data < 1) | (data > 12)] = np.nan
            
            ax_map.imshow(data, cmap=self.cmap_usda, norm=self.norm_usda, 
                        extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top),
                        interpolation='nearest')
            roi_proj.plot(ax=ax_map, facecolor='none', edgecolor='black', linewidth=1.5)
            self._setup_grid(ax_map, src.crs, src.bounds)
            
            # ScaleBar consistente com o vetor
            ax_map.add_artist(ScaleBar(1, units='m', location='lower left', box_alpha=0.7))
            
            ax_map.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.88), arrowprops=dict(facecolor='black', width=4, headwidth=10), ha='center', va='center', xycoords='axes fraction')
            ax_map.set_title(f"TEXTURA USDA - PROFUNDIDADE {band_name.replace('b','')} cm", weight='bold', pad=15)

        # Lateral Raster
        gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[0.35, 0.65], hspace=0.15)
        
        ax_leg = fig.add_subplot(gs_right[0])
        self._draw_styled_box(ax_leg, "LEGENDA USDA")
        handles = [mpatches.Patch(color=self.cfg.USDA_COLORS[i], label=f"{i}. {self.cfg.USDA_NAMES[i]}") for i in range(1, 13)]
        ax_leg.legend(handles=handles, loc='center', fontsize=7, frameon=False)

        ax_resumo = fig.add_subplot(gs_right[1])
        self._draw_styled_box(ax_resumo, "RESUMO ESTATÍSTICO")
        gs_inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_right[1], height_ratios=[0.45, 0.55])
        
        ax_txt = fig.add_subplot(gs_inner[0]); ax_txt.axis('off')
        valid = data[~np.isnan(data)]
        if valid.size > 0:
            uni, counts = np.unique(valid, return_counts=True)
            txt = "Distribuição (%):\n\n"
            for i in np.argsort(-counts):
                pct = (counts[i]/valid.size)*100
                txt += f"• {self.cfg.USDA_NAMES[uni[i]]}: {pct:.1f}%\n"
            ax_txt.text(0.1, 0.9, txt, va='top', fontsize=8, linespacing=1.5)
            
            ax_pie = fig.add_subplot(gs_inner[1])
            cols = [self.cfg.USDA_COLORS[u] for u in uni]
            def my_pct(p): return f'{p:.1f}%' if p > 3 else ''
            ax_pie.pie(counts, colors=cols, startangle=90, autopct=my_pct, wedgeprops={'edgecolor':'white'}, textprops={'fontsize':8})
        else:
            ax_txt.text(0.5, 0.5, "Sem dados", ha='center')

        return fig, valid

    def generate_final_dashboard(self, all_stats):
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle(f"PERFIL DE SOLO INTEGRADO (0-200cm)\n{self.cfg.TARGET_NAME}", weight='bold', fontsize=16)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4)
        
        ax_tbl = fig.add_subplot(gs[0]); ax_tbl.axis('off')
        data_rows = []
        for stat in all_stats:
            row = {'Prof. (cm)': stat['depth']}
            total = stat['counts'].sum()
            for c, count in zip(stat['uniques'], stat['counts']):
                row[self.cfg.USDA_NAMES[c]] = (count/total)*100
            data_rows.append(row)
        df = pd.DataFrame(data_rows).fillna(0).set_index('Prof. (cm)')
        tbl = ax_tbl.table(cellText=df.round(1).values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)
        ax_tbl.set_title("Evolução Textural em Profundidade", weight='bold')

        ax_bar = fig.add_subplot(gs[1])
        bottom = np.zeros(len(df))
        for i in range(1, 13):
            name = self.cfg.USDA_NAMES[i]
            if name in df.columns:
                vals = df[name].values
                ax_bar.bar(df.index.astype(str), vals, bottom=bottom, label=name, color=self.cfg.USDA_COLORS[i], width=0.5)
                bottom += vals
        ax_bar.set_ylabel("% Área"); ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8)
        
        return fig
# --- 6. PIPELINE HÍBRIDO ---
class HybridPipeline:
    def __init__(self, config: SoilConfig):
        self.cfg = config
        self.local_svc = LocalGeoService(config)
        self.cloud_svc = SoilService(config)
        self.renderer = HybridRenderer(config)

    def run(self):
        print(f"--- INICIANDO HYBRID TERRA SOIL: {self.cfg.ANALYSIS_MODE} ---")
        
        # 1. ROI e Vetor Local
        try:
            roi_gdf = self.local_svc.get_roi()
            target_label = self.cfg.TARGET_NAME if self.cfg.ANALYSIS_MODE == 'MUNICIPIO' else "Imovel_Rural"
            
            # Processa o Solo IBGE (Vetor)
            ibge_gdf, ibge_stats = self.local_svc.process_ibge_soils(roi_gdf)
            if ibge_gdf is None: print("   AVISO: Pular etapa IBGE (sem dados).")
        except Exception as e:
            print(f"ERRO CRÍTICO INICIAL: {e}"); return

        # 2. Download Raster (Cloud)
        self.cfg.PATH_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            files_map = self.cloud_svc.download_bands(roi_gdf, self.cfg.PATH_OUTPUT_ROOT)
        except Exception as e:
            print(f"ERRO NO GEE: {e}"); return

        # 3. Geração do PDF
        safe_name = GeoUtils.sanitize_filename(target_label)
        pdf_path = self.cfg.PATH_OUTPUT_ROOT / f"RELATORIO_HIBRIDO_{safe_name}.pdf"
        print(f">> Gerando Relatório Híbrido: {pdf_path}")
        
        stats_collector = []
        
        with PdfPages(pdf_path) as pdf:
            # PÁGINA 1: Vetor IBGE (Se houver)
            if ibge_gdf is not None:
                print("   -> Renderizando Página: Mapeamento IBGE")
                fig_vec = self.renderer.generate_vector_page(ibge_gdf, ibge_stats)
                pdf.savefig(fig_vec, bbox_inches='tight', dpi=300)
                plt.close(fig_vec)
            
            # PÁGINAS 2+: Raster USDA
            for band in self.cfg.BANDS:
                print(f"   -> Renderizando Página: Raster {band}")
                try:
                    fig, pixels = self.renderer.generate_raster_page(files_map[band], band, roi_gdf)
                    pdf.savefig(fig, bbox_inches='tight', dpi=300)
                    plt.close(fig)
                    if pixels.size > 0:
                        u, c = np.unique(pixels, return_counts=True)
                        stats_collector.append({'depth': band.replace('b',''), 'uniques': u, 'counts': c})
                except Exception as e: print(f"Erro {band}: {e}")

            # PÁGINA FINAL: Dashboard
            if stats_collector:
                print("   -> Renderizando Dashboard Final")
                fig_dash = self.renderer.generate_final_dashboard(stats_collector)
                pdf.savefig(fig_dash, bbox_inches='tight', dpi=300)
                plt.close(fig_dash)

        # Limpeza
        print(">> Limpando temporários...")
        for f in files_map.values():
            try: f.unlink()
            except: pass
        print("--- SUCESSO TOTAL ---")

if __name__ == "__main__":
    # --- CONFIGURAÇÃO ---
    cfg = SoilConfig(
        ANALYSIS_MODE='MUNICIPIO',
        TARGET_NAME='Juiz de Fora',
        TARGET_UF='MG'
        # TARGET_NAME='NOME CAR EXEMPLO', ANALYSIS_MODE='IMOVEL OU MUNICÍPIO'
    )
    
    HybridPipeline(cfg).run()
