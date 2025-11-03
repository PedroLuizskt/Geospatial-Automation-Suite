# Geospatial-Automation-Suite

A [Python](https://www.python.org/) pipeline suite for on-demand territorial intelligence. Automates complex environmental analyses (CAR, Biomes, Soils, Hydro) integrating local ([GeoPandas](https://geopandas.org/), [Rasterio](https://rasterio.readthedocs.io/en/stable/)) and cloud ([Google Earth Engine](https://earthengine.google.com/)) processing to generate PDF reports, turning hours of manual work into minutes.

The Portuguese `README.md` is available at [README.pt-br.md](https://www.google.com/search?q=README.pt-br.md).

-----

##  Objective & Solution

**The Problem:** In many environmental and governmental institutions, the generation of technical reports and situational maps is a manual, repetitive, and slow process, requiring analysts to spend hours or days processing data for a single rural property or municipality.

**The Solution:** This repository contains a suite of modular data pipelines that transform this entire workflow into an automated, scalable system. It functions as a "solution factory" that ingests raw geospatial data and outputs professional-grade PDF dossiers, complete with high-quality maps and statistical tables.

-----

##  Key Features & Analyses

The suite is composed of independent, configurable pipelines that can be run for any Area of Interest (AOI).

### CAR (Rural Environmental Registry) Analysis

  * **Municipal Dossier:** Processes the complete SICAR database (attributes and geometries) for a municipality, generating a PDF dossier with statistics for each CAR status.
  * **Property-Level Characterization:** Performs a detailed analysis of a single rural property, loading all declared layers (APP, Legal Reserve, Consolidated Use, Native Vegetation).
  * **Automatic Legal Reserve (RL) Balance:** Automatically calculates the RL deficit or surplus by cross-referencing the property with Biome and Legal Amazon boundaries to apply the correct required percentage (20%, 35%, or 80%).

### Thematic Analysis (Local Processing)

  * **Soils:** Clips and quantifies IBGE soil classes for the AOI.
  * **Geomorphology:** Clips and quantifies geomorphological units (IBGE/CPRM).
  * **Hydrography:** Analyzes the hydrographic network, identifies river springs, and calculates drainage density (km/km²).
  * **Biomes & Vegetation:** Identifies and quantifies Biomes and Phytophysiognomies.
  * **Climate:** Processes regional raster data to analyze Mean Annual Precipitation and Land Surface Temperature (LST).

### Cloud Analysis (GEE Integration)

  * **Land Use & Cover Change:** Multitemporal analysis (1985-2024) using MapBiomas to quantify the evolution of land use.
  * **Terrain Analysis (GEE):** Computes slope (steepness) from NASADEM DEMs on the fly.
  * **Soil Carbon (GEE):** Quantifies Soil Organic Carbon (SOC) using MapBiomas Soil assets.
  * **Forest Classification (GEE):** Classifies forest status (e.g., Primary, Secondary) using NASA/ORNL data.
  * **Socio-Environmental (GEE):** Analyzes "Tree Proximate People" (TPP) data from FAO to assess population density near forests.

-----

##  Technology Stack

  * **Core Data Processing:**

      * [Python 3.x](https://www.python.org/)
      * [GeoPandas](https://geopandas.org/) & [Pandas](https://pandas.pydata.org/): Vector data manipulation and statistical analysis.
      * [Rasterio](https://rasterio.readthedocs.io/en/stable/) & [Shapely](https://shapely.readthedocs.io/en/stable/): Raster processing, clipping, and masking.
      * [NumPy](https://numpy.org/): Numerical calculations.

  * **Cloud & Big Data:**

      * [Google Earth Engine (GEE)](https://earthengine.google.com/): Python API for petabyte-scale cloud processing.
      * [Geemap](https://geemap.org/): Interactive development and data export from GEE.

  * **Data Visualization & Reporting:**

      * [Matplotlib](https://matplotlib.org/): Core engine for creating static maps and charts.
      * [Contextily](https://contextily.readthedocs.io/en/latest/): Adding web map basemaps.
      * [Matplotlib-Scalebar](https://github.com/ppinard/matplotlib-scalebar): Cartographic scale bars.
      * `PdfPages`: Compiling outputs into multi-page PDF dossiers.

-----

##  Setup & Data

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/PedroLuizskt/Geospatial-Automation-Suite.git
    cd Geospatial-Automation-Suite
    ```

2.  **Install the dependencies:**
    *(Recommended to create a virtual environment: `python -m venv .venv` and `source .venv/bin/activate`)*

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the Input Data:**
    The raw geospatial data (shapefiles, GeoTIFFs, etc.) are not in this repository.

      * **Files to test the codes:** [**Download Input Data (Google Drive)**](https://drive.google.com/drive/folders/1X7DmXw88nwcVNRUOHANM8g19bBM2alZI?usp=drive_link)

-----

##  Running the Pipelines

Each script in the `/pipelines_` folder is a complete, self-contained workflow.

1.  **Configure the Script:** Open any script (e.g., `pipelines_/soil_.py`).
2.  **Adjust Paths:** At the top of the script, you will find a `@dataclass` named `Config`. Update the paths (`PATH_...`) to point to where you saved the downloaded input data.
3.  **Run the Script:**
    ```bash
    python pipelines_/soil_.py
    ```
4.  **Check Results:** The PDF dossier will be saved in your `PATH_EXPORTACAO`.

-----

## 📁 Project Structure

```
.
├── .gitignore               # Ignores unnecessary files (e.g., .venv, __pycache__)
├── LICENSE                  # MIT License
├── README.md                # This file (English)
├── README.pt-br.md          # Portuguese README
├── requirements.txt         # Python dependencies for 'pip install'
│
├── pipelines_/
│   ├── municipal_rural_property_caracterization.py
│   ├── rural_property_caracterization.py
│   ├── soil_.py
│   ├── geomorphology_.py
│   ├── hydrography_.py
│   ├── phytophysiognomies_.py
│   ├── preciptation_.py
│   ├── temperature_lst_.py
│   └── multitemporal_.py
│
├── gee_snipets/
│   ├── forest_classification_2020_V1.py
│   ├── ndvi_.py
│   ├── soil_carbon_.py
│   ├── terrain_analysis.py
│   └── treeproximatepeople_.py
│
└── exemple_outputs/
    ├── Dossie_CAR_buritizeiro.pdf
    ├── Relatorio_Carbono_Solo_...pdf
    ├── Relatorio_Declividade_...pdf
    └── ... (dozens of other example reports)
```

-----

##  Best Practices Applied

  * **Modular Code:** Each pipeline is self-contained and focused on a single task.
  * **Configuration Separation:** Uses `@dataclass` to separate paths and parameters (Config) from execution logic (Pipeline), allowing for easy reuse.
  * **Hybrid Processing:** An architecture that decides when to use local processing (GeoPandas) for small/local data and when to use cloud processing (GEE) for planetary-scale datasets (e.g., MapBiomas, NASADEM).
  * **Reproducibility:** Generates professional, automated cartographic reports with Matplotlib.

-----

##  Author

Developed by **Pedro Luiz**.
