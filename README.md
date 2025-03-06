# SCU GRID Pipeline

A comprehensive data processing and visualization pipeline for the Gamma Ray Integrated Detectors (GRID) project at Sichuan University. This tool provides automated handling, processing, and analysis of gamma-ray detector data from the TianNing satellite mission.

## Features

- Automated processing of satellite downlink data files
- Energy calibration and address correction
- Detector telemetry analysis and visualization
- Interactive data exploration through Streamlit interface
- High-quality plotting with Plotly integration
- Custom visualization toolkit (grafica)
- FITS file export for scientific analysis

## Project Structure

Directory structure explanation with key files and their purposes

```
├── app.py                # Main Streamlit application entry point
├── fix.py                # Energy calibration and address correction
├── grafica/              # Visualization package
│   ├── __init__.py
│   ├── figure.py         # Base figure class
│   ├── FigureManager.py  # Figure management system
│   ├── plotly_utils/     # Utilities for Plotly
│   │   ├── __init__.py
│   │   ├── colors.py     # Color management
│   │   └── utils.py      # Utility functions
│   ├── PlotlyFigure.py   # Plotly figure integration
│   ├── traces.py         # Trace definitions
│   └── validation.py     # Data validation
├── parse.py              # Raw data parsing utilities
├── pipeline.py           # Core processing pipeline implementation
├── plots.py              # Plotting functions and visualization tools
├── SCUGRID_stream.py     # Streamlit app entry point
├── save.py               # Data saving utilities (FITS format)
└── search.py             # Data file search functionality
```

## Installation

Steps to install and set up the pipeline

1. Clone this repository:

```bash
git clone <repository-url>
cd GRID_pipeline
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage instructions

Run the application using Streamlit:

```bash
streamlit run SCUGRID_stream.py
```

## Interface Capabilities

Description of what the user interface allows you to do

The interface allows you to:

- Process new satellite observation data
- Select detector to analyze ("天宁01" or "天宁02")
- View telemetry and scientific data
- Generate energy calibration and address correction
- Create interactive visualizations of data products
- Export data to FITS format for further analysis

## Data Flow

Explanation of how data flows through the pipeline system

1. Raw data files are detected in source directories
2. Data extraction is performed by the Parser class
3. Correction and energy reconstruction is applied by the fix module
4. Results are saved in structured FITS format
5. Interactive visualizations are generated with the grafica package

## Directory Structure

Expected directory structure for the pipeline to function correctly

The pipeline expects data in the following directory structure:

```
DownloadData_TianNing-01/    # For TianNing-01 detector
├── Source/                  # Source data files
├── Output/                  # Processed output files
│   └── [observation]/
│       ├── Extract/         # Extracted data products
│       └── Fix/             # Calibrated data products
├── Calib/                   # Calibration files
└── Science/                 # Science-ready data products

DownloadData_TianNing-02/    # For TianNing-02 detector (similar structure)
```

## Authors

- Sichuan University GRID Team

## License

This project is proprietary software of Sichuan University's GRID team.