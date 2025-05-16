# Theory-guided-data-driven-design-of-superelasticity-in-TiNi-alloys-over-wide-temperature-window
> Companion code for "Theory guided data-driven design of superelasticity in TiNi alloys over wide temperature window " Unpublished

### Core Datasets:
- `SMA.data.training2.csv` - Primary training dataset containing historical alloy characterization data
- `NormalAlloyFeature.csv` - Feature engineering dataset with normalized material properties
- `data_TiNiHfZrCu.csv` - Virtual design space for prediction (Ti-Ni-Hf-Zr-Cu pseudo-ternary system)
- `data_predict_Ap_Cu.csv` - Post-prediction virtual space with phase transformation temperature (Ap) performance metrics
- `plot_data1.csv` - Analytical results derived from `data_predict_Ap_Cu.csv` (phase diagrams and property correlations)

## 🧠 Code Components

### Main Algorithm:
- `SearchLargeSETempRangeAP.py` - Core script implementing:
  - Virtual design space exploration
  - Machine learning-driven phase transformation temperature prediction

### Custom Features:
- `Slope/` - Customized phase transition features


