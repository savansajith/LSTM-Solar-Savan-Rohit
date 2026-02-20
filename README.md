# Higher-Moment Characterization of Sunspot Cycles and Data-Driven Projections for Solar Cycle 25
## Abstract
This study presents a comprehensive statistical analysis of solar cycles 1–24 and predicts the characteristics of solar cycle 25. Using sunspot number (SSN) data from the Solar Influences Data Center (SILSO), we applied skewed Gaussian fitting to each cycle to quantify asymmetry, tailedness, and overall strength. Our analysis reveals that cycles with rapid rises exhibit higher skewness, while broader, symmetric cycles show lower skewness, highlighting the predictive value of early-cycle behavior. Furthermore, we find that even-numbered cycles display a strong correlation between skewness and kurtosis, suggesting more stable dynamics, whereas odd-numbered cycles show greater variability. To forecast solar cycle 25, we employed a Long Short Term Memory (LSTM) model, which captures nonlinear dependencies and periodic variations in the data. The model predicts a peak SSN of 184.8 in February 2025, indicating that cycle 25 will surpass cycle 24 in both intensity and duration. These findings underscore the importance of incorporating higher-order statistical moments into solar cycle prediction, providing a robust framework for advancing our understanding of solar dynamo processes.

## Dependencies
This project uses the following Python packages (no `requirements.txt` is included):

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `seaborn`
- `tensorflow`
- `plotly`
- `scikit-learn`


## Acknowledgements
This project draws inspiration from the educational content provided by [Spartificial](https://github.com/Spartificial), particularly the [YouTube Academic Projects repository](https://github.com/Spartificial/yt-acad-projs) (2021).

The Long Short-Term Memory (LSTM) model architecture used in this work is based on the implementation shared in that repository, which complements a YouTube video series focused on machine learning and space-related student projects.

> **Note:** The original repository does not include a license file. The referenced material was used strictly for academic and non-commercial research purposes.

[YouTube Academic Projects Playlist](https://www.youtube.com/playlist?list=PLRj2DdfTEVZgcNnaLAxkJQ6WjDL7qzp1N)


## Dataset Source

The dataset was obtained from the [Solar Influences Data Center (SILSO)](https://www.sidc.be/silso/), operated by the Royal Observatory of Belgium.

- **Source**: Solar Influences Data Center (SILSO)
- **Link**: [SN_m_tot_V2.0.txt](https://www.sidc.be/SILSO/DATA/SN_ms_tot_V2.0.txt)
- **Data Type**: Monthly mean total sunspot number
- **License**: All Rights Reserved. See the [LICENSE](LICENSE) file for details.
- **Attribution Required**:  
  > *"Data provided by the Solar Influences Data Center (SILSO), Royal Observatory of Belgium, Brussels"*

### Dataset Format & Structure

**Filename**: `Monthly_mean_SN_number.txt`
**Format**: Plain ASCII text  

#### Column Descriptions

| Column | Description                                                                 |
|--------|-----------------------------------------------------------------------------|
| 1      | Year (Gregorian)                                                            |
| 2      | Month (Gregorian)                                                           |
| 3      | Decimal date (middle of the corresponding month)                            |
| 4      | Monthly mean total sunspot number                                           |
| 5      | Monthly mean standard deviation of input sunspot numbers                    |
| 6      | Number of observations used to compute monthly mean                         |
| 7      | Definitive/provisional marker (` ` = definitive, `*` = provisional)         |

---

#### Line Format (Character Positions)

- `[1–4]` → Year  
- `[6–7]` → Month  
- `[9–16]` → Decimal date  
- `[19–23]` → Monthly total sunspot number  
- `[25–29]` → Standard deviation  
- `[32–35]` → Number of observations  
- `[37]` → Definitive/provisional indicator  

> **Note**: The final column shows whether the data is provisional (`*`) or finalized (blank).


## How to Run

1. **Download & Extract**  
   - Click the green **Code** button → **Download ZIP**  
   - Extract the ZIP file to any folder on your computer

2. **Install Dependencies**  
   - Open your **Terminal** / **Command Prompt** / **Anaconda Prompt**  
   - Navigate to the extracted folder:
     ```bash
     cd path/to/extracted-folder

    - Then run this single command:
      ```bash
      pip install numpy pandas matplotlib scipy seaborn tensorflow plotly scikit-learn jupyter

    - Run the project in Jupyter Notebook:
      ```bash
      jupyter notebook

    - Replace file paths in the code with the actual paths on your computer and run all cells.
