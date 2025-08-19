# Solar-Terrestrial Relations: Statistical and Predictive Analysis of Solar Activity
## Abstract
We conduct a statistical analysis of the sunspot number (SSN) for solar cycle 25, which will be generated using an algorithm based on machine learning. For the purpose we utilized data from the Solar Influences Data Center (SILSO) Observatory located in Belgium. Our research began with a detailed pattern analysis of solar cycles and their characteristics, followed by a statistical analysis of SSN series data for solar cycles 1 to 24. This involved determining key parameters, which comprises 1σ and 2σ widths, skewness, kurtosis (excess), and the definition of a strength parameter through two distinct approaches. This was achieved through the application of skew-gaussian fitting to the SSN data associated with each solar cycle. The forecasting of solar cycle 25 was achieved using a Long Short-Term Memory (LSTM) model, enabling us to plot the graphical trend of solar cycle 25. Skew-Gaussian fitting was carried out further on the data from solar cycle 25 to derive its statistical parameters including it's skewness and kurtosis. Furthermore, the classification of each solar cycle according to skewness was noted alongside the validation of a correlation between skewness and kurtosis in both odd and even cycles. Our findings tackle the challenge of uncovering essential insights into predicting uncertainty asymmetry and tail behavior through the incorporation of higher-order statistical moments (skewness, kurtosis). Practical implications of our findings include improved space weather forecasting, risk assessment for space mission projects, power grid management, and satellite planning via our asymmetric uncertainity quantification.


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
- **License**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
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

