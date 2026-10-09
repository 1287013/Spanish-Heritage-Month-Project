# Sources and indicator definitions

## Retrieval date and window
The app displays the runtime retrieval date. It requests `current year - 70` through `current year - 1`, inclusive. Actual coverage and non-null observation count depend on the selected country/indicator and are displayed from the returned data.

## Official World Bank API
API pattern:
`https://api.worldbank.org/v2/country/{ISO2}/indicator/{INDICATOR_CODE}?date={START}:{END}&format=json&per_page=20000`

Country codes:
- Mexico: MX
- Brazil: BR
- Argentina: AR
- Colombia: CO
- Chile: CL
- Peru: PE
- Venezuela: VE
- Ecuador: EC
- Uruguay: UY

Brazil is included as a Latin American comparison country.

## Indicator mappings

| Label | World Bank code | Unit in app | Definition / caution |
|---|---|---|---|
| Life expectancy at birth | `SP.DYN.LE00.IN` | years | Expected years of life for a newborn if current age-specific mortality rates persist. |
| Unemployment, national estimate | `SL.UEM.TOTL.NE.ZS` | % of labor force | Unemployment rate using national estimates. |
| GDP per capita | `NY.GDP.PCAP.CD` | current US dollars per person | GDP divided by midyear population. Not average wealth or household income. |
| Crude birth rate | `SP.DYN.CBRT.IN` | births per 1,000 people per year | Annual live births per 1,000 people. It is a rate, not a birth count. |
| Net migration | `SM.POP.NETM` | people, net | Net migration, not an emigration count. |
| Intentional homicide rate | `VC.IHR.PSRC.P5` | per 100,000 people | Intentional homicides per 100,000 people. |
| Consumer price inflation | `FP.CPI.TOTL.ZG` | % annual change | Annual percentage change in the consumer price index. |
| Central government debt | `GC.DOD.TOTL.GD.ZS` | % of GDP | Central government debt relative to GDP. |
| Total reserves including gold | `FI.RES.TOTL.CD` | current US dollars | Total reserves including gold, current US dollars. |
| External balance on goods and services | `NE.RSB.GNFS.ZS` | % of GDP | Exports less imports of goods and services as a share of GDP. |

Each indicator code links in the app to its World Bank API endpoint and indicator catalog page. The API response is the source for observations. Null values are excluded from the working table and are not imputed.

## Method references
- World Bank Indicators API documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation
- World Bank DataBank / indicators: https://data.worldbank.org/indicator
- NumPy polynomial fitting documentation: https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyfit.html
- scikit-learn R² metric: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
- scikit-learn MAE metric: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
- SymPy calculus documentation: https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html
- Streamlit documentation: https://docs.streamlit.io/
- ReportLab user guide: https://www.reportlab.com/docs/reportlab-userguide.pdf
- XlsxWriter documentation: https://xlsxwriter.readthedocs.io/
