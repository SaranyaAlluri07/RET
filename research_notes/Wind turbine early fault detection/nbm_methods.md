# NBM residual methods with statistical alarm logic for SCADA wind turbine early fault detection

Scope: a lightweight, interpretable pipeline. Steps: healthy-behaviour regression per target signal, then regime-aware residual normalisation, then EWMA/CUSUM/persistence alarms, then component diagnosis with an "unknown" outcome. Target data: CARE to Compare Wind Farm A (10-min SCADA, about 80 signals).
Research date: 2026-10-26. Roughly 20 searches and fetches. Many details come from abstracts or fetch summaries; such items are flagged where relevant.

## Q1. Target signals, input features and leakage exclusion (incl. CARE to Compare Wind Farm A context)

### Takeaway
Typical NBM targets are component temperatures: gearbox bearing and oil, generator bearings, stator windings, rotor brushes, main bearing, transformer and hydraulic oil. Inputs are operating-state drivers: active power, rotor or generator speed, ambient or nacelle temperature, wind speed, and sometimes lagged values. Wind Farm A signals are anonymised except power, reactive power and wind speed. Status labels had to be synthesised by the dataset authors, so input selection and leakage control must be done from the data (correlation clustering) rather than from signal names.

### Cited Findings
- CARE to Compare (Gück, Roelofs, Faulstich, Fraunhofer IEE; arXiv 2404.10320, also *Data* 2024) covers 36 turbines in 3 wind farms, about 89 turbine-years, 44 labelled anomaly events and 51 normal series. — [arXiv html](https://arxiv.org/html/2404.10320v2)
- **Wind Farm A**: 5 onshore turbines in Portugal (EDP open data), 86 features of which 54 are sensors. Only power, reactive power and wind speed are identifiable by name. It has 22 datasets (11 anomaly, 11 normal). Anomaly starts come from the EDP logbook, which gives only the fault start timestamp. Event starts were estimated by data analysis, so the true onset is uncertain. — [arXiv html](https://arxiv.org/html/2404.10320v2)
- Wind Farm A has no operator status data. The authors set the 14 days before each fault to status 4 (fault) and the 3 days after to status 3 (service) to keep anomalous data out of training. Status IDs 0 (normal) and 2 (idling) count as normal; 1, 3, 4 and 5 do not. — [arXiv html](https://arxiv.org/html/2404.10320v2)
- The EDP failure logbook lists component failures in the gearbox, generator (incl. generator bearing), transformer and hydraulic group. Counts differ by source: EDP says 23 records over 2016-03-03 to 2017-08-21 on 5 turbines; Mendeley says 28 rows. — [EDP OpenData details (IEA Wind)](https://iea-wind.org/wp-content/uploads/2024/07/EDP-OpenData-Details.pdf); [Mendeley description](https://data.mendeley.com/datasets/zjxjnjp3xs/2)
- The EDP SCADA signals include generator speed and temperatures, gearbox and hydraulic oil temperatures, rotor rpm, wind speed and direction, pitch angles, and active and reactive power. — [Mendeley description](https://data.mendeley.com/datasets/zjxjnjp3xs/2)
- Chesterman et al. (WES 2023) studied generator bearing replacements, generator fan replacements and rotor brush high-temperature failures. Target ("health label") signals were the related temperatures, e.g. generator bearing 1 and 2 and stator winding. Inputs were drawn from over 100 drive-train and operational 10-min signals. — [Chesterman et al., WES 8, 893 (2023)](https://wes.copernicus.org/articles/8/893/2023/)
- Letzgus (WES 2020) normalised every drivetrain temperature and oil-pressure signal with a linear regression on **active power, rotor speed and ambient temperature**. He warns that an abrupt change in one regressor can cause apparent change points in highly correlated signals. Suggested fixes: regress inputs against neighbouring turbines first, or flag simultaneous change points across signals that share a regressor. — [Letzgus, WES 5, 1375 (2020)](https://wes.copernicus.org/articles/5/1375/2020/)
- Main bearing temperature is the most commonly used signal in SCADA drivetrain diagnosis. Absolute temperatures rise late in a failure, but temperature *behaviour* can change well in advance, which is why residuals beat raw thresholds. — [Loughborough, Tautz-Weinert & Watson](https://repository.lboro.ac.uk/articles/conference_contribution/Condition_monitoring_of_wind_turbine_drive_trains_by_normal_behaviour_modelling_of_temperatures/9558500)
- A Lisbon thesis on NBM fault detection reports two findings. **Autoregressive inputs** (lags of the target) performed best. **"Simultaneity" features** (concurrent correlated signals) removed most of the fault-detection ability, which is the leakage effect. — [Felgueira thesis, IST Lisbon](https://fenix.tecnico.ulisboa.pt/downloadFile/1970719973968215/Tese_Final_Telmo_Felgueira_79004.pdf) (abstract-level)
- In Chesterman et al., adding three lags gave only marginal RMSE gains and no clear improvement in transients. — [Chesterman et al. 2023](https://wes.copernicus.org/articles/8/893/2023/)
- Lichtenstein et al. (Fraunhofer IWES, *Wind Energy* 2025) built a partly physics-based, partly empirical, interpretable NBM for drive-train temperatures. It reached R² > 0.9 for 12 of 16 temperature signals, on a turbine with a two-stage planetary gearbox and a DFIG. 10-s data were slightly better, but **10-min data were sufficient**. — [Wiley, DOI 10.1002/we.2961](https://onlinelibrary.wiley.com/doi/full/10.1002/we.2961); [Fraunhofer publica](https://publica.fraunhofer.de/handle/publica/512257)

### Inferences
- For Wind Farm A, map the anonymised sensors to components by correlation clustering. Exclude from each target's inputs any sensor in the same cluster or component group. Examples: the other generator bearing temperature when predicting one generator bearing, or gear oil and gear bearing temps against each other. This avoids the simultaneity leakage seen in the Felgueira thesis. A safe default input set is power, rotor or generator speed, wind speed, ambient and nacelle temperature, reactive power, and pitch. Add, if validated, slow-moving temperatures of *other* components that act as thermal-environment proxies.
- Lagged *target* values are risky. A slowly drifting fault gets "explained away" by its own lags. Prefer exogenous lags (e.g., power over the last 30–60 min) for thermal inertia.

### Gaps
- No source gives Wind Farm A's exact sensor-to-component mapping. Event descriptions in the CARE `event_info` file should be checked directly. They likely name components such as "gearbox", "generator bearing", "hydraulic group" and "transformer", but this was not verified here.
- No 2019–2026 paper found that systematically quantifies leakage from sibling temperatures. The evidence is the thesis finding plus expert practice.

## Q2. Which regressors work best for SCADA NBMs (2019–2026 comparisons)

### Takeaway
Evidence for non-linearity in 10-min temperature NBMs is weak. Regularised linear models (elastic net or ridge) are competitive with LightGBM and MLP, and more robust in transients such as cool-downs. GBMs are competitive with ANNs. SVR is usually worse. Autoencoders are the CARE baseline (CARE = 0.66) but are less interpretable.

### Cited Findings
- Chesterman et al. (2023) compared elastic net, LightGBM, MLP and SVR. LightGBM and MLP were marginally better than elastic net for some signals and worse for others, and the paper calls the evidence for non-linearity "weak". Elastic net gave the best healthy/unhealthy separation for rotor-brush and generator-bearing failures. LightGBM and MLP did somewhat better for generator-fan failures. SVR was never best. **During long cool-downs SVR and LightGBM degraded severely; MLP did better but still trailed elastic net.** — [Chesterman et al. 2023](https://wes.copernicus.org/articles/8/893/2023/)
- In the same study, a fleet-median correction plus elastic net cut healthy test RMSE for all signals versus the fleet median alone. Training on 2 months instead of 6 months per turbine raised RMSE and weakened detection. PCA preprocessing did not cut RMSE but avoided large erroneous errors. — [Chesterman et al. 2023](https://wes.copernicus.org/articles/8/893/2023/)
- A DTU comparison found nonlinear NN NBMs beat linear regression. Leftover seasonality in the regression residuals made abnormality hard to detect and inflated alarm levels. — [DTU Orbit](https://orbit.dtu.dk/en/publications/comparative-analysis-of-neural-network-and-regression-based-condi/) (abstract-level)
- Tautz-Weinert & Watson compared linear models, ANNs, ANFIS, SVM and Gaussian process regression on more than 100 turbines. Results were sensitive to training data and settings, and reliable monitoring without many false alarms was hard. — [Loughborough](https://repository.lboro.ac.uk/articles/conference_contribution/Condition_monitoring_of_wind_turbine_drive_trains_by_normal_behaviour_modelling_of_temperatures/9558500)
- GBMs give results competitive with ANNs for NBM fault detection. — [Felgueira thesis](https://fenix.tecnico.ulisboa.pt/downloadFile/1970719973968215/Tese_Final_Telmo_Felgueira_79004.pdf)
- XGBoost temperature NBMs for the generator and gearbox oil, with EWMA on the residuals, have been used (IEEE Access 2020). — [DOAJ record](https://doaj.org/article/17fdd44d48aa4a1eb476168e4818a1e5)
- A TPE-optimised LightGBM NBM for gearbox and performance, with a sliding-window-smoothed residual index, warned earlier than SCADA alarms (Chinese journal, 2024). — [abstract](https://search.napstic.cn/literature/periodical/010dgjs202412023)
- Multi-target NBMs, which predict several signals with one model to reduce maintenance burden, have been benchmarked on low-frequency SCADA. — [Applied Energy 2021](https://sciencedirect.com/science/article/pii/S0306261921007509?via=ihub%3D); [BFH](https://arbor.bfh.ch/bitstreams/e2848426-a155-4e2d-bb92-a172207eed01/download)
- On CARE to Compare, the Fraunhofer autoencoder baseline scored **0.66**, random scored 0.5, and Isolation Forest scored below 0.5. The Wind Farm A AE had layers 44-25-4-25-44, lr 0.0018, noise 0.06, batch 64, early stopping patience 3, and a 75/25 train/val split. — [arXiv html](https://arxiv.org/html/2404.10320v2)
- A non-peer-reviewed GitHub project reports temperature-only CARE scores across the 95 datasets: gradient boosting 0.49 and an RC thermal model 0.47. — [wind-thermal-nbm repo](https://github.com/kwwaleszczynski-del/wind-thermal-nbm) (unverified)

### Inferences
- Recommended stack: one ridge or elastic-net model per target as the interpretable baseline, plus a HistGradientBoosting/LightGBM challenger (monotone constraints on power and ambient temperature are a plausible guard). Pick per target on healthy validation separation, not just RMSE. Check residual behaviour during cool-downs and idling explicitly.
- Temperature-only NBMs probably score below the multivariate AE on CARE (unverified repo numbers). Adding power-curve or performance NBMs and hydraulic pressure targets is likely needed to cover hydraulic and transformer events.

### Gaps
- No peer-reviewed head-to-head of ridge, LightGBM and AE on CARE to Compare Wind Farm A was found.

## Q3. Regime-aware residual normalisation, robust estimators and healthy-data filtering

### Takeaway
Best practice is to train only on status-normal, power-curve-consistent data with buffers around faults and maintenance. Remove change points. Then standardise residuals robustly with median and MAD (σ ≈ 1.4826·MAD), with iterative outlier removal and optional regime conditioning. Mahalanobis-style thresholds that depend on operating condition are an established alternative.

### Cited Findings
- CARE status filtering: train only on status 0 and 2. Also mark points as not normal when wind speed is in the operating range but power is near zero. Scale power and reactive power by rated power so the power curve can be used for cleaning. — [arXiv html](https://arxiv.org/html/2404.10320v2)
- Chesterman et al.: data from 4 months before to 1 month after a failure is treated as unhealthy. Training uses the first 4380 healthy hourly samples (about 6 months) per turbine, pooled into one model per signal per farm. Data are aggregated to hourly resolution. The train/test split is chronological and made before preprocessing to avoid leakage. — [Chesterman et al. 2023](https://wes.copernicus.org/articles/8/893/2023/)
- **Fleet-median correction**: subtract the farm median per timestamp to separate a common component (weather) from a turbine-specific one. Fleet-median NaN rules: no missing values allowed if the fleet has fewer than 5 turbines, at most 20% missing for 5–9, at most 40% for 10 or more. A measurement error is flagged if the fleet-corrected deviation exceeds the signal median × 1. — [Chesterman et al. 2023](https://wes.copernicus.org/articles/8/893/2023/)
- Robust residual scaling: σ is estimated as **1.4826 × MAD**, with iterative outlier removal until thresholds stabilise (IOD–MAD). Anomaly scores are banded at **3, 4 and 5 robust σ**. — [Chesterman et al. 2023](https://wes.copernicus.org/articles/8/893/2023/)
- Letzgus training-data hygiene:
  - exclude data below 10% of rated power;
  - use daily averages, keeping only days with at least 3 h of operation;
  - remove data within **±60 days** of detected change points;
  - one-third of 600 signals from 33 turbines had change points over 2 years;
  - kernel change-point detection (Laplace kernel) reached F1 = 0.86. — [Letzgus 2020](https://wes.copernicus.org/articles/5/1375/2020/)
- Bangalore et al. use Mahalanobis distance on ANN model errors so the threshold reflects how errors vary with operating condition. A 2025 MDPI critique notes this assumes multivariate normality, which can fail (bimodal residuals) during sudden operational shifts. — [Chalmers record](https://research.chalmers.se/en/publication/248521); [MDPI JMSE 13(7):1337](https://www.mdpi.com/2077-1312/13/7/1337)
- Model-derived, turbine-specific temperature thresholds adapt to each turbine's operating envelope and may cut false alarms compared with fixed limits. — [Turnbull et al., Energies 2022](https://www.mdpi.com/1996-1073/15/14/5298)
- CARE AE baseline (Farms A and B) uses an *adaptive* threshold: a neural network predicts the expected reconstruction error ε, and a point is flagged if the error exceeds ε + γ. γ = 0.344 (A) and 0.234 (B), with 0.2–0.4 suggested. This is a learned, regime-conditioned threshold. — [arXiv html](https://arxiv.org/html/2404.10320v2)

### Inferences
- An implementable version of the above:
  1. Bin residuals by regime (power deciles plus an idle bin, and optionally ambient-temperature terciles).
  2. Per bin, compute median and 1.4826·MAD on healthy validation data, with a minimum sample count per bin and fallback to the global value.
  3. z = (r − med_bin) / σ_bin.
  4. Alternatively, fit quantile GBMs (q = 0.05/0.5/0.95) for the residual as a function of regime, and use (r − q50)/((q95 − q05)/3.29).
- The fitted adaptive-threshold result from the CARE baseline supports conditioning the scale on operating state rather than using a global σ.

### Gaps
- No source gave a canonical bin scheme (e.g., number of power bins). This must be tuned.

## Q4. EWMA/CUSUM parameters, multivariate combination and persistence rules

### Takeaway
Few wind papers publish explicit (λ, L, k, h) values. The most concrete, field-validated persistence rule is the CARE **criticality counter**: +1 per anomalous normal-status point, −1 (floored at 0) per normal point, alarm at **72** (about 12 h of 10-min data). Multi-window moving averages and CUSUM on robust z-scores (Chesterman), MEWMA (better than EWMA in one study) and Mahalanobis on residual vectors (Bangalore) are the reported multivariate and persistence options.

### Cited Findings
- **Criticality counter (CARE)**:
  - +1 when a normal-status timestamp is predicted anomalous;
  - −1 (min 0) when it is predicted normal;
  - unchanged at abnormal-status timestamps;
  - event alarm when the counter exceeds t_c = 72;
  - shorter events need lower thresholds, so t_c should be tuned per dataset. — [arXiv html](https://arxiv.org/html/2404.10320v2)
- Chesterman health score: moving averages of anomaly scores over **1, 10, 30, 90 and 180 days**, with bounds from Tukey fences across turbines. The health score is the sum over the windows: good ≤ 5, mediocre 5–10, bad > 10. — [Chesterman et al. 2023](https://wes.copernicus.org/articles/8/893/2023/)
- Chesterman CUSUM: subgroup sizes of 10, 30, 90 and 180 days, using subgroup and overall medians with MAD-based σ and three thresholds per signal. — [Chesterman et al. 2023](https://wes.copernicus.org/articles/8/893/2023/)
- Chesterman detection results:
  - IOD–MAD detected 3/4 generator-bearing and 3/3 generator-fan cases, against 2/4 and 2/3 for CUSUM. CUSUM was less sensitive and probably needed retuned thresholds.
  - IOD–MAD gave more unexplained "bad health" flags. Median false-positive ratios on healthy turbines were 0.12 (gen bearing 1) and 0.08 (gen bearing 2 and stator).
  - No lead-time figures were published. — [Chesterman et al. 2023](https://wes.copernicus.org/articles/8/893/2023/)
- An XGBoost NBM with an EWMA chart on residuals, using fixed and dynamic thresholds, gave generator-fault alarms 3 h (fixed) and 4.25 h (dynamic) ahead. This is a short lead time. — [IEEE Access 2020 (DOAJ)](https://doaj.org/article/17fdd44d48aa4a1eb476168e4818a1e5)
- Renewable Energy 2018: EWMA on residuals to remove autocorrelation, compared with MEWMA. **MEWMA was more suitable than EWMA** for early detection and avoiding errors. — [IDEAS/RePEc](https://ideas.repec.org/a/eee/renene/v115y2018icp808-816.html)
- An arXiv 2019 study on 150 turbines used seasonality-adjusted temperature residuals with a control chart, giving up to 1–2 months lead time for gearbox, generator and main-bearing failures. — [arXiv 1910.09808](https://arxiv.org/pdf/1910.09808)
- A probabilistic MLP NBM with transfer learning plus a **CUSUM chart** for early warning (power output). — [Fiocchi et al., arXiv 2404.16496](https://arxiv.org/pdf/2404.16496v2)
- Online CUSUM on power-curve residuals detected a 2% efficiency loss from gearbox degradation, but only on simulated FAST data. — [Acta Polytechnica Hungarica 2021](https://uni-obuda.hu/journal/Shaheen_AbuHanieh_Nemeth_113.pdf)
- Mahalanobis distance on predicted gearbox bearing and oil temperatures (2 × 2 MW turbines, 10-min) detected anomalies before failure without alarms in healthy operation. — [IEEE Access 2021 (DOAJ)](https://doaj.org/article/46f16bf5a30348a6b923eb19e22ee8a0)
- An EWMA on anomaly scores has also been used for main bearings. SCADA temperature alarms for the main bearing are often late because temperature propagates slowly. — [Springer 2025](https://link.springer.com/article/10.1007/s44163-025-00409-3)

### Inferences
- These defaults come from standard SPC practice, not from wind papers, and should be tuned on CARE normal datasets for Accuracy ≥ 0.9:
  - EWMA λ = 0.05–0.2 on hourly-aggregated z, with L ≈ 2.7–3.0;
  - CUSUM k = 0.5σ (to detect 1σ shifts) and h = 4–5σ;
  - criticality counter threshold 72 (10-min) as a CARE-aligned persistence rule;
  - hysteresis: clear the alarm only after the counter returns to 0 or EWMA < L/2 for 24 h.
- Multivariate combinations:
  - per component, max-|z| across that component's targets, giving interpretability;
  - fleet-level sum of squares or robust Mahalanobis (MCD covariance) on the residual vector, to catch joint shifts.
  - MEWMA is supported over univariate EWMA in one study.

### Gaps
- No wind-specific paper found with explicit λ/L/k/h values in accessible text. The IEEE Access 2020 and Renewable Energy 2018 full texts may contain them but were not retrieved.

## Q5. False alarm reduction in practice

### Takeaway
False alarms are cut mainly by four measures: status and power-curve filtering in both training and scoring, change-point removal, fleet-median correction for weather and common-mode effects, and persistence counters that freeze during non-normal status.

### Cited Findings
- The CARE criticality counter is *frozen* at abnormal-status points, so curtailment, service and fault states neither raise nor clear alarms. Wind speed and power should be checked beyond status codes, since status may miss abnormal operation. — [arXiv html](https://arxiv.org/html/2404.10320v2)
- Wind Farms B and C use 0 for missing values, so long runs of zeros must be treated with caution (a flatline or stuck-sensor issue). Status logs for B and C may be inconsistent because they are recorded only on change. — [arXiv html](https://arxiv.org/html/2404.10320v2)
- Exclude below 10% rated power because sensors report default values when not operating. Change points from maintenance or sensor swaps affect about one-third of signals and must be removed from training. — [Letzgus 2020](https://wes.copernicus.org/articles/5/1375/2020/)
- Fleet-median subtraction removes weather and farm-wide effects. Measurement-error flags apply when the fleet-corrected deviation exceeds the signal median. Elastic net strongly reduced negative error spikes during power-down cool-downs. — [Chesterman et al. 2023](https://wes.copernicus.org/articles/8/893/2023/)
- Residual "anomalies" do not always indicate impending component failure, and multiple false alarms have been reported with simple thresholding. — [Springer 2025](https://link.springer.com/article/10.1007/s44163-025-00409-3)

### Inferences
- Add a data-quality layer before scoring, which flags the point as not normal and freezes the counter:
  - stuck or flatline detection, e.g. rolling std = 0 over 6 or more consecutive 10-min points for analog temperatures;
  - physical range checks;
  - rate-of-change checks;
  - curtailment detection (power below power-curve lower band while pitch is greater than fine pitch at above-cut-in wind);
  - start-up and shutdown masks (first and last 1–2 h after state change, to cover thermal transients);
  - icing proxy (ambient below about 2 °C with power deficit).

### Gaps
- No source found giving quantified false-alarm reduction from icing or curtailment filters specifically.

## Q6. Diagnosis (residual pattern to component) and explainability

### Takeaway
The simplest interpretable diagnosis is to give each component its own target set and attribute alarms to the component whose residuals exceed thresholds. Return "unknown" when residuals of multiple unrelated components, or only the inputs, deviate. For multivariate AE models, Fraunhofer's ARCANA gives root-cause feature importance.

### Cited Findings
- ARCANA (Fraunhofer IEE, *Energy and AI* 2021) frames AE reconstruction as an optimisation that removes most anomalous properties with a few explanatory features. With an injected wind-speed sensor error, it highlighted the wind-speed sensor with significantly higher importance, unlike plain reconstruction error. — [Fraunhofer publica](https://publica.fraunhofer.de/handle/publica/267752); [XAI PdM survey, arXiv 2401.07871](https://arxiv.org/pdf/2401.07871)
- A large deviation in one input can inflate reconstruction errors across many features, which blurs attribution. This is a reason to prefer per-target NBMs for interpretability. — [arXiv 2401.07871](https://arxiv.org/pdf/2401.07871)
- Fraunhofer IEE's open-source EnergyFaultDetector framework combines an AE, preprocessing, threshold analysis and ARCANA (2026 trade article). — [bulletin.ch 2026](https://bulletin.ch/sites/default/files/2026-04/2026_3_Rehwald_Fehler_im_Betrieb_von_Windenergieanlagen_erkennen.pdf)
- Interpretable generator-bearing fault detection with ML has been demonstrated (Bergen thesis, 2023). — [BORA UiB](https://bora.uib.no/bora-xmlui/handle/11250/3085872)
- Physics-based interpretation of NBM correlations is possible (Lichtenstein et al. 2025). — [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/we.2961)
- Change points that appear simultaneously across signals sharing a regressor point to an input or regressor problem rather than a component fault. — [Letzgus 2020](https://wes.copernicus.org/articles/5/1375/2020/)

### Inferences
- Rule-based diagnosis:
  - **gearbox**: positive residuals in gear-oil or gear-bearing temps, with a stable generator group;
  - **generator bearing**: one generator-bearing residual rises, the other may follow;
  - **generator winding or cooling**: stator plus bearings rise together, or the stator alone;
  - **transformer**: transformer temperature residual rises, possibly with reactive-power anomalies;
  - **hydraulic**: hydraulic oil temperature or pressure residual, or pressure-cycling anomalies.
- Output **"unknown"** in three cases: (a) the top two component scores are within a margin, (b) input-sensor or data-quality flags are active, or (c) more than two unrelated groups alarm at once (common-mode or weather).
- For per-target GBMs, SHAP on the *prediction* explains expected temperature. To diagnose, attribute the *residual* instead, e.g. SHAP of a classifier separating recent vs. reference residual windows. This idea is not taken from a cited source.

### Gaps
- No 2019–2026 source found applying SHAP specifically to NBM residuals for wind turbines (searched; none surfaced).

## Q7. Prediction intervals for power (quantile regression, conformal) and coverage

### Takeaway
Probabilistic NBMs and conformalised quantile regression appear in recent wind work, but mostly for RUL or power NBMs rather than temperature-residual alarms. Coverage should be evaluated as the empirical fraction of healthy test points inside the interval, per regime.

### Cited Findings
- A 2026 WES preprint uses a CNN NBM on component-temperature *residuals* as a feature extractor, then a convolutional AE for health indicators, then LSTM **conformalised quantile regression** for calibrated RUL intervals. — [WES preprint wes-2026-36 / 145](https://wes.copernicus.org/preprints/wes-2026-145/wes-2026-145.pdf)
- A probabilistic MLP NBM for wind-farm power with transfer learning and CUSUM alarms. — [Fiocchi, Ladopoulou, Dellaportas, arXiv 2404.16496](https://arxiv.org/pdf/2404.16496v2)
- In a comparison of 6 regressors for SCADA power prediction, gradient boosting was best (test R² 0.922, RMSE 0.101 normalised). — [Michigan Tech / IEEE AICT 2025](https://digitalcommons.mtu.edu/michigantech-p2/2564) (abstract-level)

### Inferences
- For power: fit LightGBM or HistGB quantile models (q = 0.05, 0.5, 0.95) on wind speed, direction, density proxy (ambient temperature) and pitch. Then apply split-conformal CQR on a held-out healthy month to correct coverage.
- Report empirical coverage (target 90%) and mean interval width per wind-speed bin.
- A power point below the lower bound, persisting under the counter rule, flags underperformance. Separate this from curtailment using pitch and status.

### Gaps
- No 2019–2026 peer-reviewed study found reporting conformal coverage for SCADA power NBMs specifically.

## Q8. Recent work 2024–2026: fleet and transfer NBMs, and the July 2026 Wiley generator two-stage paper

### Takeaway
Fleet and transfer approaches show that fine-tuning a pretrained NBM on little target-turbine data, or adjusting thresholds only, works. Pooling many source turbines did not beat a single good source for AEs. The requested July 2026 Wiley generator-temperature two-stage paper **could not be found**.

### Cited Findings
- Fraunhofer IEE (*Energy & AI* 2024): cross-turbine AE transfer. Adjusting only the threshold roughly matches the baseline, and fine-tuning the decoder or the full AE does better. Models trained on several source turbines did not beat one source turbine. — [arXiv 2404.03011](https://arxiv.org/html/2404.03011v1)
- Jonas & Meyer (arXiv 2504.17709, 2025) use CycleGAN mapping of a data-poor turbine's SCADA onto well-covered turbines to reuse an NBM, with improved detection across 7 turbines. — [arXiv 2504.17709](https://arxiv.org/abs/2504.17709v2)
- Jenkel, Jonas & Meyer (*Energies* 2023) studied federated fleet NBMs on bearing temperatures. A single global model can hurt when targets differ across turbines, so per-turbine fine-tuning is needed, and training time rose about 10×. — [Energies 2023 (DOAJ)](https://doaj.org/article/fe8b6b54c0a242089becc225bf830653)
- Ulmer et al. (PHM 2020): cross-turbine CNN training on 10-min SCADA. — [ZHAW](https://srv-clst-301-data66.zhaw.ch/bitstream/11475/20824/4/2020_Ulmer_Cross-Turbine_Training_of_Convolutional_Neural_Networks_PHM_Society.pdf)
- Lichtenstein et al. (*Wind Energy* 2025, we.2961) give an empirical-physics drive-train temperature NBM. It is the closest Wiley match found, on a turbine with a **two-stage** planetary gearbox, but it is not a "two-stage modelling" paper. — [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/we.2961)
- A 2026 bagging-ensemble gearbox-temperature model (*Energy Conversion and Management: X*, vol. 29) reportedly flagged anomalies 59 days and irregularities 9 days before failure. — [METU AVESIS](https://avesis.metu.edu.tr/yayin/0b8b27ba-7330-4f61-96ba-dd2936a9b6bc/intelligent-fault-detection-in-wind-turbine-gearboxes-a-machine-learning-approach-using-scada-temperature-data) (search-snippet level)
- The CARE to Compare dataset is at Zenodo v6, published 2025-07-09. — [Zenodo](https://zenodo.org/records/14006163/latest)

### Inferences
- Wind Farm A has only 5 turbines, so fleet-median correction is possible but fragile (no missing values allowed for fewer than 5 turbines under the Chesterman rule). Use the median of the other 4 and fall back to per-turbine NBM when data are missing.

### Gaps
- **The July 2026 Wiley paper on two-stage generator temperature modelling was not found** despite two targeted searches (standard and extended). It may be too recent to be indexed, or the title or journal may differ. The caller should obtain the DOI or title.
- No peer-reviewed CARE leaderboard beyond the original AE baseline was found.
