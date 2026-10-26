# Commercial wind-farm condition monitoring, performance and digital-twin product UIs

Scope note: these notes come from about 16 searches and fetches done on 2026-10-26. Vendor pages rarely describe UIs in detail, and most claims below are vendor-reported marketing. Gated documentation, demo videos and screenshots could not be inspected (several fetches returned a PDF binary, 403 or 404). Anything not tied to a URL is under Inferences or Gaps.

## Q1. Which screens exist per product, and what health/criticality scales, colors and status vocabularies do they use?

### Takeaway
The most clearly documented screen set is Bazefield's: a fleet map and table with live status and alarm counts, drill-down from fleet to region, site and asset, live alarms and events with triage, an IEC 61400-26 availability module and planner, loss Pareto with financial impact, case management, and CMMS work orders. Other vendors cover a subset of this. CMS-focused vendors (ONYX, Siemens Gamesa, Vestas, GE SmartSignal) build around alerts, cases and reports. None of the public sources gave an exact health-scale definition or color palette. Red/amber/green status is implied but not documented.

### Cited Findings
**Bazefield (ownership: Envision bought BazeField in 2016. A 2026 profile says it is now developed by Univers, Envision's AIoT spin-out. I found no evidence of Enverus or DNV ownership.)**
- Six product pillars: Data Platform (APIs, alarm and notification services), Data Engine (SCADA/IoT ingress, time series, KPI calculation), Operations Management (fleet and event operations, Notification Hub, Availability Module and Planner, Case Management, Work Order Management), Performance Management (performance dashboards, loss analysis), Reporting & BI (standard reports, BI Manager, scheduled distribution), Administration — [Bazefield product page](https://product.bazefield.com/core.html)
- FleetView overview is a "live map and table of all sites with real-time status indicators, active alarm counts, and headline production KPIs", with drill-down from fleet to region, site and asset — [Bazefield](https://product.bazefield.com/core.html)
- Live Alarms & Events, plus Event Operations for investigation and triage. Filters are severity, domain and site — [Bazefield](https://product.bazefield.com/core.html)
- Availability Module tracks outages and allocation to IEC 61400-26, GADS or custom standards. Availability Planner covers planned and unplanned outages — [Bazefield](https://product.bazefield.com/core.html)
- Production-loss computation with financial impact per event, plus Pareto analysis by loss category. Trend analysis with SCADA overlay. Mobile app. CMMS integration (SAP PM, Maximo, IFS) as an add-on — [Bazefield](https://product.bazefield.com/core.html)
- Envision acquisition of BazeField (2016) — [elEconomista](https://www.eleconomista.es/empresas-finanzas/amp/7537075/Envision-Energy-to-Acquire-BazeField). Described as developed by Univers — [Net Zero Compare](https://netzerocompare.com/software/bazefield-univers). BazeField system was licensed into 3E SynaptiQ Wind — [Windtech](https://www.windtech-international.com/company-news/3e-partners-with-baze-to-deliver-intelligent-wind-farm-monitoring-and-operations-software)

**ONYX Insight fleetMONITOR**
- "The cloud-based condition monitoring app inside ONYX Cortex". One interface for the whole fleet, with user-configurable alarming and processing — [ONYX fleetMONITOR](https://www.onyxinsight.com/fleetmonitor/)
- Covers main bearing, generator, gearbox, blades, main shaft, foundations, towers, and yaw and pitch bearings. Fuses vibration, SCADA, oil particulate and maintenance records. Uses "failure-mode-specific algorithms" from 180+ turbine platforms — [ONYX](https://www.onyxinsight.com/fleetmonitor/)
- Alerts carry "alert context, root cause insights, and issue tracking". "Full case history, KPI dashboards and live insights when bundled with MANAGE". Site reports generated on demand. Claims "up to 24 months' early warning" — [ONYX](https://www.onyxinsight.com/fleetmonitor/)
- Service models: in-house, hybrid or full-service monitoring — [ONYX](https://www.onyxinsight.com/fleetmonitor/)
- The page does not document a severity scale or colors — [ONYX](https://www.onyxinsight.com/fleetmonitor/)

**Power Factors (Greenbyte Breeze plus Drive, merged into Unity)**
- Unity launched in September 2023 as one platform for mixed renewable fleets (wind, solar, BESS, hydro) — [Power Factors](https://powerfactors.com/?p=5335)
- Unity APM and AI Insights (October 2024) "integrate the best capabilities from... Drive and Greenbyte". Includes automatic data backfill and cleaned, validated data — [NA Clean Energy](https://www.nacleanenergy.com/alternative-energies/power-factors-launches-next-generation-ai-powered-asset-performance-management-application-on-unity-platform) (summary from a search snippet. The fetch returned 403.)
- Drive links operational events to work orders, inventory and contracts (bp Wind, 1,600 MW) — [Power Factors](https://www.powerfactors.com/blog/bp-wind-energy-selects-power-factors)
- Greenbyte user ideas forum has categories "dashboards and widgets", "calculations and metrics" and "availability contracts". Users request a lost-production-per-category graph with configurable category count, manual lost-production entries for long shutdowns, and control over warning codes that zero out availability — [Greenbyte Ideas](https://influence.powerfactors.app/forums/918697-ideas/category/352648-dashboards-and-widgets/filters/new?page=2); [Greenbyte Ideas](https://influence.powerfactors.app/forums/918697-greenbyte-ideas/filters/top?page=7); [Greenbyte Ideas](https://influence.powerfactors.app/forums/918697-greenbyte-ideas/category/352669-calculations-and-metrics/filters/hot?page=2)
- Breeze had a "Manager's Dashboard" with farm-level and financial reporting — [Windpower Engineering](https://www.windpowerengineering.com/?p=19149)

**GE Vernova (APM / SmartSignal; earlier "Digital Wind Farm" branding)**
- SmartSignal compares live sensor data against the model's prediction. A widening gap between predicted and actual (the residual) flagged an early wind main-bearing fault, a case GE values at $14M avoided — [GE Vernova blog](https://gevernova.com/software/blog/early-bearing-fault-detection-wind-turbine-prevents-14M-failure)
- 370+ prebuilt digital-twin blueprints. The 2024 brochure says 340+, so the two figures conflict. Offers "diagnostic analytics and rules that indicate probable cause", "time-to-action forecasting" that estimates when an alarm limit will be reached, and an "Analysis, case management & more" toolset. Deployable on-prem, as a cloud subscription, as a turnkey service, or inside APM — [GE Vernova SmartSignal brochure](https://gevernova.com/software/sites/default/files/2024-10/brochure-ai-powered-predictive-analytics-smartsignal.pdf) (content from a search snippet. The PDF text was not extractable.)
- APM Health plus SmartSignal enables work-order management and time-to-failure forecasting for renewables — [GE Vernova APM for renewables](https://gevernova.com/software/blog/apm-for-renewables-ge-vernova-leads-the-way)
- Industrial Managed Services runs SmartSignal on 7,000+ assets — [GE Vernova](https://gevernova.com/software/customer-stories/industrial-managed-services-predictive-analytics-customer-assets)

**Siemens Gamesa Diagnostic Center (Brande)**
- 130+ experts monitor more than 10,000 connected turbines with about 300 sensors per turbine. Around 250,000 deviations a year are filtered down to about 8,000 site visits. Failures predicted "up to three years in advance" — [SGRE Remote Services](https://siemensgamesa.com/products-and-services/service-wind/diagnostics)
- Vibration diagnostics example: 27-day warning on a high-speed pinion crack. About 70% of cases fixed remotely — [SGRE vibration diagnostics PDF](https://www.siemensgamesa.com:443/-/media/siemensgamesa/downloads/en/products-and-services/services/asset-optimization/vibration-diagnostics.pdf)
- Pythia platform with a reported 99% detection hit rate (vendor figure) — [SGRE offshore remote diagnostics](https://www.siemensgamesa.com/explore/innovations/offshore-service-logistics-part-1)

**Vestas**
- The Vestas CMS can identify service needs "up to 6 months in advance". The Remote Operations Centre issues monthly status reports and alarm reports through the VestasOnline portal — [Vestas CMS brochure](https://vestas.com/content/dam/vestas-com/global/en/brochures/Condition_Monitoring_System.pdf.coredownload.inline.pdf)

**XMPro (wind digital twin)**
- Multi-site dashboard showing turbine performance, weather and energy output. Map with color-coded status icons and a summary of active recommendations. In the turbine twin view, clicking a flagged section opens error details and linked recommendations — [XMPro multi-location windfarm](https://xmpro.com/solutions-library/intelligent-multi-location-windfarm-management/); [XMPro wind PdM](https://xmpro.com/solutions-library/renewables,use-cases/predictive-maintenance-for-wind-turbines/)
- ML scores for yaw error, efficiency, asset health and utilisation, plus failure likelihood "with confidence scoring". Ranked alerts are routed to the right responder. Operators can simulate alternative settings on a per-turbine twin before applying them — [XMPro](https://xmpro.com/solutions-library/intelligent-multi-location-windfarm-management/)
- The wind blueprint uses DTDL models based on IEC 61400-25 — [XMPro Blueprints](https://xmpro.com/blueprints)
- Recommendation alerts have discussion threads kept in the audit trail, custom forms, and work-order creation in SAP EAM — [CIO Influence](https://cioinfluence.com/itechnology-series-news/xmpro-extends-prescriptive-recommendations-and-includes-azure-digital-twins-integration-in-latest-version/)
- Basis for the Digital Twin Consortium's "Wind Farms Remote Operations Center" showcase — [DTC](https://www.digitaltwinconsortium.org/initiatives/technology-showcase/wind-farms-remote-operations-center/)

**Clir Renewables**
- AI temperature-anomaly detector (around 2019) learns a per-turbine behavior model that accounts for ambient temperature, rotor speed, ramping and seasonality. It raises events when temperatures leave the "expected probabilistic range", grades them at "multiple grades of severity", and creates follow-up actions. The CEO stressed "limited false positives — so the tool isn't wasting people's time" — [NAWindpower](https://nawindpower.com/clir-ai-software-detects-turbine-anomalies); [Windpower Engineering](https://www.windpowerengineering.com/?p=44261)
- Platform anomaly detection covers power-curve deviation, dynamic yaw, pitch, derates, icing and component health. Also yaw-misalignment and leading-edge-erosion detectors — [Clir search snippet of clir.eco/wind](https://www.clir.eco/wind) (the page returned 404 on fetch); [NAWindpower LEE](https://nawindpower.com/new-clir-software-detects-leading-edge-erosion-on-wind-blades)

**Avathon (formerly SparkCognition, rebranded October/November 2024)**
- Rebrand confirmed — [Avathon press release](https://avathon.com/press-release/avathon-aims-to-triple-workforce-in-india-within-24-months/); [Solar Power World](https://www.solarpowerworldonline.com/2024/10/sparkcognition-changes-name-to-avathon/). Use cases include ML yaw-misalignment detection and automated power-curve monitoring across fleets — [Avathon yaw](https://avathon.com/resources/case-study-identifying-yaw-misalignment/); [Avathon power curve](https://avathon.com/resources/use-case-automated-power-curve-monitoring-optimal-turbine-performance). PV Tech reports an "autonomous AI managing platform for renewables" — [PV Tech](https://pv-tech.org/avathon-launches-autonomous-ai-managing-platform-for-renewables/)

**i4SEE and Nispera**
- i4SEE Performance auto-detects curtailed operation, checks it against site or turbine curtailment rules, and flags unexpected losses — [Windtech](https://www.windtech-international.com/product-news/i4see-releases-new-version-of-i4see-performance). i4See Align diagnoses yaw misalignment, yaw control errors, yaw damage and northing errors fleet-wide each month — [Windtech](https://www.windtech-international.com/product-news/i4see-releases-new-product-to-analyse-alignment-on-fleet-scale)
- Nispera worked with ZHAW (2018–2021) on SCADA plus error-log ML fault detection that generalizes across OEMs — [ZHAW](https://zhaw.ch/en/research/project/71587)

### Inferences
- A credible product inventory, based on the Bazefield and XMPro descriptions: (1) Fleet overview as a map plus table with status, active-alarm counts and production KPIs. (2) Site/turbine drill-down. (3) Alarms/events triage with severity, domain and site filters. (4) Case management linked to (5) work orders. (6) Availability per IEC 61400-26 with a planner. (7) Loss Pareto with money values. (8) Reports. (9) Admin/config.
- The XMPro pattern of a clickable turbine schematic, where clicking a flagged component opens its error details and recommendation, is a good "digital twin" view for a 5-turbine prototype.
- CMS vendors (ONYX, SGRE, Vestas) treat the human analyst and the report as part of the product. A "monthly health report" export is expected.

### Gaps
- I found no public definitions of health-score scales, severity level names or color palettes for ONYX, Bazefield, Power Factors, GE, SGRE or Vestas. Screenshots and demos were not accessible.
- Uptake (wind), Cognite (wind), and TurbineAI or Nispera current UIs were not covered within budget. Nispera's commercial status in 2026 was not verified.
- Power Factors' Unity APM press release could not be fetched (403), so its details rest on search snippets.

## Q2. How do products present an early-warning "case"?

### Takeaway
Across vendors, a case is built from a residual (expected vs actual), a probable-cause or failure-mode diagnosis, a severity grade, a time-to-action or lead-time estimate, and a recommended action. It is tracked through an audit trail into a work order. I found no published lifecycle state names, and no explicit "confidence" except XMPro's.

### Cited Findings
- GE SmartSignal: model-vs-actual residual, then diagnostic rules giving "probable cause", then "time-to-action forecasting" (when the alarm limit will be reached), then case management and work orders — [GE brochure](https://gevernova.com/software/sites/default/files/2024-10/brochure-ai-powered-predictive-analytics-smartsignal.pdf); [GE blog](https://gevernova.com/software/blog/early-bearing-fault-detection-wind-turbine-prevents-14M-failure)
- ONYX: alert context, root-cause insights, issue tracking and full case history — [ONYX](https://www.onyxinsight.com/fleetmonitor/)
- XMPro: failure likelihood with confidence score, ranked alerts, responder routing, discussion in the audit trail, then a work order in SAP — [XMPro](https://xmpro.com/solutions-library/intelligent-multi-location-windfarm-management/); [CIO Influence](https://cioinfluence.com/itechnology-series-news/xmpro-extends-prescriptive-recommendations-and-includes-azure-digital-twins-integration-in-latest-version/)
- Clir: probabilistic expected range, multiple severity grades, and auto-created follow-up actions — [NAWindpower](https://nawindpower.com/clir-ai-software-detects-turbine-anomalies)
- Siemens Gamesa funnels about 250,000 deviations a year to about 8,000 visits, with about 70% remote fix rate. Human experts triage before dispatch — [SGRE](https://siemensgamesa.com/products-and-services/service-wind/diagnostics); [SGRE PDF](https://www.siemensgamesa.com:443/-/media/siemensgamesa/downloads/en/products-and-services/services/asset-optimization/vibration-diagnostics.pdf)
- Vendor lead-time claims: ONYX up to 24 months, SGRE up to 3 years (with a 27-day pinion-crack example), Vestas up to 6 months, Delfos 3 months on a Vestas V100 bearing — [ONYX](https://www.onyxinsight.com/fleetmonitor/); [SGRE](https://siemensgamesa.com/products-and-services/service-wind/diagnostics); [Vestas](https://vestas.com/content/dam/vestas-com/global/en/brochures/Condition_Monitoring_System.pdf.coredownload.inline.pdf); [casestudies.com](https://www.casestudies.com/company/delfos-energy/case-study/vestas-prevents-bearing-loss-with-delfos-after-3-months-of-early-warning)

### Inferences
- A case card for the prototype should contain: component and failure mode, severity, first-detected timestamp, lead time so far, projected time-to-threshold, evidence (residual chart with expected band, contributing signals), confidence, recommended action, owner, status history and comments.
- A lifecycle such as New → Acknowledged → Monitoring → Action planned (work order) → Closed (confirmed / false alarm / unknown) is consistent with these features, but it is my synthesis. No vendor's state names were found.
- Lead-time claims are vendor-reported, undefined (no definition of start and end), and not independently validated. That opens room for a product that measures lead time transparently.

### Gaps
- No public source gives exact case status names, acknowledgement semantics or feedback-labeling UI for any vendor.

## Q3. Common KPIs

### Takeaway
Products converge on production, time-based and energy-based availability (IEC 61400-26 information categories with priority rules), lost production by category (Pareto, money value), performance ratio and capacity factor. Curtailment-aware loss accounting is a known pain point.

### Cited Findings
- Bazefield KPIs: production, availability, performance ratio, capacity factor, losses with financial impact per event, and daily site/asset aggregates — [Bazefield](https://product.bazefield.com/core.html)
- IEC 61400-26-1:2019 defines an information model to derive time-based and production-based availability. Information categories with entry and exit criteria allocate time and production. Category priority resolves concurrent states. It replaces TS 61400-26-1/-2/-3 — [standard.no](https://online.standard.no/en/iec-61400-26-1-2019-ed1); [VDE preview](https://www.vde-verlag.de/iec-normen/preview-pdf/info_iec61400-26-1{ed1.0}b.pdf)
- Greenbyte users report that potential-power methods overestimate lost production under grid curtailment, and that warning codes can wrongly zero availability — [Greenbyte Ideas](https://influence.powerfactors.app/forums/918697-greenbyte-ideas/filters/top?page=7)
- i4SEE flags unexpected losses against curtailment rules — [Windtech](https://www.windtech-international.com/product-news/i4see-releases-new-version-of-i4see-performance)
- Power-curve deviation and yaw misalignment are standard performance detectors (Clir, Avathon, i4SEE) — [Avathon](https://avathon.com/resources/use-case-automated-power-curve-monitoring-optimal-turbine-performance); [Windtech i4See Align](https://www.windtech-international.com/product-news/i4see-releases-new-product-to-analyse-alignment-on-fleet-scale)

### Inferences
- Prototype KPI strip: energy produced, time-based availability, energy-based availability, lost energy (MWh and currency) by category, power-curve deviation per turbine, capacity factor, and open cases by severity.

### Gaps
- Exact IEC 61400-26 category names (for example "IAOGOOD", forced outage) and formulas were not retrievable. The standard is paywalled.

## Q4. Visual and UX conventions

### Takeaway
The public evidence supports map plus table fleet overviews with status-colored icons, drill-down hierarchies, filterable alarm lists, Pareto loss charts, trend charts with SCADA overlay, and residual/expected-band charts. Evidence on dark versus light themes was not found.

### Cited Findings
- Map plus table with real-time status indicators and alarm counts. Fleet → region → site → asset drill-down — [Bazefield](https://product.bazefield.com/core.html)
- Map with color-coded status icons. Clickable 3D or schematic turbine model — [XMPro](https://xmpro.com/solutions-library/renewables,use-cases/predictive-maintenance-for-wind-turbines/)
- Users want configurable dashboards and widgets (a Greenbyte forum category) — [Greenbyte Ideas](https://influence.powerfactors.app/forums/918697-ideas/category/352648-dashboards-and-widgets/filters/new?page=2)
- GE Vernova brand palette "Evergreen" RGB(0,94,96) and the Inter font appear in the SmartSignal brochure metadata. This is brand material, not a UI spec — [GE brochure](https://gevernova.com/software/sites/default/files/2024-10/brochure-ai-powered-predictive-analytics-smartsignal.pdf)

### Inferences
- Use a dense, light analytics theme for engineers and an optional dark control-room theme for the live overview. Status colors should be reserved for severity only. Pair the map with a sortable table, since 5 turbines fit easily in a table.

### Gaps
- No source confirmed theme defaults, density or timeline/Gantt views for specific vendors.

## Q5. Gaps and differentiators a new product could own

### Takeaway
Vendors market lead time and hit rate but publish no definitions or validation, and they rarely show false-alarm cost. A product that shows its evaluation metrics, explains each alarm, supports replay and fault injection, and treats "unknown" as a legitimate diagnosis would stand out.

### Cited Findings
- Lead-time claims vary widely (6 months to 3 years) with no measurement definitions — [ONYX](https://www.onyxinsight.com/fleetmonitor/); [SGRE](https://siemensgamesa.com/products-and-services/service-wind/diagnostics); [Vestas](https://vestas.com/content/dam/vestas-com/global/en/brochures/Condition_Monitoring_System.pdf.coredownload.inline.pdf)
- Alarm fatigue is a recognized issue: SGRE filters about 250,000 deviations a year to about 8,000 actions, and Clir says it designed for "limited false positives" — [SGRE](https://siemensgamesa.com/products-and-services/service-wind/diagnostics); [NAWindpower](https://nawindpower.com/clir-ai-software-detects-turbine-anomalies)
- XMPro offers what-if simulation on turbine twins, the closest analogue to scenario and fault injection found — [XMPro](https://xmpro.com/solutions-library/intelligent-multi-location-windfarm-management/)
- Users of loss-accounting tools want more transparency and control (manual entries, potential-power method choice) — [Greenbyte Ideas](https://influence.powerfactors.app/forums/918697-greenbyte-ideas/category/352669-calculations-and-metrics/filters/hot?page=2)

### Inferences
- Candidate differentiators: (a) a "Model evaluation" screen with per-fault detection rate, lead-time distribution, false alarms per turbine-month and alarm-to-action ratio. (b) Explainable cases with contributing-signal residual ranking. (c) Historical replay with a time scrubber and synthetic fault injection, labeled clearly as simulation. (d) An "Unknown / anomaly not matched to failure mode" diagnosis class. (e) Feedback labels (true positive / false positive / unknown) that feed the metrics.

### Gaps
- I found no source showing that any vendor publicly exposes per-model precision, lead-time distributions or alarm-fatigue metrics in the UI. Absence of evidence is not proof of absence.
