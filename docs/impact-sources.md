# Impact figures and their sources

Each figure below was checked against the primary source on 2026-10-18: the regulation text, the agency's own
table, the standard's publisher, a ministry page, or the paper itself. Figures are copied as printed. A
figure this file computes itself is labelled *computed*. Nothing here estimates reviewer time or time saved.
On real photos Drishyam's recall is about 0.2 (README, CODEBRIM section), which does not support any
time-saving claim.

| # | Figure | Status |
| --- | --- | --- |
| 1 | US National Bridge Inventory, 2026 data: 624,939 bridges, of which 41,148 are in poor condition | VERIFIED |
| 2 | 23 CFR 650.311: routine inspection at intervals not to exceed 24 months (12 or 48 under stated conditions) | VERIFIED |
| 3 | DIN 1076: main inspection every 6 years, a simple inspection 3 years after each one | VERIFIED (intervals); current edition is 2026-01, not 1999-11 |
| 4 | dacl10k paper: authorities miss inspection intervals because of staff shortages | VERIFIED |
| 5 | Drone bridge inspection: 2,536 still photos for one detailed inspection (St. Johns Bridge, Portland, OR) | VERIFIED |

## 1. US bridges and the share in poor condition

- **Figure:** All bridges 624,939; good 268,737; fair 315,054; poor 41,148 (the table's TOTALS row). NHS bridges
  in the same table: 147,243, of which 4,144 are poor.
- **Share in poor condition (computed here, not printed by FHWA):** 41,148 ÷ 624,939 = 6.584 %.
- **Source:** FHWA, Bridge Condition by Highway System 2026,
  https://www.fhwa.dot.gov/bridge/nbi/no10/condition26.cfm (linked from
  https://www.fhwa.dot.gov/bridge/nbi/condition.cfm).
- **Date:** the table is dated 8/4/2026; the page footer says "Updated: 09/02/2026".
- **Quote:** "TOTALS 624,939 268,737 315,054 41,148"
- **Notes:** FHWA classes a bridge as poor when the lowest of its deck, superstructure, substructure or culvert
  ratings is 4 or less (https://www.fhwa.dot.gov/bridge/britab.cfm). The 2025 table
  (https://www.fhwa.dot.gov/bridge/nbi/no10/condition25.cfm, dated 6/15/2025) gives 624,193 bridges and
  41,685 poor. Secondary sources (ARTBA, bridgelookup.com) print slightly different 2025 totals; cite FHWA.

## 2. US routine inspection interval: 23 CFR 650.311

- **Figure:** each bridge is inspected at regular intervals not to exceed 24 months. Bridges with a deck,
  superstructure, substructure or culvert rating, or a scour rating, of 3 or less: not to exceed 12 months.
  Bridges meeting every listed criterion, under a documented agency policy with FHWA notified in writing
  first: not to exceed 48 months. A risk-based method (Method 2, FHWA approval required) allows 12, 24, 48 or
  72 months.
- **Source:** eCFR, Title 23, § 650.311, fetched through the eCFR API:
  https://www.ecfr.gov/api/versioner/v1/full/2026-10-15/title-23.xml?part=650&section=650.311 (reader view:
  https://www.ecfr.gov/current/title-23/chapter-I/subchapter-G/part-650/subpart-C/section-650.311).
- **Date:** eCFR Title 23 is up to date as of 2026-10-15. The section was last amended on 2022-06-06 (reported
  by the research pass; not re-checked here).
- **Quote:** "Each bridge must be inspected at regular intervals not to exceed 24 months"

## 3. German inspection intervals: DIN 1076

- **Figure:** every bridge has a main inspection (Hauptprüfung) under DIN 1076 every 6 years, and a simple
  inspection (Einfache Prüfung) 3 years after each one; there are also yearly and half-yearly visual checks.
- **Source (intervals):** Federal Ministry of Transport (BMV), Zustandsnoten,
  https://www.bmv.de/SharedDocs/DE/Artikel/StB/zustandsnoten.html
- **Date:** 15.10.2024 (the page's date stamp). The page does not name an edition of the standard.
- **Quote:** "im Abstand von 6 Jahren einer Hauptprüfung nach DIN 1076"
- **Edition:** DIN 1076:2026-01, *Ingenieurbauwerke im Zuge von Straßen und Wegen - Prüfung und Überwachung*,
  is current and replaces DIN 1076:1999-11 (DIN Media, https://www.dinmedia.de/de/norm/din-1076/394559449).
  The 1999-11 edition is marked withdrawn (https://www.dinmedia.de/de/norm/din-1076/23474630).
- **UNVERIFIED:** whether the 2026-01 text keeps the 6-year and 3-year intervals. The standard is paywalled;
  DIN Media's list of changes names new sections on monitoring and imaging methods and no interval change,
  but that is not proof. Cite the BMV page for the intervals and do not call 1999-11 the current edition.
  Berlin still applies the 1999 edition by a 2026 notice
  (https://www.berlin.de/sen/uvk/_assets/verkehr/service/rechtsvorschriften/bautechnik/einfuehrung_din-1076_av-2026.pdf;
  reported by the research pass, not re-checked here).

## 4. Inspector shortage: the dacl10k paper

- **Figure:** no number. The Introduction says authorities often fail to keep up with the required inspection
  and restoration intervals because of staff shortages, budget limits and a time-consuming, analogue
  inspection process.
- **Source:** Flotzinger, Rösch and Braml, "dacl10k: Benchmark for Semantic Bridge Damage Segmentation",
  arXiv:2309.00460, https://arxiv.org/abs/2309.00460 (full text: https://arxiv.org/html/2309.00460),
  Section 1.
- **Date:** v1, 1 September 2023 (the only version).
- **Quote:** "keep up with the necessary inspection and restoration intervals due to staff shortages"
- **Note:** the abstract does not say this; only the Introduction does. The paper's "1 in 3 U.S. bridges"
  line cites the ARTBA 2023 bridge report, an industry association, so do not present it as a government
  statistic.

## 5. Images from one drone bridge inspection

- **Figure:** 2,536 high-resolution still photos for the detailed UAS inspection of the St. Johns Bridge,
  Portland, Oregon (bridge 06497), flown April 17-21, 2017. Other bridges in the same table: 197 to 401 photos.
  For storage planning the report assumes 675 photos per project, the average over the 6 bridges flown.
- **Source:** Gillins et al., *Eyes in the Sky: Bridge Inspections with Unmanned Aerial Vehicles*, Oregon DOT
  and FHWA, report FHWA-OR-RD-18-11 (SPR 787),
  https://www.oregon.gov/odot/Programs/ResearchDocuments/SPR787_Eyes_in_the_Sky.pdf, Table 4.1 and
  Section 5.3.2.
- **Date:** February 2018.
- **Quote:** "the total number of images acquired for each project is actually much higher"
- **Note:** the counts exclude thermal images, navigation-camera images and video. Supporting federal source
  (not re-checked here): FHWA-HRT-21-086, September 2021,
  https://www.fhwa.dot.gov/publications/research/infrastructure/structures/bridge/21086/21086.pdf, which says
  a whole-bridge point cloud may need thousands of images.
