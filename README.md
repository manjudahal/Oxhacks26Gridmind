# ⚡ GridMind
**Decides every 30 minutes when a grid battery should charge, discharge or hold, from live GB electricity market data
(Elexon), and explains why.** Built at OxHack 26 for the HRT challenge: Best use of Data for Predictions & Decision Making.

Live app: (paste your Streamlit link here)

## Results on 78 unseen days (real Elexon data, 07 Aug to 23 Oct 2026)
- Simulated margin GBP 543,591 vs GBP 119,263 for a fixed timetable (+356%); market-price forecast with the same optimiser: GBP 494,245
- 32% of the perfect-hindsight maximum; losing days 21% vs 42% for the fixed timetable
- Edge over the fixed timetable: GBP +5,440/day (95% block-bootstrap interval GBP 3,452 to 7,094)
- The 80% forecast range held 77% of actual prices; next-half-hour model chosen on validation: Persistence (latest known price)

Margins are simulated at Elexon imbalance (settlement) prices: a research prototype, not achievable revenue.
A real battery would need market access, dispatch and settlement integration.

## How it works
1. Real Elexon data (imbalance prices, fuel mix, market price, wind and demand forecasts), cleaned without look-ahead
2. Forecasts: 5 next-half-hour models compared walk-forward and chosen by validation profit, plus a multi-horizon LightGBM for the next 22 hours
3. Decision: a mixed-integer optimiser plans 22 hours for a 50 MW / 100 MWh battery (88% efficiency, wear cost, no simultaneous charge and discharge), acts on the first half-hour, re-plans
4. Proof: backtest on unseen months, ablation, block bootstrap, robustness, conservative mode; explanations with TreeSHAP

## 🤖 Gemini Copilot (Google Gemini API, `google-genai`)
GridMind's Python models make every forecast and the optimiser makes every decision; Gemini explains and explores them:
- **Decision briefing:** turns the current decision, its forecast range, the optimiser's 22-hour plan and the TreeSHAP drivers into a plain-English briefing.
- **What-if:** change the battery level; GridMind's optimiser recomputes the decision and Gemini explains why it changed.
- **Ask the Copilot with function calling:** Gemini can call four GridMind tools (what-if on the current decision, replay any test day with your own battery settings and risk mode, the backtest results, an overview of the test days) and answers with the numbers they return.
- **Grounded:** a system prompt restricts Gemini to provided facts and tool results, and labels all money as simulated margins.

Setup: add `GEMINI_API_KEY` in Streamlit Cloud (app → Settings → Secrets) or `.streamlit/secrets.toml` locally (git-ignored). Without a key, the rest of GridMind works unchanged.

## Files
- `streamlit_app.py`: the web app (Live decision, Gemini Copilot, Replay a day, Decision Lab, Results, Explainability, How it works)
- `gridmind_core.py`: data download, cleaning, features, models and the optimiser (exported from our notebook)
- `gridmind_model.joblib`: trained models and the settings chosen on validation months
- `gridmind_results.json`, `gridmind_replay.csv`, `gridmind_paths.npz`, `gridmind_snapshot.json`: results, replay data, planning paths, last live decision

## Run locally
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Data: Elexon Insights Solution (BMRS), used under the BMRS Data Licence.
