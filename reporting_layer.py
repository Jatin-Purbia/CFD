"""
reporting_layer.py
──────────────────
Converts raw numerical surrogate output into a structured technical report
using LangChain's prompt-template machinery.

Design decisions
────────────────
• A custom  ``PlaceholderLLM``  class implements the LangChain  ``BaseLLM``
  interface so the entire pipeline can be demonstrated *without* an API key.
  It performs deterministic, rule-based report generation from the metrics dict.
  Swapping it for ``ChatOpenAI`` or ``ChatAnthropicAI`` is a one-line change
  (see ``build_report_chain``).

• The prompt template uses clearly labelled XML-style sections so that a real
  LLM can be instructed to honour the same structure.

• ``generate_report`` is the single public entry-point consumed by the
  Streamlit dashboard.
"""

from typing import Any, Dict, List, Optional, Sequence
import os
import pathlib
import re
import json
import logging
from urllib import request as _urllib_request
from urllib.error import URLError

# Try to import LangChain pieces; provide lightweight local fallbacks
# so the dashboard can run without LangChain being installed.
try:
  from langchain.llms.base import BaseLLM
  from langchain.schema import Generation, LLMResult
  from langchain.prompts import PromptTemplate
  from langchain.chains import LLMChain
  _HAS_LANGCHAIN = True
except Exception:
  _HAS_LANGCHAIN = False

  class BaseLLM:
    pass

  class Generation:
    def __init__(self, text: str):
      self.text = text

  class LLMResult:
    def __init__(self, generations: List[List[Generation]]):
      self.generations = generations

  class PromptTemplate:
    def __init__(self, input_variables: List[str], template: str):
      self.input_variables = input_variables
      self.template = template

    def format(self, **kwargs: Any) -> str:
      return self.template.format(**kwargs)

  class LLMChain:
    def __init__(self, llm: BaseLLM, prompt: PromptTemplate):
      self.llm = llm
      self.prompt = prompt

    def run(self, **kwargs: Any) -> str:
      # Format the template and pass to the LLM's synchronous generate
      prompt_text = self.prompt.format(**kwargs)
      # Some LLM implementations expose _generate; use that signature
      result = self.llm._generate([prompt_text])
      # Extract the first generated text
      try:
        return result.generations[0][0].text
      except Exception:
        return ""


# ─────────────────────────────────────────────────────────
# Load .env (simple parser) and expose API key in environ
# ─────────────────────────────────────────────────────────
def _load_dotenv(dotenv_path: Optional[str] = None) -> None:
  """Minimal .env loader: parses KEY = VALUE lines and exports to os.environ.

  Normalises keys to uppercase and replaces non-alphanumerics with '_'.
  """
  if dotenv_path is None:
    dotenv_path = pathlib.Path.cwd() / ".env"
  else:
    dotenv_path = pathlib.Path(dotenv_path)

  if not dotenv_path.exists():
    return

  for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
      continue
    if "=" not in line:
      continue
    key, val = line.split("=", 1)
    key = key.strip()
    val = val.strip().strip('"').strip("'")
    # Normalise key: remove spaces, replace non-alnum with underscore, uppercase
    key_norm = re.sub(r"[^0-9A-Za-z]", "_", key).upper()
    os.environ[key_norm] = val


# Load .env at module import time so downstream libraries can read the key
_load_dotenv()

# Convenience names for common patterns
GORQ_API_KEY = os.environ.get("GORQ_API") or os.environ.get("GORQ_API_KEY") or os.environ.get("GORQ_APIKEY")
GORQ_API_URL = os.environ.get("GORQ_API_URL")
GORQ_MODEL = os.environ.get("GORQ_MODEL") or "llama-3.3-70b-versatile"

# Configure module logger
logger = logging.getLogger("reporting_layer")
if not logger.handlers:
  ch = logging.StreamHandler()
  fmt = logging.Formatter("[reporting_layer] %(levelname)s: %(message)s")
  ch.setFormatter(fmt)
  logger.addHandler(ch)
level = os.environ.get("REPORTING_LOG_LEVEL", "INFO").upper()
try:
  logger.setLevel(getattr(logging, level))
except Exception:
  logger.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────
# 1.  Placeholder LLM  (fully offline, deterministic)
# ─────────────────────────────────────────────────────────
class PlaceholderLLM(BaseLLM):
    """
    A drop-in LangChain LLM that generates a structured technical report
    purely from the metrics embedded in the prompt.  No network call.

    To swap for a real model:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
        chain = LLMChain(llm=llm, prompt=REPORT_TEMPLATE)
    """

    @property
    def _llm_type(self) -> str:
        return "placeholder-deterministic"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._build_report(prompt)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    async def _agenerate(self, prompts, stop=None, **kwargs):
        return self._generate(prompts, stop, **kwargs)

    # ── core report logic ─────────────────────────────────
    @staticmethod
    def _build_report(prompt: str) -> str:
        """
        Parse key metrics out of the prompt text (injected by the template)
        and assemble a professional report.  Falls back to generic text when
        a value is missing.
        """

        def _extract(label: str, default: str = "N/A") -> str:
            """Pull the first value after ``label:`` in the prompt."""
            for line in prompt.split("\n"):
                if label in line:
                    parts = line.split(label, 1)
                    if len(parts) > 1:
                        return parts[1].strip().rstrip(".")
            return default

        # ── Pull values ───────────────────────────────────
        T_a       = _extract("Temperature A:")
        T_b       = _extract("Temperature B:")
        u_a       = _extract("Velocity A:")
        u_b       = _extract("Velocity B:")
        max_a     = _extract("Max Concentration A:")
        max_b     = _extract("Max Concentration B:")
        mean_a    = _extract("Mean Concentration A:")
        mean_b    = _extract("Mean Concentration B:")
        ratio     = _extract("Concentration Ratio (B/A):")
        spread_a  = _extract("Plume Spread A:")
        spread_b  = _extract("Plume Spread B:")
        l2_diff   = _extract("Spatial L2 Difference:")
        uncert_a  = _extract("Model Uncertainty A:")
        uncert_b  = _extract("Model Uncertainty B:")
        rf_r2     = _extract("RF R²:")
        mlp_r2    = _extract("MLP R²:")

        # ── Derive qualitative statements ─────────────────
        try:
            ratio_f = float(ratio)
            if ratio_f > 1.3:
                temp_effect = ("Condition B produced significantly higher peak "
                               "concentrations, indicating that the combined "
                               "effect of elevated temperature and/or altered "
                               "velocity substantially increases local NH₃ "
                               "accumulation.")
            elif ratio_f < 0.7:
                temp_effect = ("Condition B resulted in notably lower peak "
                               "concentrations.  Increased advective transport "
                               "or enhanced diffusion disperses the plume more "
                               "effectively under these parameters.")
            else:
                temp_effect = ("Peak concentrations remained relatively stable "
                               "between conditions, suggesting that the "
                               "parameter changes did not drastically alter the "
                               "dominant transport mechanism.")
        except (ValueError, TypeError):
            temp_effect = ("Qualitative comparison could not be computed.")

        # ── Uncertainty commentary ────────────────────────
        try:
            ua = float(uncert_a)
            ub = float(uncert_b)
            max_u = max(ua, ub)
            if max_u < 0.01:
                uncert_note = ("Ensemble disagreement between RF and MLP "
                               "surrogates is low across both conditions, "
                               "indicating high prediction confidence.")
            else:
                uncert_note = ("Non-trivial disagreement between the RF and "
                               "MLP surrogates is observed.  Results in the "
                               "high-uncertainty region should be validated "
                               "against a full CFD solve.")
        except (ValueError, TypeError):
            uncert_note = "Uncertainty analysis not available."

        # ── Assemble report ──────────────────────────────
        report = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║          NEURAL CFD SURROGATE – TECHNICAL ANALYSIS REPORT               ║
║          NH₃ Advection-Diffusion Comparative Study                      ║
╚══════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. SIMULATION CONDITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────┬──────────────────┬──────────────────┐
  │  Parameter          │  Condition A     │  Condition B     │
  ├─────────────────────┼──────────────────┼──────────────────┤
  │  Temperature (K)    │  {T_a:<16} │  {T_b:<16} │
  │  Velocity (m/s)     │  {u_a:<16} │  {u_b:<16} │
  │  Max Conc (mol/m³)  │  {max_a:<16} │  {max_b:<16} │
  │  Mean Conc (mol/m³) │  {mean_a:<16} │  {mean_b:<16} │
  │  Plume Spread       │  {spread_a:<16} │  {spread_b:<16} │
  └─────────────────────┴──────────────────┴──────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  2. COMPARATIVE ANALYSIS – Effect of Temperature & Velocity on NH₃
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Concentration Ratio (B / A) : {ratio}
  Spatial L2 Difference        : {l2_diff}

  {temp_effect}

  The plume spread metric (fraction of domain above 10 % of peak) shifted
  from {spread_a} (Condition A) to {spread_b} (Condition B).  A larger
  spread indicates more effective atmospheric dispersion, which reduces
  localised exposure risk but extends the hazard perimeter.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3. MODEL CONFIDENCE & UNCERTAINTY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Surrogate Performance (test-set):
    • Random Forest  R² = {rf_r2}
    • MLP            R² = {mlp_r2}

  Ensemble Uncertainty (mean |RF − MLP|):
    • Condition A : {uncert_a}
    • Condition B : {uncert_b}

  {uncert_note}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  4. ENGINEERING RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Ventilation design should account for the plume-spread ratio observed
    between conditions to size extraction systems appropriately.
  • Temperature excursions above 350 K warrant full-resolution CFD
    validation before facility commissioning.
  • Consider adding boundary-layer turbulence models (k-ε) for industrial
    environments where Reynolds numbers exceed 10⁵.

"""
        return report


# ─────────────────────────────────────────────────────────
# Optional GorqLLM: calls an HTTP endpoint using the API key if configured
# ─────────────────────────────────────────────────────────
class GorqLLM(BaseLLM):
    """Light wrapper that forwards the prompt to a simple HTTP API.

    The URL should be provided via the `GORQ_API_URL` environment variable.
    The API key is taken from `GORQ_API_KEY`. If either is missing, this
    falls back to the `PlaceholderLLM` behaviour.
    """

    @property
    def _llm_type(self) -> str:
        return "gorq-http"

    def _post_prompt(self, prompt: str) -> str:
        if not GORQ_API_KEY or not GORQ_API_URL:
              # No remote API configured
              logger.info("GORQ API key present=%s, URL present=%s; using PlaceholderLLM fallback",
              bool(GORQ_API_KEY), bool(GORQ_API_URL))
              return PlaceholderLLM._build_report(prompt)

        payload = json.dumps({"prompt": prompt, "model": GORQ_MODEL}).encode("utf-8")
        req = _urllib_request.Request(
            GORQ_API_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GORQ_API_KEY}",
            },
            method="POST",
        )

        logger.info("Attempting Gorq request to %s using model %s (key present: %s)",
              GORQ_API_URL, GORQ_MODEL, bool(GORQ_API_KEY))
        try:
          with _urllib_request.urlopen(req, timeout=15) as resp:
            status = getattr(resp, 'getcode', lambda: None)()
            body = resp.read().decode("utf-8")
            logger.info("Gorq response status=%s, body_len=%d", status, len(body))
            try:
              js = json.loads(body)
              # Expecting either {'text': '...'} or {'choices':[{'text': '...'}]}
              if isinstance(js, dict) and "text" in js:
                return js["text"]
              if isinstance(js, dict) and "choices" in js and js["choices"]:
                return js["choices"][0].get("text", "")
            except Exception:
              # If response isn't JSON, return raw body
              return body
        except URLError as e:
          logger.warning("Gorq request failed: %s; falling back to PlaceholderLLM", e)
          return PlaceholderLLM._build_report(prompt)

        return PlaceholderLLM._build_report(prompt)

    def _generate(self, prompts: List[str], stop: Optional[Sequence[str]] = None, **kwargs: Any) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._post_prompt(prompt)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    async def _agenerate(self, prompts, stop=None, **kwargs):
        return self._generate(prompts, stop, **kwargs)


# ─────────────────────────────────────────────────────────
# 2.  LangChain prompt template
# ─────────────────────────────────────────────────────────
REPORT_TEMPLATE = PromptTemplate(
    input_variables=[
        "T_a", "T_b", "u_a", "u_b",
        "max_a", "max_b", "mean_a", "mean_b",
        "ratio", "spread_a", "spread_b",
        "l2_diff", "uncert_a", "uncert_b",
        "rf_r2", "mlp_r2",
    ],
    template="""You are a senior chemical-engineering analyst.  Generate a
structured technical report based on the following surrogate-model outputs
for an NH₃ advection-diffusion comparison study.

Temperature A: {T_a}
Temperature B: {T_b}
Velocity A: {u_a}
Velocity B: {u_b}
Max Concentration A: {max_a}
Max Concentration B: {max_b}
Mean Concentration A: {mean_a}
Mean Concentration B: {mean_b}
Concentration Ratio (B/A): {ratio}
Plume Spread A: {spread_a}
Plume Spread B: {spread_b}
Spatial L2 Difference: {l2_diff}
Model Uncertainty A: {uncert_a}
Model Uncertainty B: {uncert_b}
RF R²: {rf_r2}
MLP R²: {mlp_r2}

Report:"""
)


# ─────────────────────────────────────────────────────────
# 3.  Public API
# ─────────────────────────────────────────────────────────
def build_report_chain() -> LLMChain:
    """
    Constructs a LangChain ``LLMChain`` wired to the PlaceholderLLM.
    SWAP ``PlaceholderLLM()`` for ``ChatOpenAI(...)`` for production use.
    """
    # Prefer remote GorqLLM when an API key is present; otherwise use the
    # local PlaceholderLLM. GorqLLM will itself fall back to the placeholder
    # if a URL is not configured.
    if GORQ_API_KEY and "GorqLLM" in globals():
      try:
        llm = globals()["GorqLLM"]()
      except Exception as e:
        logger.warning("Failed to instantiate GorqLLM: %s; using PlaceholderLLM", e)
        llm = PlaceholderLLM()
    else:
      if GORQ_API_KEY and not GORQ_API_URL:
        logger.info("GORQ_API_KEY present but GORQ_API_URL missing; using PlaceholderLLM")
      llm = PlaceholderLLM()

    chain = LLMChain(llm=llm, prompt=REPORT_TEMPLATE)
    logger.info("Built report chain with LLM: %s", type(llm).__name__)
    return chain


def generate_report(comparison_metrics: Dict[str, Any], model_metrics: Dict[str, Any]) -> str:
    """
    High-level entry point called by the Streamlit dashboard.

    Parameters
    ----------
    comparison_metrics : dict   – output of SurrogateRegistry.compare_conditions()
    model_metrics      : dict   – output of SurrogateRegistry.train()  (R², RMSE …)

    Returns
    -------
    str – the full formatted report.
    """
    chain = build_report_chain()

    template_vars = {
        "T_a":      f"{comparison_metrics['params_a']['T']:.1f} K",
        "T_b":      f"{comparison_metrics['params_b']['T']:.1f} K",
        "u_a":      f"({comparison_metrics['params_a']['u_x']:.2f}, {comparison_metrics['params_a']['u_y']:.2f}) m/s",
        "u_b":      f"({comparison_metrics['params_b']['u_x']:.2f}, {comparison_metrics['params_b']['u_y']:.2f}) m/s",
        "max_a":    f"{comparison_metrics['max_conc_a']:.4e}",
        "max_b":    f"{comparison_metrics['max_conc_b']:.4e}",
        "mean_a":   f"{comparison_metrics['mean_conc_a']:.4e}",
        "mean_b":   f"{comparison_metrics['mean_conc_b']:.4e}",
        "ratio":    f"{comparison_metrics['conc_ratio']:.3f}",
        "spread_a": f"{comparison_metrics['spread_a']:.3f}",
        "spread_b": f"{comparison_metrics['spread_b']:.3f}",
        "l2_diff":  f"{comparison_metrics['spatial_diff_l2']:.4e}",
        "uncert_a": f"{comparison_metrics['uncertainty_a']:.4e}",
        "uncert_b": f"{comparison_metrics['uncertainty_b']:.4e}",
        "rf_r2":    f"{model_metrics.get('rf_r2', 0.0):.4f}",
        "mlp_r2":   f"{model_metrics.get('mlp_r2', 0.0):.4f}",
    }

    report = chain.run(**template_vars)
    return report


# ─────────────────────────────────────────────────────────
# 4.  Quick smoke-test
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy_comp = {
        "params_a":   {"T": 300.0, "u_x": 1.0,  "u_y": 0.0,  "D_base": 3e-5},
        "params_b":   {"T": 380.0, "u_x": -0.5, "u_y": 1.5,  "D_base": 5e-5},
        "max_conc_a": 0.042, "max_conc_b": 0.065,
        "mean_conc_a":0.003, "mean_conc_b":0.005,
        "conc_ratio": 1.548,
        "spatial_diff_l2": 0.12,
        "spread_a": 0.22,    "spread_b": 0.31,
        "uncertainty_a": 0.002, "uncertainty_b": 0.004,
    }
    dummy_model = {"rf_r2": 0.94, "mlp_r2": 0.91}

    print(generate_report(dummy_comp, dummy_model))
