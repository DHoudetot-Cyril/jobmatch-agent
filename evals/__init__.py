"""DeepEval suite for jobmatch-agent (fictional dataset, local Qwen judge).

Importing this package first makes sure DeepEval never phones home and never falls back to
OpenAI: telemetry, update checks and Confident AI uploads are disabled before `deepeval`
is imported anywhere.
"""

import os

for _key, _value in {
    "DEEPEVAL_TELEMETRY_OPT_OUT": "1",
    "DEEPEVAL_UPDATE_WARNING_OPT_IN": "0",
    "DEEPEVAL_DISABLE_DOTENV": "1",  # don't let deepeval read our .env
    "DEEPEVAL_DISABLE_TIMEOUTS": "1",  # local judge is slow; httpx timeout still applies
}.items():
    os.environ.setdefault(_key, _value)
# A stray key in the environment must not route anything to the cloud.
for _key in ("OPENAI_API_KEY", "CONFIDENT_API_KEY", "DEEPEVAL_API_KEY"):
    os.environ.pop(_key, None)
