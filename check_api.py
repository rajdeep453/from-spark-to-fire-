"""Run this FIRST. python check_api.py  -> confirms your provider works with the project's client shim."""
import sys, os, time
if "--env" in sys.argv:
    for line in open(sys.argv[sys.argv.index("--env") + 1], encoding="utf-8-sig"):
        line = line.split("#")[0].strip()
        if "=" in line:
            k, v = line.split("=", 1); os.environ[k.strip()] = v.strip()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prop_dynamics.openai_compat import configure_openai, DEFAULT_MODEL
openai = configure_openai(None)
model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
print("base_url:", os.getenv("OPENAI_BASE_URL"), "| model:", model)
t = time.time()
r = openai.ChatCompletion.create(model=model, temperature=0, max_tokens=150,
        messages=[{"role": "user", "content": "Reply with exactly: #TAG:ABC-1234#"}])
txt = r["choices"][0]["message"]["content"]
print("reply:", txt.strip(), f"({time.time()-t:.1f}s, tokens={r.get('usage',{}).get('total_tokens')})")
print("OK" if txt else "EMPTY REPLY - check model name")
