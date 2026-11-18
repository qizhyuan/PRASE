"""Exercise an installed wheel: the native engine loads and agrees with the NumPy engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fixtures import graph_pair
from prase import Paris, ParisConfig, evaluate, methods
from prase.paris import native

assert native.available(), native.load_error()
left, right, reference = graph_pair(60, seed=1)
compiled = Paris(left, right, ParisConfig(engine="native")).run()
python = Paris(left, right, ParisConfig(engine="python")).run()
assert compiled.alignment.pairs() == python.alignment.pairs()
assert evaluate(compiled.alignment, reference).f1 > 0.8
result = methods.paris().run(left, right, reference=reference)
assert result.steps[0].values["engine"] == "native"
print("wheel smoke test passed")
