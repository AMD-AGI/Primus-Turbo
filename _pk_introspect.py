import inspect

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import Vector as Vec

print("fx float-ish:", [n for n in dir(fx) if "loat" in n or "BF" in n or "Bf" in n])
print("Vec api:", [n for n in dir(Vec) if not n.startswith("_")])
try:
    import flydsl.expr.vector as v

    print("vector mod:", [n for n in dir(v) if not n.startswith("_")])
except Exception as e:
    print("no flydsl.expr.vector:", e)
print(inspect.getsource(rocdl.raw_ptr_buffer_atomic_fadd))
try:
    from flydsl.expr import buffer_ops

    print(inspect.signature(buffer_ops.create_buffer_resource))
    print(inspect.signature(buffer_ops.buffer_store))
except Exception as e:
    print("buffer_ops:", e)
