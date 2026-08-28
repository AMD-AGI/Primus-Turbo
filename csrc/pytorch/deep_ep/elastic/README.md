# `deep_ep` elastic (V2) host side -- placeholder

Reserved for the upstream DeepEP V2 ("elastic") host buffer, so it can sit
beside `../legacy/` the way it does upstream.

Upstream source: `deepseek-ai/DeepEP` `csrc/elastic/` (`buffer.hpp`,
`utils.hpp`).

`../python_api.cpp` already contains upstream's registration of both halves:

```cpp
deep_ep::legacy::register_apis(m);
deep_ep::elastic::register_apis(m);   // needs "elastic/buffer.hpp"
```

so the `#include "elastic/buffer.hpp"` there stays unresolved until this
directory is filled in.
