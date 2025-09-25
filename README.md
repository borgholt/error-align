<div style="text-align: center; margin-bottom: 30px;">
  <img src=".github/assets/logo.svg" alt="ErrorAlign Logo" width="60%"/>
</div>

Text-to-text alignment algorithm for speech recognition error analysis.

:construction: **Work-in-progress**: C++ version with Python bindings.

### Install

```
pip install error_align
```

### Use
```
from error_align import ErrorAlign

ref = "Some things are worth noting!"
hyp = "Something worth nothing period?"

alignments = ErrorAlign(ref, hyp).align()
```

Resulting `alignments`:
```
Alignment(SUBSTITUTE: Some -> Some-),
Alignment(SUBSTITUTE: things -> -thing),
Alignment(DELETE: are),
Alignment(MATCH: worth == worth),
Alignment(SUBSTITUTE: noting -> nothing),
Alignment(INSERT: period)
```