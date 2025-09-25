<p align="center">
  <img src=".github/assets/logo.svg" alt="ErrorAlign Logo" width="50%"/>
</p>
<br/>
Text-to-text alignment algorithm for speech recognition error analysis.


---

### Install

```
pip install error_align
```
---

### Use
```python
from error_align import ErrorAlign

ref = "Some things are worth noting!"
hyp = "Something worth nothing period?"

alignments = ErrorAlign(ref, hyp).align()
```

Resulting `alignments`:
```python
Alignment(SUBSTITUTE: "Some" -> "Some"-),
Alignment(SUBSTITUTE: "things" -> -"thing"),
Alignment(DELETE: "are"),
Alignment(MATCH: "worth" == "worth"),
Alignment(SUBSTITUTE: "noting" -> "nothing"),
Alignment(INSERT: "period")
```
---
<p style="text-align:center;">
:contruction: **Work-in-progress**: C++ version with Python bindings.
</p>