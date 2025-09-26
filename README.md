<p align="center">
  <img src=".github/assets/logo.svg" alt="ErrorAlign Logo" width="58%"/>
</p>
<br/>

**Text-to-text alignment algorithm for speech recognition error analysis.** ErrorAlign helps you dig deeper into your speech recognition projects by accurately aligning each word in a reference transcript with the model-generated transcript. Unlike traditional methods, such as Levenshtein-based alignment, it is not restricted to simple one-to-one alignment, but can map a single reference word to multiple words or subwords in the model output. This enables quick and reliable identification of error patterns in rare words, names, or domain-specific terms that matter most for your application.

### Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Work-in-Progress](#wip)
- [Citation and Research](#citation)



---

<a name="installation">

### Installation

```
pip install error_align
```
---

<a name="usage">

### Usage
```python
from error_align import error_align

ref = "Some things are worth noting!"
hyp = "Something worth nothing period?"

alignments = error_align(ref, hyp)
```

Resulting alignments:
```python
Alignment(SUBSTITUTE: "Some" -> "Some"-),
Alignment(SUBSTITUTE: "things" -> -"thing"),
Alignment(DELETE: "are"),
Alignment(MATCH: "worth" == "worth"),
Alignment(SUBSTITUTE: "noting" -> "nothing"),
Alignment(INSERT: "period")
```


```python
from error_align import error_align

ref = "Some things are worth noting!"
hyp = "Something worth nothing period?"

alignments = error_align(ref, hyp)
```

---

<a name="wip">

### Work-in-Progress

🚧 Optimization for longform text.

🚧 Efficient word-level first-pass.

🚧 C++ version with Python bindings.


---

<a name="citation">

### Citation and Research

_Coming soon_

---
