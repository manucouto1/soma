---
title: The Python API
description: One package, nine modules and 110 public names — where each one lives, and why the pages under this one are generated rather than written.
---

`pip install somatize` puts one package on the path. It is nine modules, and
which one a thing is in follows the same split the rest of this documentation
does: the graph and how to run it are the package itself, and everything that is
*about* a run — read back, diagnosed, searched over — is a module of its own.

## The nine

| module | names | what is in it |
|---|---|---|
| [`somatize`](/soma/reference/python/somatize/) | 11 | `Graph`, `Node`, `Store`, `Ctx`, `Opaque`, `Recorder`, and the two that reach another machine — `Broker` and `Worker` |
| [`somatize.data`](/soma/reference/python/data/) | 9 | where the data comes from, and whether the model learnt what you meant |
| [`somatize.foreseen`](/soma/reference/python/foreseen/) | 5 | what a node's answer will be called before anything runs, and what an edit did |
| [`somatize.health`](/soma/reference/python/health/) | 11 | whether what happened was healthy — an opinion, at bounds you can move |
| [`somatize.reasoning`](/soma/reference/python/reasoning/) | 11 | what somebody was trying to find out, read back and drawn |
| [`somatize.record`](/soma/reference/python/record/) | 13 | what happened, read back: runs, forwards, curves, a fleet, a timeline |
| [`somatize.study`](/soma/reference/python/study/) | 24 | a search over configurations, and what each trial did |
| [`somatize.torch`](/soma/reference/python/torch/) | 19 | everything that needs torch in front of you, training first |
| [`somatize.worker`](/soma/reference/python/worker/) | 7 | the process on the other machine, and what it will agree to open |

Only `somatize.torch` imports torch, and only `somatize.data` reaches for
Arrow. The package itself needs neither: a worker that tokenises text is a
machine with no torch on it at all, and that is a thing this framework does
rather than a thing it tolerates.

## These pages are generated

Everything under this one is rendered from the package's own docstrings, and
nothing on them was typed twice. That is a measurement rather than a taste:
**110 public names and not one of them is undocumented**, with `Trainer`'s
docstring at 1.6 KB and every module's own between 1.1 and 2.3 KB — worked
example, table of flags and all.

The alternative was tried, by the library this one re-derives. Its reference was
876 hand-written lines describing `Filter`, `DifferentiableFilter` and `board` —
an API that had stopped existing. **A wrong reference is the most damaging page
a site can have**, because it is the page people copy out of, and a second copy
of prose is exactly how one is made. So there is one copy, and it is the one the
interpreter answers with.

What the generator decides is layout, not content, and it decides two things:

- **Members are grouped by shape** — constructors, then methods, then
  properties. Nothing is left out. `Graph` has thirty-one methods and some of
  them are plumbing, but a list of what to hide would be a second declaration
  of the surface, which is the thing being avoided. Each docstring says who it
  is for; `plan_json`'s says *"for whoever draws it"*.
- **An inherited member is linked, not repeated.** `.at()`, `.on()`,
  `.cached()`, `.frozen()` and `.mapped()` reach `Node`, `Parquet`, `Learning`
  and `Split` alike, so they are written up once under
  [`Node`](/soma/reference/python/somatize/#node) and named on the other three.

## How to read a signature

Two things on these pages are worth knowing before you meet them.

**Some classes say they are not constructed.** `Sampler`, `Pruner`,
`Partition` and `Point` answer `TypeError: No constructor defined` — they are
built through a class method, `Sampler.tpe(...)` or `Partition.kfold(...)`, and
the page says so rather than printing an empty `()` that would throw. `Bound` is
the other shape: you are handed one, by `Store.bound()`.

**`self` is not written.** A method reads `Space.read(said)`, which is how it is
called.

## Where the prose is

A reference says what each name does; it is a poor place to learn what to reach
for. That is what the rest of this site is:
[the model](/soma/model/overview/) for the vocabulary,
[running one](/soma/running/the-plan/) for a graph and its plan,
[looking at it](/soma/looking/the-record/) for the record and the diagnosis, and
[searching](/soma/searching/a-study/) for a study. The
[Rust reference](/soma/api/rust/) is the other half of the same surface, for the
eight published crates.
