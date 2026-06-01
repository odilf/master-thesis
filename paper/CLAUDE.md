# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build commands

```sh
latexmk -pdf main.tex  # one-shot build
```

There are also `just` commands, but those are for the user to use interactively. Bots should just build the file normally to check that it works.

Build output goes to `.build/`; the engine is LuaLaTeX (`lualatex`).

## Editing

Use `jj` to make and track edits, not git branches.

## Structure

- `main.tex` — document root; loads preamble, body, and frontmatter
- `body/main.tex` — all thesis chapters (Introduction, Background, Computation of EL solutions, Forced Systems, Lie Groups)
- `frontmatter/` — cover, abstract, dedication, table of contents
- `uc3m-preamble.tex` — UC3M university template (don't modify unless fixing template issues)
- `utils.tex` — custom macros (see below)
- `references.bib` — BibTeX bibliography (don't modify, generated from Zotero)

## Custom macros (utils.tex)

| Macro | Purpose |
|---|---|
| `\todo{text}` | Red inline TODO marker |
| `\tc` | Shorthand for `\todo{Cite}` — marks missing citations |
| `\newterm{word}` | Introduce a new term (italic + highlighted) |
| `\faint{text}` | Gray text (used for annotations in derivations) |
| `\Ld` | Discrete Lagrangian $L_d$ |
| `\RR` | Real numbers $\mathbb{R}$ |
| `\mdif{i}` | Partial derivative w.r.t. $i$-th argument |

## Thesis topic

**Parallel variational integrators in forced systems and Lie groups** — a Master's thesis (TFM) at UC3M.

Core idea: variational integration discretizes the configuration space $Q$ rather than the Euler-Lagrange ODE, preserving geometric structure. The paper presents a parallel Jacobi-Newton algorithm for solving the discrete Euler-Lagrange (DEL) equations, then extends it to forced systems (Lagrange-D'Alembert) and Lie groups (via reduction to the Lie algebra and discrete Euler-Poincaré equations). The novel contributions are convergence criteria for the forced and Lie group cases.

Key references: Ferraro et al. (parallel iterative method), Marsden & West (discrete mechanics and variational integrators), Sato Martín de Almagro (discrete mechanics for forced systems).

## Assistance style

The users wants to keep control over the project. It is fundemental that the user understands all the topics, and every step in every development present in the paper, and 95% of steps in most of the topics surrounding but not directly on the paper.

Changes should be as minimal and simple as possible, since the user will want to edit most of the changes the bot does.

## References

References can be accessed from the `file` entry in `references.bib`, which points to the Zotero storage.
