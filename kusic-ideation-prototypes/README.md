# Kevin-C pathology research workbench

Four working prototypes based on the interviews in `../ideas folder/IDEAS.txt`.
The portfolio, concerns, score anchors and progress conclusions are in
[`../IDEAS.txt`](../IDEAS.txt). The existing interview notes are preserved.

Run from `D:/ashbyprize` with Python 3.11 or newer:

```powershell
python kusic-ideation-prototypes/run.py
```

The desktop needs Python's Tcl/Tk installation. It opens locally and needs no
account, server, API key, model download or third-party Python runtime package.
On this computer `python3` is a nonfunctional Windows Store alias; use `python`.
The managed sandbox can prevent native Tcl initialization; a normal terminal was
used to validate the desktop workflows. The command interface works without Tk.

Choose a workflow, load its sample, and choose **Run input**. The **Result** tab
explains the measurements; **Evidence JSON** contains the full result. **Open
JSON** loads your own compatible local export, and **Input JSON** lets you edit
it. A new selection loads that workflow's sample. **Compare all ideas** runs all
four synthetic experiments and enables a comparison CSV export.

For the hint workflow, **Next hint** advances through the authored lesson. For
retrieval, **Inspect prior source** accepts a returned case reference and shows
its source string and one-based local input row after rechecking the current
input's eligibility. It never opens a source URL or file. Edited input needs a
new run before it has a new result; result exports contain the completed run.

## Workflows and input contracts

| ID | Workflow | Input | Result |
| --- | --- | --- | --- |
| K01 | Calibrated second-look gate | Three disjoint labeled-logit cohorts | Temperature, NLL/Brier/ECE, coverage, retained error and review reasons |
| K02 | Patient-prior retrieval | Query, authorization claims, dated record embeddings and source metadata | Eligible ranked priors, exclusion reasons and aggregate unrestricted baseline |
| K03 | Trainee hint ladder | Authored lesson, source/permission/review metadata and an explicit intent tag | One progressive hint with evidence, or a reason to withhold it |
| K04 | Paired robustness audit | Complete clean/perturbed labeled probability pairs and subgroup labels | Accuracy/probability shifts, prediction flips and subgroup evidence |

Generate an exact editable input template and run a local JSON file:

```powershell
python kusic-ideation-prototypes/run.py template --idea K02 > tests/prior-input.json
python kusic-ideation-prototypes/run.py run --idea K02 --input tests/prior-input.json --output tests/prior-result.json
python kusic-ideation-prototypes/run.py demo --output tests/results
python tests/run_kevin_evaluation.py
```

The template output is UTF-8; Windows PowerShell 5's `>` may write UTF-16. In
that shell, use `| Out-File -Encoding utf8 tests/prior-input.json` instead. The
desktop also provides templates without shell encoding concerns.

Input JSON is limited to 2 MB. Duplicate keys, NaN and Infinity are rejected.
Finite numerics, class dimensions and identity/pairing constraints are checked
within each workflow. Errors are displayed and do not produce a success result.

K01 rows have `id`, `logits`, `label`; top-level `calibration`, `selection`,
`evaluation`, `scope`, `provenance` and optional `target_error`. Supported scopes
are `synthetic_research` and `authorized_nonclinical_research`. Provenance has
`model_id`, `dataset_id`, `allowed_scope`. Case IDs must be disjoint across all
three cohorts; this check does not establish patient independence. Calibration
uses a stable unclipped objective and bounds temperature to 0.05–100. Reported
probability NLL clips probabilities below 1e-15; ECE has ten confidence bins;
Brier sums squared class errors per example, then averages examples. Thresholds
maximize selection coverage subject to an empirical selection-error target.
An empty retained set has `accepted_error: null`. Held-out labels never tune the
temperature or threshold. The gate evaluates exported values, not live cases.

K02 has `query`, `records`, optional `purpose` (`research` or `education`) and
`top_k` (1–50). Timestamps require seconds and a timezone. Embeddings are finite,
nonzero, equally sized numeric lists. Patient, institution, chronology, consent,
permitted use and source are checked before guarded ranking. All supplied
records must already be lawfully accessible: input authorization is a caller
claim, and the aggregate baseline examines all supplied records. Default export
contains opaque references, not patient identifiers or embeddings. Those
references are pseudonyms; provenance, timestamps or subgroup labels can still
be sensitive. This prototype does not provide login, verified consent, access
logging, record linkage, retention enforcement or an authenticated data boundary.

K03 requires `intent`, `lesson_id`, `step`, `as_of`, `lessons` and optional
`max_review_age_days` (1–730). The sample teaches observation using fictional
shapes; it contains no actual pathology teaching claims. The intent tag must be
`education` or `patient_diagnosis`; the latter is withheld. This is a caller
contract, not semantic detection of clinical requests. Source/license/reviewer
metadata are assertions, not proof of rights or expert review.

K04 requires `clean`, `perturbed`, `provenance`, optional `subgroup_minimum`
(default 5). Each row has `id`, `probabilities`, `label`, `subgroup`. Probabilities
sum to 1 within 1e-6. Paired IDs, labels and group membership must match exactly.
Groups below the minimum retain their counts but have no performance metrics.
Larger groups are descriptive only; the threshold is not a power calculation.
Opaque case IDs do not establish experimental blinding.

## Evidence and interpretation

The saved evaluation is `../tests/results/evaluation.json`; scores and primary
measurements are also in `../tests/results/comparison.csv`. Reports include input
or source hashes, author, status, four 1–10 scores, rationales, evidence, limits
and recommendations. Embedded evidence references use JSON pointers so exported
reports stay portable. The evaluation harness saves hashes of its test scripts.

Complexity and legal burden favor lower scores. Friendliness and creativity
favor higher scores. Complexity/friendliness/creativity assess the scoped
prototype product design; legal burden assesses its intended real deployment.
Present synthetic-only legal burden is separately recorded as 1 for every idea.
Scores are reasoned design assessments without a participant usability study.
The equal-weight screening index is documented in IDEAS; it is not a scientific
ranking and does not establish clinical value. K04 is the recommended lead demo
because it directly addresses the interview emphasis on testing and verifiability.

The synthetic controls show improved K01 probability error without improved
classification accuracy; its held-out retained error still exceeds the target.
K02 enforces the supplied metadata policy; K03 withholds five deliberately
invalid scenarios; K04 distinguishes stable and deliberately unstable score
vectors. These measurements establish software behavior on invented fixtures.
They do not establish patient benefit, real-model accuracy, learning gain,
clinical robustness or deployment readiness. K02 also supports optional local
CONCH image embedding, described below. Realistic pathology perturbations,
expert content and external cohorts remain future validation work. The user has
supplied a weight file and will locate the Ashby pathology data later.

The official [Ashby challenge](https://ai.ncsa.illinois.edu/education/2026-ashby-prize-in-computational-science-hackathon/)
focuses on pathology workflows and augmenting foundation models. Before integrating
CONCH, review the exact version's [publisher terms](https://github.com/mahmoodlab/CONCH#license-and-terms-of-use),
including gated access, use and redistribution restrictions. Intended use needs
review under the [FDA CDS guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software);
human review alone does not establish exclusion. Identifiable records need an
appropriate basis under [HHS research rules](https://www.hhs.gov/hipaa/for-professionals/special-topics/research/index.html).
No source provides legal clearance for this tool; the scores record concerns.

## Optional local CONCH image input

`models/conch.bin` is the user-supplied 802,235,437-byte weight file. `models/` is
ignored by Git. The adapter reads the visual tower with `weights_only=True`,
strict tensor keys, and CPU inference. It never downloads a model or requests a
token. The exact tested publisher source revision is
`141cc09c7d4ff33d8eda562bd75169b457f71a62`; timm is `0.9.16`.
These optional packages are installed under ignored `tests/_conch_runtime` on
this computer; PyTorch, torchvision and Pillow are already in the selected Python.

```powershell
$env:PYTHONPATH = 'tests/_conch_runtime'
python kusic-ideation-prototypes/run.py
```

Choose **Embed local images**, then an image-manifest JSON and your weight file.
An image manifest is the normal K02 template with every `embedding` replaced
by an `image_path`; relative paths resolve beside the manifest. All supplied
images must already be authorized for this offline research cohort, including
records that the query's eligibility policy will later exclude. A runnable
generated-shape manifest is `tests/generated_conch/image_manifest.json`.

The command interface prepares a reusable K02 request and a provenance sidecar,
then prints retrieval results:

```powershell
python kusic-ideation-prototypes/run.py embed-priors --input tests/generated_conch/image_manifest.json --weights models/conch.bin --output tests/generated_conch/my-priors.json
python tests/run_kevin_conch_smoke.py
```

The adapter processes at most 16 query-plus-record images, one per batch, on two
CPU threads. Image dimensions are bounded to 4096 per side, compressed files to
64 MiB, and multi-frame images are rejected. Preprocessing follows the publisher's
448-pixel transform and OpenAI normalization. Features use the contrastive-head
projection normalized to 512 values. Local paths are checked before opening;
URLs and network paths are unsupported. Original images/weights are preserved.
Exports refuse aliases to declared input assets, including existing hard links.

`tests/results/conch_smoke.json` records the real-weight integration test on
generated shapes, source/runtime/model/image hashes, working-set peak and timing.
It confirms model loading, feature compatibility and the retrieval path, without
measuring pathology accuracy. Normal pytest uses backend test doubles for adapter
failure controls and never loads the large checkpoint.

## Verification and repository boundaries

All testing scripts and generated results are kept under the gitignored `tests/`
folder. Runtime code stays in `kusic-ideation-prototypes/`. Existing missing
UMUD/tabular files are not restored, and the old training job remains paused.
`python train.py --smoke-test` cannot run because `train.py` is absent. The new
end-to-end equivalent is `python tests/run_kevin_evaluation.py` plus the real
CLI and hidden desktop workflow tests. Raw data and credentials are untouched.

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

Ruff is installed locally under ignored `tests/_tooling` on this machine. Set
`$env:PYTHONPATH = 'tests/_tooling'` for those Ruff commands if it is not otherwise
installed. Pytest is available in the installed Python. For a fresh environment,
install Ruff and pytest in your own virtual environment; the core runtime is
stdlib, with separate optional dependencies for CONCH image input.
Ignored tests need explicit Ruff paths with `--no-respect-gitignore` to include
them. Desktop tests skip only when native Tcl/Tk cannot initialize; run from a
normal terminal to validate the interface. For an isolated temp area, use a new
unused path such as `--basetemp=tests/tmp-run-001` (pytest owns that directory).
