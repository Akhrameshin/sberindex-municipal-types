PY ?= python3
export OPENBLAS_NUM_THREADS = 2
export OMP_NUM_THREADS = 2
.PHONY: all core null synthetic models experiments authors sources test report article dashboard verify package
all: authors
	$(PY) scripts/build_atlas.py --stage all --extended
	$(PY) src/atlas_assignment.py
	$(MAKE) null synthetic test report dashboard verify package
core:
	$(PY) scripts/build_atlas.py --stage all
	$(PY) src/atlas_assignment.py
	$(MAKE) null synthetic test report dashboard verify package
null:
	$(PY) src/atlas_null.py
synthetic:
	$(PY) src/synthetic.py
models:
	$(PY) scripts/build_atlas.py --stage models
experiments:
	$(PY) scripts/build_atlas.py --stage experiments --extended
authors:
	$(PY) scripts/fetch_external.py
sources:
	$(PY) scripts/download_validation.py
test:
	$(PY) -m pytest -q tests/test_methods.py tests/tests_icvi.py tests/tests_avu.py tests/test_atlas_v2.py > results/v2/tests.log
	cat results/v2/tests.log
report:
	$(PY) scripts/build_report.py
article:
	$(PY) scripts/make_figures.py
	$(PY) scripts/build_report.py
	$(PY) scripts/build_pdf.py --overleaf
dashboard:
	$(PY) scripts/render_story.py
verify:
	$(PY) scripts/verify_release.py
package:
	$(PY) scripts/package_release.py
