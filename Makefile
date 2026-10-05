.PHONY: setup data train eval serve demo test params bench
PY ?= python

setup:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

data:
	bash scripts/download_meld.sh && $(PY) scripts/prepare_meld.py --config configs/data.yaml

train:
	$(PY) -m meld_robot.train.train --stage text   --config configs/train.yaml
	$(PY) -m meld_robot.train.train --stage audio  --config configs/train.yaml
	$(PY) -m meld_robot.train.train --stage fusion --config configs/train.yaml

eval:
	$(PY) -m meld_robot.eval.offline_eval --ckpt checkpoints/fusion/best.pt --split test
	$(PY) -m meld_robot.eval.streaming_eval --ckpt checkpoints/fusion/best.pt --split test --limit 300
	$(PY) -m meld_robot.eval.grounding_eval --config configs/stream.yaml

serve:
	$(PY) -m meld_robot.server.app --config configs/stream.yaml

demo:
	$(PY) demo/replay_demo.py --config configs/stream.yaml --clip $(CLIP)

test:
	$(PY) -m pytest -q

params:
	$(PY) scripts/count_params.py --config configs/stream.yaml

bench:
	$(PY) scripts/benchmark.py --config configs/stream.yaml
