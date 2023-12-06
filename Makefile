NAME ?= ocr_recognition
GPUS ?= all

.PHONY: build
build:
	docker build -t $(NAME) .

.PHONY: run
run:
	docker run --rm -it \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME)

.PHONY: stop
stop:
	-docker stop $(NAME)
	-docker rm $(NAME)


.PHONY: train
train:
	docker run --rm \
		--gpus=$(GPUS) \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		python ./bin/train.py

.PHONY: style
style:
	git config --global --add safe.directory /workdir && pre-commit run --verbose --files ocr_recognition/*

.PHONY: test-cov
test-cov:
	docker run --rm \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		pytest \
			-p no:logging \
			--cache-clear \
			--cov ocr_recognition/builder \
			--cov ocr_recognition/data \
			--cov ocr_recognition/metrics \
			--cov ocr_recognition/model \
			--cov ocr_recognition/modules \
			--cov ocr_recognition/visualizers \
			--junitxml=pytest.xml \
			--cov-report term-missing:skip-covered \
			--cov-report xml:coverage.xml \
			tests
