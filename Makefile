NAME ?= ocr_recognition
GPUS ?= all

build:
	docker build -t $(NAME) .

run:
	docker run --rm -it \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME)

train:
	docker run --rm \
		--gpus=$(GPUS) \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		python ./bin/train.py
