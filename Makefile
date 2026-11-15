.PHONY: build build-local build-docker package run run-local run-gpu stop test

build:
	./scripts/build.sh --all

build-local:
	./scripts/build.sh --local

build-docker:
	./scripts/build.sh --docker

package:
	./scripts/package-single-binary.sh

run:
	./scripts/run.sh --docker

run-local:
	./scripts/run.sh --local

run-gpu:
	./scripts/run.sh --gpu

stop:
	./scripts/stop.sh

test:
	./scripts/test-e2e.sh
