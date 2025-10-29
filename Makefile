#!/usr/bin/make -f
##
# Makefile for Disturbance Tracker (DTrack)
##

# Create the 'dtrack' binary (main application)
dtrack:
	cd ./src && go build -o ../dtrack .

dtrack-headless:
	cd ./src && go build -o ../dtrack-headless -tags headless .

test: dtrack
	cd ./src && go test ./...
	cd ./src && go vet ./...
	cd ./src && go-staticcheck ./...

clean:
	podman rmi dtrack_builder 2>/dev/null || true
	$(RM) dtrack


##
# Container-Based Development
##

# Run Make targets from within 'builder'
builder-%:
	@$(MAKE) builder-shell BUILDER_CMD="make -C /repo $*"

# Run an arbitrary command inside 'builder'
BUILDER_CMD ?= /bin/bash
builder-shell: builder
	podman run --rm --tty \
		-v "$(shell go env GOCACHE):/root/.cache/go-build" \
		-v "$(PWD):/repo" \
		dtrack_builder $(BUILDER_CMD)

# Container used to compile (build) final binary
builder:
	podman build \
		-t dtrack_builder \
		-f .builder


.PHONY: test clean builder builder-shell
