# go-whisper-ct2 Makefile
# Build orchestration for C++ library and Go bindings

.PHONY: all build build-cpp build-go build-cli build-benchmark clean test test-cpp test-go install-cpp help

# Configuration
BUILD_DIR := csrc/build
BIN_DIR := bin
INSTALL_PREFIX ?= /usr/local
CMAKE_BUILD_TYPE ?= Release
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Default target
all: build

# Create build directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build C++ library
build-cpp: $(BUILD_DIR)
	@echo "==> Building C++ library..."
	cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX) \
		-DBUILD_SHARED_LIBS=ON
	$(MAKE) -C $(BUILD_DIR) -j$(NPROC)
	@echo "==> C++ library built successfully"

# Build Go package (requires C++ library)
build-go: build-cpp
	@echo "==> Building Go package..."
	CGO_CFLAGS="-I$(PWD)/csrc/include" \
	CGO_LDFLAGS="-L$(PWD)/$(BUILD_DIR) -lwhisper_ct2 -Wl,-rpath,$(PWD)/$(BUILD_DIR)" \
	go build ./pkg/whisper/...
	@echo "==> Go package built successfully"

# Build CLI application
build-cli: build-go $(BIN_DIR)
	@echo "==> Building CLI..."
	CGO_CFLAGS="-I$(PWD)/csrc/include" \
	CGO_LDFLAGS="-L$(PWD)/$(BUILD_DIR) -lwhisper_ct2 -Wl,-rpath,$(PWD)/$(BUILD_DIR)" \
	go build -o $(BIN_DIR)/whisper-ct2 ./cmd/whisper-ct2
	@echo "==> CLI built: $(BIN_DIR)/whisper-ct2"

# Build benchmark tool
build-benchmark: build-go $(BIN_DIR)
	@echo "==> Building benchmark tool..."
	CGO_CFLAGS="-I$(PWD)/csrc/include" \
	CGO_LDFLAGS="-L$(PWD)/$(BUILD_DIR) -lwhisper_ct2 -Wl,-rpath,$(PWD)/$(BUILD_DIR)" \
	go build -o $(BIN_DIR)/whisper-benchmark ./cmd/benchmark
	@echo "==> Benchmark tool built: $(BIN_DIR)/whisper-benchmark"

# Build everything
build: build-cli build-benchmark

# Run C++ tests
test-cpp: build-cpp
	@echo "==> Running C++ tests..."
	cd $(BUILD_DIR) && ctest --output-on-failure

# Run Go tests
test-go: build-cpp
	@echo "==> Running Go tests..."
	CGO_CFLAGS="-I$(PWD)/csrc/include" \
	CGO_LDFLAGS="-L$(PWD)/$(BUILD_DIR) -lwhisper_ct2" \
	LD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR) \
	go test -v ./pkg/whisper/...

# Run all tests
test: test-go

# Install C++ library system-wide (requires sudo)
install-cpp: build-cpp
	@echo "==> Installing C++ library to $(INSTALL_PREFIX)..."
	$(MAKE) -C $(BUILD_DIR) install
	ldconfig 2>/dev/null || true
	@echo "==> Installation complete"

# Install CLI to GOPATH/bin
install: build-cli
	@echo "==> Installing CLI..."
	cp $(BIN_DIR)/whisper-ct2 $(GOPATH)/bin/ 2>/dev/null || \
	cp $(BIN_DIR)/whisper-ct2 ~/go/bin/ 2>/dev/null || \
	cp $(BIN_DIR)/whisper-ct2 /usr/local/bin/
	@echo "==> CLI installed"

# Clean build artifacts
clean:
	@echo "==> Cleaning..."
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)
	go clean ./...
	@echo "==> Clean complete"

# Deep clean including Go cache
distclean: clean
	go clean -cache ./...

# Format code
fmt:
	go fmt ./...

# Lint code
lint:
	golangci-lint run ./...

# Generate documentation
docs:
	godoc -http=:6060 &
	@echo "Documentation available at http://localhost:6060/pkg/github.com/xPrimeTime/go-whisper-ct2/"

# Help
help:
	@echo "go-whisper-ct2 build targets:"
	@echo ""
	@echo "  make              - Build everything (default)"
	@echo "  make build        - Build C++ library, Go package, CLI, and benchmark tool"
	@echo "  make build-cpp    - Build C++ library only"
	@echo "  make build-go     - Build Go package only"
	@echo "  make build-cli    - Build CLI application"
	@echo "  make build-benchmark - Build benchmark tool"
	@echo ""
	@echo "  make test         - Run all tests"
	@echo "  make test-cpp     - Run C++ tests"
	@echo "  make test-go      - Run Go tests"
	@echo ""
	@echo "  make install-cpp  - Install C++ library system-wide (requires sudo)"
	@echo "  make install      - Install CLI to GOPATH/bin"
	@echo ""
	@echo "  make clean        - Remove build artifacts"
	@echo "  make distclean    - Deep clean including Go cache"
	@echo ""
	@echo "  make fmt          - Format Go code"
	@echo "  make lint         - Lint Go code"
	@echo "  make docs         - Start local documentation server"
	@echo ""
	@echo "Configuration variables:"
	@echo "  CMAKE_BUILD_TYPE  - CMake build type (default: Release)"
	@echo "  INSTALL_PREFIX    - Installation prefix (default: /usr/local)"
	@echo ""
	@echo "Example:"
	@echo "  make CMAKE_BUILD_TYPE=Debug build"
