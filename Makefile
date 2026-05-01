# ==============================================================================
# Makefile for panjep
# Cross-platform: macOS (Apple Silicon / Intel) and Linux (x86_64 / aarch64).
#
# Overridable variables (pass on the command line):
#   CXX     – compiler           default: g++ on Linux, clang++ on macOS
#   OMP     – 0 to disable OpenMP  default: 1
#   PREFIX  – install prefix      default: /usr/local
#   VERBOSE – 1 to print detected platform / flags
#
# Examples:
#   make                        # release build with auto-detected settings
#   make CXX=g++                # force GCC
#   make OMP=0                  # single-threaded build
#   make VERBOSE=1              # show resolved compiler / OpenMP / arch
#   make install PREFIX=~/.local
# ==============================================================================

OS   := $(shell uname -s)
ARCH := $(shell uname -m)

# ── Compiler ──────────────────────────────────────────────────────────────────
# Default: g++ on Linux (ships with OpenMP/libgomp), clang++ on macOS.
# Override at any time: make CXX=clang++
#
# On macOS, ignore conda-forge's cross-compiler wrapper
# (e.g. arm64-apple-darwin20.0.0-c++) inherited from CXX in an activated conda
# env: it injects -isystem $CONDA_PREFIX/include, which pulls in conda's libc++
# headers while the binary still links against the system libc++.dylib. The
# resulting std::string ABI mismatch produces "pointer being freed was not
# allocated" malloc aborts at runtime.
ifeq ($(OS),Linux)
    CXX ?= g++
else ifeq ($(OS),Darwin)
    ifneq (,$(findstring apple-darwin,$(CXX)))
        CXX := clang++
    endif
    CXX ?= clang++
else
    # FreeBSD / NetBSD etc. — clang is the BSD default.
    CXX ?= c++
endif

# ── Base compilation flags ────────────────────────────────────────────────────
CXXFLAGS := -std=c++17 -O3 \
            -fno-math-errno -fno-trapping-math -fno-signed-zeros \
            -Wall -Wextra -Wpedantic

# ARM family: macOS arm64, Linux aarch64, plus the rarer 32-bit aliases.
# Targets ARMv8-A baseline so the binary runs on Graviton / Apple Silicon /
# Raspberry Pi 4+; -flax-vector-conversions silences NEON-intrinsic type
# punning warnings on both clang and recent GCC.
_ARM := $(filter arm64 aarch64 armv8% armv7%,$(ARCH))
ifneq (,$(_ARM))
    CXXFLAGS += -march=armv8-a -flax-vector-conversions
else
    CXXFLAGS += -march=native
endif
LDFLAGS  :=

# ── OpenMP ────────────────────────────────────────────────────────────────────
# Disable with: make OMP=0
OMP    ?= 1
OMP_OK := no

ifeq ($(OMP),1)

  ifeq ($(OS),Darwin)
    # ------------------------------------------------------------------
    # macOS: Xcode CLT does not ship OpenMP.  Probe Homebrew libomp in
    # priority order: brew --prefix query → ARM64 path → Intel path.
    # ------------------------------------------------------------------
    _LIBOMP := $(shell brew --prefix libomp 2>/dev/null)
    ifeq ($(_LIBOMP),)
      _LIBOMP := $(shell test -d /opt/homebrew/opt/libomp && \
                         echo /opt/homebrew/opt/libomp)
    endif
    ifeq ($(_LIBOMP),)
      _LIBOMP := $(shell test -d /usr/local/opt/libomp && \
                         echo /usr/local/opt/libomp)
    endif

    ifneq ($(_LIBOMP),)
      # Homebrew libomp found (works with Apple Clang and Homebrew GCC).
      CXXFLAGS += -Xpreprocessor -fopenmp -I$(_LIBOMP)/include
      LDFLAGS  += -L$(_LIBOMP)/lib -lomp
      OMP_OK   := yes (Homebrew libomp at $(_LIBOMP))
    else
      # Last resort: try the compiler's own -fopenmp (e.g. Homebrew GCC).
      _OMP_TEST := $(shell echo 'int main(){}' | \
                     $(CXX) -fopenmp -x c++ - -o /dev/null 2>/dev/null && echo yes)
      ifeq ($(_OMP_TEST),yes)
        CXXFLAGS += -fopenmp
        LDFLAGS  += -fopenmp
        OMP_OK   := yes ($(CXX) -fopenmp)
      else
        $(warning [OpenMP] libomp not found on macOS.)
        $(warning          Install with:  brew install libomp)
        $(warning          Building single-threaded.)
      endif
    endif

  else
    # ------------------------------------------------------------------
    # Linux / *BSD: GCC accepts -fopenmp (libgomp); Clang accepts it too
    # but needs libomp-dev (Debian/Ubuntu) or libomp-devel (Fedora/RHEL).
    # Run a compile-test so we emit a clear warning instead of a linker
    # error when the OpenMP runtime is missing.
    # ------------------------------------------------------------------
    _OMP_TEST := $(shell echo 'int main(){}' | \
                   $(CXX) -fopenmp -x c++ - -o /dev/null 2>/dev/null && echo yes)
    ifeq ($(_OMP_TEST),yes)
      CXXFLAGS += -fopenmp
      LDFLAGS  += -fopenmp
      OMP_OK   := yes ($(CXX) -fopenmp)
    else
      $(warning [OpenMP] $(CXX) does not support -fopenmp.)
      $(warning   Debian/Ubuntu: sudo apt install g++   (or libomp-dev for clang))
      $(warning   Fedora/RHEL:   sudo dnf install gcc-c++  (or libomp-devel for clang))
      $(warning   Arch:          sudo pacman -S gcc       (or openmp for clang))
      $(warning          Building single-threaded.)
    endif

  endif
endif   # OMP=1

# ── Sources ───────────────────────────────────────────────────────────────────
TARGET := panjep
SRCDIR := src
SRCS   := $(SRCDIR)/main.cpp $(SRCDIR)/panjep.cpp
HDRS   := $(SRCDIR)/panjep.hpp

# ── Primary targets ───────────────────────────────────────────────────────────
.PHONY: all clean test bench install uninstall help info

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	@echo "[panjep] $(OS)/$(ARCH)  CXX=$(CXX)  OpenMP=$(OMP_OK)"
ifeq ($(VERBOSE),1)
	@echo "  CXXFLAGS=$(CXXFLAGS)"
	@echo "  LDFLAGS =$(LDFLAGS)"
endif
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) $(LDFLAGS)

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "panjep build targets:"
	@echo "  make            release build"
	@echo "  make OMP=0      single-threaded build"
	@echo "  make CXX=g++    force a specific compiler"
	@echo "  make VERBOSE=1  show full compile/link flags"
	@echo "  make test       run correctness tests"
	@echo "  make bench      run synthetic benchmarks (n=500/1000/2000)"
	@echo "  make install [PREFIX=...]"
	@echo "  make clean"

info:
	@echo "OS    = $(OS)"
	@echo "ARCH  = $(ARCH)"
	@echo "CXX   = $(CXX)"
	@echo "OMP   = $(OMP_OK)"
	@echo "FLAGS = $(CXXFLAGS)"
	@echo "LDFL  = $(LDFLAGS)"

# ── Install / uninstall ───────────────────────────────────────────────────────
PREFIX ?= /usr/local

install: $(TARGET)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/$(TARGET)

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/$(TARGET)

# ── Quick correctness tests ───────────────────────────────────────────────────
test: $(TARGET)
	@echo "=== Small correctness test (PHYLIP) ==="
	./$(TARGET) -v test/sample5.phy
	@echo ""
	@echo "=== Wikipedia example (PHYLIP) ==="
	./$(TARGET) -v test/wiki5.phy
	@echo ""
	@echo "=== Protein FASTA (5 seqs) ==="
	./$(TARGET) -v test/sample5.faa
	@echo ""
	@echo "=== Nucleotide FASTA (5 seqs) ==="
	./$(TARGET) -v test/sample_nucl.fna

# ── Benchmarks with synthetic data ───────────────────────────────────────────
# BENCH_DIR defaults to $TMPDIR (set by macOS / sandboxes), then /tmp.
BENCH_DIR ?= $(or $(TMPDIR),/tmp)

bench: $(TARGET) gen_test
	@mkdir -p $(BENCH_DIR)
	@echo "=== Benchmark n=500  (BENCH_DIR=$(BENCH_DIR)) ==="
	./gen_test 500  > $(BENCH_DIR)/bench500.phy  && \
	  ./$(TARGET) -v -e 0 $(BENCH_DIR)/bench500.phy  > /dev/null
	@echo "=== Benchmark n=1000 ==="
	./gen_test 1000 > $(BENCH_DIR)/bench1000.phy && \
	  ./$(TARGET) -v -e 0 $(BENCH_DIR)/bench1000.phy > /dev/null
	@echo "=== Benchmark n=2000 ==="
	./gen_test 2000 > $(BENCH_DIR)/bench2000.phy && \
	  ./$(TARGET) -v -e 0 $(BENCH_DIR)/bench2000.phy > /dev/null

gen_test: tools/gen_test.cpp
	$(CXX) -std=c++17 -O2 -o $@ $<

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f $(TARGET) gen_test
