CC ?= gcc
CXX ?= g++
CFLAGS ?= -O2 -Wall -Wextra -std=c11
CXXFLAGS ?= -O2 -Wall -Wextra -std=c++11
LDFLAGS ?=
LDLIBS ?= -lm

BIN_DIR := bin
OBJ_DIR := build
SRC_DIR := src

COMMON_OBJS := \
	$(OBJ_DIR)/csv.o \
	$(OBJ_DIR)/util.o

DSP_OBJS := \
	$(OBJ_DIR)/dsp.o \
	$(OBJ_DIR)/mfcc.o \
	$(OBJ_DIR)/frame_io.o

SVM_INC_DIR := $(shell if [ -f /usr/include/svm.h ]; then echo /usr/include; \
	elif [ -f /usr/include/libsvm/svm.h ]; then echo /usr/include/libsvm; \
	elif [ -f /usr/local/include/svm.h ]; then echo /usr/local/include; \
	elif [ -f /usr/local/include/libsvm/svm.h ]; then echo /usr/local/include/libsvm; \
	fi)
SVM_CPPFLAGS ?= $(if $(SVM_INC_DIR),-I$(SVM_INC_DIR),)
HAVE_SVM := $(if $(SVM_INC_DIR),yes,)

ALL_BINS := \
	$(BIN_DIR)/organize_dataset \
	$(BIN_DIR)/preprocess \
	$(BIN_DIR)/extract_features \
	$(BIN_DIR)/split_normalize \
	$(BIN_DIR)/train \
	$(BIN_DIR)/evaluate \
	$(BIN_DIR)/plot_confusion \
	$(BIN_DIR)/cross_validate

ifeq ($(HAVE_SVM),yes)
ALL_BINS += $(BIN_DIR)/svm_baseline
endif

all: $(ALL_BINS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/csv.o: $(SRC_DIR)/csv.c $(SRC_DIR)/csv.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/util.o: $(SRC_DIR)/util.c $(SRC_DIR)/util.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/dsp.o: $(SRC_DIR)/dsp.c $(SRC_DIR)/dsp.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/mfcc.o: $(SRC_DIR)/mfcc.c $(SRC_DIR)/mfcc.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/frame_io.o: $(SRC_DIR)/frame_io.c $(SRC_DIR)/frame_io.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR)/organize_dataset: $(SRC_DIR)/organize_dataset.c $(COMMON_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/preprocess: $(SRC_DIR)/preprocess.c $(OBJ_DIR)/frame_io.o $(OBJ_DIR)/util.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lsndfile $(LDLIBS)

$(BIN_DIR)/extract_features: $(SRC_DIR)/extract_features.c $(COMMON_OBJS) $(DSP_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/split_normalize: $(SRC_DIR)/split_normalize.c $(COMMON_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/train: $(SRC_DIR)/train.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lfann $(LDLIBS)

$(BIN_DIR)/evaluate: $(SRC_DIR)/evaluate.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lfann $(LDLIBS)

$(BIN_DIR)/plot_confusion: $(SRC_DIR)/plot_confusion.c $(OBJ_DIR)/csv.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/cross_validate: $(SRC_DIR)/cross_validate.c $(OBJ_DIR)/csv.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lfann $(LDLIBS)

ifeq ($(HAVE_SVM),yes)
$(BIN_DIR)/svm_baseline: $(SRC_DIR)/svm_baseline.c | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SVM_CPPFLAGS) $^ -o $@ $(LDFLAGS) -lsvm $(LDLIBS)
else
$(BIN_DIR)/svm_baseline:
	@echo "libsvm headers not found; install libsvm-dev or set SVM_CPPFLAGS to build svm_baseline"
	@exit 1
endif

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)

best_run: all
	./scripts/run_best_pipeline.sh

.PHONY: all clean best_run
