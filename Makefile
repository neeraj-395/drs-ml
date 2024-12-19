# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -I./libs/nlohmann

# Directories
SRC_DIR = src
BUILD_DIR = build
DATA_DIR = data
STATIC_DIR = static
MODELS_DIR = models

# Files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
EXEC = $(BUILD_DIR)/main.out

# Targets
.PHONY: all clean run

all: $(EXEC)

# Build the executable
$(EXEC): $(OBJS)
	@echo "Linking objects to create executable..."
	$(CXX) $(OBJS) -o $@

# Compile each .cpp file to an object file
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	@echo "Cleaning up build files..."
	rm -rf $(BUILD_DIR)

# Train the model
train: all
	@echo "Running the program..."
	$(EXEC)
