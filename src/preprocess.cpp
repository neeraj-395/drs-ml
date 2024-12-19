#include "preprocess.hpp"

#include <iostream>
#include <fstream>
#include <random>
#include <array>
#include <algorithm>

inline uint32_t byteswap32(const uint32_t& value) {
    return ((value & 0x000000FF) << 24) | ((value & 0x0000FF00) << 8) |
            ((value & 0x00FF0000) >> 8) | ((value & 0xFF000000) >> 24);
}

static std::vector<double> normalizer(const MNISTImage &image) {
    std::vector<double> normalized(image.pixels.size());
    for (size_t i = 0; i < image.pixels.size(); ++i) {
        normalized[i] = image.pixels[i] / 255.0;
    }
    return normalized;
}

static std::array<double, 10> one_hot_encoder(const MNISTLabel &label) {
    std::array<double, 10> target = { 0.0 };
    target[label.label] = 1.0;
    return target;
}

std::vector<MNISTImage> MNIST::load_images(const std::string &filename) {
    std::ifstream file(filename, std::ios_base::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    uint32_t magic_num = 0, num_of_img = 0, rows = 0, cols = 0;

    file.read(reinterpret_cast<char *>(&magic_num), 4);
    file.read(reinterpret_cast<char *>(&num_of_img), 4);
    file.read(reinterpret_cast<char *>(&rows), 4);
    file.read(reinterpret_cast<char *>(&cols), 4);

    if (byteswap32(magic_num) != 2051) {
        throw std::runtime_error("Invalid magic number in MNIST image file.");
    }

    num_of_img = byteswap32(num_of_img);
    rows = byteswap32(rows);
    cols = byteswap32(cols);

    std::vector<MNISTImage> images(num_of_img);
    for (size_t i=0; i < num_of_img; ++i) {
        MNISTImage image;
        image.pixels.resize(rows * cols);
        file.read(reinterpret_cast<char *>(image.pixels.data()), rows * cols);
        images[i] = image;
    }

    file.close();
    return images;
}

std::vector<MNISTLabel> MNIST::load_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    uint32_t magic_num = 0, num_of_labels = 0;

    file.read(reinterpret_cast<char *>(&magic_num), 4);
    file.read(reinterpret_cast<char *>(&num_of_labels), 4);

    if (byteswap32(magic_num) != 2049) {
        throw std::runtime_error("Invalid magic number in MNIST label file.");
    }

    num_of_labels = byteswap32(num_of_labels);
    std::vector<MNISTLabel> labels(num_of_labels);
    file.read(reinterpret_cast<char *>(labels.data()), num_of_labels);

    file.close();
    return labels;
}

MatrixD MNIST::load_processed_images(const std::string& filename) {
    std::vector<MNISTImage> images = load_images(filename);
    MatrixD normalized_images;

    for (const auto &image : images) {
        normalized_images.push_back(normalizer(image));
    }   return normalized_images;
}

MatrixD MNIST::load_processed_labels(const std::string& filename) {
    std::vector<MNISTLabel> labels = load_labels(filename);
    MatrixD encoded_labels;

    for (const auto &label : labels) {
        auto one_hot_vector = one_hot_encoder(label);
        encoded_labels.emplace_back(one_hot_vector.begin(), 
                                    one_hot_vector.end());
    }   return encoded_labels;
}


DataSplit MNIST::split_data(const MatrixD& inputs, const MatrixD& targets, 
                            const double& train_ratio) {
    size_t total_size = inputs.size();
    size_t train_size = static_cast<size_t>(inputs.size() * train_ratio);

    // Create indices and shuffle them
    std::vector<size_t> indices(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        indices[i] = i;
    }

    std::mt19937 g(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), g);

    MatrixD train_inputs, train_targets;
    MatrixD val_inputs, val_targets;
    train_inputs.reserve(train_size);
    train_targets.reserve(train_size);
    val_inputs.reserve(total_size - train_size);
    val_targets.reserve(total_size - train_size);

    // Split data based on shuffled indices
    for (size_t i = 0; i < train_size; ++i) {
        train_inputs.push_back(inputs[indices[i]]);
        train_targets.push_back(targets[indices[i]]);
    }
    for (size_t i = train_size; i < total_size; ++i) {
        val_inputs.push_back(inputs[indices[i]]);
        val_targets.push_back(targets[indices[i]]);
    }

    return (DataSplit) { train_inputs, train_targets,
                         val_inputs, val_targets };
}
