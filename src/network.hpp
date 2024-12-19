#pragma once
#include "layer.hpp"
#include "../libs/nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <algorithm>

typedef std::vector<Layer> Layers;
using json = nlohmann::json;

template<size_t input_size, size_t output_size>
class Network {

public:
    Network(const std::vector<unsigned>& topology)
    :output_layer(topology.empty() ? 0 : topology.back(), output_size, SIGMOID)
    {
        if (topology.empty()) {
            throw std::invalid_argument("Topology must have at least one hidden layer.");
        }

        hidden_layers.emplace_back(input_size, topology.front(), RELU);

        for (size_t i = 1; i < topology.size(); ++i) {
            hidden_layers.emplace_back(topology[i - 1], topology[i], RELU);
        }
    }

    VectorD feedforward(const VectorD& inputs)
    {
        VectorD hidden_output = hidden_layers.front().forward(inputs);
        for (size_t i = 1; i < hidden_layers.size(); ++i) {
            hidden_output = hidden_layers[i].forward(hidden_output);
        }
        return output_layer.forward(hidden_output);
    }

    void backpropagate(const VectorD& output_errors, const double& learning_rate) {
        VectorD prevl_errors = output_layer.backward(output_errors,learning_rate);
        for (int i = hidden_layers.size() - 1; i >= 0; --i) {
            prevl_errors = hidden_layers[i].backward(prevl_errors, learning_rate);
        }
    }

    void train(const MatrixD& inputs, const MatrixD& targets, const unsigned& epochs, 
               const double& learning_rate = 0.1)
    {
        if (inputs.size() != targets.size()) {
            throw std::invalid_argument("The number of inputs must match the number of targets");
        }

        for (unsigned epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {

                VectorD final_output = feedforward(inputs[i]);

                VectorD output_errors = errors(final_output, targets[i]);

                backpropagate(output_errors, learning_rate);
            }
        }
    }

    int accuracy_score(const MatrixD& inputs, const MatrixD& targets) {
        unsigned correct_predictions = 0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            VectorD output = feedforward(inputs[i]);

            auto max_output_it = std::max_element(output.begin(), output.end());

            int predicted_class = std::distance(output.begin(), max_output_it);

            auto max_target_it = std::max_element(targets[i].begin(), targets[i].end());

            int true_class = std::distance(targets[i].begin(), max_target_it);

            if (predicted_class == true_class) {
                correct_predictions++;
            }
        }
        return correct_predictions;
    }

    void save_network(const std::string& file_path) {
        std::ofstream file(file_path, std::ios::out);

        if (!file) {
            std::cerr << "Error creating file: " << file_path << std::endl;
            return;
        }

        json network_data;

        for (size_t i = 0; i < hidden_layers.size(); ++i) {
            std::string layer_name = "hidden_layer_" + std::to_string(i);
            network_data[layer_name]["weights"] = hidden_layers[i].get_weights();
            network_data[layer_name]["biases"] = hidden_layers[i].get_biases();
        }

        network_data["output_layer"]["weights"] = output_layer.get_weights();
        network_data["output_layer"]["biases"] = output_layer.get_biases();

        file << network_data.dump(4);

        file.close();
    }

private:
    Layer output_layer;
    Layers hidden_layers;

    VectorD errors(const VectorD& outputs, const VectorD& targets) {
        VectorD output_errors(outputs.size());
        for(size_t i = 0; i < outputs.size(); ++i) {
            output_errors[i] = targets[i] - outputs[i];
        }
        return output_errors;
    }

    double mean_squared_error(const VectorD& outputs, const VectorD& targets) {
        double error = 0.0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            error += (targets[i] - outputs[i]) * (targets[i] - outputs[i]);
        }
        return error / outputs.size();
    }
};