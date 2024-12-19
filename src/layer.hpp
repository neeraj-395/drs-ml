#pragma once
#include "activation.hpp"

#include <vector>
#include <random>

typedef std::vector<double> VectorD;
typedef std::vector<VectorD> MatrixD;

class Layer {
private:
    VectorD biases;
    VectorD inputs;
    MatrixD weights;
    VectorD outputs;
    const size_t num_inputs;
    const size_t num_neurons;
    ActFunc activation;
    ActFuncPrime activation_prime;

public:
    Layer(const size_t& __num_inputs, const size_t& __num_neurons, Activation type)
    :num_inputs(__num_inputs), num_neurons(__num_neurons), activation(activation_func(type)), 
     activation_prime(activation_prime_func(type))
    {
        std::mt19937 rand(0);
        double scale_factor = std::sqrt(2.0 / static_cast<double>(__num_inputs));
        std::uniform_real_distribution<double> dist(-scale_factor, scale_factor);
        weights.resize(num_neurons, VectorD(num_inputs));
        outputs.resize(num_neurons);
        biases.resize(num_neurons);
        inputs.resize(num_inputs);

        for (size_t i = 0; i < num_neurons; ++i) {
            for (size_t j = 0; j < num_inputs; ++j) {
                weights[i][j] = dist(rand);
            } biases[i] = dist(rand);
        }
    }

    VectorD forward(const VectorD& _inputs) 
    {
        inputs = _inputs;
        outputs.resize(num_neurons, 0.0);
        for (size_t i = 0; i < num_neurons; ++i) {
            double activation_input = biases[i];
            for (size_t j = 0; j < num_inputs; ++j) {
                activation_input += weights[i][j] * inputs[j];
            }
            outputs[i] = activation(activation_input);
        }
        return outputs;
    }

    VectorD backward(const VectorD& d_outputs, const double& learning_rate)
    {
        VectorD d_inputs(num_inputs, 0.0);
        for (size_t i = 0; i < num_neurons; ++i) {
            const double delta = d_outputs[i] * activation_prime(outputs[i]);
            for (size_t j = 0; j < num_inputs; ++j) {
                d_inputs[j] += weights[i][j] * delta;
                weights[i][j] += learning_rate * delta * inputs[j];
            }
            biases[i] += learning_rate * delta;
        }
        return d_inputs;
    }

    const MatrixD& get_weights() const { return weights; }
    const VectorD& get_biases() const { return biases; }
};