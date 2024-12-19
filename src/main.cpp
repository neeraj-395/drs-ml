#include "preprocess.hpp"
#include "network.hpp"

int main() {
    MatrixD inputs = MNIST::load_processed_images("data/train-images.idx3-ubyte");
    MatrixD targets = MNIST::load_processed_labels("data/train-labels.idx1-ubyte");

    DataSplit mnist = MNIST::split_data(inputs, targets);

    Network<784, 10> net({10, 10});

    std::cout << "Training Neural Network...." << std::endl;
    net.train(mnist.train_inputs, mnist.train_targets, 10, 0.01);

    std::cout << "Validating Neural Network...." << std::endl;
    int score = net.accuracy_score(mnist.val_inputs, mnist.val_targets);

    double accuracy = static_cast<double>(score) / mnist.val_inputs.size();
    std::cout << "Validation Accuracy: " << accuracy * 100 << "%" << std::endl;

    std::cout << "Saving Neural Network...." << std::endl;
    net.save_network("models/network.json");
    std::cout << "The trained model is saved to 'models/network.json'." << std::endl;

    return 0;
}
