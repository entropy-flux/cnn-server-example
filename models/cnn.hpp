#include <tannic.hpp>
#include <tannic/reductions.hpp>
#include <tannic/serialization.hpp>
#include <tannic-nn.hpp> 
#include <tannic-nn/functional.hpp> 
#include <tannic-nn/convolutional.hpp> 

using namespace tannic;    
 
struct CNN : nn::Module {
    nn::Convolutional2D convolutional_layer;
    nn::Linear output_layer;

    constexpr CNN(type dtype, size_t input_channels, size_t hidden_channels, size_t output_size, bool bias = true) 
    :   convolutional_layer(dtype, input_channels, hidden_channels, 3, 1, 1, true)
    ,   output_layer(dtype, hidden_channels * 28 * 28, output_size, true) {}

    void initialize(nn::Parameters& parameters) const {
        convolutional_layer.initialize("convolutional_layer", parameters);
        output_layer.initialize("output_layer", parameters);
    }

    Tensor forward(Tensor features) const {     
        features = features.view(features.size(0), 1, 28, 28);
        features = convolutional_layer(features);
        features = nn::relu(features);
        features = flatten(features, 1);  // keep dim 0 (batch), flatten from dim 1 onward
        return output_layer(features); 
    }
};
  