#include <cstdint>
#include <iostream>
#include <cstring>
#include <memory>
#include <tannic.hpp>
#include <tannic/reductions.hpp>
#include <tannic/serialization.hpp>
#include <tannic-nn.hpp> 
#include <tannic-nn/functional.hpp> 
#include <tannic-nn/convolutional.hpp>
#include "server.hpp" 
#include "models/cnn.hpp"

using namespace tannic;    
 
constexpr CNN model(float32, 1, 32, 10);

int main() {  
    nn::Parameters parameters;  
    parameters.initialize("../data/tannic/CNN");
    model.initialize(parameters);

    Server server(8080);
    while (true) {
        Socket socket = server.accept();  

        try {
            while (true) {
                Header header{};
                if (!server.read(socket, &header, sizeof(Header))) {
                    std::cout << "Client disconnected.\n";
                    break; 
                }

                if (header.magic != MAGIC) {
                    std::cerr << "Invalid magic! Closing connection.\n";
                    break;
                }

                Metadata<Tensor> metadata{};  
                if (!server.read(socket, &metadata, sizeof(Metadata<Tensor>))) {
                    std::cout << "Client disconnected.\n";
                    break;
                }

                Shape shape; 
                size_t size;
                for (uint8_t dimension = 0; dimension < metadata.rank; dimension++) {
                    if (!server.read(socket, &size, sizeof(size_t))) {
                        std::cout << "Client disconnected.\n";
                        break;
                    }
                    shape.expand(size);
                }

                std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(metadata.nbytes);
                if (!server.read(socket, buffer->address(), metadata.nbytes)) {
                    std::cout << "Client disconnected.\n";
                    break;
                }

                Tensor input(dtypeof(metadata.dcode), shape, 0, buffer);  
                Tensor output = argmax(model(input), -1); 

                header = headerof(output);
                metadata = metadataof(output);

                server.write(socket, &header, sizeof(Header));
                server.write(socket, &metadata, sizeof(Metadata<Tensor>)); 
                server.write(socket, output.shape().address(), output.shape().rank() * sizeof(size_t));
                server.write(socket, output.bytes(), output.nbytes());
            }

        } catch (const std::exception& e) {
            std::cerr << "Unexpected client error: " << e.what() << "\n";
        }
    } 
}