# Tannic-NN Example: Serving a simple CNN

This repository contains a minimal example of how to use [Tannic-NN](https://github.com/entropy-flux/Tannic-NN) to serve a **Convolutional Neural Network (CNN)** model over a simple socket-based server.  

It is intended as a **baseline reference implementation**, a ground to build upon showing the essential steps of serving a model with Tannic-NN in the simplest possible way.  
The Tannic tensor library is still unoptimized, **it is not optimized and therefore runs relatively slow**. The goal is to provide a transparent starting point, not a production-ready server.

This example demonstrates:  
- Initializing model parameters.
- Running inference with a CNN model built with **Tannic-NN**.
- Sending and receiving tensors through a custom TCP server with a python client.

---

## Example

The code sets up a `Server` listening on port `8080` and exposes a **CNN** for inference.  
Clients can connect, send input tensors (e.g., images), and receive the model’s predictions.

Key components:
- **CNN Model** (`include/vit.hpp`): Defined using Tannic-NN building blocks.  
- **Server** (`include/server.hpp`): Minimal TCP server for exchanging tensors.    
- **Client** (`evaluate.py`): Minimal python client sending a picture an printing probabilities. 


### Demo: Evaluating digit MNIST accuracy. 

There is a simple CNN trained for a few epochs in the models folder. Since the model is small, the weights are uploaded to github, no need to download them separately.

You can run an evaluation to evaluate the C++ model over network, using the evaluate script:

```bash
python evaluate.py 
Accuracy: 0.9766
Accuracy: 0.9805
Accuracy: 0.9844
Accuracy: 0.9785
Accuracy: 0.9742
...
Done!
```
 
---

## 📦 Dependencies 

- A C++23 compiler (GCC ≥ 14, Clang ≥ 16, etc.)
- CMake ≥ 3.30

You will need PyTorch and PyTannic (Just for the example, the server doesn't rely on python.)
PyTannic is a binding I created to easily send torch tensors to the server, you can install it with pip just as:

```bash
pip install pytannic
```

Torcheval is used for calculation of metrics in the `evaluation.py` script, install it as:

```bash
pip install torcheval
```

For the example you should also install the `requests` package, since it will try to download an image from internet. 

```bash
pip install requests
```

For training the model you will need `mltracker` and `torchsystem`. They are also easily installable from pip:

```bash
pip install mltracker torchsystem
``` 
---

## 🛠️ Build Instructions
  
Clone this repo (with submodules) alongside **Tannic-NN**:

```bash
git clone https://github.com/your-username/vit-server-example.git
cd vit-server-example
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make -j$(nproc) 
```  

Then from the build directory run the executable:

```bash
./vit-server
```

This will setup a `Server` listening on port `8080` exposing the model. You can try it now just running:

```bash
python evaluate.py
```

It will download mnist digits dataset and evaluate the CNN over it. 

⚠️ Note: The Tannic tensor library is still unoptimized, so inference may take a few minutes.
This repository is meant as a baseline reference implementation clear and minimal rather than a fast production-ready server.