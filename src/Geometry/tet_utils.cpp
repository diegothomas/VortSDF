#include <torch/extension.h>
#include <vector>

void helloworld() {
    std::cout << "Hello" << std::endl; 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("helloworld", &helloworld, "helloworld (CPP)");
}