############################################
# 1.) Base: NVIDIA TensorFlow 25.02 (CUDA 12.8, Python 3.12)
############################################
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

############################################
# 2.) Working directory
############################################
WORKDIR /workspace

############################################
# 3.) System-level dependencies
############################################	
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake git curl python3-dev python3-pip \
        graphviz libgl1 libglib2.0-0 \
        libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

############################################
# 4.) Ensure python symlink exists (forced)
############################################
RUN ln -sf /usr/bin/python3 /usr/bin/python

############################################
# 5.) Install Node.js
############################################
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest

############################################
# 6.) Build-time environment
############################################
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

############################################
# 7.) CUDA build policy toggle
# Default ON for early Blackwell (sm_120).
# Set to "off" later to test llama.cpp native kernels.
############################################
ARG FORCE_CUBLAS=on

############################################
# 8.) Copy requirements for caching
############################################
COPY dockerfiles/rtx-5060_dev-requirements.txt .

############################################
# 9.) Build python dependencies for sm_120
############################################
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    # Create a constraints file to lock the entire NVIDIA compatibility layer
    echo "numpy==1.26.4" > /tmp/constraints.txt && \
    echo "ml-dtypes==0.4.1" >> /tmp/constraints.txt && \
    \
    # I. Install main requirements using the constraints and filtering
    grep -vE "llama-cpp-python|tensorflow|numpy|ml-dtypes" rtx-5060_dev-requirements.txt > /tmp/reqs.txt || true && \
    python3 -m pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/reqs.txt && \
    \
    # II. Build llama-cpp-python
    # Using the constraint here prevents the build-wheels process from pulling NumPy 2.x
    export CMAKE_ARGS="-DGGML_CUDA=on \
                       -DGGML_CUDA_FORCE_CUBLAS=${FORCE_CUBLAS} \
                       -DCMAKE_CUDA_ARCHITECTURES=120 \
                       -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" && \
    export FORCE_CMAKE=1 && \
    python3 -m pip install llama-cpp-python --no-cache-dir -c /tmp/constraints.txt --force-reinstall && \
    \
    # III. Force-reinstall both to ensure they own the /usr/local/ path
    python3 -m pip install --no-cache-dir "numpy==1.26.4" "ml-dtypes==0.4.1" --force-reinstall

############################################
# 10.) Runtime optimization
############################################
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_MODULE_LOADING=LAZY
ENV TF_ENABLE_ONEDNN_OPTS=0

############################################
# 11.) Verification (smoke test) - C-level check
############################################
RUN echo "import torch; \
import llama_cpp.llama_cpp as llm; \
cap = torch.cuda.get_device_capability(0); \
print(f'Detected: {torch.cuda.get_device_name(0)} | sm_{cap[0]}{cap[1]}'); \
assert cap[0] >= 12, 'Blackwell GPU not found'; \
print('Testing llama-cpp GPU backend...'); \
# This reaches into the compiled GGML binary to check support \
has_gpu = llm.llama_supports_gpu_offload(); \
print(f'llama-cpp GPU Support: {has_gpu}'); \
assert has_gpu, 'llama-cpp was built without CUDA support!'; \
print('Hardware verification passed!')" > /usr/local/bin/verify_gpu.py

############################################
# 12.) Ports exposed to the host
############################################
EXPOSE 8888 5173 8000

############################################
# 13.) Default entry
############################################
CMD ["bash"]