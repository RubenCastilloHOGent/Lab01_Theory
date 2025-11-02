# Lab Report: 01 - DOCKER LAB

## Student information

- Student name: RUBEN CASTILLO SAINZ
- Student code: 410529

## Assignment description

This lab teaches how to use Docker in MLOps for building, deploying and orchestrating ML models.
The lab is split into three main parts:

### Part 1: Docker Basics in ML Context

This first part of the lab shows you how to run a small machine learning model inside a Docker container by using a Flask application and making it available via an API.

The goal of this part is to understand the basics of Docker, how to create and run containers for your own models and in a practical way, see in first hand why Docker is a valuable tool in MLOps.

The Flask application Python code and the TensorFlow model is provided in the resources of this lab so the focus is purely on getting them up and running in Docker.

To do so, the main focus of this lab is on writting a Dockerfile that:

    - Imports a lightweight Python base image (Slim version);
    - Copies over the requirements and installs the dependencies;
    - Exposes port 5000;
    - Sets the working directory inside the container to /APP;
    - Copies the saved 'model_repository' directory and the Flask-based app.py into the container (Both based on the provided scripts in the resources of this lab);
    - Runs the Flask application, which serves the TensorFlow model via a web API.
    - If everything went fine, you should see the results of the model in the API response.

Once the Docker image is generated and everything is running fine, we learn how to push the Docker Image to a Docker Registry like Docker Hub.

`Key ake away`: Understanding tha basics of Docker by deploying a model in a container. 
Also, pushing the Docker image to a Docker registry to show case real life scenarios where Docker Images are not only run in your local machine.

### Part 2: Triton Serving

In the second part of this lab, we are introduced to NVIDA Triton Inference Server. 
An open source software developed by NVIDIA that simplifies deploying and running AI models in production.

In the first part of Part 2, we had to deploy the model used in Part 1 to the Triton Server, run the Triton Server and test its endpoints.

Then, in the second part of Part 2, we were given the chance to choose an existing public model and deploy it in Triton Server.
By doing this, we could see how Triton allows you to run multiple models simultaneously.

We could choose the model from:
    - https://catalog.ngc.nvidia.com/models
    - https://tfhub.dev/
    - https://huggingface.co/models

Based on certain criterias:
    - Choose a model that's compatible with Triton (ONNX, TensorRT, or TensorFlow SavedModel format)
    - Look for models with clear documentation and download instructions
    - Consider model size - smaller models will download and load faster for this exercise
    - Popular categories include image classification, object detection, or text processing models

`Key take away`: Learning how to use a real production server for deploying Machine Learning models which is capable of running multiple models simultaneously and handling performance in a scalable manner.

### Part 3: Docker Compose

In the third part of this lab, we learn how to orchestrate and manage multiple containers.
Starting each container with a `docker run` is not sustainable in real life and production environments so we learn how to use Docker Compose to orchestrate both services in one command by putting several containers setups inn one file called `docker-compose.yml`

To showcase it, we build a docker-compose.yml file where we put together both services:
    - ml-flask-app of Part 1.
    - Triton-server sevice of Part 2.

Then we can run both services together with one command.

`Key take away`: Simplifying the way you manage multi container setups and applications by "bundeling" them together.

## Proof of work done

Include screenshots, code snippets, etc. to prove that you completed the assignment. If you have a lot of code, you can link to a separate file or repository.

## Evaluation criteria

### Part 1: Docker Basics

- [x] Show that you installed Docker and Docker Compose by running the following commands:  
  `docker --version` and `docker compose version`

    Terminal output of executing docker --version and docker compose version in my Macbook Air terminal:

    ```bash
    (base) rubencastillosainz@Rubens-MacBook-Air 01-dockerlab % docker --version
    Docker version 28.5.1, build e180ab8

    (base) rubencastillosainz@Rubens-MacBook-Air 01-dockerlab % docker compose version
    Docker Compose version v2.40.2-desktop.1
    ```

- [x] Show that you created a Dockerfile to host an ML model using Flask  
  A Dockerfile was created using a lightweight Python base image, installing dependencies, and serving a TensorFlow model via Flask.

    !["Docker File"](.img/01-docker-lab/AcceptanceCriteria_1_2.png)

- [x] Show that you can build the Docker image using your Dockerfile  
  The image 'ml-flask-app' was built successfully.

    Building the image:

    ```bash
    (venv) (base) rubencastillosainz@Rubens-MacBook-Air 01-dockerlab % docker build -t ml-flask-app .
    [+] Building 75.0s (11/11) FINISHED                                                                                                                                     docker:desktop-linux
    => [internal] load build definition from Dockerfile                                                                                                                                    0.0s
    => => transferring dockerfile: 568B                                                                                                                                                    0.0s
    => [internal] load metadata for docker.io/library/python:3.12-slim                                                                                                                     0.5s
    => [internal] load .dockerignore                                                                                                                                                       0.0s
    => => transferring context: 2B                                                                                                                                                         0.0s
    => [1/6] FROM docker.io/library/python:3.12-slim@sha256:e97cf9a2e84d604941d9902f00616db7466ff302af4b1c3c67fb7c522efa8ed9                                                               0.0s
    => => resolve docker.io/library/python:3.12-slim@sha256:e97cf9a2e84d604941d9902f00616db7466ff302af4b1c3c67fb7c522efa8ed9                                                               0.0s
    => [internal] load build context                                                                                                                                                       0.0s
    => => transferring context: 904B                                                                                                                                                       0.0s
    => CACHED [2/6] WORKDIR /app                                                                                                                                                           0.0s
    => [3/6] COPY requirements.txt .                                                                                                                                                       0.0s
    => [4/6] RUN pip install --no-cache-dir -r requirements.txt                                                                                                                           42.0s
    => [5/6] COPY model_repository/ ./model_repository/                                                                                                                                    0.1s 
    => [6/6] COPY app.py .                                                                                                                                                                 0.0s 
    => exporting to image                                                                                                                                                                 32.1s 
    => => exporting layers                                                                                                                                                                26.6s 
    => => exporting manifest sha256:1f9fca83f78787c5ea148f1ca9a9cd9b5d825a039e437d8f470e1b4bf277f084                                                                                       0.0s 
    => => exporting config sha256:91ebe17b084caf9eb4c3d2082d81f9cf96cbe78d3c1210875241900288461849                                                                                         0.0s 
    => => exporting attestation manifest sha256:1168cab598837c3b4537d6045c3380cc91dbf61aee7c8397034bdc9e7863e0b9                                                                           0.0s
    => => exporting manifest list sha256:7b07bebe45389a7d0820d934db8653c88fc2bf718540d03babb95259926fb4f7                                                                                  0.0s
    => => naming to docker.io/library/ml-flask-app:latest                                                                                                                                  0.0s
    => => unpacking to docker.io/library/ml-flask-app:latest                                                                                                                               5.5s
    (venv) (base) rubencastillosainz@Rubens-MacBook-Air 01-dockerlab % docker run -d -p 5000:5000 ml-flask-app
    * Serving Flask app 'app'
    * Debug mode: on
    WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
    * Running on all addresses (0.0.0.0)
    * Running on http://127.0.0.1:5000
    * Running on http://172.17.0.2:5000
    Press CTRL+C to quit
    * Restarting with stat
    * Debugger is active!
    * Debugger PIN: 765-927-578
    192.168.65.1 - - [26/Oct/2025 11:46:00] "GET / HTTP/1.1" 404 -
    192.168.65.1 - - [26/Oct/2025 11:46:00] "GET /favicon.ico HTTP/1.1" 404 -
    192.168.65.1 - - [26/Oct/2025 11:46:13] "GET / HTTP/1.1" 404 -
    ```


- [x] Show that you can run the Docker container from your image  
  The container was run with port mapping, allowing access to the Flask app via `http://localhost:5000/health`.

    Health check:

    ```bash
    curl http://localhost:5000/health
    {
    "model_loaded": true,
    "status": "healthy"
    }
    ```

    Model result:

    ```bash
    curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [1.2, 3.4, 5.6, 7.8]}'
    {
    "confidence": 0.995094895362854,
    "prediction": 1,
    "raw_output": 0.995094895362854
    }
    ```

- [x] Show how you push and pull the Docker image to your container registry  

  The image was tagged as `0.0.1` and pushed to Docker Hub: [rubencastillohogent/ml-flask-app](https://hub.docker.com/repository/docker/rubencastillohogent/ml-flask-app/general). 

    Tagging the image:

    ```bash
    docker tag ml-flask-app rubencastillohogent/ml-flask-app:0.0.1
    ```

    Pushing the image:

    ```bash
    docker push rubencastillohogent/ml-flask-app:0.0.1
    ```

    !["Image pushed to Docker Hub Registry."](./img/01-docker-lab/AcceptanceCriteria_1_4.png)

#### Questions:
❓ Why do we copy requirements.txt before copying the application code? How does this improve Docker layer caching?

    Because every instruction in a Dockerfile creates a layer. Docker caches the layers to avoid re-running the same commands if nothing has changed.

    This way, if you later on change something on your Python Application but requirements.txt stays the same, Docker won’t execute requirements.txt and it will reused the cached layer instead of reinstalling all the dependencies.

    This makes builds so much faster and efficient.


❓ What is the difference between python:3.12 and python:3.12-slim? What are the trade-offs?

    - Python:3.12 is the full Python version which is quite heavy (around 1GB).
    - Python:3.12-slim is a slimmed down version of the full image which is lighter because it includes only what is minimally required to run Python (10x smaller)

    The slim version is much faster due to its size and it is recommended for Production to it lightweight version. The full Python version is more recommended for Development/Testing environments rather than Production environments.

❓ What does the -t flag do in the docker build command? Why is it useful to tag your images?

   `-t` stands for TAG.

    Tagging an image gives it a human readable name to your image which helps humans to quickly identify an image instead of by their ID.

    It also allows you to quickly give it version controls if needed.

❓ What does the -p 5000:5000 flag do? 
    -p maps a port in your host machine to a port inside the container.

    5000:5000 means:

    - Host Port is 5000: the port you use in your browser or API requests
    - Container Port is 5000: the port your Flask app is listening on inside the container

❓ What happens if you try to run the container without the -p flag? Can you still access the API?

    In that case, you are not mapping the ports from your local machine to the container. The Container still runs but is not accessible from your host machine because they are isolated. So the answer is no, I won't be able to access the API.

    This is why we need to map the ports by using `-p`.

    In the past you could also use the EXPOSE command within your Dockerfile but this is no longer the case and nowadays is just for documentation purposes.

❓ Run docker images after building. What information does this show you about your image?

    ```bash
    docker images
    REPOSITORY                         TAG             IMAGE ID       CREATED         SIZE
    01-dockerlab-ml-flask-app          latest          3544e60e48f8   5 days ago      2.84GB
    ml-flask-app                       latest          8a60a9030c0c   5 days ago      2.84GB
    rubencastillohogent/ml-flask-app   0.0.1           0e484881b3b9   6 days ago      2.66GB
    <none>                             <none>          7b07bebe4538   6 days ago      2.66GB
    mysite-backend                     latest          7cfe8e798e81   7 days ago      332MB
    mysite-frontend                    latest          2956d0d1b7c8   7 days ago      275MB
    mysite                             latest          3e234693a59c   7 days ago      275MB
    nginx                              latest          029d4461bd98   3 weeks ago     244MB
    hello-world                        latest          6dc565aa6309   2 months ago    16.9kB
    mongo                              7.0.12          ae1cf99fa7bf   16 months ago   1.01GB
    mongo-express                      1.0.2           1b23d7976f02   20 months ago   285MB
    nvcr.io/nvidia/tritonserver        23.10-py3       83513cb05c7e   2 years ago     17.2GB
    nvcr.io/nvidia/tritonserver        23.10-py3-sdk   3e0d1a0336bf   2 years ago     14.5GB
    ```

❓ Use docker ps to see running containers. 

    `docker ps` shows all running containers at this moment in my machine:

    ```bash
    docker ps
    CONTAINER ID   IMAGE                                   COMMAND                  CREATED             STATUS             PORTS                                                             NAMES
    9bae43b0a21d   nvcr.io/nvidia/tritonserver:23.10-py3   "/opt/nvidia/nvidia_…"   About an hour ago   Up About an hour   0.0.0.0:8000-8002->8000-8002/tcp, [::]:8000-8002->8000-8002/tcp   nice_bose

    ```
    If I run `docker ps -a` then I can see all containers running and stopped in my machine:

    ```bash
    docker ps -a
    CONTAINER ID   IMAGE                                   COMMAND                  CREATED             STATUS                           PORTS                                                             NAMES
    9bae43b0a21d   nvcr.io/nvidia/tritonserver:23.10-py3   "/opt/nvidia/nvidia_…"   About an hour ago   Up About an hour                 0.0.0.0:8000-8002->8000-8002/tcp, [::]:8000-8002->8000-8002/tcp   nice_bose
    35161704a794   nvcr.io/nvidia/tritonserver:23.10-py3   "/opt/nvidia/nvidia_…"   About an hour ago   Exited (0) About an hour ago                                                                       distracted_black
    f73300c372ce   nvcr.io/nvidia/tritonserver:23.10-py3   "/opt/nvidia/nvidia_…"   About an hour ago   Created                                                                                            intelligent_raman
    f7735a3ae3e2   ml-flask-app                            "python app.py"          2 hours ago         Created                                                                                            condescending_napier
    3d57443a2f7d   01-dockerlab-ml-flask-app               "python app.py"          4 hours ago         Exited (0) About an hour ago                                                                       ml-flask-app
    a1014d410d2c   nvcr.io/nvidia/tritonserver:23.10-py3   "/opt/nvidia/nvidia_…"   4 hours ago         Exited (137) About an hour ago                                                                     triton-server
    49744caa17a4   nvcr.io/nvidia/tritonserver:23.10-py3   "/opt/nvidia/nvidia_…"   5 days ago          Exited (137) 5 days ago                                                                            nostalgic_lovelace
    99cd91b068e4   nvcr.io/nvidia/tritonserver:23.10-py3   "/opt/nvidia/nvidia_…"   5 days ago          Exited (0) 5 days ago                                                                              cranky_cartwright
    2ec6ca3c7234   nvcr.io/nvidia/tritonserver:23.10-py3   "/opt/nvidia/nvidia_…"   5 days ago          Exited (1) 5 days ago                                                                              keen_chebyshev
    97b95b0dcc96   nvcr.io/nvidia/tritonserver:23.10-py3   "/opt/nvidia/nvidia_…"   5 days ago          Exited (1) 5 days ago                                                                              festive_bouman
    8d42d3768a41   nvcr.io/nvidia/tritonserver:23.10-py3   "/opt/nvidia/nvidia_…"   6 days ago          Exited (1) 5 days ago                                                                              nice_tu
    e158538c1eb1   nvcr.io/nvidia/tritonserver:23.10-py3   "/opt/nvidia/nvidia_…"   6 days ago          Created                                                                                            kind_taussig
    2d59894290e6   nvcr.io/nvidia/tritonserver:23.10-py3   "/opt/nvidia/nvidia_…"   6 days ago          Created                                                                                            mystifying_gauss
    b35fa3735f11   0e484881b3b9                            "python app.py"          6 days ago          Exited (137) 5 days ago                                                                            adoring_vaughan
    3c80377c9397   0e484881b3b9                            "python app.py"          6 days ago          Exited (0) 6 days ago                                                                              stoic_moore
    b3dd1bfdba79   0e484881b3b9                            "python app.py"          6 days ago          Created                                                                                            boring_boyd
    b7559f771de9   7b07bebe4538                            "-d"                     6 days ago          Created                                                                                            serene_cartwright
    06e4090107b1   7b07bebe4538                            "python app.py"          6 days ago          Exited (0) 6 days ago                                                                              magical_germain
    4a8b358cd5e4   mysite-frontend                         "/docker-entrypoint.…"   7 days ago          Exited (0) 6 days ago                                                                              mysite-frontend
    f298a6050f38   mysite-backend                          "uvicorn mysite.main…"   7 days ago          Exited (0) 6 days ago                                                                              mysite-backend
    fec8a9462eb7   mongo:7.0.12                            "docker-entrypoint.s…"   7 days ago          Exited (0) 6 days ago                                                                              mysite-mongodb
    cfa0458be037   mongo-express:1.0.2                     "/sbin/tini -- /dock…"   7 days ago          Exited (143) 6 days ago                                                                            mysite-mongo-express
    3202a0e7f980   14be83748aa1                            "/docker-entrypoint.…"   7 days ago          Exited (0) 7 days ago                                                                              peaceful_swanson
    452ea52f8df4   21c41ca0d0ed                            "uvicorn mysite.main…"   7 days ago          Exited (0) 7 days ago                                                                              jolly_shamir
    45643a1dc233   mysite                                  "/docker-entrypoint.…"   7 days ago          Exited (0) 7 days ago                                                                              adoring_leavitt
    18cda663315a   hello-world                             "/hello"                 9 days ago          Exited (0) 9 days ago                                                                              condescending_perlman

    ```

❓ What is the purpose of tagging an image before pushing? 
    - Allowing version control to keep track of different versions of an application.
    - Allowing referencing to a specific image.
    - Create environment distinctions too.
    - Create roll backs to a latest version if the new version fails.

❓ What naming conventions should you follow for production images?
    - Use always lower case letters because Docker is case sensitive.
    - Don’t use spaces. In case a space is needed, better to use “_” or “-”.
    - Include **version tags** or semantic versioning (`v1.0.0`) instead of only `latest`.

❓ **What are the benefits of using container registries?** 

    By using container registries, you make sure that you have everything stored in one `place of truth`.
    By having one unique place to host all your containers, you allow teams to collaborate as they can pull and push images reliably, allowing version control.
    On top of it, you can then define security scans around those images in one unique place.

❓How do they fit into a CI/CD pipeline?

    Docker images encapsulate your app and its environment, registries store them safely, and CI/CD pipelines use them to automate testing, deployment, and rollback — making software delivery reliable and repeatable.

    1. CI (Continuous Integration)

    You make sure your code is identical to PROD eliminating the problems with local machines.

    As key elements and steps to make sure everything works fine:

    - Build a new Docker image whenever code is pushed.
    - Run automated tests inside the container.
    - Tag the image with version or commit SHA.

    Once the code is ready, you can then push it to the registry making sure your code is working fine once it gets to PROD:

    - After successful tests, push the tagged image to the container registry.

    2. CD (Continuous Deployment)

    You make sure your are deploying the exact tested image to production.

    Key steps are:

    - Deployment systems (Kubernetes, Docker Compose, etc.) will pull the image from the registry.
    - Ensures production runs the exact same image that passed tests = Environment is consistent!

### Part 2: Triton Serving

- [x] Show that you have deployed a TensorFlow model via a Triton container  
    Executing the below command shows how the Tensorflow models have been deployed via a Triton container.
    In the end of the logs we can see how both models `example_model` and `all-MiniLM-L6-v2` are READY.

```bash
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models

=============================
== Triton Inference Server ==
=============================

NVIDIA Release 23.10 (build 72127510)
Triton Server Version 2.39.0

Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use the NVIDIA Container Toolkit to start this container with GPU support; see
   https://docs.nvidia.com/datacenter/cloud-native/ .

W1101 12:33:30.361935 1 pinned_memory_manager.cc:237] Unable to allocate pinned system memory, pinned memory pool will not be available: CUDA driver version is insufficient for CUDA runtime version
I1101 12:33:30.361992 1 cuda_memory_manager.cc:117] CUDA memory pool disabled
I1101 12:33:30.373922 1 model_lifecycle.cc:461] loading: example_model:1
I1101 12:33:30.374487 1 model_lifecycle.cc:461] loading: all-MiniLM-L6-v2:1
I1101 12:33:30.595637 1 tensorflow.cc:2577] TRITONBACKEND_Initialize: tensorflow
I1101 12:33:30.595670 1 tensorflow.cc:2587] Triton TRITONBACKEND API version: 1.16
I1101 12:33:30.595672 1 tensorflow.cc:2593] 'tensorflow' TRITONBACKEND API version: 1.16
I1101 12:33:30.595674 1 tensorflow.cc:2617] backend configuration:
{"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}}
I1101 12:33:30.595725 1 tensorflow.cc:2683] TRITONBACKEND_ModelInitialize: example_model (version 1)
2025-11-01 12:33:30.596526: I tensorflow/cc/saved_model/reader.cc:45] Reading SavedModel from: /models/example_model/1/model.savedmodel
2025-11-01 12:33:30.597243: I tensorflow/cc/saved_model/reader.cc:91] Reading meta graph with tags { serve }
2025-11-01 12:33:30.597256: I tensorflow/cc/saved_model/reader.cc:132] Reading SavedModel debug info (if present) from: /models/example_model/1/model.savedmodel
I1101 12:33:30.597470 1 onnxruntime.cc:2608] TRITONBACKEND_Initialize: onnxruntime
I1101 12:33:30.597490 1 onnxruntime.cc:2618] Triton TRITONBACKEND API version: 1.16
I1101 12:33:30.597497 1 onnxruntime.cc:2624] 'onnxruntime' TRITONBACKEND API version: 1.16
I1101 12:33:30.597499 1 onnxruntime.cc:2654] backend configuration:
{"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}}
I1101 12:33:30.605134 1 onnxruntime.cc:2719] TRITONBACKEND_ModelInitialize: all-MiniLM-L6-v2 (version 1)
2025-11-01 12:33:30.611366: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:375] MLIR V1 optimization pass is not enabled
2025-11-01 12:33:30.612116: I tensorflow/cc/saved_model/loader.cc:233] Restoring SavedModel bundle.
2025-11-01 12:33:30.634256: E tensorflow/core/framework/node_def_util.cc:676] NodeDef mentions attribute debug_name which is not in the op definition: Op<name=VarHandleOp; signature= -> resource:resource; attr=container:string,default=""; attr=shared_name:string,default=""; attr=dtype:type; attr=shape:
shape; attr=allowed_devices:list(string),default=[]; is_stateful=true> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node Variable}}                                                                                         2025-11-01 12:33:30.637159: I tensorflow/cc/saved_model/loader.cc:217] Running initialization op on SavedModel bundle at path: /models/example_model/1/model.savedmodel
2025-11-01 12:33:30.642686: I tensorflow/cc/saved_model/loader.cc:334] SavedModel load for tags { serve }; Status: success: OK. Took 46159 microseconds.
I1101 12:33:30.645960 1 tensorflow.cc:2732] TRITONBACKEND_ModelInstanceInitialize: example_model_0 (CPU device 0)
2025-11-01 12:33:30.646851: I tensorflow/cc/saved_model/reader.cc:45] Reading SavedModel from: /models/example_model/1/model.savedmodel
2025-11-01 12:33:30.647901: I tensorflow/cc/saved_model/reader.cc:91] Reading meta graph with tags { serve }
2025-11-01 12:33:30.647921: I tensorflow/cc/saved_model/reader.cc:132] Reading SavedModel debug info (if present) from: /models/example_model/1/model.savedmodel
2025-11-01 12:33:30.649149: I tensorflow/cc/saved_model/loader.cc:233] Restoring SavedModel bundle.
2025-11-01 12:33:30.670424: I tensorflow/cc/saved_model/loader.cc:217] Running initialization op on SavedModel bundle at path: /models/example_model/1/model.savedmodel
2025-11-01 12:33:30.675104: I tensorflow/cc/saved_model/loader.cc:334] SavedModel load for tags { serve }; Status: success: OK. Took 28257 microseconds.
I1101 12:33:30.675471 1 tensorflow.cc:2732] TRITONBACKEND_ModelInstanceInitialize: example_model_1 (CPU device 0)
2025-11-01 12:33:30.676961: I tensorflow/cc/saved_model/reader.cc:45] Reading SavedModel from: /models/example_model/1/model.savedmodel
2025-11-01 12:33:30.678036: I tensorflow/cc/saved_model/reader.cc:91] Reading meta graph with tags { serve }
2025-11-01 12:33:30.678054: I tensorflow/cc/saved_model/reader.cc:132] Reading SavedModel debug info (if present) from: /models/example_model/1/model.savedmodel
2025-11-01 12:33:30.678943: I tensorflow/cc/saved_model/loader.cc:233] Restoring SavedModel bundle.
2025-11-01 12:33:30.699287: I tensorflow/cc/saved_model/loader.cc:217] Running initialization op on SavedModel bundle at path: /models/example_model/1/model.savedmodel
2025-11-01 12:33:30.704225: I tensorflow/cc/saved_model/loader.cc:334] SavedModel load for tags { serve }; Status: success: OK. Took 27267 microseconds.
I1101 12:33:30.704728 1 model_lifecycle.cc:818] successfully loaded 'example_model'
W1101 12:33:30.829575 1 onnxruntime.cc:813] autofilled max_batch_size to 4 for model 'all-MiniLM-L6-v2' since batching is supporrted but no max_batch_size is specified in model configuration. Must specify max_batch_size to utilize autofill with a larger max batch size
I1101 12:33:30.832072 1 onnxruntime.cc:2784] TRITONBACKEND_ModelInstanceInitialize: all-MiniLM-L6-v2_0 (CPU device 0)
I1101 12:33:30.832095 1 onnxruntime.cc:2784] TRITONBACKEND_ModelInstanceInitialize: all-MiniLM-L6-v2_1 (CPU device 0)
I1101 12:33:31.116971 1 model_lifecycle.cc:818] successfully loaded 'all-MiniLM-L6-v2'
I1101 12:33:31.117047 1 server.cc:592] 
+------------------+------+
| Repository Agent | Path |
+------------------+------+
+------------------+------+

I1101 12:33:31.117076 1 server.cc:619] 
+-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Backend     | Path                                                            | Config                                                                                                                                                        |
+-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
| tensorflow  | /opt/tritonserver/backends/tensorflow/libtriton_tensorflow.so   | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}} |
| onnxruntime | /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime.so | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}} |
+-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+

I1101 12:33:31.117096 1 server.cc:662] 
+------------------+---------+--------+
| Model            | Version | Status |
+------------------+---------+--------+
| all-MiniLM-L6-v2 | 1       | READY  |
| example_model    | 1       | READY  |
+------------------+---------+--------+

I1101 12:33:31.117189 1 metrics.cc:710] Collecting CPU metrics
I1101 12:33:31.117284 1 tritonserver.cc:2458] 
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option                           | Value                                                                                                                                                                                                           |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| server_id                        | triton                                                                                                                                                                                                          |
| server_version                   | 2.39.0                                                                                                                                                                                                          |
| server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_shared_memory cuda_shared_memory binary_tensor_data parameters statistics trace logging |
| model_repository_path[0]         | /models                                                                                                                                                                                                         |
| model_control_mode               | MODE_NONE                                                                                                                                                                                                       |
| strict_model_config              | 0                                                                                                                                                                                                               |
| rate_limit                       | OFF                                                                                                                                                                                                             |
| pinned_memory_pool_byte_size     | 268435456                                                                                                                                                                                                       |
| min_supported_compute_capability | 6.0                                                                                                                                                                                                             |
| strict_readiness                 | 1                                                                                                                                                                                                               |
| exit_timeout                     | 30                                                                                                                                                                                                              |
| cache_enabled                    | 0                                                                                                                                                                                                               |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

I1101 12:33:31.118400 1 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
I1101 12:33:31.118523 1 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
I1101 12:33:31.169391 1 http_server.cc:270] Started Metrics Service at 0.0.0.0:8002
^CI1101 12:33:38.038845 1 server.cc:293] Waiting for in-flight requests to complete.
I1101 12:33:38.038876 1 server.cc:309] Timeout 30: Found 0 model versions that have in-flight inferences
Signal (2) received.
I1101 12:33:38.039208 1 tensorflow.cc:2770] TRITONBACKEND_ModelInstanceFinalize: delete instance state
I1101 12:33:38.039264 1 server.cc:324] All models are stopped, unloading models
I1101 12:33:38.039321 1 server.cc:331] Timeout 30: Found 2 live models and 0 in-flight non-inference requests
I1101 12:33:38.039649 1 onnxruntime.cc:2836] TRITONBACKEND_ModelInstanceFinalize: delete instance state
I1101 12:33:38.040891 1 tensorflow.cc:2770] TRITONBACKEND_ModelInstanceFinalize: delete instance state
I1101 12:33:38.040919 1 tensorflow.cc:2709] TRITONBACKEND_ModelFinalize: delete model state
I1101 12:33:38.042856 1 model_lifecycle.cc:603] successfully unloaded 'example_model' version 1
I1101 12:33:38.043729 1 onnxruntime.cc:2836] TRITONBACKEND_ModelInstanceFinalize: delete instance state
I1101 12:33:38.046781 1 onnxruntime.cc:2760] TRITONBACKEND_ModelFinalize: delete model state
I1101 12:33:38.046833 1 model_lifecycle.cc:603] successfully unloaded 'all-MiniLM-L6-v2' version 1
I1101 12:33:39.045441 1 server.cc:331] Timeout 29: Found 0 live models and 0 in-flight non-inference requests

```

- [x] Show that you can get an inference from your model using the Triton HTTP endpoint 

    To have a more visual representation, I have used Postman to hit the Triton server HTTP endpoint and analyse the response back.
    In the screenshot below we can see the GET body payload and the 200 response I got back from the endpoint with the `model` given back a result of `0.995094895362854`

    !["Postman POST method response."](./img/01-docker-lab/AcceptanceCriteria_2_2.png) 
    !["Postman Triton server check by hitting /repository/index endpoint."](./img/01-docker-lab/AcceptanceCriteria_2_4.png)

- [x] Show how you can run a publicly available model of your choice  doc

    As seen in these screenshots, I have been able to run the MiniLM-L6-v2 public model on the Triton Inference server

    !["Postman body payload passed to the MiniLM-L6-v2 public model](./img/01-docker-lab/AcceptanceCriteria_2_6.png)

    !["Postman response to executing all-MiniLM-L6-v2 model."](./img/01-docker-lab/AcceptanceCriteria_2_8.png)

    `Full model response`:

    ```JSON
    {
    "model_name": "all-MiniLM-L6-v2",
    "model_version": "1",
    "outputs": [
        {
            "name": "last_hidden_state",
            "datatype": "FP32",
            "shape": [
                1,
                8,
                384
            ],
            "data": [
                0.08915802836418152,
                -0.1256694495677948,
                0.04061371833086014,
                0.19689913094043733,
                0.11653585731983185,
                -0.09995909780263901,
                -0.05629037320613861,
                0.08638057857751847,
                0.06806296855211258,
                -0.02570907026529312,
                -0.004496562294661999,
                0.046590499579906467,
                0.03480847179889679,
                0.0781482607126236,
                0.00946887955069542,
                -0.031144622713327409,
                0.11741386353969574,
                -0.10201671719551087,
                0.16417953372001649,
                -0.0896780863404274,
                -0.013701647520065308,
                -0.23846307396888734,
                0.091797836124897,
                -0.09268878400325775,
                0.015320129692554474,
                0.1743014007806778,
                -0.032283395528793338,
                0.15604731440544129,
                -0.044103533029556277,
                -0.9295317530632019,
                -0.1786917746067047,
                -0.018226269632577897,
                -0.24804824590682984,
                0.09736083447933197,
                0.11087580770254135,
                0.17985524237155915,
                0.2405264526605606,
                -0.011296117678284645,
                0.20381252467632295,
                -0.011593197472393513,
                0.2799202799797058,
                -0.17145821452140809,
                0.35591816902160647,
                0.03900473192334175,
                -0.08805408328771591,
                0.005404278635978699,
                -0.06297460198402405,
                0.04294099286198616,
                -0.19434046745300294,
                -0.4248834252357483,
                -0.035290010273456576,
                -0.10122504085302353,
                -0.11137459427118302,
                -0.17144164443016053,
                0.10216817259788513,
                0.22040092945098878,
                0.233604297041893,
                -0.08296292275190354,
                0.14596764743328095,
                0.016579216346144677,
                0.11258461326360703,
                0.02898373082280159,
                -0.3205018639564514,
                -0.04479376599192619,
                -0.19718091189861298,
                -0.06921570003032685,
                -0.17795874178409577,
                -0.054914217442274097,
                -0.22332552075386048,
                0.10817854851484299,
                0.01742057502269745,
                0.10317898541688919,
                -0.06551338732242584,
                0.06837151199579239,
                -0.05021360144019127,
                0.3166945278644562,
                0.18944001197814942,
                -0.14558537304401399,
                0.32891473174095156,
                0.135955810546875,
                -0.2575477361679077,
                -0.05393708124756813,
                -0.049953408539295199,
                0.22417499125003816,
                0.0857505053281784,
                0.21447964012622834,
                0.19297270476818086,
                -0.15508480370044709,
                0.07790842652320862,
                0.06681505590677262,
                -0.15756909549236298,
                0.1991960108280182,
                0.10733546316623688,
                -0.19222143292427064,
                -0.4773619771003723,
                -0.030023274943232538,
                -0.1634027659893036,
                -0.16531579196453095,
                -0.057268816977739337,
                5.580479145050049,
                0.12244260311126709,
                0.19168028235435487,
                0.16972941160202027,
                0.11162067949771881,
                -0.20214268565177918,
                -0.0735655426979065,
                0.14009302854537965,
                -0.17462444305419923,
                0.376282662153244,
                -0.09034810960292816,
                0.021573277190327646,
                -0.055954709649086,
                0.28077438473701479,
                -0.00469460291787982,
                0.06901594996452332,
                -0.1706823706626892,
                -0.23613981902599336,
                0.09735636413097382,
                0.09577333182096482,
                0.08468534052371979,
                0.21506521105766297,
                0.13411135971546174,
                -0.13252584636211396,
                0.010043561458587647,
                0.2888087332248688,
                -1.028277039527893,
                0.22317267954349519,
                -5.735823509343001e-32,
                0.12440352141857147,
                -0.11906582117080689,
                -0.07598254829645157,
                0.05110696330666542,
                0.05760648101568222,
                0.18373160064220429,
                0.1419234722852707,
                0.32237547636032107,
                -0.19174084067344666,
                -0.16232571005821229,
                0.15254828333854676,
                -0.2033504694700241,
                -0.018941355869174005,
                0.04811748117208481,
                0.11290498077869415,
                0.03096214309334755,
                0.17134621739387513,
                -0.024446405470371248,
                0.04361451789736748,
                0.2712639570236206,
                -0.032792966812849048,
                0.3556396961212158,
                0.057189151644706729,
                -0.20124474167823792,
                -0.16255167126655579,
                -0.16706180572509767,
                0.06629684567451477,
                0.09072380512952805,
                -0.18807807564735413,
                -0.24616894125938416,
                -0.3395356833934784,
                0.20100590586662293,
                -0.3100636303424835,
                -0.07811615616083145,
                0.09979146718978882,
                -0.1626814901828766,
                -0.06348767876625061,
                -0.05893925577402115,
                0.08770342916250229,
                0.11563216894865036,
                -0.004216864705085754,
                -0.06352054327726364,
                -0.1459931582212448,
                -0.06101253628730774,
                0.14151820540428163,
                0.260783851146698,
                -0.01285020262002945,
                -0.07508889585733414,
                0.06675146520137787,
                0.11470312625169754,
                0.008168017491698265,
                0.11251123249530792,
                0.11128789931535721,
                0.027087274938821794,
                -0.17871904373168946,
                0.05961104854941368,
                0.05509289354085922,
                0.14610283076763154,
                -0.05830240249633789,
                0.39843758940696719,
                -0.07557306438684464,
                -0.0007496960461139679,
                -0.12369716167449951,
                0.31327956914901736,
                0.07117582112550736,
                -0.23733223974704743,
                0.01853128895163536,
                -0.24042730033397675,
                0.04967004060745239,
                0.1257200539112091,
                -0.030799295753240587,
                -0.1524357795715332,
                -0.13244526088237763,
                -0.08195395022630692,
                -0.03822612017393112,
                -0.1412486732006073,
                -0.1411779522895813,
                0.4819423258304596,
                0.05963684618473053,
                0.17424015700817109,
                0.23024173080921174,
                -0.36540454626083376,
                0.029812565073370935,
                -0.1706271916627884,
                0.19462838768959046,
                -0.03254910930991173,
                -0.0008370243012905121,
                -0.07191192358732224,
                0.08898964524269104,
                0.0808589905500412,
                0.00789518654346466,
                -0.11541800200939179,
                0.19025319814682008,
                -0.008644988760352135,
                0.0369986891746521,
                4.7789217370109247e-32,
                -0.08042377978563309,
                -0.0746908187866211,
                -0.27128133177757265,
                0.035580113530159,
                -0.00015434622764587402,
                0.08190853148698807,
                0.19178654253482819,
                -0.10236063599586487,
                -0.25743672251701357,
                0.1806458979845047,
                0.04323985427618027,
                0.059298209846019748,
                -0.10915400087833405,
                0.049617499113082889,
                0.048401571810245517,
                -0.06470029056072235,
                0.08692682534456253,
                0.014163890853524208,
                0.10991768538951874,
                -0.01710008829832077,
                -0.082857646048069,
                0.07324311137199402,
                0.04600020498037338,
                0.11630384624004364,
                0.07177722454071045,
                0.21258868277072907,
                -0.050788432359695438,
                0.011753168888390065,
                0.04139123857021332,
                0.09285766631364823,
                0.018765421584248544,
                0.20104733109474183,
                0.039866894483566287,
                0.22462546825408936,
                -0.089157335460186,
                0.15960294008255006,
                0.09810321033000946,
                -0.21892306208610536,
                0.039182573556900027,
                0.08661625534296036,
                0.07586131244897843,
                0.001355547457933426,
                0.050594694912433627,
                0.12114432454109192,
                -0.0036627687513828279,
                0.14219406247138978,
                -0.08663690090179444,
                -0.22207674384117127,
                -0.005074668675661087,
                0.11292963474988938,
                -0.17261749505996705,
                0.15570054948329926,
                -0.04354436323046684,
                0.01785219833254814,
                -0.11878721415996552,
                -0.14679458737373353,
                -0.3728785216808319,
                -0.08062278479337692,
                0.017773013561964036,
                0.23346762359142304,
                -0.06571262329816818,
                0.2717916667461395,
                -0.15089885890483857,
                -0.08952701091766358,
                -0.16375888884067536,
                -0.07546330988407135,
                -0.1183447539806366,
                -0.027409140020608903,
                -0.15026898682117463,
                -0.0695001482963562,
                -0.13867366313934327,
                -0.09364808350801468,
                -0.057115040719509128,
                0.10962784290313721,
                0.019273564219474794,
                -0.04438422620296478,
                -0.2565158009529114,
                -0.06508078426122666,
                -0.09294559061527252,
                0.0048079416155815128,
                0.006384342908859253,
                0.05787087604403496,
                -0.17887432873249055,
                -0.029311757534742357,
                0.010237127542495728,
                0.000628393143415451,
                0.0569329559803009,
                0.037683285772800449,
                -0.024133026599884034,
                0.01889895834028721,
                0.14818690717220307,
                -0.01591290533542633,
                -0.047419264912605289,
                -0.19163760542869569,
                -0.20016293227672578,
                -8.790958361259982e-8,
                0.003499454353004694,
                -0.0838657021522522,
                -0.09297121316194534,
                0.15277250111103059,
                -0.2127939909696579,
                -0.1280333399772644,
                -0.130239337682724,
                -0.11375290155410767,
                -0.002573911100625992,
                0.100953109562397,
                0.020146256312727929,
                -0.11332696676254273,
                0.10112499445676804,
                0.12979575991630555,
                0.06279810518026352,
                -0.30129334330558779,
                -0.17256353795528413,
                -0.034421421587467197,
                0.13167709112167359,
                0.040506720542907718,
                0.1016802191734314,
                0.2573453187942505,
                -0.13903850317001344,
                0.0744628757238388,
                -0.1527688205242157,
                0.2456696331501007,
                0.0534711554646492,
                -0.40032458305358889,
                0.09835317730903626,
                0.17618979513645173,
                -0.011823631823062897,
                -0.046748124063014987,
                -0.3731957972049713,
                -0.0932641476392746,
                -0.05179446563124657,
                0.052759621292352679,
                -0.10952749848365784,
                0.0014478862285614014,
                0.3055954575538635,
                0.1578826755285263,
                -0.48903200030326846,
                -0.06982879340648651,
                -0.0075268191285431389,
                -0.01568404957652092,
                -0.07706937193870545,
                -0.33922991156578066,
                -0.37105458974838259,
                -0.43283936381340029,
                -0.09067536890506745,
                -0.3339008688926697,
                -0.039259426295757297,
                -0.04860540106892586,
                -0.09317146241664887,
                0.03071020171046257,
                0.12947149574756623,
                0.10824311524629593,
                -0.15592831373214723,
                0.02334265410900116,
                -0.28707337379455569,
                -0.13384994864463807,
                0.24363984167575837,
                -0.09261682629585266,
                -0.12134413421154022,
                -0.2924383580684662,
                0.6506202816963196,
                0.5280399918556213,
                0.07148247957229614,
                0.5712524652481079,
                0.36172717809677126,
                -0.2945345938205719,
                0.2750833332538605,
                -0.23885205388069154,
                1.221483588218689,
                0.10364039242267609,
                0.3055416941642761,
                -0.11461567878723145,
                -0.0144285187125206,
                0.5856934189796448,
                0.08712562173604965,
                0.14645035564899445,
                -0.06504322588443756,
                0.5632609128952026,
                -0.7989971041679382,
                0.5221282243728638,
                0.5242021083831787,
                0.5587581396102905,
                -0.26213744282722475,
                -0.2264299988746643,
                0.426999032497406,
                -0.2286853790283203,
                -0.5807126760482788,
                -0.23472121357917787,
                0.6302641034126282,
                -0.48659828305244448,
                -0.08547265827655792,
                -0.32090407609939577,
                0.4946112036705017,
                -0.16045235097408296,
                0.5539903044700623,
                0.25976455211639407,
                0.36786866188049319,
                0.2623133659362793,
                -0.39803212881088259,
                0.2196371853351593,
                0.18720132112503053,
                -0.8658680319786072,
                -0.4041961431503296,
                0.2982224225997925,
                0.037502940744161609,
                -0.14418503642082215,
                -0.07895372062921524,
                -0.1506751924753189,
                -0.003539297729730606,
                0.13203662633895875,
                -0.757469654083252,
                -0.1312406212091446,
                -0.5196964740753174,
                -0.8402342200279236,
                0.3277979791164398,
                0.2493000328540802,
                -0.2528342604637146,
                0.03695012629032135,
                0.4423811733722687,
                0.5165656805038452,
                -0.17832967638969422,
                0.1912553906440735,
                -0.015987426042556764,
                0.2735540270805359,
                0.6759511232376099,
                0.2889855206012726,
                -0.14360809326171876,
                -0.6394320130348206,
                0.07206163555383682,
                -0.2607221007347107,
                -0.07557849586009979,
                0.5409190058708191,
                0.19693882763385774,
                -0.8654502034187317,
                -0.09473946690559387,
                -0.04762345179915428,
                -0.3836895823478699,
                -0.912520706653595,
                0.6850987076759338,
                0.1870122253894806,
                -0.828036904335022,
                0.18129344284534455,
                -0.02382492460310459,
                -0.12537036836147309,
                -0.37453916668891909,
                -1.0170155763626099,
                0.06039272993803024,
                0.23653174936771394,
                -0.15372392535209657,
                -0.13184604048728944,
                0.06054951995611191,
                -0.22731690108776093,
                0.4692295491695404,
                0.23676779866218568,
                -0.6470509767532349,
                0.1376849114894867,
                -0.5418495535850525,
                -0.1858462691307068,
                -0.18786212801933289,
                -0.8221875429153442,
                -0.3456646203994751,
                0.07120843231678009,
                -0.4201831817626953,
                0.03990376740694046,
                -0.23437371850013734,
                -0.7685751914978027,
                -0.37829500436782839,
                -0.46999767422676089,
                0.10124174505472183,
                0.3685244023799896,
                -0.2518496811389923,
                -0.19146345555782319,
                -0.19754061102867127,
                0.2613569498062134,
                0.06849819421768189,
                0.23647688329219819,
                -1.2080895900726319,
                0.18463465571403504,
                -0.13867470622062684,
                0.6813976764678955,
                0.2740183174610138,
                0.3684122562408447,
                -0.4467082619667053,
                -0.15622633695602418,
                -0.6387444138526917,
                -0.7459837198257446,
                -0.07186157256364823,
                -5.163796219885404e-32,
                0.1267445683479309,
                -0.551529586315155,
                0.04766014218330383,
                -0.3123025894165039,
                0.5106776356697083,
                -0.1413375288248062,
                -0.12727870047092439,
                0.25715118646621706,
                -0.47376465797424319,
                -1.216741919517517,
                0.3219414949417114,
                -0.5677676200866699,
                0.33610379695892336,
                0.21568003296852113,
                0.5326412916183472,
                0.18918347358703614,
                -0.38046306371688845,
                0.14351940155029298,
                -1.1088693141937256,
                0.4870733618736267,
                -0.016042139381170274,
                0.4455752372741699,
                -0.18536056578159333,
                -0.28841790556907656,
                -0.47556981444358828,
                -0.38955456018447878,
                -0.10767427086830139,
                0.24986542761325837,
                -0.8968291282653809,
                0.37126749753952029,
                -0.009919745847582817,
                0.17608781158924104,
                0.33182787895202639,
                0.15396898984909059,
                -0.13059386610984803,
                0.011653713881969452,
                0.13773882389068604,
                -0.8667026162147522,
                -0.030698347836732866,
                -0.1368068903684616,
                -0.0046358052641153339,
                -0.49765756726264956,
                -0.30021074414253237,
                -0.021161029115319253,
                -0.08635717630386353,
                0.2980095148086548,
                0.29054802656173708,
                -0.3761909604072571,
                0.17406585812568665,
                0.2579471468925476,
                -0.4322754442691803,
                -0.03767465054988861,
                -0.264659583568573,
                0.03701876848936081,
                0.2519468665122986,
                0.26356181502342226,
                0.26271823048591616,
                -0.540234386920929,
                -0.32852035760879519,
                -0.17502273619174958,
                -0.41934892535209658,
                -0.12021538615226746,
                -0.8712946176528931,
                0.058896586298942569,
                0.24119699001312257,
                0.09099659323692322,
                -0.024752438068389894,
                -0.07884740084409714,
                0.25335219502449038,
                0.17162787914276124,
                -0.3252038061618805,
                0.06604079902172089,
                0.06652775406837464,
                0.5112522840499878,
                0.30764564871788027,
                -0.828007161617279,
                0.23223763704299928,
                0.3906058073043823,
                -0.22098368406295777,
                -0.06172487884759903,
                -0.21428368985652924,
                -0.5748966336250305,
                0.1834941804409027,
                0.023140188306570054,
                -0.1540614366531372,
                -0.04200407490134239,
                -0.22817453742027284,
                -0.8412536382675171,
                0.008405586704611779,
                0.5169680714607239,
                -0.11737962067127228,
                -0.3466007113456726,
                0.6354206204414368,
                0.22741185128688813,
                0.0683828592300415,
                -4.090676259823405e-34,
                -0.23577970266342164,
                0.43550026416778567,
                -0.4511314034461975,
                0.44750693440437319,
                0.35963794589042666,
                -0.24228018522262574,
                0.39130833745002749,
                -0.005443025846034288,
                0.6795130968093872,
                1.2251622676849366,
                -0.052718210965394977,
                0.25095415115356448,
                -0.21597889065742494,
                -0.11760566383600235,
                0.727189302444458,
                0.2901418209075928,
                0.4680567681789398,
                -0.14613279700279237,
                0.13898806273937226,
                -0.40823620557785036,
                -0.15630263090133668,
                0.36751845479011538,
                0.5168854594230652,
                1.3957701921463013,
                -0.31739407777786257,
                0.06608898937702179,
                -0.043122366070747378,
                0.1344885230064392,
                -0.3566931188106537,
                0.33434128761291506,
                -0.1686984747648239,
                -0.11639311909675598,
                0.16433531045913697,
                -0.33448612689971926,
                -0.46188855171203616,
                0.5578742623329163,
                0.19959187507629395,
                0.31275755167007449,
                -0.4260978698730469,
                -0.04098956286907196,
                -0.4651680886745453,
                0.22110745310783387,
                0.40351271629333498,
                0.43695932626724245,
                0.19775646924972535,
                -0.39068132638931277,
                -0.11189018189907074,
                -1.0793342590332032,
                -0.5203929543495178,
                0.16555635631084443,
                -0.40904685854911806,
                -0.16250485181808473,
                0.0066879987716674809,
                0.04785957559943199,
                -0.47374457120895388,
                0.42233848571777346,
                -0.20770233869552613,
                0.12839190661907197,
                0.28367918729782107,
                -0.13577096164226533,
                -0.5818447470664978,
                0.48877784609794619,
                -0.19474291801452638,
                -0.40084564685821535,
                0.21221324801445008,
                -0.009588146582245827,
                -0.5490806698799133,
                0.5878849029541016,
                0.17965713143348695,
                -0.41888153553009035,
                0.7669208645820618,
                0.27962934970855715,
                -1.4222475290298463,
                -0.6754990220069885,
                -0.2774050831794739,
                -0.5479835271835327,
                -0.4561987519264221,
                0.2142096608877182,
                -0.18031255900859834,
                -0.7050886154174805,
                0.07651474326848984,
                -0.11186518520116806,
                0.5217323899269104,
                -0.04335155710577965,
                -0.39237141609191897,
                0.3554299473762512,
                -0.29414913058280947,
                0.227609321475029,
                -0.3179846405982971,
                -0.15987129509449006,
                0.1859089583158493,
                -0.11436136066913605,
                0.9207650423049927,
                -0.24959015846252442,
                -0.28641897439956667,
                -1.0505462455512316e-7,
                0.0825321227312088,
                -0.021032672375440599,
                -0.6608214974403381,
                0.10704674571752548,
                0.20174191892147065,
                0.4624377489089966,
                0.14303594827651978,
                0.19217708706855775,
                -0.03029128722846508,
                -0.30724266171455386,
                0.05049074441194534,
                0.5961029529571533,
                -0.34184330701828005,
                0.024469712749123575,
                0.5326708555221558,
                -0.19605031609535218,
                0.3447822630405426,
                -0.3733549416065216,
                0.2906944453716278,
                0.3432352840900421,
                -0.991590142250061,
                0.00809454545378685,
                0.6087693572044373,
                -0.5661936402320862,
                -0.07422354817390442,
                0.11357567459344864,
                0.9461599588394165,
                0.4725968837738037,
                -0.2846853733062744,
                0.5163206458091736,
                0.48622190952301028,
                0.9233198761940002,
                0.020545264706015588,
                -0.08302326500415802,
                -0.6100929975509644,
                -0.049593813717365268,
                0.12355822324752808,
                -0.4597671329975128,
                -0.1194443479180336,
                -0.4147211015224457,
                -0.8583738803863525,
                0.2824038863182068,
                -0.11628255993127823,
                0.6020562648773193,
                -0.09708398580551148,
                0.1281958967447281,
                -0.035448119044303897,
                0.1655745506286621,
                -0.23503205180168153,
                -0.11315755546092987,
                -0.16336394846439362,
                0.46049612760543826,
                0.7951378226280212,
                -0.14349400997161866,
                0.4233027696609497,
                -0.4437696933746338,
                -0.240972638130188,
                -0.011253457516431809,
                0.09375162422657013,
                0.2772569954395294,
                0.07829539477825165,
                0.20133298635482789,
                0.5378787517547607,
                0.26390475034713747,
                0.08374261856079102,
                -0.046691201627254489,
                0.004410280846059322,
                0.07050120085477829,
                0.04958822950720787,
                0.42055249214172366,
                -0.10601409524679184,
                -0.16728414595127107,
                0.1005069836974144,
                0.49167630076408389,
                0.23830819129943849,
                -0.7905313372612,
                0.13700918853282929,
                -0.09750384092330933,
                -0.20179836452007295,
                -0.38172420859336855,
                0.5096937417984009,
                -0.5219301581382752,
                -0.40598464012145998,
                0.2186419814825058,
                0.3490672707557678,
                -0.0003542378544807434,
                -0.2203332632780075,
                -0.020444434136152269,
                0.20259828865528108,
                -0.021255455911159517,
                -0.2306273877620697,
                -0.23830288648605348,
                0.2935783565044403,
                0.38434746861457827,
                -0.18040251731872559,
                0.14482203125953675,
                0.05622032657265663,
                0.3300495445728302,
                -0.15817441046237946,
                -0.11370086669921875,
                0.2044834941625595,
                -0.06933360546827316,
                0.03199287876486778,
                0.004512554965913296,
                -0.03677745908498764,
                -0.3342299461364746,
                -0.04892526939511299,
                0.04106203094124794,
                -0.046555377542972568,
                0.19455265998840333,
                -0.08556129783391953,
                -0.09415166825056076,
                -0.001736808568239212,
                -0.2782731056213379,
                0.06022527813911438,
                -0.15957550704479218,
                -0.31036508083343508,
                0.5259721279144287,
                0.25016456842422488,
                -0.09763193130493164,
                0.254773885011673,
                0.20143671333789826,
                0.16692684590816499,
                0.08811034262180329,
                0.19266489148139954,
                -0.10438281297683716,
                -0.04574330896139145,
                0.16445840895175935,
                0.21351732313632966,
                0.23223258554935456,
                -0.2215646356344223,
                -0.18171574175357819,
                -0.12472221255302429,
                -0.6913856267929077,
                0.37787172198295596,
                0.3370632529258728,
                0.37765827775001528,
                0.2332412600517273,
                0.1897677630186081,
                0.053079456090927127,
                0.06071522831916809,
                -0.16717714071273805,
                0.2910078167915344,
                0.492621511220932,
                -0.2988255023956299,
                -0.7207691073417664,
                -0.13107599318027497,
                0.1412929743528366,
                -0.008209669962525368,
                0.14750328660011292,
                0.265569269657135,
                -0.4605540931224823,
                -0.1539682000875473,
                -0.12821240723133088,
                -0.19260385632514954,
                -0.08611835539340973,
                -0.33288103342056277,
                -0.21259309351444245,
                -0.06383878737688065,
                -0.10005007684230805,
                -0.1730336993932724,
                -0.10717594623565674,
                -0.07875268906354904,
                -0.5432756543159485,
                -0.16417427361011506,
                0.010966831818223,
                0.04270421713590622,
                0.04152324050664902,
                -0.19513018429279328,
                0.14001242816448213,
                0.4308936595916748,
                0.27492794394493105,
                0.1023719385266304,
                -0.06362734735012055,
                -0.1773516684770584,
                -0.17113402485847474,
                0.02018607407808304,
                0.028251176699995996,
                -0.17202264070510865,
                -0.19652926921844483,
                -0.5071138739585877,
                -0.016950517892837526,
                -0.207743301987648,
                0.5913376212120056,
                0.2395426481962204,
                0.41322797536849978,
                0.17135775089263917,
                -0.06368102133274079,
                -0.0049913921393454079,
                -0.0862119048833847,
                0.2848326861858368,
                3.430083699491804e-32,
                -0.23255640268325807,
                0.24114300310611726,
                -0.14904633164405824,
                0.6664965152740479,
                -0.016537463292479516,
                -0.09471704810857773,
                -0.002190442755818367,
                0.08428115397691727,
                -0.28859031200408938,
                -0.28992459177970889,
                0.05439511314034462,
                -0.1117851734161377,
                0.10817226767539978,
                -0.047262173146009448,
                0.12370339781045914,
                0.3252158761024475,
                -0.1737343668937683,
                0.6178460717201233,
                0.12787708640098573,
                0.02192741632461548,
                -0.3528774678707123,
                0.15360978245735169,
                0.0039336904883384708,
                -0.7091814875602722,
                -0.48417991399765017,
                -0.15627619624137879,
                -0.39733168482780459,
                0.09319484978914261,
                -0.30142033100128176,
                -0.1853364109992981,
                -0.005528407171368599,
                0.26899483799934389,
                -0.3376825451850891,
                0.13966313004493714,
                -0.28163942694664,
                0.037556953728199008,
                0.10003218799829483,
                -0.152268186211586,
                -0.34500813484191897,
                0.36777448654174807,
                -0.12618288397789002,
                -0.20007038116455079,
                0.08583749085664749,
                -0.08658219128847122,
                0.1744007021188736,
                0.2040979266166687,
                -0.15620876848697663,
                -0.4592922329902649,
                0.09834732115268707,
                0.25445953011512759,
                0.23469269275665284,
                0.04393533617258072,
                0.20047076046466828,
                -0.19239501655101777,
                0.34327301383018496,
                0.5194835066795349,
                0.22408241033554078,
                0.33235737681388857,
                0.06596073508262634,
                0.08122240751981735,
                0.036696501076221469,
                0.21189074218273164,
                -0.26399168372154238,
                -0.2941287159919739,
                0.0865042507648468,
                0.22814062237739564,
                -0.15654292702674867,
                0.07166233658790589,
                0.4790588915348053,
                0.0325855016708374,
                0.04066034033894539,
                -0.0985632836818695,
                0.4533510208129883,
                0.5949884653091431,
                0.1778503656387329,
                -0.457780122756958,
                -0.12913036346435548,
                0.2942756116390228,
                0.00012066401541233063,
                0.5274897217750549,
                -0.18748746812343598,
                -0.063540980219841,
                0.27776363492012026,
                0.11834155768156052,
                -0.49631622433662417,
                -0.3809436559677124,
                -0.1555022895336151,
                -0.3001704514026642,
                -0.09303531050682068,
                0.11991062760353089,
                -0.7283881306648254,
                0.16062328219413758,
                -0.03387697786092758,
                -0.4117264449596405,
                -0.020951680839061738,
                -4.378052890251982e-32,
                -0.3771207332611084,
                0.5522217154502869,
                -0.1322147101163864,
                0.34640881419181826,
                0.12290173023939133,
                0.11632134020328522,
                0.3066893219947815,
                0.2241312563419342,
                -0.3086148202419281,
                -0.15293258428573609,
                -1.0358366966247559,
                -0.06373374164104462,
                -0.15421168506145478,
                -0.0005850666202604771,
                0.03828434646129608,
                0.4173007309436798,
                0.1274494081735611,
                -0.08892980962991715,
                -0.22003789246082307,
                -0.02417657896876335,
                0.013782540336251259,
                1.052484393119812,
                0.2603921890258789,
                0.09195054322481156,
                0.1258927881717682,
                -0.3697143495082855,
                -0.43015801906585696,
                -0.42839860916137698,
                -0.46478432416915896,
                0.26052847504615786,
                -0.19245195388793946,
                -0.18367786705493928,
                -0.41443586349487307,
                -0.128008171916008,
                0.2546162009239197,
                -0.08440160006284714,
                0.4786950945854187,
                -0.17865610122680665,
                -0.4577144384384155,
                -0.2641490399837494,
                -0.14009544253349305,
                0.4363119602203369,
                0.14945432543754579,
                0.4270535707473755,
                0.2972468435764313,
                -0.5098008513450623,
                0.09138106554746628,
                -0.04365207254886627,
                -0.3677714169025421,
                0.28286850452423098,
                -0.8698756694793701,
                -0.0230429507791996,
                0.3257412910461426,
                0.6413044333457947,
                -0.3030303716659546,
                0.7643249034881592,
                -0.4156167507171631,
                -0.18159186840057374,
                -0.21513007581233979,
                -0.1581457257270813,
                -0.205260768532753,
                0.07189415395259857,
                0.20632269978523255,
                0.15236054360866548,
                0.5826767086982727,
                -0.3836083710193634,
                -0.3535539507865906,
                0.022435639053583146,
                -0.14563731849193574,
                -0.1753349006175995,
                -0.010514810681343079,
                0.06592126190662384,
                -0.8462050557136536,
                -0.24802470207214356,
                0.24534469842910767,
                -0.449633926153183,
                -0.1184600293636322,
                -0.09973382204771042,
                -0.03811081871390343,
                -0.45443227887153628,
                0.21881240606307984,
                -0.42811232805252077,
                -0.08062674105167389,
                0.08416318893432617,
                -0.6031453013420105,
                0.20432458817958833,
                0.08597360551357269,
                -0.22003374993801118,
                -0.06552684307098389,
                -0.06924855709075928,
                -0.1989155262708664,
                -0.6347854733467102,
                0.42204079031944277,
                0.2648865878582001,
                0.07493528723716736,
                -1.0094858282627684e-7,
                0.13895711302757264,
                -0.39950141310691836,
                0.21889473497867585,
                -0.11041940748691559,
                -0.44162148237228396,
                0.20696936547756196,
                0.3347407579421997,
                -0.18265868723392487,
                -0.2014981210231781,
                -0.03022707998752594,
                0.02423916570842266,
                0.6107453107833862,
                -0.04655397683382034,
                0.17295539379119874,
                -0.10342695564031601,
                -0.10184399783611298,
                0.11050761491060257,
                -0.18758085370063783,
                -0.0362127460539341,
                0.2627788782119751,
                0.1658320426940918,
                -0.29550206661224368,
                0.36467528343200686,
                -0.08751995861530304,
                -0.12034444510936737,
                0.25834113359451296,
                0.3470073342323303,
                0.4361193776130676,
                0.06411607563495636,
                -0.03451818600296974,
                0.6359290480613709,
                0.5217463374137878,
                -0.02897309511899948,
                0.2595807611942291,
                0.10119514167308808,
                0.25896358489990237,
                0.3413190543651581,
                -0.12611596286296845,
                0.12857386469841004,
                -0.19179555773735047,
                -0.19701234996318818,
                0.3074585795402527,
                -0.2439769059419632,
                0.7056429386138916,
                -0.5691766738891602,
                0.14279213547706605,
                0.17121179401874543,
                0.18912330269813538,
                -0.3744979798793793,
                -0.11908268928527832,
                -0.05849664658308029,
                -0.36259108781814577,
                0.17379432916641236,
                0.030374087393283845,
                -0.31169965863227847,
                -0.22827550768852235,
                -0.03865070641040802,
                0.3178218603134155,
                0.0008918996900320053,
                -0.35213369131088259,
                0.2374366968870163,
                0.3124717175960541,
                0.6754423379898071,
                -0.18311357498168946,
                -0.02848883718252182,
                0.13413751125335694,
                0.14714762568473817,
                0.39070558547973635,
                -0.23226594924926759,
                -0.12514784932136537,
                -0.34661567211151125,
                0.2077658772468567,
                0.22956973314285279,
                0.3943953514099121,
                0.3948502242565155,
                -0.3792349100112915,
                -0.013084661215543747,
                0.03823086991906166,
                0.18611323833465577,
                -0.2853110432624817,
                -0.027358803898096086,
                -0.5707694292068481,
                -0.07507778704166413,
                0.3583866059780121,
                -0.10980921983718872,
                0.09165380895137787,
                -0.31283900141716006,
                0.11613604426383972,
                -0.11826662719249726,
                -0.2365519106388092,
                -0.23915448784828187,
                0.07101261615753174,
                -0.02359016239643097,
                0.0855371281504631,
                -0.1394520103931427,
                0.16959227621555329,
                0.8428006768226624,
                0.12026525288820267,
                0.2634725272655487,
                0.1669386476278305,
                0.47974786162376406,
                0.14579662680625916,
                0.3026675283908844,
                0.09703837335109711,
                0.20459701120853425,
                -0.3072989881038666,
                0.02620682865381241,
                0.2334994226694107,
                -0.1412544697523117,
                0.06978816539049149,
                -0.12145502865314484,
                -0.09412405639886856,
                -0.05714159086346626,
                -0.03350312262773514,
                -0.4357204735279083,
                -0.2059754580259323,
                -0.3914191424846649,
                -0.1795295923948288,
                0.19397202134132386,
                0.002853561192750931,
                0.18446247279644013,
                0.4256770610809326,
                -0.05110593140125275,
                -0.049175478518009189,
                0.010544123128056527,
                -0.34744733572006228,
                -0.19337549805641175,
                0.4809582531452179,
                0.8033810257911682,
                0.21871143579483033,
                -0.27491751313209536,
                -0.1919652670621872,
                -0.2001306116580963,
                -0.046412210911512378,
                0.4283847212791443,
                -0.10776180028915405,
                0.33006715774536135,
                -0.08825285732746124,
                -0.1345808357000351,
                -0.21827396750450135,
                -0.21609915792942048,
                -0.5674816370010376,
                0.5592813491821289,
                0.4160751700401306,
                -0.7076536417007446,
                -0.09112255275249481,
                -0.05300673842430115,
                0.16164304316043855,
                -0.1605963408946991,
                -0.4325008988380432,
                0.5859357118606567,
                -0.5806612372398377,
                -0.17147080600261689,
                0.10278141498565674,
                -0.21102175116539002,
                0.102244533598423,
                -0.07846204191446305,
                -0.11648785322904587,
                0.09335868060588837,
                -0.20750148594379426,
                -0.6559212803840637,
                -0.3059733211994171,
                0.07023552805185318,
                -0.318179190158844,
                -0.12119075655937195,
                0.25679486989974978,
                -0.18019947409629823,
                0.4652625024318695,
                -0.32996293902397158,
                -0.3886968493461609,
                0.023401588201522828,
                -0.07949754595756531,
                0.21549999713897706,
                0.11604157090187073,
                -0.19659097492694856,
                -0.2728550434112549,
                0.27846723794937136,
                0.38556385040283205,
                0.04048396646976471,
                0.08301326632499695,
                -0.3765159845352173,
                0.006756100803613663,
                0.20558232069015504,
                -0.09633445739746094,
                0.3157905042171478,
                0.24810875952243806,
                0.056933045387268069,
                -0.21203064918518067,
                -0.16468863189220429,
                0.08173220604658127,
                0.14140962064266206,
                -9.261344419683716e-32,
                -0.13827592134475709,
                -0.1678885519504547,
                -0.10744357854127884,
                0.5425553917884827,
                -0.035520970821380618,
                0.1239279955625534,
                -0.329129695892334,
                0.11988630890846253,
                -0.010975141078233719,
                0.16749334335327149,
                -0.21568769216537476,
                -0.4399576187133789,
                0.18371988832950593,
                0.7003604173660278,
                0.7334481477737427,
                0.5580013394355774,
                0.058792125433683398,
                0.497619092464447,
                -0.056837234646081927,
                0.06686154752969742,
                -0.34252646565437319,
                0.3771110773086548,
                -0.0003306567668914795,
                -0.34111911058425906,
                -0.5265526175498962,
                -0.5855478644371033,
                -0.08576096594333649,
                -0.2014431208372116,
                0.020140986889600755,
                0.2917412519454956,
                0.030104754492640497,
                0.13838370144367219,
                0.0493115596473217,
                0.13585884869098664,
                0.0980573445558548,
                -0.8490942716598511,
                -0.06310491263866425,
                -0.15204206109046937,
                -0.023674238473176957,
                0.23960258066654206,
                -0.2547023892402649,
                -0.20737813413143159,
                0.08782833814620972,
                -0.028455743566155435,
                0.7118542194366455,
                -0.0996929332613945,
                0.3514862060546875,
                -0.00964394211769104,
                -0.29363036155700686,
                0.032930564135313037,
                0.022076722234487535,
                0.01572217419743538,
                0.4214252233505249,
                -0.1288817673921585,
                0.08030182123184204,
                0.14348699152469636,
                0.10985007882118225,
                0.21930481493473054,
                -0.3662647604942322,
                0.541174054145813,
                0.1080983355641365,
                0.4044322967529297,
                -0.7223277688026428,
                0.19451123476028443,
                -0.3965388238430023,
                0.04896317794919014,
                -0.27938348054885867,
                -0.04548891633749008,
                0.3404145836830139,
                0.14161205291748048,
                0.008442177437245846,
                -0.26040762662887576,
                0.3209611773490906,
                0.22656883299350739,
                -0.17746801674365998,
                -0.4707360565662384,
                0.053197234869003299,
                0.08110172301530838,
                -0.4106200337409973,
                0.3667081594467163,
                -0.3142055869102478,
                -0.5632793307304382,
                0.10482878237962723,
                -0.21281756460666657,
                -0.3877635598182678,
                -0.039323169738054278,
                -0.2599848806858063,
                -0.15150786936283112,
                0.3908410668373108,
                -0.3794419765472412,
                -0.3386286497116089,
                0.10122542828321457,
                -0.24285656213760377,
                -0.9723166823387146,
                0.3039468824863434,
                5.027604032146309e-32,
                -0.3427288234233856,
                0.6336103081703186,
                -0.5823118090629578,
                0.6488182544708252,
                0.3259713649749756,
                -0.2074112892150879,
                0.3515007197856903,
                -0.26614949107170107,
                -0.4578016996383667,
                0.5247145295143127,
                -0.6244078278541565,
                -0.09781074523925781,
                0.20307494699954987,
                -0.2768641710281372,
                0.5384394526481628,
                -0.008408892899751664,
                0.11025292426347733,
                -0.262752890586853,
                0.3130338788032532,
                -0.48524171113967898,
                -0.3292539417743683,
                0.5196507573127747,
                0.11283878237009049,
                0.6015441417694092,
                -0.17443624138832093,
                0.2736496329307556,
                0.2675088346004486,
                -0.3713352084159851,
                -0.2103913128376007,
                -0.20104272663593293,
                -0.008148428052663803,
                -0.09798051416873932,
                -0.09483463317155838,
                -0.006821157410740852,
                -0.09080762416124344,
                -0.18927142024040223,
                0.661952018737793,
                -0.32905903458595278,
                -0.4561026394367218,
                0.12499631941318512,
                0.03244219720363617,
                0.13693228363990785,
                0.353346586227417,
                0.9300281405448914,
                0.15266764163970948,
                0.1946955919265747,
                -0.23676744103431703,
                -0.7276939153671265,
                0.23594117164611817,
                0.3169470429420471,
                -0.39437001943588259,
                0.0418190136551857,
                0.21579544246196748,
                0.36500632762908938,
                -0.4872887134552002,
                0.5640631914138794,
                -0.0043810829520225529,
                -0.16266028583049775,
                -0.2099086195230484,
                0.19822753965854646,
                -0.6127076148986816,
                0.15636733174324037,
                0.0770849883556366,
                0.26597777009010317,
                -0.03639848902821541,
                -0.22598405182361604,
                -0.5747027397155762,
                0.10499877482652664,
                -0.001816343516111374,
                -0.3566783368587494,
                0.6170508861541748,
                0.43805623054504397,
                -0.5731617212295532,
                -0.0993996188044548,
                0.33372026681900027,
                -0.3813726603984833,
                -0.25136038661003115,
                -0.36490586400032046,
                -0.06655547022819519,
                -0.37150314450263979,
                -0.5570715665817261,
                -0.26377609372138979,
                0.32307374477386477,
                -0.3393421471118927,
                -0.23805683851242066,
                0.05683388561010361,
                -0.038610003888607028,
                -0.08783320337533951,
                -0.1570328027009964,
                -0.003021087497472763,
                0.3886701762676239,
                -0.10834714770317078,
                0.4517412781715393,
                -0.10682202130556107,
                -0.06528495252132416,
                -9.537781409107993e-8,
                -0.1000843346118927,
                0.17706462740898133,
                0.20971275866031648,
                -0.27605780959129336,
                -0.04522378742694855,
                0.23169392347335816,
                0.526934027671814,
                -0.2882899045944214,
                -0.265726774930954,
                -0.147418811917305,
                0.4339762330055237,
                0.4421871304512024,
                -0.559902012348175,
                0.35658228397369387,
                0.5525475740432739,
                -0.29631561040878298,
                0.2765853703022003,
                -0.29020240902900698,
                -0.2580663859844208,
                0.7474703788757324,
                -0.48206377029418948,
                0.3706360459327698,
                -0.16853556036949159,
                0.20750652253627778,
                -0.1423671841621399,
                -0.0647570788860321,
                0.039046596735715869,
                -0.15516047179698945,
                0.17694512009620667,
                -0.006932415999472141,
                0.3743382692337036,
                0.1626509130001068,
                0.021986667066812516,
                -0.06216482073068619,
                -0.5918117761611939,
                0.1272856444120407,
                0.5266847014427185,
                -0.6662335991859436,
                0.13053753972053529,
                -0.517169713973999,
                -0.1536455899477005,
                0.36934804916381838,
                -0.1280197948217392,
                1.1673709154129029,
                -0.3436949849128723,
                -0.27312952280044558,
                -0.10420330613851547,
                0.049970634281635287,
                -0.28443440794944765,
                -0.13015379011631013,
                0.1688654124736786,
                0.214748814702034,
                0.0686783641576767,
                -0.02682378515601158,
                0.24114057421684266,
                -0.32414400577545168,
                -0.05112423002719879,
                -0.48753249645233157,
                -0.04549913480877876,
                0.06776956468820572,
                0.15500566363334657,
                -0.2668072581291199,
                0.4537571966648102,
                0.2905944585800171,
                0.3894382119178772,
                0.31051698327064516,
                -0.33391445875167849,
                0.3534696400165558,
                -0.1344999521970749,
                0.11359336972236633,
                -0.13007473945617677,
                -0.3589877188205719,
                -0.19935257732868195,
                -0.15460503101348878,
                0.5048960447311401,
                -1.0781632661819459,
                0.24654236435890199,
                0.7057123780250549,
                0.4237769544124603,
                -0.5484112501144409,
                0.04674862325191498,
                0.09666547924280167,
                -0.1367127150297165,
                0.41363611817359927,
                -0.18338023126125337,
                -0.22368812561035157,
                -0.07786418497562409,
                0.3172871768474579,
                -0.0021016765385866167,
                -0.12503571808338166,
                -0.44329357147216799,
                0.28461509943008425,
                0.4317765533924103,
                -0.11592070758342743,
                0.1752614974975586,
                0.2393801063299179,
                0.3044579029083252,
                0.7842288017272949,
                0.4762391746044159,
                -0.6998271346092224,
                -0.16360126435756684,
                0.21963882446289063,
                0.6426216959953308,
                -0.01080181635916233,
                0.011371247470378876,
                -0.8476556539535523,
                0.70741206407547,
                -0.23349407315254212,
                0.24521851539611817,
                0.5602309107780457,
                -0.38604798913002016,
                0.5515596866607666,
                -0.4432528018951416,
                -0.3259661793708801,
                0.6922584772109985,
                -0.8115991353988648,
                -0.27283087372779848,
                -0.602891206741333,
                -0.7134100198745728,
                0.07449791580438614,
                0.17892257869243623,
                0.07645504176616669,
                -0.10766598582267761,
                0.8025110363960266,
                -0.5610936880111694,
                -0.40268322825431826,
                0.12071225047111511,
                -0.3526515066623688,
                1.315990686416626,
                0.32631316781044009,
                -0.3531639575958252,
                -0.40248745679855349,
                0.2920255661010742,
                0.4054906666278839,
                -0.17183014750480653,
                -0.04887355864048004,
                0.8507730960845947,
                0.27614158391952517,
                0.514061450958252,
                -0.2305200695991516,
                0.032983772456645969,
                -0.8399688601493836,
                -0.08239452540874481,
                -0.09038649499416352,
                -0.5669577121734619,
                -1.468306541442871,
                0.17796260118484498,
                0.36456599831581118,
                0.5884693264961243,
                0.8612180352210999,
                0.8804711699485779,
                -0.6519759893417358,
                -0.9396104216575623,
                0.23923306167125703,
                0.4878431260585785,
                -0.35811662673950198,
                -1.2817201614379883,
                0.4469250440597534,
                -0.09519772976636887,
                0.19266264140605927,
                0.21274657547473908,
                -0.01103050448000431,
                -0.048082321882247928,
                -0.007574789226055145,
                0.6139225363731384,
                -0.5865355730056763,
                0.3389526903629303,
                -0.07128322869539261,
                -0.7817275524139404,
                -0.2948776185512543,
                0.4532521963119507,
                -0.9670005440711975,
                0.2297751009464264,
                -0.2246972918510437,
                0.07137411832809448,
                0.01725301891565323,
                0.32435211539268496,
                0.4294761121273041,
                0.28694799542427065,
                -0.7965933680534363,
                -1.346443772315979,
                0.5085179209709168,
                -0.4910483956336975,
                0.7205045819282532,
                0.8457509875297546,
                -0.0786370262503624,
                0.05781964957714081,
                -0.04639705270528793,
                0.08239016681909561,
                0.2379142791032791,
                0.9700558185577393,
                2.3294427718147657e-32,
                -0.22795122861862184,
                -0.7537648677825928,
                0.9335417747497559,
                1.1048296689987183,
                -0.4265088737010956,
                -0.061783138662576678,
                -0.3059937655925751,
                1.380932092666626,
                -0.05817946791648865,
                0.17751149833202363,
                -0.17982862889766694,
                -0.1758047342300415,
                -0.03796929866075516,
                -0.25814861059188845,
                0.3436054289340973,
                0.8960521817207336,
                -1.1258002519607545,
                -0.1386878788471222,
                -0.625044584274292,
                -0.037302203476428989,
                0.2412639707326889,
                -0.8721104860305786,
                0.032961342483758929,
                -0.47455817461013796,
                -0.7026269435882568,
                -0.7832337021827698,
                -0.38819384574890139,
                -0.08691540360450745,
                0.006249774247407913,
                0.019107047468423845,
                0.22009314596652986,
                -0.19854964315891267,
                -1.2380073070526124,
                1.5767637491226197,
                -0.35322311520576479,
                0.8523069024085999,
                0.43691563606262209,
                0.3098922669887543,
                -0.541305422782898,
                -0.18057149648666383,
                -0.9980296492576599,
                0.35276687145233157,
                0.8294113278388977,
                0.5548853874206543,
                0.22526021301746369,
                -0.7639250755310059,
                -0.4246352016925812,
                0.3392816185951233,
                0.9055688977241516,
                -0.04819883033633232,
                -0.3669106662273407,
                0.5864011645317078,
                -0.269476056098938,
                -0.09392014890909195,
                0.542833149433136,
                0.6251004338264465,
                0.2570515275001526,
                -0.19628237187862397,
                -0.09109511226415634,
                0.5768703818321228,
                0.06330413371324539,
                -0.004105143249034882,
                0.23493748903274537,
                1.3728694915771485,
                -0.21599575877189637,
                0.27239668369293215,
                -0.6124306321144104,
                -1.2234675884246827,
                0.3306127190589905,
                0.43596982955932619,
                0.16712117195129395,
                -0.8858569264411926,
                -0.6852110624313355,
                0.8400818109512329,
                0.01732943393290043,
                -0.12679222226142884,
                -0.02611479163169861,
                0.5629050731658936,
                0.004254987463355064,
                -1.0337895154953004,
                1.383658528327942,
                -0.2396637201309204,
                -0.1266912817955017,
                -1.0192285776138306,
                -1.0782796144485474,
                -0.4407467842102051,
                0.0583379790186882,
                -0.853515088558197,
                0.5971606969833374,
                -1.0091453790664673,
                0.16330401599407197,
                0.32237905263900759,
                -0.02540673315525055,
                -0.39751672744750979,
                1.707327127456665,
                -1.5586036819922133e-32,
                -0.28353068232536318,
                0.6961038112640381,
                0.2796996235847473,
                1.0944440364837647,
                0.5912005305290222,
                0.16426771879196168,
                1.0509127378463746,
                -0.11322839558124542,
                -0.6214461922645569,
                1.8378454446792603,
                0.830061137676239,
                0.12251030653715134,
                -0.27094006538391116,
                -0.24508683383464814,
                -0.11097817867994309,
                0.15498651564121247,
                -0.8420482873916626,
                -0.6244221329689026,
                -0.5075727105140686,
                0.19688501954078675,
                -0.9064576625823975,
                1.70216703414917,
                0.44030216336250307,
                0.039846792817115787,
                -0.9320281744003296,
                0.002041853964328766,
                0.7674928903579712,
                -0.555304765701294,
                0.22511489689350129,
                0.014853137545287609,
                0.5549730062484741,
                0.2138926386833191,
                -0.3982841372489929,
                0.7245775461196899,
                0.5378742218017578,
                -0.39359480142593386,
                1.6640626192092896,
                -0.17974530160427094,
                -0.14907388389110566,
                0.4549044966697693,
                0.4034898281097412,
                0.3742386996746063,
                -0.33224332332611086,
                -0.32605451345443728,
                -0.2104167938232422,
                0.7403643131256104,
                0.7000395059585571,
                -0.765968918800354,
                0.06231135129928589,
                0.5676470994949341,
                0.2962799072265625,
                -0.04939989373087883,
                0.05588390678167343,
                0.12626540660858155,
                -0.08391837775707245,
                -0.1618400514125824,
                -0.5049074292182922,
                0.11568447947502136,
                -0.6696765422821045,
                0.12932418286800385,
                -0.39326173067092898,
                0.5594993829727173,
                0.18775632977485658,
                0.6855105757713318,
                -0.7502402663230896,
                -0.597560465335846,
                -0.4015011191368103,
                0.6408573389053345,
                1.2148048877716065,
                0.03429638221859932,
                -0.6127548813819885,
                0.20881322026252747,
                -0.02490028738975525,
                -0.8143033981323242,
                -0.2272966206073761,
                -0.5831689238548279,
                -0.22927051782608033,
                -0.2081073522567749,
                0.14104659855365754,
                -0.09393731504678726,
                -0.33177322149276736,
                -0.3829447329044342,
                -0.778586208820343,
                1.0190352201461793,
                -0.10777964442968369,
                0.6247885227203369,
                -0.05471610650420189,
                -0.037067703902721408,
                0.3075559139251709,
                0.5394055843353272,
                0.8177947402000427,
                -0.011291757225990296,
                -0.31230443716049197,
                -0.2940394878387451,
                -0.10085977613925934,
                -9.28512093878453e-8,
                -0.5744401216506958,
                -0.7941571474075317,
                0.2617530822753906,
                -0.2650687098503113,
                -0.45836779475212099,
                -0.21187087893486024,
                -0.739494800567627,
                -0.5425136685371399,
                -0.6259490847587586,
                -0.5307319164276123,
                0.29878076910972597,
                -0.048161573708057407,
                -0.03797583281993866,
                0.6938958168029785,
                0.0953495055437088,
                -0.18789923191070558,
                -0.010412191972136498,
                0.3067915439605713,
                -0.08625678718090058,
                0.33038780093193056,
                -0.3687931001186371,
                0.39412152767181399,
                0.3019355535507202,
                0.9919730424880981,
                -0.30636921525001528,
                0.6219334006309509,
                0.6864567995071411,
                1.0571249723434449,
                -0.14600712060928346,
                -0.5463977456092835,
                1.1797728538513184,
                0.27261027693748476,
                -0.612309455871582,
                -0.5708702802658081,
                0.6187672019004822,
                0.657008171081543,
                -0.5651035904884338,
                -0.06082867830991745,
                -0.10749395191669464,
                1.5772759914398194,
                -0.6237338185310364,
                -0.28229573369026186,
                -0.11203958094120026,
                -0.2630878984928131,
                -0.41261249780654909,
                -0.7407177686691284,
                -0.9302734732627869,
                -0.4357568919658661,
                -0.3020924925804138,
                -0.8617072701454163,
                -0.5068978667259216,
                -0.47037816047668459,
                -0.2763807475566864,
                -0.08968955278396607,
                -0.1299860030412674,
                -0.30126744508743288,
                0.2287369668483734,
                -0.37037473917007449,
                -1.3099539279937745,
                0.5800127983093262,
                0.619743824005127,
                0.7162157893180847,
                0.3974592387676239,
                -0.8321610689163208,
                0.35330361127853396,
                0.5998180508613586,
                0.4242449104785919,
                0.945661187171936,
                -0.3797282576560974,
                0.8825353980064392,
                0.347637802362442,
                -0.5444936156272888,
                -0.23077930510044099,
                0.03476869687438011,
                1.6793911457061768,
                -0.156342014670372,
                0.21762847900390626,
                -0.4830709397792816,
                0.12105128169059754,
                0.3963160812854767,
                0.17878732085227967,
                -0.5575876235961914,
                -0.3084894120693207,
                -0.3652607500553131,
                1.0142236948013306,
                0.6887761354446411,
                0.6596108078956604,
                0.22282375395298005,
                -0.24052296578884126,
                0.6434443593025208,
                -0.18342480063438416,
                0.132218137383461,
                0.9299465417861939,
                0.05331326276063919,
                -0.4049243628978729,
                -0.031077001243829728,
                -0.040773529559373859,
                0.6779357194900513,
                -0.06693492829799652,
                -0.3355293869972229,
                -0.26996728777885439,
                0.03142556920647621,
                0.11176569759845734,
                -0.39340710639953616,
                -0.3074522018432617,
                -0.3675272464752197,
                0.503991425037384,
                0.36952099204063418,
                0.3702585697174072,
                -0.4658111333847046,
                -0.6183658242225647,
                -0.21141603589057923,
                0.2996637523174286,
                -0.43021029233932497,
                -0.5957843065261841,
                -0.168859601020813,
                -0.612859308719635,
                0.5359165072441101,
                -0.1350734680891037,
                0.5991060137748718,
                0.2826325297355652,
                0.11745969206094742,
                -0.1293514370918274,
                -0.0647178515791893,
                -0.33976539969444277,
                -0.188390851020813,
                -1.4005059003829957,
                -0.19163675606250764,
                1.3754279613494874,
                -0.2096291035413742,
                0.13583984971046449,
                -1.2114613056182862,
                -0.6402503252029419,
                0.5946738123893738,
                -0.00419265404343605,
                0.8393058776855469,
                -0.4442216753959656,
                0.5610324144363403,
                -0.5214608907699585,
                0.6768717169761658,
                0.30222785472869875,
                -0.23995669186115266,
                1.1677249670028687,
                0.546051561832428,
                0.3220168650150299,
                -1.7469637393951417,
                0.07420588284730911,
                0.8875441551208496,
                0.4589070975780487,
                -0.5079702138900757,
                0.3811577558517456,
                -1.177929401397705,
                -0.98191237449646,
                0.21657708287239076,
                -0.13494664430618287,
                -1.067687749862671,
                0.4849909245967865,
                0.31572338938713076,
                -0.2519247829914093,
                -0.46665337681770327,
                -0.7720581293106079,
                0.10743994265794754,
                0.8750199675559998,
                -0.13609588146209718,
                0.11265822499990463,
                0.03920949250459671,
                0.40526503324508669,
                -0.2744123041629791,
                -0.7956770658493042,
                -0.5386636257171631,
                -0.13576415181159974,
                -0.4415760338306427,
                -0.21280066668987275,
                -0.762557864189148,
                0.03970620408654213,
                0.5224325656890869,
                0.3666166365146637,
                -0.38692113757133486,
                -1.010405421257019,
                0.2079332023859024,
                -0.7448342442512512,
                0.6928860545158386,
                -0.2732078731060028,
                -0.22411610186100007,
                0.17651310563087464,
                0.728863537311554,
                -0.6894481778144836,
                0.093876913189888,
                -0.8256320357322693,
                -0.26943832635879519,
                0.484383761882782,
                2.4195955478095548e-32,
                -0.3502908945083618,
                -0.2676079273223877,
                0.2046971321105957,
                -0.18877169489860536,
                -0.470842182636261,
                0.2358596920967102,
                -0.04649932309985161,
                0.4695040285587311,
                -0.27991873025894167,
                0.32014715671539309,
                -0.5845527648925781,
                -1.667178750038147,
                0.4473734200000763,
                0.6446110010147095,
                -0.17814785242080689,
                0.4117039144039154,
                -0.15472480654716493,
                1.287308931350708,
                0.3171912431716919,
                0.6933290958404541,
                0.5243046283721924,
                -0.32957923412323,
                0.017880981788039209,
                -0.4442598521709442,
                -0.11982811987400055,
                -0.7800136208534241,
                0.5027564167976379,
                -0.29842865467071535,
                -0.6017605066299439,
                0.3145046830177307,
                -0.0804135873913765,
                -0.6366478204727173,
                0.24619843065738679,
                -0.006815922446548939,
                0.7546831369400024,
                0.9044265747070313,
                0.46002545952796938,
                -0.32060906291007998,
                0.13424964249134065,
                -0.01074080541729927,
                -0.5371200442314148,
                -0.1936718374490738,
                -0.2764829397201538,
                0.11269579082727432,
                0.3394826650619507,
                -0.26930439472198489,
                -0.3896336555480957,
                0.4212973117828369,
                0.6000418663024902,
                0.28141218423843386,
                0.04656528681516647,
                0.7104458212852478,
                0.5205278396606445,
                -0.1683111935853958,
                0.9581893086433411,
                0.5334323644638062,
                0.14808066189289094,
                0.7540419101715088,
                0.6047973036766052,
                0.10043083131313324,
                0.4800248444080353,
                0.661862850189209,
                0.013050206005573273,
                0.09582315385341644,
                0.7623546719551086,
                0.20079174637794496,
                0.050186656415462497,
                -0.44294872879981997,
                1.0699394941329957,
                0.4197533428668976,
                -0.7175852060317993,
                -0.25005531311035159,
                -0.8580552935600281,
                0.5631731152534485,
                -0.059158481657505038,
                0.13968658447265626,
                0.7821635603904724,
                -0.03456559777259827,
                -0.18263328075408936,
                -0.15012599527835847,
                0.3707154393196106,
                -0.8821276426315308,
                0.13918432593345643,
                0.9488967657089233,
                -0.7358802556991577,
                0.24646462500095368,
                0.3146522641181946,
                -0.21309560537338258,
                -0.23411482572555543,
                -0.22807742655277253,
                0.7990178465843201,
                -0.035234175622463229,
                0.7691605091094971,
                -0.07149368524551392,
                0.4449818730354309,
                -1.9612692718663879e-32,
                -0.17471203207969666,
                0.5517823696136475,
                -0.33037734031677248,
                -0.4141364097595215,
                0.761204183101654,
                -0.7439022064208984,
                -0.29485517740249636,
                -1.0250178575515748,
                -0.15984001755714417,
                0.28773292899131777,
                0.26826733350753786,
                0.3217011094093323,
                -0.16989901661872865,
                0.04954686760902405,
                0.1786087304353714,
                0.5550249218940735,
                0.6318030953407288,
                0.031847819685935977,
                -0.24694333970546723,
                0.8632058501243591,
                -0.2616916596889496,
                0.4945564270019531,
                -0.2934386134147644,
                -0.27033188939094546,
                -0.19288258254528047,
                -0.2779443562030792,
                -0.38909614086151125,
                -0.40341633558273318,
                -1.2694908380508423,
                0.04685196280479431,
                0.2881724238395691,
                -0.5447119474411011,
                -0.4937672019004822,
                0.3536761403083801,
                0.7790550589561462,
                -0.15545836091041566,
                0.31983861327171328,
                -0.6960188150405884,
                -0.5293192267417908,
                1.8948615789413453,
                -0.14896604418754579,
                -0.6159010529518127,
                0.19697114825248719,
                -0.06449215859174729,
                0.2626532316207886,
                -0.4879067540168762,
                -1.0838762521743775,
                -0.38371071219444277,
                -0.2317076474428177,
                0.7585934400558472,
                -0.46588513255119326,
                0.35060474276542666,
                -1.0435937643051148,
                -0.061040833592414859,
                -0.21201089024543763,
                -0.1746743768453598,
                -0.2750746011734009,
                -0.39702075719833376,
                -0.06441254168748856,
                -0.19876325130462647,
                -0.359611839056015,
                0.35613125562667849,
                0.057120803743600848,
                -0.12305065989494324,
                1.5797756910324097,
                -0.968510627746582,
                0.2676442861557007,
                0.696770191192627,
                0.07309927046298981,
                -0.7805717587471008,
                -0.12598858773708344,
                0.13339565694332124,
                -0.37750813364982607,
                0.8401539325714111,
                -0.7142421007156372,
                -1.161677360534668,
                -0.11945576965808869,
                -0.4664500057697296,
                -0.4783673882484436,
                -0.5697513222694397,
                0.7904021739959717,
                -0.35399049520492556,
                0.0878000259399414,
                -0.3081904351711273,
                -0.5164710879325867,
                -0.13522063195705415,
                0.46323224902153017,
                -0.497185617685318,
                0.20940448343753816,
                0.9839490652084351,
                -0.22139890491962434,
                -0.3043484091758728,
                0.24042712152004243,
                0.16223648190498353,
                -0.18810570240020753,
                -9.367082043354458e-8,
                -0.2323288768529892,
                0.13619863986968995,
                -0.4727030098438263,
                0.14996281266212464,
                0.020356636494398118,
                0.21975667774677277,
                1.0631234645843506,
                -0.4560871720314026,
                -0.3027549386024475,
                0.07837269455194473,
                0.5049428939819336,
                0.5718187093734741,
                -0.7701569199562073,
                -0.22857975959777833,
                -0.46219301223754885,
                -0.11683189868927002,
                -0.37388062477111819,
                -0.16489794850349427,
                0.09539876878261566,
                0.18085691332817079,
                0.27874261140823367,
                0.1145908534526825,
                -0.05094270035624504,
                0.8608376979827881,
                -0.10392507165670395,
                0.3422601521015167,
                0.6771668791770935,
                0.6723223328590393,
                0.18970027565956117,
                0.7272946238517761,
                0.8358654975891113,
                0.8643932342529297,
                -0.3585262894630432,
                0.12650884687900544,
                0.21613061428070069,
                0.22530730068683625,
                0.6965866088867188,
                -0.010611047968268395,
                0.5856994986534119,
                -0.48316630721092226,
                -0.29278022050857546,
                -0.11928916722536087,
                -0.2589260935783386,
                0.48702165484428408,
                0.5602065920829773,
                -0.35236164927482607,
                -0.29723960161209109,
                -0.05166718363761902,
                0.0809892788529396,
                -0.4464084208011627,
                0.24184337258338929,
                -0.5763502717018127,
                -0.20557880401611329,
                -0.2734088897705078,
                0.5468353629112244,
                0.44107675552368166,
                0.23210251331329347,
                -0.35185903310775759,
                -1.0815402269363404,
                -0.39534440636634829,
                0.6527233719825745,
                -0.14847828447818757,
                0.2671165466308594,
                -1.6597894430160523,
                0.7676731944084168,
                0.4344428777694702,
                -0.24066093564033509,
                0.6789773106575012,
                0.29690366983413699,
                -0.7027360200881958,
                -0.543782651424408,
                0.1547357589006424,
                0.1021546795964241,
                0.24370531737804414,
                0.5995134711265564,
                0.4028584957122803,
                -0.17153522372245789,
                0.3027194142341614,
                -0.0122258011251688,
                -0.446537584066391,
                0.050837431102991107,
                -0.6579862833023071,
                0.1548657864332199,
                0.2819405794143677,
                0.6829910278320313,
                0.6508842706680298,
                -0.05960182473063469,
                0.3762063682079315,
                0.08306241035461426,
                0.044652171432971957,
                -0.260405957698822,
                0.1425667256116867,
                -0.33800533413887026,
                0.49337446689605715,
                -0.5563245415687561,
                -0.15679456293582917,
                0.1257190853357315,
                -0.22687619924545289,
                -0.19988031685352326,
                0.23077060282230378,
                0.6862329244613648,
                0.3212391138076782,
                -0.5105114579200745,
                0.24662470817565919,
                0.18916985392570496,
                -0.571397602558136,
                -0.2581574618816376,
                0.08602935075759888,
                -0.09631957858800888,
                0.061536453664302829,
                0.14195770025253297,
                0.07697213441133499,
                0.08190369606018067,
                0.09394200146198273,
                -0.4486941695213318,
                -0.5275091528892517,
                0.026358410716056825,
                0.08666907250881195,
                0.04883170500397682,
                -0.258047878742218,
                0.13492052257061006,
                -0.22861775755882264,
                0.6251978278160095,
                0.20127111673355103,
                -0.34660500288009646,
                -0.09250453114509583,
                -0.632421612739563,
                2.26654314994812,
                0.6261004209518433,
                0.5899518132209778,
                -0.22860650718212129,
                -0.43508046865463259,
                -0.278664767742157,
                0.035361774265766147,
                0.4034963846206665,
                -0.5646915435791016,
                0.28240537643432619,
                0.05900390446186066,
                -0.10373976826667786,
                -0.5649967193603516,
                0.12936978042125703,
                -0.573345422744751,
                0.9935505390167236,
                0.4011586010456085,
                -0.6440178751945496,
                0.4041654169559479,
                -0.2493087202310562,
                0.0452500656247139,
                -0.34300294518470766,
                -0.5479019284248352,
                0.35080939531326296,
                0.8683427572250366,
                -0.2637566924095154,
                0.22718091309070588,
                -0.6062103509902954,
                0.1348685473203659,
                0.41629865765571597,
                0.013335774652659893,
                -0.24947822093963624,
                0.0629529282450676,
                -0.5589937567710877,
                0.008212999440729618,
                -0.1133892610669136,
                -1.2157617807388306,
                -0.013436585664749146,
                0.5131303071975708,
                -0.05626959353685379,
                1.1235724687576295,
                -0.5626686215400696,
                -0.6435989737510681,
                -0.27480292320251467,
                -0.06009949743747711,
                -0.1831679344177246,
                0.10445216298103333,
                -0.3799043595790863,
                0.043904419988393787,
                0.47634679079055788,
                -0.06352123618125916,
                -0.06064879521727562,
                0.6239752173423767,
                -0.2967524230480194,
                -0.2604717016220093,
                0.1036355197429657,
                0.0841154232621193,
                0.2665804326534271,
                0.7799540758132935,
                0.036873720586299899,
                -0.10959865897893906,
                -0.4776659905910492,
                -0.5727196931838989,
                0.03428442031145096,
                -7.749270183643307e-32,
                -0.18113557994365693,
                -0.2500230669975281,
                0.502473771572113,
                0.5543073415756226,
                0.3323201835155487,
                -0.7055015563964844,
                -0.13124965131282807,
                0.15373890101909638,
                0.2753092348575592,
                0.409247487783432,
                0.14067652821540833,
                -0.18789060413837434,
                -0.19316868484020234,
                -0.08548503369092941,
                0.8162241578102112,
                0.5981311202049255,
                -0.5753440856933594,
                0.19700005650520326,
                -0.39108267426490786,
                0.059203650802373889,
                -0.6475054621696472,
                0.6086475253105164,
                -0.2608051598072052,
                -0.09864397346973419,
                -0.5009469985961914,
                -0.6793068647384644,
                0.2372276932001114,
                0.16843678057193757,
                0.23351657390594483,
                0.037626784294843677,
                0.29183489084243777,
                0.49850907921791079,
                0.24487318098545075,
                0.24524620175361634,
                0.21626754105091096,
                -0.07225736230611801,
                0.2965903878211975,
                -0.6762599349021912,
                -0.22464552521705628,
                -0.07981263101100922,
                -0.11020507663488388,
                -0.09325731545686722,
                -0.5076656341552734,
                0.018070850521326066,
                1.0613075494766236,
                -0.16985172033309937,
                -0.07569979131221771,
                -0.4232069253921509,
                -1.0260416269302369,
                0.15256431698799134,
                -0.25321337580680849,
                0.29922133684158327,
                1.2875115871429444,
                -0.3895619213581085,
                -0.21546345949172975,
                0.056844279170036319,
                0.3198340833187103,
                -0.37351202964782717,
                -0.5222975611686707,
                0.5360217094421387,
                -0.7534033060073853,
                0.46024832129478457,
                -0.1128472089767456,
                0.14606621861457826,
                -0.32137906551361086,
                0.04790150374174118,
                -0.3398304581642151,
                0.24233032763004304,
                -0.18618713319301606,
                0.15168318152427674,
                0.28724977374076846,
                -0.6416199803352356,
                -0.589028537273407,
                -0.11166156083345413,
                -0.37863022089004519,
                -0.3892950415611267,
                -0.045521773397922519,
                -0.18032701313495637,
                0.636570394039154,
                0.35615432262420657,
                -0.29464229941368105,
                -0.7363148331642151,
                0.2780818045139313,
                -0.7643568515777588,
                -0.42724886536598208,
                0.4732123911380768,
                -0.15686070919036866,
                -0.2242860198020935,
                0.04229775443673134,
                0.03939387947320938,
                0.45596739649772646,
                0.4456591010093689,
                0.22353395819664002,
                0.28881508111953738,
                -0.9779438376426697,
                4.6031136264075926e-32,
                0.3228108584880829,
                -0.057317350059747699,
                -0.592494785785675,
                0.6819571256637573,
                0.5540664792060852,
                -0.15337076783180238,
                0.32681283354759219,
                -0.18804313242435456,
                -0.04518445208668709,
                0.5151382088661194,
                -0.3505914509296417,
                0.34323132038116457,
                0.15311862528324128,
                -0.25586700439453127,
                -0.168685644865036,
                -0.3849745988845825,
                -0.17035289108753205,
                -0.29922527074813845,
                -0.14904744923114777,
                -0.3522968292236328,
                -0.1556011438369751,
                0.45946115255355837,
                -0.026458919048309327,
                1.0271353721618653,
                -0.04260767251253128,
                0.05399768427014351,
                0.25648021697998049,
                -0.3891659379005432,
                -0.31368669867515566,
                0.22465164959430695,
                0.027000943198800088,
                -0.5249308347702026,
                -1.1960827112197877,
                -0.6088312268257141,
                -0.40197426080703738,
                0.1743047833442688,
                0.2382335662841797,
                -0.9434306621551514,
                0.05504569411277771,
                -0.6843193769454956,
                0.20525966584682465,
                0.0951148271560669,
                0.72835773229599,
                2.021235466003418,
                -0.13815444707870484,
                0.43748366832733157,
                0.022721320390701295,
                -1.0178704261779786,
                -0.6487738490104675,
                -0.2505216598510742,
                -0.127615287899971,
                0.2652725577354431,
                -0.000056471675634384158,
                -0.055777110159397128,
                -0.3466324806213379,
                -0.4931449592113495,
                0.14661714434623719,
                -0.32285448908805849,
                -0.1914777308702469,
                -0.04411676898598671,
                -0.6486060619354248,
                0.14126235246658326,
                0.21753472089767457,
                0.06146020069718361,
                -0.33002230525016787,
                -0.3004666566848755,
                -0.6177195906639099,
                0.7985106706619263,
                -0.21239054203033448,
                -0.17847973108291627,
                1.8430405855178834,
                -0.1569182425737381,
                -0.07435252517461777,
                -0.3905876874923706,
                -0.2959181070327759,
                0.03345569968223572,
                -0.7786704897880554,
                -0.0037330854684114458,
                0.075076162815094,
                -0.4340701997280121,
                -0.3934592008590698,
                -0.25654762983322146,
                0.009612915106117726,
                0.048038359731435779,
                -0.4238366484642029,
                0.07404747605323792,
                -0.44841131567955019,
                -0.27101758122444155,
                -0.3083727955818176,
                0.28132376074790957,
                -0.8088775277137756,
                -0.5122926235198975,
                -0.2287425547838211,
                -0.293658584356308,
                -0.1794423907995224,
                -8.963203868006531e-8,
                -0.00844682939350605,
                0.13584449887275697,
                0.43020620942115786,
                0.1686670035123825,
                0.5006487965583801,
                0.6327803134918213,
                0.6943822503089905,
                0.5637688040733337,
                0.18289442360401154,
                0.30889907479286196,
                0.5018983483314514,
                0.17297622561454774,
                -1.3259116411209107,
                -0.7080926299095154,
                0.5881845951080322,
                0.17397290468215943,
                0.36727648973464968,
                0.09950119256973267,
                -0.1955532729625702,
                0.47587865591049197,
                -0.20013809204101563,
                0.5142917633056641,
                -0.183190256357193,
                0.2178468406200409,
                0.37421318888664248,
                0.16657817363739015,
                0.468108594417572,
                0.3405163884162903,
                -0.0019351784139871598,
                0.42133891582489016,
                0.3524259030818939,
                -0.04264551028609276,
                -0.09526530653238297,
                -0.2792825698852539,
                0.09650860726833344,
                0.8994303941726685,
                0.7090333700180054,
                -0.4454326927661896,
                -0.16081497073173524,
                0.4827938377857208,
                0.31673938035964968,
                1.1412014961242676,
                0.19116604328155518,
                -0.10027626156806946,
                0.20424900949001313,
                -0.2966447174549103,
                0.15872186422348023,
                -0.4646945595741272,
                0.34111931920051577,
                0.05428621545433998,
                0.04066018760204315,
                0.8236393332481384,
                0.5409248471260071,
                -0.17150437831878663,
                0.12322933971881867,
                -0.3225708603858948,
                -0.08346830308437348,
                -0.14356030523777009,
                -0.0653674453496933,
                0.3981473743915558,
                0.7354548573493958,
                -0.26994788646698,
                0.2200424075126648,
                -0.5317423343658447,
                0.9532397985458374,
                0.40574562549591067,
                0.06037788838148117,
                0.8833091855049133,
                0.1955648511648178,
                -0.8841359615325928,
                -0.09279362112283707,
                0.27204853296279909,
                0.2729395031929016,
                0.20415686070919038,
                0.3150346279144287,
                0.25115475058555605,
                -0.1707945317029953,
                0.4557205140590668,
                0.10984431207180023,
                -0.4536248743534088,
                0.10307484865188599,
                -0.43385106325149538,
                0.10360820591449738,
                0.6313861012458801,
                0.4154942035675049,
                0.5902491807937622,
                -0.19702880084514619,
                0.223180890083313,
                0.4059304893016815,
                0.12477871775627136,
                -0.3069896101951599,
                -0.0309490617364645,
                -0.0619419626891613,
                0.34112945199012759,
                -0.787070095539093,
                -0.1875854730606079,
                0.9683545827865601,
                0.2359931468963623,
                -0.32160767912864687,
                0.1886691451072693,
                0.6737374067306519,
                0.4621899724006653,
                -0.6117752194404602,
                0.42139193415641787,
                -0.08051827549934387,
                -0.6894972324371338,
                -0.1437285989522934,
                0.019933100789785386,
                0.13920170068740846,
                -0.13531705737113954,
                -0.023332122713327409,
                0.293605774641037,
                0.1312507838010788,
                0.17144940793514253,
                -0.5118836760520935,
                -0.6227210760116577,
                -0.03277796506881714,
                0.11852806806564331,
                0.14521224796772004,
                0.10152844339609146,
                0.36350467801094057,
                0.033404525369405749,
                0.8305764198303223,
                0.1367591768503189,
                -0.2271173596382141,
                -0.02652657777070999,
                -0.2770538330078125,
                0.6169585585594177,
                0.9436415433883667,
                0.710844874382019,
                -0.334053099155426,
                -0.3581928610801697,
                -0.6182308197021484,
                0.4245925545692444,
                -0.1130528599023819,
                -0.34247344732284548,
                0.15866915881633759,
                0.13523167371749879,
                0.05372702330350876,
                -0.5648167729377747,
                -0.14197640120983125,
                -0.9163562655448914,
                0.8442267179489136,
                0.059246160089969638,
                -0.45602017641067507,
                0.4474963843822479,
                -0.39804840087890627,
                0.07827571034431458,
                -0.5841413736343384,
                -0.9652947187423706,
                0.24346685409545899,
                0.5945135951042175,
                -0.47677963972091677,
                0.2488965392112732,
                -0.5700446963310242,
                0.280565083026886,
                0.4200229346752167,
                -0.3164346516132355,
                -0.2807326912879944,
                -0.2302534133195877,
                -0.5578022003173828,
                0.5680168867111206,
                -0.5029423236846924,
                -0.8728393912315369,
                -0.035318247973918918,
                0.4892759919166565,
                -0.011272352188825608,
                1.0057891607284547,
                -0.689005434513092,
                -0.8312308192253113,
                -0.24512241780757905,
                -0.3115054666996002,
                0.16858376562595368,
                0.2531853914260864,
                -0.3116704523563385,
                0.026675205677747728,
                0.7532036304473877,
                0.10739210247993469,
                0.29597237706184389,
                0.3876926302909851,
                -0.04583879932761192,
                -0.207985520362854,
                0.09580081701278687,
                0.0694444552063942,
                0.04576266556978226,
                0.4328845143318176,
                -0.20209556818008424,
                -0.02574576437473297,
                -0.6698071956634522,
                -0.7360023856163025,
                0.058548182249069217,
                -7.18449863306616e-32,
                -0.11992944031953812,
                -0.7128020524978638,
                0.32623526453971865,
                0.4423999786376953,
                0.4454081058502197,
                -0.48549485206604006,
                -0.02100784331560135,
                -0.143925279378891,
                0.14163929224014283,
                0.5915327072143555,
                0.2106803059577942,
                -0.3142317831516266,
                -0.3107242286205292,
                0.35279810428619387,
                0.8255383968353272,
                0.7862129807472229,
                -0.784352719783783,
                0.10778709501028061,
                -0.11394380033016205,
                0.5844687223434448,
                -0.7391705513000488,
                0.7348217964172363,
                -0.37934863567352297,
                0.026554344221949579,
                -0.5565614104270935,
                -0.7853492498397827,
                0.2580389380455017,
                0.3098090887069702,
                -0.09129170328378678,
                0.18420585989952088,
                0.3052809238433838,
                0.561339259147644,
                -0.021740669384598733,
                0.15749333798885346,
                0.38053297996520998,
                -0.0765191838145256,
                0.05943147838115692,
                -1.105035662651062,
                -0.05541378632187843,
                0.08115026354789734,
                -0.3413921892642975,
                -0.19801631569862367,
                -0.16996286809444428,
                0.001499587669968605,
                1.1231495141983033,
                0.07042843103408814,
                -0.1337141990661621,
                -0.42969000339508059,
                -0.76861971616745,
                -0.18774938583374024,
                -0.26744216680526736,
                -0.0318220779299736,
                0.839858889579773,
                -0.38178375363349917,
                -0.5384589433670044,
                0.2984263300895691,
                0.5298572182655335,
                -0.7489229440689087,
                -0.6467719078063965,
                0.4810550808906555,
                -0.49217647314071658,
                0.7198278903961182,
                -0.4743991494178772,
                0.2879674434661865,
                0.03590342029929161,
                0.07144978642463684,
                -0.31456950306892397,
                0.1337875872850418,
                -0.28832441568374636,
                -0.1996038854122162,
                -0.03087536245584488,
                -0.598579466342926,
                -0.5433133244514465,
                -0.10276979953050614,
                -0.35101160407066347,
                -0.462692528963089,
                -0.15079337358474732,
                -0.2716574966907501,
                0.4343196153640747,
                0.19953079521656037,
                -0.0687602311372757,
                -0.9086378216743469,
                0.20505467057228089,
                -0.9638794660568237,
                -0.542161762714386,
                0.19417670369148255,
                -0.31835153698921206,
                -0.7678333520889282,
                0.14487208425998689,
                -0.09977197647094727,
                0.25972780585289,
                0.36457011103630068,
                0.0003987625241279602,
                0.04148028790950775,
                -0.7077435255050659,
                4.728511249523086e-32,
                -0.15274575352668763,
                -0.20325219631195069,
                -0.6945550441741943,
                0.504854679107666,
                0.6546149849891663,
                -0.1807481348514557,
                0.2209261953830719,
                0.18456663191318513,
                -0.27847588062286379,
                0.7965016961097717,
                -0.8216132521629334,
                0.3402889370918274,
                0.40006643533706667,
                -0.057079121470451358,
                0.27061712741851809,
                -0.5230779647827148,
                0.00605111476033926,
                -0.519798219203949,
                -0.14520923793315888,
                -0.6976168751716614,
                -0.3881266117095947,
                0.21855531632900239,
                -0.3963104486465454,
                0.9306173920631409,
                -0.1490081250667572,
                0.18378180265426637,
                0.17451828718185426,
                -0.6185550689697266,
                -0.5158134698867798,
                0.009634491056203843,
                0.2977958917617798,
                -0.2550506591796875,
                -1.466958999633789,
                -0.5522478818893433,
                -0.1129622757434845,
                0.2669261395931244,
                0.2983464002609253,
                -0.721650242805481,
                -0.07683532685041428,
                -0.3745743930339813,
                0.19635862112045289,
                0.41279280185699465,
                1.0297154188156129,
                2.001431465148926,
                -0.03225702419877052,
                0.3391275405883789,
                -0.24024698138237,
                -1.2070993185043336,
                -0.1992669552564621,
                -0.2887876033782959,
                -0.005793644115328789,
                0.34905004501342776,
                -0.007462384179234505,
                -0.4112597107887268,
                -0.5943936705589294,
                -0.33520403504371645,
                0.4015749394893646,
                -0.24515333771705628,
                0.0804118663072586,
                -0.11187252402305603,
                -0.9070234894752502,
                0.28725871443748476,
                -0.1356983482837677,
                0.6907176375389099,
                -0.2071085125207901,
                -0.1974690705537796,
                -0.6368651986122131,
                0.8615604639053345,
                0.0688234493136406,
                -0.3868708908557892,
                1.6392022371292115,
                0.2686614394187927,
                -0.8336239457130432,
                0.319071501493454,
                -0.3681918978691101,
                -0.04880324751138687,
                -0.9512152671813965,
                0.31179338693618777,
                -0.10645240545272827,
                -0.5838937163352966,
                -0.5078170895576477,
                -0.3203014135360718,
                0.2775390148162842,
                0.15879516303539277,
                -0.8772176504135132,
                0.30865228176116946,
                0.058649469166994098,
                -0.23343725502490998,
                -0.038354285061359408,
                0.1172235757112503,
                -0.9659915566444397,
                -0.4299235939979553,
                0.08976257592439652,
                -0.18236449360847474,
                -0.09241412580013275,
                -9.148835289352064e-8,
                -0.031499698758125308,
                0.3967353105545044,
                0.6721820831298828,
                0.3305008113384247,
                0.46375343203544619,
                1.0055333375930787,
                0.7271708250045776,
                0.597147524356842,
                0.06608837842941284,
                0.4956960082054138,
                0.6510422229766846,
                0.3351703882217407,
                -0.8053209781646729,
                -0.4413580894470215,
                1.0660336017608643,
                -0.033039260655641559,
                0.39520901441574099,
                -0.17755499482154847,
                -0.04487866535782814,
                0.8776014447212219,
                -0.37748414278030398,
                0.48241153359413149,
                -0.2237437516450882,
                0.09815450012683869,
                0.41061195731163027,
                0.023342398926615716,
                0.6324443817138672,
                0.06142895668745041,
                0.03342953324317932,
                0.4853501617908478,
                0.4093213379383087,
                -0.17093296349048615,
                0.0025599217042326929,
                -0.517045259475708,
                -0.3497457206249237,
                1.0089865922927857,
                0.46990957856178286,
                -0.7431812286376953,
                0.12958809733390809,
                0.1324687898159027,
                0.3052644431591034,
                1.163149118423462,
                0.11924490332603455,
                0.4392565190792084,
                0.3036620020866394,
                -0.1850873827934265,
                0.18706151843070985,
                -0.16194170713424684,
                0.27838030457496645,
                -0.13450197875499726,
                -0.025829674676060678,
                0.7733514904975891,
                0.7372460961341858,
                -0.24069160223007203,
                0.014222852885723114,
                -0.20751789212226869,
                -0.025805188342928888,
                -0.15234965085983277,
                -0.07179667055606842,
                0.4586966633796692,
                0.7479405403137207,
                -0.27546077966690066,
                0.7349324226379395,
                -0.8841629028320313
            ]
        }
    ]
}
    ```

- [x] Explain the model repository structure and the `config.pbtxt` file  

    The model repository is how Triton knows what models exist, how to run them and how to manage the versions in a safe, scalable and predictable way.
    Since Triton can serve many models at a time, it helps Triton keeping things organized and standardized.

    A famous analogy would be a Library where:
        - model_repository ~> is the Library
        - all-MiniLM-L6-v2 (model name)~> is the book
        - 1 ~> First version of the book
        - model.onnx ~> The content of the book
        - config.pbtxt ~> The table of content of the book.

    The `config.pbtxt` file tells Triton:
        - name: This is the **model identifier** in Triton.
        - platform: Tells Triton which **backend** to use. In this case, the config file is telling Triton to use ONXX Runtime to run an ONNX model.
        - max_batch_size: Defines the maximum number of inputs Triton can process in one single batch. In this case, the model can handle from 1 to 8 sentences at a time. If the input would contain more than 8 sentences, it would fail.
        - input: must match the input tensor name in the ONXX model, the data type, in this case Integer 64 bits and the shape (dimensions) of the input.
        - output: must match the output tensor name in the ONXX model, the data type, in this case Float 32 bits and the shape (dimensions) of the output


#### Questions:

❓ Why is the model stored in a folder named 1? What does this number represent?
    It represents first version of the model. It must always be a positive integer.
    In this case it is stored in the folder name 1 because it is the first version of the model.

❓ What is the purpose of the config.pbtxt file? Why is it essential for Triton to understand how to serve your model? :question: Analyze the config.pbtxt file. What does each field represent?
    It is the bridge between my model file and Triton Inference server.
    The `config.pbtxt` file tells Triton:
            - name: This is the **model identifier** in Triton.
            - platform: Tells Triton which **backend** to use. In this case, the config file is telling Triton to use ONXX Runtime to run an ONNX model.
            - max_batch_size: Defines the maximum number of inputs Triton can process in one single batch. In this case, the model can handle from 1 to 8 sentences at a time. If the input would contain more than 8 sentences, it would fail.
            - input: must match the input tensor name in the ONXX model, the data type, in this case Integer 64 bits and the shape (dimensions) of the input.
            - output: must match the output tensor name in the ONXX model, the data type, in this case Float 32 bits and the shape (dimensions) of the output

❓ What is the purpose of the volume mapping (-v option)?
    Connecting a folder in my local machine to the container. This way the container accesses files while keeping them persistent and editable from the host.

❓ What information does the model status endpoint provide? How can you use this to debug model loading issues?
    !["Postman POST method response."](./img/01-docker-lab/AcceptanceCriteria_2_7.png) 

    This provides back the model metadata from Triton. Basically, it tells everything Triton knows about the model like: model name, model version, platform, input tensors the model expects and the output tensors Triton will return.

    In other words, it is the structure inference requests must have.

❓ Test the inference endpoint and analyze the response. What format does the output take? How does it differ from the Flask API response?
    The main difference is that Flask API model outputs a single scalar while MiniLM is multidimensional shown as a flattened list of values (as seen in the acceptance criteria response payloads).
    !["Testing MiniLM-L6-v2 inference endpoint via Postman (Body payload).](./img/01-docker-lab/AcceptanceCriteria_2_6.png)
    !["Testing MiniLM-L6-v2 inference endpoint via Postman (Response payload)."](./img/01-docker-lab/AcceptanceCriteria_2_8.png)

    As seen in the MiniLM response above, it produces a high dimensional response where as the Flask API application produces a single number.

❓ Triton also supports gRPC What is the difference between HTTP and gRPC for model inference? When would you choose one over the other?
    The main difference is the way the serialization is passed through: In HTTP is normally a human readable format like JSON or XML while in gRPC is binary.
    This causes for example that HTTP payload sizes are normally larger than gRPC (due to its binary format).

    HTTP normally uses HTTP/1 protocol while gRPC uses HTTP/2 protocol.

    You choose HTTP when you want to use public APIs by using HTTP/REST calls.
    Since the payload is human readable makes it also easier to develop and debug in case of issues.
    Cases when latency or volume are not relevant.

    You choose gRPC when high performance and low latency are critical for your application, there's large and frequent data transfer needed or if your application needs to stream data.


### Part 3: Docker Compose

- [x] Show that you created a 'docker-compose.yml' file to orchestrate your services  
This image shows the `docker-compose.yml` I created to orchestrate both services together.

    !["Docker Compose yml file."](./img/01-docker-lab/AcceptanceCriteria_3_1.png)

- [x] Show that you can start, stop, and view logs of your services using Docker Compose

    #### Starting the services

    By executing `docker compose up` we can see how the services are started in one go:

    ```bash
    docker compose up
    WARN[0000] /Users/rubencastillosainz/Documents/Benciowski/Personal/HOGent/MLOps/LABS/mlops-2526-RubenCastilloHOGent/resources/01-dockerlab/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
    Attaching to ml-flask-app, triton-server
    triton-server  | 
    triton-server  | =============================
    triton-server  | == Triton Inference Server ==
    triton-server  | =============================
    triton-server  | 
    triton-server  | NVIDIA Release 23.10 (build 72127510)
    triton-server  | Triton Server Version 2.39.0
    triton-server  | 
    triton-server  | Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
    triton-server  | 
    triton-server  | Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
    triton-server  | 
    triton-server  | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
    triton-server  | By pulling and using the container, you accept the terms and conditions of this license:
    triton-server  | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
    triton-server  | 
    triton-server  | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
    triton-server  |    Use the NVIDIA Container Toolkit to start this container with GPU support; see
    triton-server  |    https://docs.nvidia.com/datacenter/cloud-native/ .
    triton-server  | 
    triton-server  | W1101 12:32:20.094389 1 pinned_memory_manager.cc:237] Unable to allocate pinned system memory, pinned memory pool will not be available: CUDA driver version is insufficient for CUDA runtime version
    triton-server  | I1101 12:32:20.094724 1 cuda_memory_manager.cc:117] CUDA memory pool disabled
    triton-server  | I1101 12:32:20.139006 1 model_lifecycle.cc:461] loading: example_model:1
    triton-server  | I1101 12:32:20.140271 1 model_lifecycle.cc:461] loading: all-MiniLM-L6-v2:1
    triton-server  | I1101 12:32:20.441110 1 tensorflow.cc:2577] TRITONBACKEND_Initialize: tensorflow
    triton-server  | I1101 12:32:20.441137 1 tensorflow.cc:2587] Triton TRITONBACKEND API version: 1.16
    triton-server  | I1101 12:32:20.441139 1 tensorflow.cc:2593] 'tensorflow' TRITONBACKEND API version: 1.16
    triton-server  | I1101 12:32:20.441140 1 tensorflow.cc:2617] backend configuration:
    triton-server  | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}}
    triton-server  | I1101 12:32:20.441181 1 tensorflow.cc:2683] TRITONBACKEND_ModelInitialize: example_model (version 1)
    triton-server  | 2025-11-01 12:32:20.442871: I tensorflow/cc/saved_model/reader.cc:45] Reading SavedModel from: /models/example_model/1/model.savedmodel
    triton-server  | I1101 12:32:20.443664 1 onnxruntime.cc:2608] TRITONBACKEND_Initialize: onnxruntime
    triton-server  | I1101 12:32:20.443681 1 onnxruntime.cc:2618] Triton TRITONBACKEND API version: 1.16
    triton-server  | I1101 12:32:20.443704 1 onnxruntime.cc:2624] 'onnxruntime' TRITONBACKEND API version: 1.16
    triton-server  | I1101 12:32:20.443843 1 onnxruntime.cc:2654] backend configuration:
    triton-server  | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}}
    triton-server  | 2025-11-01 12:32:20.445545: I tensorflow/cc/saved_model/reader.cc:91] Reading meta graph with tags { serve }
    triton-server  | 2025-11-01 12:32:20.445574: I tensorflow/cc/saved_model/reader.cc:132] Reading SavedModel debug info (if present) from: /models/example_model/1/model.savedmodel
    triton-server  | I1101 12:32:20.453023 1 onnxruntime.cc:2719] TRITONBACKEND_ModelInitialize: all-MiniLM-L6-v2 (version 1)
    triton-server  | 2025-11-01 12:32:20.462960: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:375] MLIR V1 optimization pass is not enabled
    triton-server  | 2025-11-01 12:32:20.463574: I tensorflow/cc/saved_model/loader.cc:233] Restoring SavedModel bundle.
    triton-server  | 2025-11-01 12:32:20.499684: E tensorflow/core/framework/node_def_util.cc:676] NodeDef mentions attribute debug_name which is not in the op definition: Op<name=VarHandleOp; signature= -> resource:resource; attr=container:string,default=""; attr=shared_name:string,default=""; attr=dtype:
    type; attr=shape:shape; attr=allowed_devices:list(string),default=[]; is_stateful=true> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node Variable}}                                                                        triton-server  | 2025-11-01 12:32:20.505308: I tensorflow/cc/saved_model/loader.cc:217] Running initialization op on SavedModel bundle at path: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 12:32:20.520550: I tensorflow/cc/saved_model/loader.cc:334] SavedModel load for tags { serve }; Status: success: OK. Took 77781 microseconds.
    triton-server  | I1101 12:32:20.534009 1 tensorflow.cc:2732] TRITONBACKEND_ModelInstanceInitialize: example_model_0 (CPU device 0)
    triton-server  | 2025-11-01 12:32:20.534975: I tensorflow/cc/saved_model/reader.cc:45] Reading SavedModel from: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 12:32:20.539871: I tensorflow/cc/saved_model/reader.cc:91] Reading meta graph with tags { serve }
    triton-server  | 2025-11-01 12:32:20.539907: I tensorflow/cc/saved_model/reader.cc:132] Reading SavedModel debug info (if present) from: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 12:32:20.545030: I tensorflow/cc/saved_model/loader.cc:233] Restoring SavedModel bundle.
    triton-server  | 2025-11-01 12:32:20.567754: I tensorflow/cc/saved_model/loader.cc:217] Running initialization op on SavedModel bundle at path: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 12:32:20.572660: I tensorflow/cc/saved_model/loader.cc:334] SavedModel load for tags { serve }; Status: success: OK. Took 37719 microseconds.
    triton-server  | I1101 12:32:20.572919 1 tensorflow.cc:2732] TRITONBACKEND_ModelInstanceInitialize: example_model_1 (CPU device 0)
    triton-server  | 2025-11-01 12:32:20.573802: I tensorflow/cc/saved_model/reader.cc:45] Reading SavedModel from: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 12:32:20.575220: I tensorflow/cc/saved_model/reader.cc:91] Reading meta graph with tags { serve }
    triton-server  | 2025-11-01 12:32:20.575242: I tensorflow/cc/saved_model/reader.cc:132] Reading SavedModel debug info (if present) from: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 12:32:20.576913: I tensorflow/cc/saved_model/loader.cc:233] Restoring SavedModel bundle.
    triton-server  | 2025-11-01 12:32:20.594931: I tensorflow/cc/saved_model/loader.cc:217] Running initialization op on SavedModel bundle at path: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 12:32:20.600803: I tensorflow/cc/saved_model/loader.cc:334] SavedModel load for tags { serve }; Status: success: OK. Took 27004 microseconds.
    triton-server  | I1101 12:32:20.601216 1 model_lifecycle.cc:818] successfully loaded 'example_model'
    triton-server  | W1101 12:32:20.725351 1 onnxruntime.cc:813] autofilled max_batch_size to 4 for model 'all-MiniLM-L6-v2' since batching is supporrted but no max_batch_size is specified in model configuration. Must specify max_batch_size to utilize autofill with a larger max batch size
    triton-server  | I1101 12:32:20.728707 1 onnxruntime.cc:2784] TRITONBACKEND_ModelInstanceInitialize: all-MiniLM-L6-v2_0 (CPU device 0)
    triton-server  | I1101 12:32:20.728716 1 onnxruntime.cc:2784] TRITONBACKEND_ModelInstanceInitialize: all-MiniLM-L6-v2_1 (CPU device 0)
    triton-server  | I1101 12:32:20.983049 1 model_lifecycle.cc:818] successfully loaded 'all-MiniLM-L6-v2'
    triton-server  | I1101 12:32:20.983149 1 server.cc:592] 
    triton-server  | +------------------+------+
    triton-server  | | Repository Agent | Path |
    triton-server  | +------------------+------+
    triton-server  | +------------------+------+
    triton-server  | 
    triton-server  | I1101 12:32:20.983175 1 server.cc:619] 
    triton-server  | +-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
    triton-server  | | Backend     | Path                                                            | Config                                                                                                                                                        |
    triton-server  | +-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
    triton-server  | | tensorflow  | /opt/tritonserver/backends/tensorflow/libtriton_tensorflow.so   | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}} |
    triton-server  | | onnxruntime | /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime.so | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}} |
    triton-server  | +-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
    triton-server  | 
    triton-server  | I1101 12:32:20.983196 1 server.cc:662] 
    triton-server  | +------------------+---------+--------+
    triton-server  | | Model            | Version | Status |
    triton-server  | +------------------+---------+--------+
    triton-server  | | all-MiniLM-L6-v2 | 1       | READY  |
    triton-server  | | example_model    | 1       | READY  |
    triton-server  | +------------------+---------+--------+
    triton-server  | 
    triton-server  | I1101 12:32:20.983301 1 metrics.cc:710] Collecting CPU metrics
    triton-server  | I1101 12:32:20.983483 1 tritonserver.cc:2458] 
    triton-server  | +----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    triton-server  | | Option                           | Value                                                                                                                                                                                                           |
    triton-server  | +----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    triton-server  | | server_id                        | triton                                                                                                                                                                                                          |
    triton-server  | | server_version                   | 2.39.0                                                                                                                                                                                                          |
    triton-server  | | server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_shared_memory cuda_shared_memory binary_tensor_data parameters statistics trace logging |
    triton-server  | | model_repository_path[0]         | /models                                                                                                                                                                                                         |
    triton-server  | | model_control_mode               | MODE_NONE                                                                                                                                                                                                       |
    triton-server  | | strict_model_config              | 0                                                                                                                                                                                                               |
    triton-server  | | rate_limit                       | OFF                                                                                                                                                                                                             |
    triton-server  | | pinned_memory_pool_byte_size     | 268435456                                                                                                                                                                                                       |
    triton-server  | | min_supported_compute_capability | 6.0                                                                                                                                                                                                             |
    triton-server  | | strict_readiness                 | 1                                                                                                                                                                                                               |
    triton-server  | | exit_timeout                     | 30                                                                                                                                                                                                              |
    triton-server  | | cache_enabled                    | 0                                                                                                                                                                                                               |
    triton-server  | +----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    triton-server  | 
    triton-server  | I1101 12:32:20.985516 1 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
    triton-server  | I1101 12:32:20.985655 1 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
    triton-server  | I1101 12:32:21.034328 1 http_server.cc:270] Started Metrics Service at 0.0.0.0:8002
    ml-flask-app   |  * Serving Flask app 'app'
    ml-flask-app   |  * Debug mode: on
    ml-flask-app   | WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
    ml-flask-app   |  * Running on all addresses (0.0.0.0)
    ml-flask-app   |  * Running on http://127.0.0.1:5000
    ml-flask-app   |  * Running on http://172.19.0.3:5000
    ml-flask-app   | Press CTRL+C to quit
    ml-flask-app   |  * Restarting with stat
    ml-flask-app   |  * Debugger is active!
    ml-flask-app   |  * Debugger PIN: 121-885-266
    Gracefully Stopping... press Ctrl+C again to force
    Container ml-flask-app  Stopping
    Container ml-flask-app  Stopped
    Container triton-server  Stopping
    ml-flask-app exited with code 0
    triton-server  | Signal (15) received.
    triton-server  | I1101 12:32:27.138738 1 server.cc:293] Waiting for in-flight requests to complete.
    triton-server  | I1101 12:32:27.138762 1 server.cc:309] Timeout 30: Found 0 model versions that have in-flight inferences
    triton-server  | I1101 12:32:27.138840 1 server.cc:324] All models are stopped, unloading models
    triton-server  | I1101 12:32:27.138846 1 server.cc:331] Timeout 30: Found 2 live models and 0 in-flight non-inference requests
    triton-server  | I1101 12:32:27.139174 1 onnxruntime.cc:2836] TRITONBACKEND_ModelInstanceFinalize: delete instance state
    triton-server  | I1101 12:32:27.139188 1 tensorflow.cc:2770] TRITONBACKEND_ModelInstanceFinalize: delete instance state
    triton-server  | I1101 12:32:27.140807 1 tensorflow.cc:2770] TRITONBACKEND_ModelInstanceFinalize: delete instance state
    triton-server  | I1101 12:32:27.140825 1 tensorflow.cc:2709] TRITONBACKEND_ModelFinalize: delete model state
    triton-server  | I1101 12:32:27.141260 1 model_lifecycle.cc:603] successfully unloaded 'example_model' version 1
    triton-server  | I1101 12:32:27.141474 1 onnxruntime.cc:2836] TRITONBACKEND_ModelInstanceFinalize: delete instance state
    triton-server  | I1101 12:32:27.145090 1 onnxruntime.cc:2760] TRITONBACKEND_ModelFinalize: delete model state
    triton-server  | I1101 12:32:27.145167 1 model_lifecycle.cc:603] successfully unloaded 'all-MiniLM-L6-v2' version 1
    triton-server  | I1101 12:32:28.141059 1 server.cc:331] Timeout 29: Found 0 live models and 0 in-flight non-inference requests
    Container triton-server  Stopped
    triton-server exited with code 137

    ```

    #### Seeing Docker Compose Logs

    By executing `docker compose logs -f` we can see real time logs of the services:

    ```bash
    docker compose logs -f
    WARN[0000] /Users/rubencastillosainz/Documents/Benciowski/Personal/HOGent/MLOps/LABS/mlops-2526-RubenCastilloHOGent/resources/01-dockerlab/docker-compose.yml: the attribute `version` is obsolete, i
    t will be ignored, please remove it to avoid potential confusion                                                                                                                                     ml-flask-app  |  * Serving Flask app 'app'
    ml-flask-app  |  * Debug mode: on
    ml-flask-app  | WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
    ml-flask-app  |  * Running on all addresses (0.0.0.0)
    ml-flask-app  |  * Running on http://127.0.0.1:5000
    ml-flask-app  |  * Running on http://172.19.0.3:5000
    ml-flask-app  | Press CTRL+C to quit
    ml-flask-app  |  * Restarting with stat
    ml-flask-app  |  * Debugger is active!
    ml-flask-app  |  * Debugger PIN: 116-238-659
    triton-server  | 
    triton-server  | =============================
    triton-server  | == Triton Inference Server ==
    triton-server  | =============================
    triton-server  | 
    triton-server  | NVIDIA Release 23.10 (build 72127510)
    triton-server  | Triton Server Version 2.39.0
    triton-server  | 
    triton-server  | Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
    triton-server  | 
    triton-server  | Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
    triton-server  | 
    triton-server  | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
    triton-server  | By pulling and using the container, you accept the terms and conditions of this license:
    triton-server  | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
    triton-server  | 
    triton-server  | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
    triton-server  |    Use the NVIDIA Container Toolkit to start this container with GPU support; see
    triton-server  |    https://docs.nvidia.com/datacenter/cloud-native/ .
    triton-server  | 
    triton-server  | W1101 10:17:39.116276 1 pinned_memory_manager.cc:237] Unable to allocate pinned system memory, pinned memory pool will not be available: CUDA driver version is insufficient for CUD
    A runtime version                                                                                                                                                                                    triton-server  | I1101 10:17:39.116373 1 cuda_memory_manager.cc:117] CUDA memory pool disabled
    triton-server  | I1101 10:17:39.133367 1 model_lifecycle.cc:461] loading: example_model:1
    triton-server  | I1101 10:17:39.134006 1 model_lifecycle.cc:461] loading: all-MiniLM-L6-v2:1
    triton-server  | I1101 10:17:39.500712 1 tensorflow.cc:2577] TRITONBACKEND_Initialize: tensorflow
    triton-server  | I1101 10:17:39.500772 1 tensorflow.cc:2587] Triton TRITONBACKEND API version: 1.16
    triton-server  | I1101 10:17:39.500775 1 tensorflow.cc:2593] 'tensorflow' TRITONBACKEND API version: 1.16
    triton-server  | I1101 10:17:39.500777 1 tensorflow.cc:2617] backend configuration:
    triton-server  | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}}
    triton-server  | I1101 10:17:39.500841 1 tensorflow.cc:2683] TRITONBACKEND_ModelInitialize: example_model (version 1)
    triton-server  | 2025-11-01 10:17:39.502002: I tensorflow/cc/saved_model/reader.cc:45] Reading SavedModel from: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 10:17:39.503351: I tensorflow/cc/saved_model/reader.cc:91] Reading meta graph with tags { serve }
    triton-server  | 2025-11-01 10:17:39.503380: I tensorflow/cc/saved_model/reader.cc:132] Reading SavedModel debug info (if present) from: /models/example_model/1/model.savedmodel
    triton-server  | I1101 10:17:39.503454 1 onnxruntime.cc:2608] TRITONBACKEND_Initialize: onnxruntime
    triton-server  | I1101 10:17:39.503467 1 onnxruntime.cc:2618] Triton TRITONBACKEND API version: 1.16
    triton-server  | I1101 10:17:39.503474 1 onnxruntime.cc:2624] 'onnxruntime' TRITONBACKEND API version: 1.16
    triton-server  | I1101 10:17:39.503475 1 onnxruntime.cc:2654] backend configuration:
    triton-server  | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}}
    triton-server  | I1101 10:17:39.514372 1 onnxruntime.cc:2719] TRITONBACKEND_ModelInitialize: all-MiniLM-L6-v2 (version 1)
    triton-server  | 2025-11-01 10:17:39.520589: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:375] MLIR V1 optimization pass is not enabled
    triton-server  | 2025-11-01 10:17:39.522109: I tensorflow/cc/saved_model/loader.cc:233] Restoring SavedModel bundle.
    triton-server  | 2025-11-01 10:17:39.547480: E tensorflow/core/framework/node_def_util.cc:676] NodeDef mentions attribute debug_name which is not in the op definition: Op<name=VarHandleOp; signatur
    e= -> resource:resource; attr=container:string,default=""; attr=shared_name:string,default=""; attr=dtype:type; attr=shape:shape; attr=allowed_devices:list(string),default=[]; is_stateful=true> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node Variable}}                                                         triton-server  | 2025-11-01 10:17:39.550344: I tensorflow/cc/saved_model/loader.cc:217] Running initialization op on SavedModel bundle at path: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 10:17:39.555229: I tensorflow/cc/saved_model/loader.cc:334] SavedModel load for tags { serve }; Status: success: OK. Took 53266 microseconds.
    triton-server  | I1101 10:17:39.559665 1 tensorflow.cc:2732] TRITONBACKEND_ModelInstanceInitialize: example_model_0 (CPU device 0)
    triton-server  | 2025-11-01 10:17:39.560545: I tensorflow/cc/saved_model/reader.cc:45] Reading SavedModel from: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 10:17:39.561701: I tensorflow/cc/saved_model/reader.cc:91] Reading meta graph with tags { serve }
    triton-server  | 2025-11-01 10:17:39.561954: I tensorflow/cc/saved_model/reader.cc:132] Reading SavedModel debug info (if present) from: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 10:17:39.562948: I tensorflow/cc/saved_model/loader.cc:233] Restoring SavedModel bundle.
    triton-server  | 2025-11-01 10:17:39.576424: I tensorflow/cc/saved_model/loader.cc:217] Running initialization op on SavedModel bundle at path: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 10:17:39.580988: I tensorflow/cc/saved_model/loader.cc:334] SavedModel load for tags { serve }; Status: success: OK. Took 20447 microseconds.
    triton-server  | I1101 10:17:39.581297 1 tensorflow.cc:2732] TRITONBACKEND_ModelInstanceInitialize: example_model_1 (CPU device 0)
    triton-server  | 2025-11-01 10:17:39.581887: I tensorflow/cc/saved_model/reader.cc:45] Reading SavedModel from: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 10:17:39.583336: I tensorflow/cc/saved_model/reader.cc:91] Reading meta graph with tags { serve }
    triton-server  | 2025-11-01 10:17:39.583351: I tensorflow/cc/saved_model/reader.cc:132] Reading SavedModel debug info (if present) from: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 10:17:39.584327: I tensorflow/cc/saved_model/loader.cc:233] Restoring SavedModel bundle.
    triton-server  | 2025-11-01 10:17:39.598012: I tensorflow/cc/saved_model/loader.cc:217] Running initialization op on SavedModel bundle at path: /models/example_model/1/model.savedmodel
    triton-server  | 2025-11-01 10:17:39.602313: I tensorflow/cc/saved_model/loader.cc:334] SavedModel load for tags { serve }; Status: success: OK. Took 20427 microseconds.
    triton-server  | I1101 10:17:39.602734 1 model_lifecycle.cc:818] successfully loaded 'example_model'
    triton-server  | W1101 10:17:39.708960 1 onnxruntime.cc:813] autofilled max_batch_size to 4 for model 'all-MiniLM-L6-v2' since batching is supporrted but no max_batch_size is specified in model con
    figuration. Must specify max_batch_size to utilize autofill with a larger max batch size                                                                                                             triton-server  | I1101 10:17:39.712125 1 onnxruntime.cc:2784] TRITONBACKEND_ModelInstanceInitialize: all-MiniLM-L6-v2_0 (CPU device 0)
    triton-server  | I1101 10:17:39.712126 1 onnxruntime.cc:2784] TRITONBACKEND_ModelInstanceInitialize: all-MiniLM-L6-v2_1 (CPU device 0)
    triton-server  | I1101 10:17:39.991847 1 model_lifecycle.cc:818] successfully loaded 'all-MiniLM-L6-v2'
    triton-server  | I1101 10:17:39.991918 1 server.cc:592] 
    triton-server  | +------------------+------+
    triton-server  | | Repository Agent | Path |
    triton-server  | +------------------+------+
    triton-server  | +------------------+------+
    triton-server  | 
    triton-server  | I1101 10:17:39.991945 1 server.cc:619] 
    triton-server  | +-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------
    ------------------------------------------------------------+                                                                                                                                        triton-server  | | Backend     | Path                                                            | Config                                                                                            
                                                                |                                                                                                                                        triton-server  | +-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------
    ------------------------------------------------------------+                                                                                                                                        triton-server  | | tensorflow  | /opt/tritonserver/backends/tensorflow/libtriton_tensorflow.so   | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-co
    mpute-capability":"6.000000","default-max-batch-size":"4"}} |                                                                                                                                        triton-server  | | onnxruntime | /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime.so | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-co
    mpute-capability":"6.000000","default-max-batch-size":"4"}} |                                                                                                                                        triton-server  | +-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------
    ------------------------------------------------------------+                                                                                                                                        triton-server  | 
    triton-server  | I1101 10:17:39.991967 1 server.cc:662] 
    triton-server  | +------------------+---------+--------+
    triton-server  | | Model            | Version | Status |
    triton-server  | +------------------+---------+--------+
    triton-server  | | all-MiniLM-L6-v2 | 1       | READY  |
    triton-server  | | example_model    | 1       | READY  |
    triton-server  | +------------------+---------+--------+
    triton-server  | 
    triton-server  | I1101 10:17:39.992100 1 metrics.cc:710] Collecting CPU metrics
    triton-server  | I1101 10:17:39.992229 1 tritonserver.cc:2458] 
    triton-server  | +----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------
    -----------------------------------------------------------------+                                                                                                                                   triton-server  | | Option                           | Value                                                                                                                                          
                                                                    |                                                                                                                                   triton-server  | +----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------
    -----------------------------------------------------------------+                                                                                                                                   triton-server  | | server_id                        | triton                                                                                                                                         
                                                                    |                                                                                                                                   triton-server  | | server_version                   | 2.39.0                                                                                                                                         
                                                                    |                                                                                                                                   triton-server  | | server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_shared_memory cuda_shar
    ed_memory binary_tensor_data parameters statistics trace logging |                                                                                                                                   triton-server  | | model_repository_path[0]         | /models                                                                                                                                        
                                                                    |                                                                                                                                   triton-server  | | model_control_mode               | MODE_NONE                                                                                                                                      
                                                                    |                                                                                                                                   triton-server  | | strict_model_config              | 0                                                                                                                                              
                                                                    |                                                                                                                                   triton-server  | | rate_limit                       | OFF                                                                                                                                            
                                                                    |                                                                                                                                   triton-server  | | pinned_memory_pool_byte_size     | 268435456                                                                                                                                      
                                                                    |                                                                                                                                   triton-server  | | min_supported_compute_capability | 6.0                                                                                                                                            
                                                                    |                                                                                                                                   triton-server  | | strict_readiness                 | 1                                                                                                                                              
                                                                    |                                                                                                                                   triton-server  | | exit_timeout                     | 30                                                                                                                                             
                                                                    |                                                                                                                                   triton-server  | | cache_enabled                    | 0                                                                                                                                              
                                                                    |                                                                                                                                   triton-server  | +----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------
    -----------------------------------------------------------------+                                                                                                                                   triton-server  | 
    triton-server  | I1101 10:17:39.993724 1 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
    triton-server  | I1101 10:17:39.993841 1 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
    triton-server  | I1101 10:17:40.044569 1 http_server.cc:270] Started Metrics Service at 0.0.0.0:8002
    ^C     
    ```

    `Checking running services`

    ```bash
    docker compose ps
    WARN[0000] /Users/rubencastillosainz/Documents/Benciowski/Personal/HOGent/MLOps/LABS/mlops-2526-RubenCastilloHOGent/resources/01-dockerlab/docker-compose.yml: the attribute `version` is obsolete, i
    t will be ignored, please remove it to avoid potential confusion                                                                                                                                     NAME            IMAGE                                   COMMAND                  SERVICE         CREATED         STATUS         PORTS
    ml-flask-app    01-dockerlab-ml-flask-app               "python app.py"          ml-flask-app    2 minutes ago   Up 2 minutes   0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp
    triton-server   nvcr.io/nvidia/tritonserver:23.10-py3   "/opt/nvidia/nvidia_…"   triton-server   2 minutes ago   Up 2 minutes   0.0.0.0:8000-8002->8000-8002/tcp, [::]:8000-8002->8000-8002/tcp
    ```

    !["Active containers in Docker Desktop."](./img/01-docker-lab/AcceptanceCriteria_3_2.png)

    #### Stopping and removing the services using Docker Compose

    By executing `docker compose stop` we can stop the services:

    ```bash
    docker compose stop
    WARN[0000] /Users/rubencastillosainz/Documents/Benciowski/Personal/HOGent/MLOps/LABS/mlops-2526-RubenCastilloHOGent/resources/01-dockerlab/docker-compose.yml: the attribute `version` is obsolete, i
    t will be ignored, please remove it to avoid potential confusion                                                                                                                                     [+] Stopping 2/2
    ✔ Container ml-flask-app   Stopped                                                                                                                                                             0.6s 
    ✔ Container triton-server  Stopped                                                                                                                                                             1.3s 
    ```

    By executing `docker compose down` we can remove the services:

    ```bash
    docker compose down
    WARN[0000] /Users/rubencastillosainz/Documents/Benciowski/Personal/HOGent/MLOps/LABS/mlops-2526-RubenCastilloHOGent/resources/01-dockerlab/docker-compose.yml: the attribute `version` is obsolete, i
    t will be ignored, please remove it to avoid potential confusion                                                                                                                                     [+] Running 3/3
    ✔ Container ml-flask-app        Removed                                                                                                                                                        0.8s 
    ✔ Container triton-server       Removed                                                                                                                                                        1.3s 
    ✔ Network 01-dockerlab_default  Removed                                                                                                                                                        0.1s 

    ```

#### Questions:

❓ How can you view the logs of the services?
    The main command is  `docker logs` or `docker-compose logs`. This will show you logs for all services.
    
    If you would like to watch logs for a specific service, then you can execute the same command by passing the service name through: `docker-compose logs <serviceName>`

    Useful options when running these commands are:
        - `-f`: To see real time logs
        - `-t`: Shows timestamps for each log.
        - `--tail XX`: Shows only the last XX lines where XX is the amount of lines you want to see.

❓ What does the -d flag do in docker compose up -d? When would you use it vs. not using it?
     The `-d` stands for **detached mode**. Basically, it starts all the services in the background and it will keep your terminal free for other work rather keeping your terminal busy with the services you just started.

    You use it when multitasking or when you execute it in Production or Staging environments when you don't need to see the logs constantly in real time.
    Also, a tipical case is in CI/CD pipelines so automated scripts start the services in the background, run tests and stops them.

    You don't use if your main purpose is debugging or troubleshooting issues because it allows you seeing logs in real time.

❓ How would you stop the services? What command would you use to stop and remove all containers, networks, and volumes defined in the docker-compose.yml file?
    - To stop running services I would use `docker-compose stop`
    - To stop and remove everything and completely clean up containers and networks I would use `docker-compose down`
    - If I also want to remove volumnes, then I'd use `docker-compose down -v`

## Issues

### Part 1: Docker Basics

"none".
It was the first time running Docker on a MacOS machine and despite that, I must say that the configuration was much easier than the time I installed it on my old Windows machine.

### Part 2: Triton Serving 

At first, I had an issue with running this command with the parameter `--gpus all`. Since my Macbook does not have an Nvidia GPU, I needed to check out what the error message was to find out I just needed to remove it from the command line.

```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v ${PWD}/model_repository:/models `
  nvcr.io/nvidia/tritonserver:23.10-py3 `
  tritonserver --model-repository=/models
```

Once the Triton server was up and running, I wasn't sure what the content of the`config.pbtxt` file should be until I found out that I could see the metadata loaded by Triton by just hitting the PORT 8000 with the model name.

Once the `config.pbtxt` file was created, I had issues trying to get a succesfull response back from the Public model.
This was due to me sending a wrong request in the POST body payload that did not match the input schema in the `config.pbtxt` file. At this point, I needed to check some documentation and read a bit more about what was actually doing to fully understand the error. Once I was on the same page, it was clear to me how to solve it and I ended up getting succesfull responsed from the public model I chose.

### Part 3: Docker Compose

"none". 
The containers were orchestrated without any issues and it all ran fine at the first attempt.

## Reflection

Parts 1 and 3 of the lab were very clear to me since I was more familiar with Docker and Docker Compose. Despite never using them in my professional career yet I already knew what the purpose of them are. This helped to understand what the goal of each steps were and what was I doing when following the steps in the lab. Also, I had already read all the theory material and watched and done the exericeses in the Youtube videos provided in the course so it was a bit redoing what was shown in the theory part.

Part 2 was more challenging because I never heard of NVIDIA Triton Inference server before.
When following the steps in the lab, at first I was getting everything up and running with no issues until I hit some issues trying to run the public model I chose. 
At that moment, I needed some time to take some steps back and read more about Triton server to understand better what the goal was. Once I did that, I was able to understand the root cause of my issues and fix them.

Nevertheless, at the moment I was writting this report (few days after I did the lab) and I answered all the questions in the lab and acceptance criteria parts, made me realize that I still needed to read a bit deeper about the Triton Inference and its purpose to fully understand the questions and make sure I was giving the right answers.

In general words, I have like thid this lab as it helped me understand Docker basics and how to run applications in containers, orchestrate them together and also 'pretend' a production case scenario where ML models are run in a server rather than on my local machine.

## Resources

`Docker Basics:`
https://www.docker.com/

https://docs.docker.com/get-docker/

https://www.docker.com/products/docker-desktop/

https://hub.docker.com

https://docs.docker.com/develop/develop-images/dockerfile_best-practices/


`TRITON Inference Server:`
https://docs.nvidia.com/deeplearning/triton-inference-server

https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md

https://www.supermicro.com/en/glossary/triton-inference-server

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main/onnx

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html

`Docker Compose:`
https://docs.docker.com/compose/

https://docs.docker.com/compose/compose-file/

https://www.geeksforgeeks.org/devops/docker-compose/

`Course Theory`
The Future of Linux Containers: https://youtu.be/wW9CAH9nSLs

At PyCon 2013, Solomon Hykes shows docker to the public for the first time.

Docker in 100 Seconds: https://www.youtube.com/watch?v=Gjnup-PuquQ

The intro to Docker I wish I had when I started: https://www.youtube.com/watch?v=Ud7Npgi6x8E

Docker Tutorial for Beginners: https://www.youtube.com/watch?v=b0HMimUb4f0

Docker Crash Course playlist: https://www.youtube.com/watch?v=31ieHmcTUOk&list=PL4cUxeGkcC9hxjeEtdHFNYMtCpjNBm3h7