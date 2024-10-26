# GPN Project

This project is a Graph-based Point Cloud Network (GPN) for 3D object classification and segmentation. It includes various graph-based neural network models and utilities for processing point cloud data.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To use the GPN project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/zxk2022/gpn.git
    cd gpn
    ```

2. Prepare your dataset and configure the paths in the configuration files.

3. Train the model:
    ```bash
    python main.py --cfgs <path_to_config_file>
    ```

4. Evaluate the model:
    ```bash
    python main.py --cfgs <path_to_config_file> --mode val
    ```

## Contributing

We welcome contributions to the GPN project. To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.

Please ensure that your code follows the project's coding standards and includes appropriate tests.
