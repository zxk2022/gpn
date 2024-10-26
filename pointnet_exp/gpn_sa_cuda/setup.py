from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointnet2_ggcn',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            'pointnet2_ggcn',
            [
                'src/pointnet2_ggcn_api.cpp',
                'src/gabriel_graph.cpp',
                'src/gabriel_graph_gpu.cu',
                'src/message_passing_sum.cpp',
                'src/message_passing_sum_gpu.cu',
                'src/aggregate_sum.cpp',
                'src/aggregate_sum_gpu.cu',
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
