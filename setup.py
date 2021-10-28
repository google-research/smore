# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup
from distutils.command.build import build
from setuptools.command.install import install

from setuptools.command.develop import develop
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

import os
import subprocess
BASEPATH = os.path.dirname(os.path.abspath(__file__))

extlib_path = 'smore/common/torchext'
compile_args = ['-I%s/third_party/ThreadPool' % BASEPATH, '-Wno-deprecated-declarations']
link_args = []

ext_modules=[CppExtension('extlib', 
                          ['%s/extlib.cpp' % extlib_path],
                          extra_compile_args=compile_args,
                          extra_link_args=link_args)]

# build cuda lib
import torch
if torch.cuda.is_available():
    ext_modules.append(CUDAExtension('extlib_cuda',
                                    ['%s/%s' % (extlib_path, x) for x in [
                                        'extlib_cuda.cpp', 'ext_cuda_kernel.cu'
                                    ]],
                                    extra_compile_args=compile_args))

class custom_develop(develop):
    def run(self):
        original_cwd = os.getcwd()

        # build custom ops
        folders = [
           os.path.join(BASEPATH, 'smore/cpp_sampler'),
        ]
        for folder in folders:
            os.chdir(folder)
            subprocess.check_call(['make'])

        os.chdir(original_cwd)

        super().run()


setup(name='smore',
      py_modules=['smore'],
      ext_modules=ext_modules,
      install_requires=[
      ],
      cmdclass={
          'build_ext': BuildExtension,
          'develop': custom_develop,
        }
)
