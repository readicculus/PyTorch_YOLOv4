from setuptools import setup

with open('requirements.txt') as f:
   requirements = f.read().splitlines()
setup(
   name='PyTorchYOLOv4',
   version='0.0.1',
   packages=['PyTorchYOLOv4', 'PyTorchYOLOv4.utils'],
   scripts=[],
   url='git@github.com:readicculus/PyTorch_YOLOv4.git',
   description='Torch YoloV4',
   long_description=open('README.md').read(),
   install_requires=requirements
)