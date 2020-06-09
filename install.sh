CUDA=${1:-"cu100"}
pip3 install torch==1.4.0+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip3 install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip3 install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip3 install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip3 install torch-geometric==1.4.3
pip3 install -r requirements.txt
echo "install.sh finished!"
