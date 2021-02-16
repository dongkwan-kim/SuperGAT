CUDA=${1:-"cu100"}
pip3 install torch==1.4.0+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip3 install torch-sparse==0.6.1 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip3 install torch-cluster==1.5.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip3 install torch-geometric==1.4.3
pip3 install -r requirements.txt
echo "install.sh finished!"
