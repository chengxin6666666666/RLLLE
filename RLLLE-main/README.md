## Project Setup
conda create --name py36 python=3.6
source activate py36
Clone the repo and install the complementary requirements:
pip install -r requirements.txt

### Train
python -m torch.distributed.launch --nproc_per_node 1 --master_port 4320 train3.py -opt options/train/LOLv1.yml --launcher pytorch

### Test
python test.py -opt options/test/LOLv1.yml



