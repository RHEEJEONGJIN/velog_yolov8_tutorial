# VOTE
## run code
```bash
"""
Enter Key -> Start Predict
q Key -> Exit Windows
"""
python src/run.py
```
## train code
```bash
# vote datasets에서 train, test 폴더가 없을 때 split_train_test.py 실행 / 지금은 있음.
# python src/split_train_test.py 
yolo detect train cfg=cfg/custom.yaml
```
## requirements
```bash
pip install --upgrade pip
pip install ultralytics
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install shapely
pip install scikit-learn
```