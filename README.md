# Sensing-Assisted Long-Term Beam Tracking
Simulator for paper ''Knowledge Distillation for Sensing-Assisted Long-Term Beam Tracking in mmWave Communications''

## Dataset Preparation

1. Download the project and extract it to your local machine.

2. Download **Scenario 9** from:  
   https://www.deepsense6g.net/scenarios/Scenarios%201-9/scenario-9

3. Extract the dataset to form the file structure:

```text
dataset/
└── scenario9/
    ├── unit1/
    └── scenario9.csv
 ```
5. Run the preprocessing scripts CSV_process.py and gen_data_seq.py in order

## Training model:

-- run train_image.py to train model. 

-- key parameters
1) args.kd_mode: kd_mode=0: no KD; kd_mode=1: KD
2) args.seq_length_student: Sequence length for student model (8 or 5 or 3)
3) args.attention=True: Use attention for teacher model
4) create model for training, the model can be student model or teacher model 

## Testing model
All trained model along with the hyparameters are under the folder: All_models/

-- run test_model_image.py to test the model

-- key parameters
1) args.model_arch: Model architecture
2) args.kd_mode:0 for no KD; 1 for KD
3) args.seq_length_student: Sequence length for student model (8 or 5 or 3)
4) args.attention=True: Use attention for teacher model
5) 
### Models and hyperparameters:
Nine models contained: 
1) Teacher_noAtten.pth: Best teacher model without attention mechanism
2) Teacher_withAtten.pth: Best teacher model with attention mechanism
3) Teacher_selfKD.pth: Best teacher model (including attention mechanism) with self-KD refinement
4) StudentL8_noKD.pth: Student model without KD for input sequence length 8
5) StudentL8_KD.pth: Student model with KD for input sequence length 8
6) StudentL5_noKD.pth: Student model without KD for input sequence length 5
7) StudentL5_KD.pth: Student model with KD for input sequence length 5
8) StudentL3_noKD.pth: Student model without KD for input sequence length 3
9) StudentL3_KD.pth: Student model with KD for input sequence length 3
   
The hyperparameters are shown in the txt files.
