## German to English language translation model using Transformer
Trained on MULTI30K dataset.
BLEU score = 30
Peformance can be improved by training for larger number of Epochs.
Created an API for the model using FAST API.
#### Pytorch version used = 1.7.0
Download the checkpoint from here: *https://drive.google.com/file/d/1pQRyS6yIaKXeTqWn9NNebjeAJU2R931W/view?usp=sharing*
 
## Procedure to run:
1. Download the checkpoint from the link given.
2. Open the api.py file and change the location in load_model function to the location of checkpoint.
3. Run api.py.
4. Go to http://127.0.0.1:8080 .
5. If not working go to http://localhost:8080/ . For Swagger UI add docs after "/". http://localhost:8080/docs

