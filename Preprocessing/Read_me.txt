Here I am working on a pre-processing module. I have already
acquired a dataset known as BETH (explained in Beth-data_choosing).

This dataset contains thousand of cowries logs, categorized
on the basis of IPs. Additionally, they have a pre-build 
dataset for ML model training. I would like to train my model
on this dataset as a prototype to judge the behavior of the
AI-model, since I have no knowledge in this field, and am 
studying this field as well.

Before all that, I realized that there are some useless
attributes in the dataset, which are not necessary for training.

I believe that these are the timestamp (why would my model need
to know when the attack happened), the processID (Since, the OS
assigns a new processID to every process when it runs).

The attributes that I am suspicious about are:
-event_id
-mountNameSpace
-StackAddress

To run the complete the preprocessing module, do the following prompts:
1. Get the data-set from the link mentioned in BETH-data_choosing.txt.
This will create an archive folder in your directory.

2. Now run the proprocessor.py file, this will convert the training,
testing, validation datasets into json format. Note that, you may 
have to change the names of the input file.

3. Now run the TF-IDF.py file. This will create the .npy files in Vectorized_tfidf foder, for the training 
of the models. Note that currently TF-IDF is only going to take the 500 features out of each dataset.
You may need to change it in the future.

4. Now run the event_profiler.py file. Check from gpt, that this is going to guess the profiles
from the previous.npy files, then add these profiles in the .json files. The resulting files 
will be stored in the profiled folder.

5. Now run the data_prepare.py file. This will create the dataset for the training of the DL models
in the DL_Data folder.

6. Workflow of everything:
    1. Download the Beth dataset.
    2. Run preprocessor.py on the training, testing, and validation datasets.
    3. Run TF-IDF.py on the outputs of the previous python file.
    4. Now run event_profiler.py on the output of previous files.
    5. Now run data_prepare.py on the output of previous files.
    6. This will generate data on which models can operate.
In order for just testing, only do the above tasks for test data.