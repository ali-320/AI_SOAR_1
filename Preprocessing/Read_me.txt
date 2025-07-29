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