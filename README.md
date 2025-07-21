# AI_SOAR_1
Here we are working on an AI based Security Orchestration, Automation and Response

Tu run this you have to do the followin steps:
# 1. nai_stix
The purpose of this module, is to collect TI reports from websites and send them to ELK.
To observe its working. You first need to install WSL from command prompt by typing "wsl --install", then install the docker desktop from the official website.
Then insall python for running python script. After doing all this download and store the nai_stix folder somewhere in your computer. Also check whether you have all of these installed or from cmd. You should also check curl, if you don't have it ask AI to help you install it.

Now, open cmd/powershell and go to the nai_stix folder using "cd" command. Once you're in, type "docker compose up -d". If an error occurs, its likely that docker desktop is not running. Type docker desktop in windows search and just open it for running.
Once it is running you can see it in the taskbar near the speaker and wifi icons, then you can try "docker compose up -d" (only if it didn't run on the first time).

Now, run the python script to fetch data by typing "python fetch_and_send_stix.py". Now, nai_stix is officially up and running. You can check this by typing "http://localhost:5601/". This will open kibana.
But now we need to create a kind of pointer in kibana that points to the data-set that out python file just fetched. For this go to the menu bar, then stack management (at the end), then data view in the kibana section. Here create new data view by the name of 
stix_object*. This will collect the fetched data which you can see in menu -> discover. 

Now to stop the nai_stix, just open cmd with the same path as before and type "docker compose down". This will shut down the containers.
Also, if you want to run kibana on some other port, just opend the docker-compose.yml file and you can change the port numbers of kibana, elastic search and logstash in the portnumber section. I recommend that you don't do anything to logstash and elastic search and just change the port number of kibana for display.

#Important point about ELK:
If you delete the past data present in elastic search, it will not be shown on kibana as well. To delete the past data from elastic-search. First make sure that it is running and then use curl to dele it. This can be done by typing the following command in cmd:
CURL -X DELETE "http://localhost:9200/stix_objects"
