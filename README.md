# Containerized Question Answering Service
Simple Flask REST service for English question answering. Leverages the transformer API and a RoBERTa model with 
question answering-head pretrained by deepset on the SQuAD2.0 dataset 
(learn more [here](https://huggingface.co/deepset/roberta-base-squad2)). Runs on CPU only.


## Setup and Start Service
This service can be either set up in a Python 3.8 environment or build as Docker image. 


### Python Environment
Install all necessary requirements in your Python environment via:

```
pip install -r requirements.txt
```

To start the server run:

```
python server.py -p <PORT>
```

With the optional parameter <code>-p</code> the port can be changed from the default port 5000.


### Docker Container
To build the docker image run the following command: 

```
docker build -t qaservice:final .
```

To start a container with the previously built image run the following command or use your favorite framework for 
deployment.

```
docker run -d -p <PORT>:5000 --name qaservice qaservice:final
```

Other than in the [Python Environment](#Python Environment) the parameter <code>-p</code> is mandatory since the 
internal port has to be mapped to a port of the host.


## API Usage
The service exposes a single REST endpoint: <code>POST /inference</code> which expects a JSON object of the following 
format

```
{
    impossible: boolean [optional, default: True]
    top_k: integer [optional, default: 1]
    data: [
        {
            questions: [
                'question1',
                'question2',
                ...
            ]
            context: 'context'
        },
        ...
    ]
}
```

The parameter <code>impossible</code> can be set to <code>False</code> if all questions can be answered with the given 
context and the model should be forced to return a valid answer. The parameter <code>top_k</code> can be set to an 
arbitrary value <code>k >= 1</code> to get the best <code>k</code> answers for each question.

The questions and contexts are provided under the parameter <code>data</code> in a nested form in which all questions 
are grouped by the corresponding context.

Example requests can be found in the folder [requests](./requests) and can be sent with <code>python requests.py -p 
<PORT></code>. The parameter <code>-p</code> has to be set if the port on the host differs from the default port 5000.
Responses will be logged in an automatically created [build](./build) folder.