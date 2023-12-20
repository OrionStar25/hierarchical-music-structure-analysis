## Installation

```bash
$ conda activate llm
$ pip install transformers torch accelerate
$ huggingface-cli login
```

By running `huggingface-cli login`, you will be asked to input an access token. Generate yours here: https://huggingface.co/settings/tokens .

## Prompt

Write prompt in `prompt.txt`. This prompt gets read by the LLM.

## Inference

Run:
```bash
$ python inference.py
```

## Demo

```
Result: Below are lyrics of a song delimited by triple backticks. Identify the singer and song name.

Lyrics: '''
I said, "No one has to know what we do"
His hands are in my hair, his clothes are in my room
And his voice is a familiar sound
Nothing lasts forever
But this is getting good now

He's so tall and handsome as hell
He's so bad, but he does it so well
And when we've had our very last kiss
My last request is

Say you'll remember me
Standing in a nice dress
Staring at the sunset, babe
Red lips and rosy cheeks
Say you'll see me again
Even if it's just in your wildest dreams, ah-ah, ha (ha-ah, ha)
Wildest dreams, ah-ah, ha
'''

Answer:  The singer is Taylor Swift and the song is "Wildest Dreams".
```
