# How to get triviaQA and convert it to SQuAD format


1. Get the triviaQA dataset


`
curl https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz > data/triviaQA/triviaqa-rc.tar.gz
`

This will yield a tar.gz file containing the wikipedia/web dump and the QA files for wikipedia and web questions.


2. Unzip it

`
tar -xvzf triviaqa-rc.tar.gz
`

This will yield wikipedia and web dir and qa files under qa dir.

3. Convert to SQuAD Format using triviaQA_to_squad scripts :)
``
python3 src/triviaQA_to_squad/convert_to_squad_format.py --triviaqa_file=/home/jupyter/SEB-workspace/data/triviaQA/qa/verified-web-dev.json --squad_file=/home/jupyter/SEB-workspace/data/triviaQA/verified_web_dev_squad_fmt.json --wikipedia_dir=/home/jupyter/SEB-workspace/data/triviaQA/evidence/wikipedia --web_dir=/home/jupyter/SEB-workspace/data/triviaQA/evidence/web
``

QA FILENAME is the name of the qa file we want to convert. 
SQUAD FILENAME is the name of the output, SQuAD formatted file
