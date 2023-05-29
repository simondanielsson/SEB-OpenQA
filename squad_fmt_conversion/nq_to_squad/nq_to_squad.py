#!/usr/bin/python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Convert the Natural Questions dataset into SQuAD JSON format.

To use this utility, first follow the directions at the URL below to download
the complete training dataset.

    https://ai.google.com/research/NaturalQuestions/download

Next, run this program, specifying the data you wish to convert. For instance,
the invocation:

    python nq_to_squad.py\
        --data_pattern=/usr/local/data/tnq/v1.0/train/*.gz\
        --output_file=/usr/local/data/tnq/v1.0/train.json

will process all training data and write the results into `train.json`. This
file can, in turn, be provided to squad_eval.py using the --squad argument.
"""

import argparse
import glob
import gzip
import json
import logging
import os
import re


def clean_text(start_token, end_token, doc_tokens, doc_bytes,
               ignore_final_whitespace=True):
  """Remove HTML tags from a text span and reconstruct proper spacing."""
  text = ""
  for index in range(start_token, end_token):
    token = doc_tokens[index]
    if token["html_token"]:
      continue
    text += token["token"]
    # Add a single space between two tokens iff there is at least one
    # whitespace character between them (outside of an HTML tag). For example:
    #
    #   token1 token2                           ==> Add space.
    #   token1</B> <B>token2                    ==> Add space.
    #   token1</A>token2                        ==> No space.
    #   token1<A href="..." title="...">token2  ==> No space.
    #   token1<SUP>2</SUP>token2                ==> No space.
    next_token = token
    last_index = end_token if ignore_final_whitespace else end_token + 1
    for next_token in doc_tokens[index + 1:last_index]:
      if not next_token["html_token"]:
        break
    chars = (doc_bytes[token["end_byte"]:next_token["start_byte"]]
             .decode("utf-8"))
    # Since some HTML tags are missing from the token list, we count '<' and
    # '>' to detect if we're inside a tag.
    unclosed_brackets = 0
    for char in chars:
      if char == "<":
        unclosed_brackets += 1
      elif char == ">":
        unclosed_brackets -= 1
      elif unclosed_brackets == 0 and re.match(r"\s", char):
        # Add a single space after this token.
        text += " "
        break
  return text


def nq_to_squad(record):
  """Convert a Natural Questions record to SQuAD format."""

  doc_bytes = record["document_html"].encode("utf-8")
  doc_tokens = record["document_tokens"]

  # NQ training data has one annotation per JSON record.
  annotation = record["annotations"][0]

  short_answers = annotation["short_answers"]
  # Skip examples that don't have exactly one short answer.
  # Note: Consider including multi-span short answers.
  if not short_answers:
    short_answers = [{'end_byte': -1, 'end_token': -1, 'start_byte': -1, 'start_token': -1}]
  # instead extract all answers
  #else:
  #  short_answer = short_answers[0]

  long_answer = annotation["long_answer"]
  # Skip examples where annotator found no long answer.
  #if long_answer["start_token"] == -1:
    #return
  # Skip examples corresponding to HTML blocks other than <P>.
  long_answer_html_tag = doc_tokens[long_answer["start_token"]]["token"]
  #if long_answer_html_tag != "<P>":
    #return

  paragraph = clean_text(
      long_answer["start_token"], long_answer["end_token"], doc_tokens,
      doc_bytes)
  answers = []
  # extract all legal answers
  for short_answer in short_answers:
    answer = clean_text(
        short_answer["start_token"], short_answer["end_token"], doc_tokens,
        doc_bytes
    )
    before_answer = clean_text(
        long_answer["start_token"], short_answer["start_token"], doc_tokens,
        doc_bytes, ignore_final_whitespace=False
    )
    answer_dict = {'answer_start': len(before_answer), 'text': answer}
    answers.append(answer_dict)
      

  return {"title": record["document_title"],
          "paragraphs":
              [{"context": paragraph,
                #"qas": [{"answers": [{"answer_start": len(before_answer),
                #                      "text": answer}],
                "qas": [{"answers": answers,
                         "id": record["example_id"],
                         "question": record["question_text"],
                         "is_impossible": not any(answers)}]}]}


def main():
  parser = argparse.ArgumentParser(
      description="Convert the Natural Questions to SQuAD JSON format.")
  parser.add_argument("--data_pattern", dest="data_pattern",
                      help=("A file pattern to match the Natural Questions "
                            "dataset."),
                      metavar="PATTERN", required=True)
  parser.add_argument("--version", dest="version",
                      help="The version label in the output file.",
                      metavar="LABEL", default="nq-train")
  parser.add_argument("--output_file", dest="output_file",
                      help="The name of the SQuAD JSON formatted output file.",
                      metavar="FILE", default="nq_as_squad.json")
  parser.add_argument("--as_jsonl", action='store_true', default=False,
                      help=("True if result should be saved as jsonl"))
  args = parser.parse_args()

  root = logging.getLogger()
  root.setLevel(logging.DEBUG)

  records = 0
  nq_as_squad = {"version": args.version, "data": []}

  for file in sorted(glob.iglob(args.data_pattern)):
    logging.info("opening %s", file)
    file_opener = None
    if str(file).endswith('gz'):
        file_opener = gzip.GzipFile
    elif str(file).endswith('.json') or str(file).endswith('.jsonl'):
        file_opener = open
    else:
        raise ValueError(f'Cannot open file {file}')
    with file_opener(file, "r") as f:
      for line in f:
        nq_record = json.loads(line)
        try:
            squad_record = nq_to_squad(nq_record)
        except UnicodeError as e:
            logging.info(f'Unicode Error after sample {records}:\n\t{e.reason}')
            continue
        records += 1
        if squad_record:
          nq_as_squad["data"].append(squad_record)
        if records % 1000 == 0:
          logging.info("processed %s records", records)
          logging.info("processed %s records", len(nq_as_squad["data"]))
  print("Converted %s NQ records into %s SQuAD records." %
        (records, len(nq_as_squad["data"])))
  with open(args.output_file, "w") as f:
    if args.as_jsonl and args.output_file.endswith('.jsonl'):
        logging.info('Saving as jsonl')
        nq_as_squad = nq_as_squad['data']
    json.dump(nq_as_squad, f)


if __name__ == "__main__":
  main()
