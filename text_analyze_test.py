# produces json files of analyzed text.


#from server import analyze
#import torch
#from server import get_all_projects

# projects = {}
# print(str(torch.cuda.is_available()) + 'text analyze test')
#req = {
#  "project": "new",
#  "text": "The following is a transcript from The Guardian."
#}


#projects = get_all_projects()
#ret = analyze(req)
#print(ret)

import json
from backend import api
import jsonlines

def remove_symbols_from_text(text):
    # given string, removes symbols
    for char in text:
        if char in "“”":
            text = text.replace(char, '"')
        if char in "‘’":
            text = text.replace(char, '\'')
        if char in "–—–":
            text = text.replace(char, '-')
        if char in "……":
            text = text.replace(char, '...')
        if char in "Θθ":
            text = text.replace(char, '(theta)')
        if char in "Ωω":
            text = text.replace(char, '(omega)')
    text.replace('\u2014', '-')
    text.replace('\u2026', '...')
    return text


'''raw_text = """
    In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

    The scientist named the population, after their distinctive horn, Ovid's Unicorn. These four-horned, silver-white unicorns were previously unknown to science.

    Now, after almost two centuries, the mystery of what sparked this odd phenomenon is finally solved.––

    Dr. Jorge Pérez, an evolutionary biologist from the University of La Paz, and several companions, were exploring the Andes Mountains when they found a small valley, with no other animals or humans. Pérez noticed that the valley had what appeared to be a natural fountain, surrounded by two peaks of rock and silver snow.

    Pérez and the others then ventured further into the valley. "By the time we reached the top of one peak, the water looked blue, with some crystals on top," said Pérez.

    Pérez and his friends were astonished to see the unicorn herd. These creatures could be seen from the air without having to move too much to see them - they were so close they could touch their horns.

    While examining these bizarre creatures the scientists discovered that the creatures also spoke some fairly regular English. Pérez stated, "We can see, for example, that they have a common 'language,' something like a dialect or dialectic."

    Dr. Pérez believes that the unicorns may have originated in Argentina, where the animals were believed to be descendants of a lost race of people who lived there before the arrival of humans in those parts of South America.

    While their origins are still unclear, some believe that perhaps the creatures were created when a human and a unicorn met each other in a time before human civilization. According to Pérez, "In South America, such incidents seem to be quite common."

    However, Pérez also pointed out that it is likely that the only way of knowing for sure if unicorns are indeed the descendants of a lost alien race is through DNA. "But they seem to be able to communicate in English quite well, which I believe is a sign of evolution, or at least a change in social organization," said the scientist.
    """   '''
# raw_text = """
#     In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.
#     """

raw_text= "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.\nThe scientist named the population, after their distinctive horn, Ovid's Unicorn. These four-horned, silver-white unicorns were previously unknown to science.\n\nNow, after almost two centuries, the mystery of what sparked this odd phenomenon is finally solved.\n\nDr. Jorge Pérez, an evolutionary biologist from the University of La Paz, and several companions, were exploring the Andes Mountains when they found a small valley, with no other animals or humans. Pérez noticed that the valley had what appeared to be a natural fountain, surrounded by two peaks of rock and silver snow.\n\nPérez and the others then ventured further into the valley. \"By the time we reached the top of one peak, the water looked blue, with some crystals on top,\" said Pérez.\n\nPérez and his friends were astonished to see the unicorn herd. These creatures could be seen from the air without having to move too much to see them – they were so close they could touch their horns.\n\nWhile examining these bizarre creatures the scientists discovered that the creatures also spoke some fairly regular English. Pérez stated, \"We can see, for example, that they have a common 'language,' something like a dialect or dialectic.\"\n\nDr. Pérez believes that the unicorns may have originated in Argentina, where the animals were believed to be descendants of a lost race of people who lived there before the arrival of humans in those parts of South America.\n\nWhile their origins are still unclear, some believe that perhaps the creatures were created when a human and a unicorn met each other in a time before human civilization. According to Pérez, \"In South America, such incidents seem to be quite common.\"\n\nHowever, Pérez also pointed out that it is likely that the only way of knowing for sure if unicorns are indeed the descendants of a lost alien race is through DNA. \"But they seem to be able to communicate in English quite well, which I believe is a sign of evolution, or at least a change in social organization,\" said the scientist."
# raw_text = "getting \"no\" from"
# print(raw_text)
'''
Tests for BERT
'''
"""lm = api.BERTLM()
start = api.time.time()
payload_bert = lm.check_probabilities(raw_text, topk=5)
end = api.time.time()
print("{:.2f} Seconds for a run with BERT".format(end - start))"""
# print("SAMPLE:", sample)

'''
Tests for GPT-2
'''

raw_text = remove_symbols_from_text(raw_text)
raw_text = (raw_text.encode('ascii', 'ignore')).decode("utf-8")
lm = api.LM()
start = api.time.time()
payload = lm.check_probabilities(raw_text, topk=20)
end = api.time.time()
print("{:.2f} Seconds for a check with GPT-2".format(end - start))
'''for item in payload["pred_topk"]:
  print(lm.postprocess(item[0][0]))'''



"""start = api.time.time()
sample = lm.sample_unconditional()
end = api.time.time()
print("{:.2f} Seconds for a sample from GPT-2".format(end - start))"""
# print("SAMPLE:", sample)

res = {
        "request": {'project': "new", 'text': raw_text},
        "result": payload
    }

print(res)

#with open('test_json.json', 'w') as outfile:
#    json.dump(res, outfile)

output = []

with jsonlines.open('gpt-2.medium-345M-k40.train.jsonl') as reader:
    for obj in reader:
        print(str(obj["id"]))
        raw_text = obj["text"]
        raw_text = remove_symbols_from_text(raw_text)
        # print(raw_text + "\n\n" + str(obj["id"]))
        payload = lm.check_probabilities(raw_text, topk=10)
        res = {
            "request": {'project': "new", 'text': raw_text},
            "result": payload
        }
        output = output.append(res)
        if obj["id"] == 10:
            break

with open('gpt2.analyzed.medk40train.json', 'w') as outfile:
    json.dump(str(output), outfile)

