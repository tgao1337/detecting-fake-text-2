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
import torch
from guppy import hpy


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
# raw_text = "As the final day of the presidential campaign approached Wednesday morning, polls suggested the race was closer than many had expected.\n\nPolls have shown Hillary Clinton leading Donald Trump by 9 percentage points since late September, with the Republican leading by 7 percent.\n\nThe final polls of early voting show Mrs. Clinton with an 18-point advantage and a 9-point lead over Mr. Trump. By early afternoon, Mr. Clinton had narrowed the lead to 13 points, while Mr. Trump held just a 1-point lead for the final three days among likely voters.\n\nThe Associated Press projected a margin between Mrs. Clinton and Mr. Trump of 5 percentage points on its Web site.\n\nBut as of 6:20 a.m., the first three days of polling in Iowa had closed more than two percentage points behind the final results, with Mrs. Clinton leading by six percentage points. The latest RealClearPolitics polling average had the two leading candidates up by one percentage point for the week. A Quinnipiac University poll gave Mrs. Clinton a 7-percentage-point edge.\n\nIn the latest AP-Quinnipiac tracking poll from early Wednesday morning, Mrs. Clinton had a 10-point lead over Mr. Trump, 50 - 39 percent. In the other three-day survey conducted on Sept. 28-30, Mrs. Clinton and Mr. Trump were up by 13 points among likely voters, with her leading 47 - 44 percent. In the Sept. 21 AP-Quinnipiac poll, Mrs. Clinton had a four-point lead.\n\nThe first day of the 2016 presidential election is Sept. 26. Some states on the map that were deemed safe for Hillary Clinton (and for Democrats) are shown in red, while states like Iowa are in blue. Photo by Michael S. Williamson/The Washington Post.\n\nMany experts expect Tuesday's first-in-the-nation Iowa caucus to deliver the decisive vote Tuesday to Republicans to take control of the House, Senate and presidency for the first time since the 2010 census. Republicans are in control in just two states at this point \u2014 Arizona and Missouri \u2014 and control four more \u2014 New Hampshire, Arizona and Indiana.\n\nIowa is the first state in Iowa's electoral history to be a contested primary. Republicans are expected to hold a narrow advantage in Iowa, with voters in three of the state's four counties who did not participate in last year's caucuses making up a solid majority, according to a University of Iowa poll published earlier this year.\n\nThat has Democrats feeling optimistic.\n\n\"I suspect that people are more confident and focused this time around,\" said Jim Bell, the president of the Iowa Democratic Party, who did not participate in the current Democratic caucuses.\n\nThe final four days of early voting will begin at 8 a.m. on Wednesday, when Iowa's Republican primary will be held; if Mrs. Clinton wins either that early primary or the fall general election, she would head into November with control of the presidency. If it was close, the Republican establishment might make a late push for an upset in the March 1 general election, just as they did in 2012.\n\nIn a CNN-ORC International poll Tuesday, voters said they wanted to take back the White House, but that their support would not \"crack\" before Oct. 26. That poll showed Mrs. Clinton ahead by six percentage points or more among likely caucus-goers.\n\nThe latest CNN-ORC Poll of Iowa also found Clinton and Mr. Trump tied among a potential third-party candidate, with independent Angus King the only Republican in the race.\n\nOther polls, however, showed Mrs. Clinton widening the gap.\n\nA Wall Street Journal/NBC News poll released on Sept. 23 found that 48 percent of likely voters supported Mrs. Clinton; 36 percent backed Mr. Trump. The poll also surveyed 738 likely Democratic caucus-goers, who have about as many choices in which party to support. In the September poll, a slim 46 percent supported Mr. Trump, compared to 41 percent for Mrs. Clinton.\n\nThe last early vote tally was posted by the Clinton campaign and the Democratic National Committee on Wednesday morning. The first results will be in a few hours, when the first day of early voting \u2014 and how many voters actually showed up at polling places \u2014 will start to take shape for the first time.\n\nA Fox News poll released last week suggested an even split of support, one in which 39 percent favored Mrs. Clinton, 33 percent backed Mr. Trump and 13 percent favored neither. The third-party candidates appeared to have strong momentum on both sides Wednesday as voting began, with support for Gary Johnson at 41 percent, as well as Green Party candidate Jill Stein at just 3 percent and 8 percent, respectively.\n\nThe final two days of early voting will be conducted by the National Association for Research & Therapy of Posttraumatic Stress Disorder on Aug. 1 and 4 \u2014 the first two days in the Sept. 26-29 race \u2014"
# raw_text = "In a shocking incident that is already being seen as a turning point in women's rights in Afghanistan with the death penalty against sexual harassment, a woman was attacked because of her attire last weekend at a restaurant when she tried to urinate in her cup of tea without a padlock.\n\nAn Afghan official has said that the incident was a result of her having been embarrassed about her dress.\n\nPolice officer Mohammed Shah, told Reuters \"we arrested the culprit behind this attack and brought him to our police station for questioning.\"\n\nShah reportedly told Reuters by telephone this woman had been at a restaurant last Saturday when some employees at nearby shops stopped them from removing their pants. \"These customers started laughing when they saw I was in this style,\" Shah added, asking not to be named as he has no choice in his official duties.\n\nThe woman, 30, was on her way home from lunch when the incident occurred. She was attacked by an Arab driver who attempted to take her to a nearby pub, he said. She was badly injured but made a full recovery.\n\nWomen's rights activists and local officials are trying to raise awareness about the issue, while one woman on social media warned other women going to eat to remove their pants. \"For those who think wearing pants is a thing of the past, just know that you really can be attacked by any man in Afghanistan if you take off your dress. I'm tired of looking like this and I'm sick of thinking girls in my country will have to wear only their underwear,\" wrote Shafieen Bibi, 21.\n\nAfghanistan has no law that deals specifically with women's clothing in Afghanistan, only that the wearing of clothing that reveals \"such appearance or aspect of [the woman's] body\" can lead to criminal charges being laid against her.\n\nShah's statement did not specify any punishment for the man but local police were quick to respond by saying that assault will not be tried as such. \"Women who are attacked in the city and in rural areas, we will not make any comment on such incident,\" Shah said.\n\nWomen continue to receive very little from the government in regards to improving the lives of girls. However, the situation on the ground in the country is not improving either and a growing number of women's activists are turning to the courts with the hope that such legislation, if passed by lawmakers, would improve society's perception of women as equal, not just victims of sexual assault.\n\n\"It's all I do, no other thing. It's my life, but I am completely uneducated about my rights, how I should dress and what my rights should be,\" Nour Mariam, 23, told Al Jazeera."
# raw_text = "As a recent graduate in my field of computer science, one thing that surprised me is how pervasive information processing is in life. As I've been growing in my field, it's become clear that the ability to process information using the mind and the power of communication is extremely precious \u2014 it's the difference between having a very useful set of skills and only having the ability to process it once. For me, this was very exciting and helped me become aware of some new concepts and methods I had not considered when working in various other fields. In order to understand how information processing is useful in many ways, I decided to use my own application of information technology \u2014 learning to code on OpenCV.\n\nIn the early 1980's, I used a Commodore 64 to learn to code the simple OpenCV program. I have since used several Commodore 32c's, as well as my own computer for a number of years. In the years since, I've used several different computers for tasks like programming to write software applications, which I've often found to be easier to learn with a more traditional computer.\n\nIn 1984, I started learning to program to play the classic game 'Tic-Tac-Toe' in my Commodore 64 with some help from another student. A few years later, when I was about 25 years old, I got into computer programming using my Commodore 2K1, a piece of kit I bought back in 1982 when I worked at a software company.\n\nLearning to code with the Commodore 64 proved to be difficult for a number of reasons. By learning on my own, I had trouble developing and maintaining programs that worked perfectly. My programs had limited syntax and limited features that the user knew how to use. These things added up over 30+ years, and while there were also plenty of tutorials to guide me through the process, I was finding it challenging. This is where information technology comes into play! I became comfortable learning on a number of different platforms, from Linux to the Free Software in the '90s movement to Mac OS X to the MS-DOS and UNIX.\n\nWhen I started looking at my computer again in the middle of last year, I couldn't think of anything I didn't love. This prompted a search on the internet to find out more about how computer programmers learn and the information architecture that makes learning easier. My next search, came up with some of my favourite articles and webpages that I started looking into and started to learn and use. In a few months, I had a basic understanding of the basic principles, concepts, and principles of how programmers learn.\n\nA few years ago, I came across something called the \"Computer Architecture Handbook\". This book has been written by Andrew Jurgens. I thought this would be a nice resource to learn about how programming is done on computers. The concept of how we design machines is very significant to computer scientists, and learning to program computers in the right places has allowed me to create various apps and games that work well.\n\nAs I began writing this, I started looking at some of the various tutorials and articles available online, and realized several of them had something in common. One of these articles taught me how to use Windows NT. I also started reading a lot of tutorials on how to read C libraries through the C compiler which is the software application compiler used for many programming languages. This provided the core concepts I needed to read more deeply into information processing concepts and learn to understand the information being conveyed to my computer.\n\nWhile many of the lessons from this book helped me further my knowledge, the greatest joy was to be able to see what information-processing techniques were already known. I could actually use my own computers, my own language tools, and a couple of free resources like RCS, xkcd or Wikipedia. All of it has made learning to program with my computer a lot more affordable, fun and enjoyable for me. This is one area that still seems to be on the rise. Just about every year I find myself spending more time at my computer and spending more time writing code! It may seem like there isn't much to write about here in the first post of my blog, but what is there to write about when you can spend as much time as you want writing code?\n\nI'm hoping this article will have started a spark in you, as a programmer and aspiring developer. If you are a programmer, or have recently started learning to code, then you should go read what I've written here, and give these learning resources a play! Just keep in mind that we are still quite a ways away from seeing \"real\" computing with our own technology, so this is not a definitive list of things you should do by yourself. For developers, there are still a lot of things to learn. You can learn from a beginner's view of programming when it comes to how to implement a simple function using your command line. In this post, I'll share with you some of the things I learned using these two tutorials above. I hope"
# raw_text = "It is almost Halloween and now we know where to find the most adorable Halloween costume from Disney Parks. You will find that the best costumes are sold in all the Disney Stores of the world and the Disney Parks are open year round and open at all hours to guests. So get ready for more beautiful costumes for your Halloween, this is your chance to be the ultimate Halloween costume fanatic!\nThere are a lot of costumes available and you might want to choose it to dress a particular character, but before you rush to purchase, check it out and find those perfect Halloween costume for you. Disney is always giving out special discounts on many types of costumes and a big part of that is giving us great deals on discounts. For instance, if you are in the Orlando area and buying a Disney Disney Springs ticket, there may be a discount of $25 off in addition to the standard Disney Park admission.\nSo which ones are worth it to go through? Well the costumes are all available in Disney parks and many of them, are worth the price of admission. Here are a few of our favorite and popular Disney costumes.\nIf you want some ideas, check out more Disney Halloween costumes, just don't choose a dress because the look on most of their people's faces isn't what you want or you will end up regretting it!\nFrozen Halloween Costume \u2013\nIf you want to go for the more authentic look for winter, you can always choose to go with one of the iconic costumes of Elsa (Snow Queen Elsa) from Frozen to Frozen. This costume was inspired by Elsa's famous red dress she wore while visiting the castle as Frozen's main character. It is known as the Frozen Frozen costume in Disney parks and is popular throughout the Disney parks.\nHalloween Costume Options:\nThere are many things that you can do in order to dress for the Halloween season and you really have to plan on how you will dress from beginning to end because if you do not make the costume plan it will be very difficult to complete.\nThe best Halloween costumes are not just for Halloween, they are a part of every month to go. The Halloween costumes are the perfect and easy costume to prepare for the occasion. So why don't we have a list, you will know exactly which costumes are in stock and can be ordered. If you plan to wear a costume every week, go ahead, but if you want something from the regular line-up, take one look over at the top of this page and think hard about which one suits you best.\nIf you really want to take advantage of the best Halloween costume deals, there are the items for men. There are the top-selling brands, like Nike and Adidas, among others that will also make you extremely happy with their clothing and a large selection of other interesting items as well. The same may apply to women. There will be many women costumes that you can choose from and make sure to research them and select one or the other, you wouldn't get the chance for a great outfit to choose from. Of course, some costumes may also be part of the special events or themed ones and the selection of themed costumes will be even wider and more creative. So pick the one that fits you best and don't skip out on getting to know the Disney Disney Store's best prices.\u00a0\nCheck out this page to find Halloween merchandise, which has everything you need to plan on buying your very own Halloween costume.\nHalloween costumes are a popular part of the Disney Experience because it gives you a chance to live out your imagination, and in so doing, help your family and friends have the best of Halloween and other special events and events. It's a perfect time to visit the Disney resorts and Disney Kingdoms and for your Halloween activities, you can get a special gift. When planning your next trick-or-treating, think a lot about buying a Halloween accessory for your children or a Halloween gift, just in case things get tough after all. Also, make sure to select the best Halloween costumes that will work for you as well. Make sure that these Halloween Costume Ideas will provide you with the best Halloween costume from Disney Parks.\nSee the full list of Disney Halloween Costumes :\nSo there you have it, the three most popular Halloween Halloween costumes in Disney Parks, the ones you'll want to consider when planning your next Disney Halloween plan. Remember that no matter your level, if you choose the more authentic or less official Disney Halloween costumes, you will have great options."


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

#raw_text = remove_symbols_from_text(raw_text)
#raw_text = (raw_text.encode('ascii', 'ignore')).decode("utf-8")
#lm = api.LM()
#start = api.time.time()
#payload = lm.check_probabilities(raw_text, topk=20)
#end = api.time.time()
#print("{:.2f} Seconds for a check with GPT-2".format(end - start))
'''for item in payload["pred_topk"]:
  print(lm.postprocess(item[0][0]))'''



"""start = api.time.time()
sample = lm.sample_unconditional()
end = api.time.time()
print("{:.2f} Seconds for a sample from GPT-2".format(end - start))"""
# print("SAMPLE:", sample)

#res = {
#        "request": {'project': "new", 'text': raw_text},
#        "result": payload
#    }

#print(res)

#with open('test_json.json', 'w') as outfile:
#    json.dump(res, outfile)
output = []


# # analyze gpt2
# counter = 0
# lm = api.LM()
# with jsonlines.open('gpt-2.webtext.train.jsonl') as reader:
#     for obj in reader:
#         if obj["id"] > 15103:
#             raw_text = obj["text"]
#             if obj["length"] > 1023:  # if longer than 1023, cut short
#                 dot = [i for i in range(1024) if
#                        raw_text.startswith('.', i) or raw_text.startswith('\n', i)]
#                 # print(dot)
#                 if len(dot) == 0:
#                     dot = [i for i in range(1024) if
#                            raw_text.startswith(' ', i)]
#                     if len(dot) == 0:
#                         raw_text = raw_text[:1024]
#                     else:
#                         raw_text = raw_text[:dot[-1]]
#                 else:
#                     raw_text = raw_text[:dot[-1]]
#             counter = counter + 1
#             print(obj["id"])
#
#             raw_text = remove_symbols_from_text(raw_text)
#             raw_text = (raw_text.encode('ascii', 'ignore')).decode("utf-8")
#             # print(raw_text)
#             # print(raw_text + "\n\n" + str(obj["id"]))
#             payload = lm.check_probabilities(raw_text, topk=20)
#             # print(payload)
#             res = {
#                 "request": {'project': "new", 'text': raw_text},
#                 "result": payload
#             }
#             # print(res)
#             # print(output)
#             # output.append(res)
#             with jsonlines.open('gpt2.analyzed.human-10000.jsonl', mode='a') as writer:
#                 writer.write(res)
#             # break
#             torch.cuda.empty_cache()
#             h = hpy()
#             print(h.heap())
#             if obj["id"] == 25000:
#                 break

#with open('gpt2.analyzed.webtext-5000.json', 'w') as outfile:
#    json.dump(output, outfile)


counter = 0
lm = api.LM()
# # analyze gpt3
# with jsonlines.open('gpt-3.175b_samples.jsonl') as reader:
#     for obj in reader:
#         counter = counter + 1
#         #print(obj)
#         #print(len(obj))
#         #print(type(obj))
#         dot = [i for i in range(1024) if
#                obj.startswith('.', i) or obj.startswith('\n', i)]
#         #print(dot)
#         if len(dot) == 0:
#             dot = [i for i in range(1024) if obj.startswith(' ', i)]
#             if len(dot) == 0:
#                 obj = obj[:1024]
#             else:
#                 obj = obj[:dot[-1]]
#         else:
#             obj = obj[:dot[-1]]
#         #print(len(obj))
#         #print((obj))
#
#         raw_text = remove_symbols_from_text(obj)
#         raw_text = (raw_text.encode('ascii', 'ignore')).decode("utf-8")
#         # print(raw_text)
#         # print(raw_text + "\n\n" + str(obj["id"]))
#         payload = lm.check_probabilities(raw_text, topk=20)
#             # print(payload)
#         res = {
#             "request": {'project': "new", 'text': raw_text},
#             "result": payload
#         }
#         # print(res)
#         # print(output)
#         output.append(res)
#         # break
#         torch.cuda.empty_cache()
#         print(counter)
#         #if counter == 100:
#             #break
# with open('gpt3.analyzed.machine-485.json', 'w') as outfile:
#     json.dump(output, outfile)


# # analyze grover human and machine
# with jsonlines.open('generator=mega_dataset=p0.94.jsonl') as reader:
#     for obj in reader:
#         if obj["label"] == "human":  # change this to human or machine
#             counter = counter + 1
#             # print(obj)
#             # print(len(obj))
#             # print(type(obj))
#             dot = [i for i in range(1024) if
#                    obj["article"].startswith('.', i) or obj["article"].startswith('\n', i)]
#             # print(dot)
#             if len(dot) == 0:
#                 dot = [i for i in range(1024) if obj["article"].startswith(' ', i)]
#                 if len(dot) == 0:
#                     obj = obj["article"][:1024]
#                 else:
#                     obj = obj["article"][:dot[-1]]
#             else:
#                 obj = obj["article"][:dot[-1]]
#             # print(len(obj))
#             # print((obj))
#
#             raw_text = remove_symbols_from_text(obj)
#             raw_text = (raw_text.encode('ascii', 'ignore')).decode("utf-8")
#             # print(raw_text)
#             # print(raw_text + "\n\n" + str(obj["id"]))
#             payload = lm.check_probabilities(raw_text, topk=20)
#             # print(payload)
#             res = {
#                 "request": {'project': "new", 'text': raw_text},
#                 "result": payload
#             }
#             # print(res)
#             # print(output)
#             #output.append(res)
#             # break
#             torch.cuda.empty_cache()
#             print(counter)
#             if counter == 1000:
#                 break
#
# h = hpy()
# print(h.heap())
# with open('grover.analyzed.human-1000.json', 'w') as outfile:
#     json.dump(output, outfile)




# # analyze gpt2 machine
# counter = 0
# lm = api.LM()
# with jsonlines.open('gpt-2.medium-345M-k40.train.jsonl') as reader:
#     for obj in reader:
#         if obj["id"] > 14988:
#             raw_text = obj["text"]
#             if obj["length"] > 1023:  # if longer than 1023, cut short
#                 dot = [i for i in range(1024) if
#                        raw_text.startswith('.', i) or raw_text.startswith('\n', i)]
#                 # print(dot)
#                 if len(dot) == 0:
#                     dot = [i for i in range(1024) if
#                            raw_text.startswith(' ', i)]
#                     if len(dot) == 0:
#                         raw_text = raw_text[:1024]
#                     else:
#                         raw_text = raw_text[:dot[-1]]
#                 else:
#                     raw_text = raw_text[:dot[-1]]
#             counter = counter + 1
#             print(obj["id"])
#
#             raw_text = remove_symbols_from_text(raw_text)
#             raw_text = (raw_text.encode('ascii', 'ignore')).decode("utf-8")
#             # print(raw_text)
#             # print(raw_text + "\n\n" + str(obj["id"]))
#             payload = lm.check_probabilities(raw_text, topk=20)
#             # print(payload)
#             res = {
#                 "request": {'project': "new", 'text': raw_text},
#                 "result": payload
#             }
#             # print(res)
#             # print(output)
#             # output.append(res)
#             with jsonlines.open('gpt2.analyzed.machine-10000.jsonl', mode='a') as writer:
#                 writer.write(res)
#             # break
#             torch.cuda.empty_cache()
#             h = hpy()
#             print(h.heap())
#             if obj["id"] == 25000:
#                 break


# analyze grover human and machine
with jsonlines.open('generator=mega_dataset=p0.94.jsonl') as reader:
    for obj in reader:
        if obj["label"] == "human":  # change this to human or machine
            print(counter)
            counter = counter + 1
            if counter > -1:

                # print(obj)
                # print(len(obj))
                # print(type(obj))
                dot = [i for i in range(1024) if
                       obj["article"].startswith('.', i) or obj["article"].startswith('\n', i)]
                # print(dot)
                if len(dot) == 0:
                    dot = [i for i in range(1024) if obj["article"].startswith(' ', i)]
                    if len(dot) == 0:
                        obj = obj["article"][:1024]
                    else:
                        obj = obj["article"][:dot[-1]]
                else:
                    obj = obj["article"][:dot[-1]]
                # print(len(obj))
                # print((obj))

                raw_text = remove_symbols_from_text(obj)
                raw_text = (raw_text.encode('ascii', 'ignore')).decode("utf-8")
                # print(raw_text)
                # print(raw_text + "\n\n" + str(obj["id"]))
                payload = lm.check_probabilities(raw_text, topk=20)
                # print(payload)
                res = {
                    "request": {'project': "new", 'text': raw_text},
                    "result": payload
                }
                # print(res)
                # print(output)
                #output.append(res)
                # break
                with jsonlines.open('grover.analyzed.human-10000.jsonl', mode='a') as writer:
                    writer.write(res)

                torch.cuda.empty_cache()
                print(counter)
                if counter == 25000:
                    break
