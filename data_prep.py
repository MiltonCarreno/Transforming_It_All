import re

with open("JLBorges.txt", "r", encoding="utf-8") as fd:
    text = fd.read()

# Select only stories, omit rest (i.e. forwords, afterwords, etc)
start = ["Tlön, Uqbar, Orbis Tertius\nI.\n",
         "Funes, His Memory\*\nI recall",
         "The Immortal\nSolomon saith: T",
         "The Maker\*\nHe had",
         "On Exactitude in Science\nIn that Empire,",
         "The Ethnographer\nI was told a",
         "The Interloper\n2 Reyes",
         "The Other\nThe incident",
         "Shakespeare’s Memory\nAugust 25"]
end = ["my endless contrition, and my weariness.",
       "manage, and steps out into the plains.",
       "For Estela Canto",
       "world is exceedingly complex for the simplicity of men.",
       "to put an end to men and their wondrous, fragile life.",
       "his death, he had been in heaven.",
       "remedy this report has the temerity to suggest.",
       "street the library's on.\*",
       "fleeting memories that are perhaps authentic."]

# Function to only select the stories
def getSection(start, end, txt):
    pos = ""
    l = 0
    for i, p in enumerate(zip(start[:], end[:])):
        a = re.search(p[0], txt).span()[0]
        b = re.search(p[1], txt).span()[1] + 1
        print(f"{i+1}th - Start: {a} End: {b} Len:{b-a}")
        pos += txt[a:b]
        l += (b - a)
    print(f"Text Length: {l}")
    print(f"NEW Text Length: {len(pos)}")
    return pos

text = getSection(start, end, text)
chars = sorted(list(set(text))) # Get list of unique characters in data
print(f"Pre-pruned Vocab Size: {len(chars)}\n{chars}")

# Replace language-specific characters, except for 'ç', 'ñ', with closests counterparts 
dic = {}
dic['A'] = ['Á', 'Ä']
dic['AE'] = ['Æ']
dic['ss'] = ['ß']
dic['a'] = ['à', 'á', 'â', 'ã', 'ä']
dic['e'] = ['è', 'é', 'ê', 'ë']
dic['i'] = ['í', 'î', 'ï',]
dic['o'] = ['ò', 'ó', 'ô', 'ö']
dic['oe'] = ['œ']
dic['u'] = ['ú', 'û', 'ü']
dic['"'] = ['“', '”']
dic["'"] = ['’']
dic[','] = ['»']

# Function to replace characters
def multiple_replace(dic, txt):
    for i in dic.items():
        pattern = re.compile("|".join(map(re.escape, i[1])))
        txt = re.sub(pattern, i[0], txt)
    return txt
    
text = multiple_replace(dic, text)
chars = sorted(list(set(text)))
vocab_size = len(chars) # Number of unique characters
print(f"Post-pruned Vocab Size: {len(chars)}\n{chars}")

# Write processed text to file 'tiny_borges.txt'
with open("tiny_borges.txt", "x") as f:
    f.write(text)