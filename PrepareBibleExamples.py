
# coding: utf-8

# Quick notebook to prepae examples (not the tfrecord kind) from Project Gutenborgs Bible text

# In[1]:


import re


# In[2]:


txt = open('./pg10.txt').read()


# In[3]:


split_reg = re.compile('\n{4}')


# # split the raw text of the bible into books

# In[4]:


books = split_reg.split(txt)


# In[5]:


books = books[4:] # The first 4 are preamble from project gutenberg


# Split each book into the name of the book and its text

# In[6]:


book_verse_spliter = re.compile('\n{3}(?=1:1)',flags=re.MULTILINE) #The look ahead makes sure the book starts at chapter 1 verse 1
book,verses = book_verse_spliter.split(books[1])


# In[7]:


books[1][:100]


# In[8]:


verses[:100]


# Split all the text in one book into verses. Make it a dict of chapter, verse and text

# In[9]:


verses_splitter = re.compile('(?P<chapter>\d+):(?P<verse>\d+)(?P<text>.+?)(?=\d+\:\d+)',)


# In[10]:


gen = verses_splitter.finditer(verses.replace("\n",""))


# In[11]:



next(gen).groupdict()


# Lets run all of that on the entire bible

# In[12]:


examples= []
book_id = 0
book_map={}
for num,book in enumerate(books):
        splitted  = book_verse_spliter.split(book)
        if len(splitted) >1:
            book_name, book_text = splitted
            book_name = book_name.strip().replace('\n', ' ')
            if book_name.startswith("The "): #This filters out other junk in the dataset
                for verse_regex_match in verses_splitter.finditer(book_text.replace("\n"," ")):
                    example = verse_regex_match.groupdict()
                    example.update({"book":book_name,"book_id":book_id,"text":example["text"].strip()})
                    examples.append(example)
            book_map[book_name] =book_id
            book_id+=1
            


# In[13]:


len(examples)


# Lets save it

# In[14]:


import pickle
pickle.dump(examples,open('./bible_data.pkl','wb'))


# # Now we make them into TF records

# In[15]:


import tensorflow as tf
from preppy import BibPreppy


# In[16]:


import random
random.shuffle(examples)
val,train = examples[:3000], examples[3000:]


# In[17]:


BP =BibPreppy(tokenizer_fn=list) #Charecter level tokenization
for (data,path) in [(val,'./val.tfrecord'),(train,'./train.tfrecord')]:
    with open(path,'w') as f:
        writer = tf.python_io.TFRecordWriter(f.name)
    for example in data:
        record = BP.sequence_to_tf_example(sequence=example["text"],book_id=example["book_id"])
        writer.write(record.SerializeToString())


# In[18]:


BP.update_reverse_vocab()
BP.book_map.update(book_map)


# In[19]:


pickle.dump(BP,open('./preppy.pkl','wb'))


# In[20]:


len(BP.vocab),len(BP.book_map)


# In[23]:


BP.vocab["<START>"]


# In[24]:


BP.vocab

