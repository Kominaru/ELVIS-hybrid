# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
from genericpath import exists
from os import makedirs, mkdir
import pickle
import PIL
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np 
from transformers import BertTokenizer, TFBertModel
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Input
from keras.preprocessing import image
from PIL import Image
from keras.models import Model
import requests
from io import BytesIO
from time import sleep

# Initialize BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
text_model = TFBertModel.from_pretrained("bert-base-uncased",output_hidden_states=True)

# Initialize INCEPTION model
base_model = InceptionResNetV2(weights='imagenet', include_top=True)
image_model = Model(inputs=base_model.input,outputs=base_model.get_layer('avg_pool').output)

def extract_user_id(raw_user_id):
    return raw_user_id.split('_')[1][:-4]



city="losangeles"

makedirs(f"{city}/IMGMODEL/data_10+10/",exist_ok=True)

#Read CSV
df=pd.read_csv(f"{city}.csv")

#Drop irrelevant columns
df=df.drop(['parse_count','author','sample','rating_review','title_review','review_preview','url_review','date','city','url_restaurant'],axis=1)
df = df[df['user_id'].notna()] #Purge invalid user ids

# df = df[df['rating_review']>3]

#Created fabricated user_ids
df["user_id"]=df["user_id"].apply(extract_user_id)
df = df.assign(user_id=(df["user_id"]).astype('category').cat.codes)

#Create fabricated restaurant IDs
df = df.assign(restaurant_id=(df["restaurant_name"]).astype('category').cat.codes)
df = df.drop('restaurant_name',axis=1)

#Split photos column into actual lists
df = df[df["user_id"].isin(df[df["photos"].notna()]['user_id'].unique())]
df = df.assign(photos=df.photos.str.split(','))
df["photos"]=df["photos"].apply(lambda x: [""] if x is NaN else x+[""] ) #We add an additional element to each list so we can have a row for the review as well

#Explode the dataset to have one row per image or text
df=pd.DataFrame({
  col:np.repeat(df[col].values, df["photos"].str.len())
 for col in df.columns.difference(["photos"])
  }).assign(**{"photos":np.concatenate(df["photos"].values)})[df.columns.tolist()]


#Collapse the review and image columns into one content column
df["content"]=df.apply(lambda x : x.review_full if x["photos"]=="" else x["photos"],axis=1)
df = df.drop(["review_full","photos"],axis=1)

#Fabricate review IDs
df["content_id"]=np.arange(0,df.shape[0]).tolist()
df = df.drop('review_id',axis=1)

print(min(df["content_id"]))
input()
#Add text/image flag
df=df[df["content"].notna()]
df["is_image"]=df.apply(lambda x: 1 if x.content[:5]=="https" else 0,axis=1)

#Separate the contents to start translation, drop column from df
contents=df["content"].to_numpy()
df = df.drop('content',axis=1)

print("Total number of contents:", df.shape[0])
#Get separate image and text arrays
images=contents[df["is_image"]==1]
texts=contents[df["is_image"]==0]

print("Total number of images:", images.shape[0])
print("Total number of texts:", texts.shape[0])

def translate(text): 
	return text_model(tokenizer(list(text), return_tensors='tf',padding=True, truncation=True))[1].numpy()

def loadImage(URL):
    errored=False
    while True:
        try:
            response = requests.get(URL,timeout=3)
            img_bytes = BytesIO(response.content)
            break
        except:
            if not errored: print("\nError",end="") 
            else: print(".",end="")
            errored=True
            sleep(1)
    if errored: print("")
            
    # print("got")
    
    img = Image.open(img_bytes)
    img = img.convert('RGB')
    img = img.resize((299,299), Image.NEAREST)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    return img

def embed_image(batch):
    loaded_batch=np.array(list(map(loadImage,batch)))
    loaded_batch=np.squeeze(loaded_batch)
    return image_model.predict(loaded_batch)


texts=np.array_split(texts, texts.shape[0]//64+1)
images=np.array_split(images, images.shape[0]//16+1)

new_texts=[]
new_images=[]

for i,batch in enumerate(images):
    print(f"\033[K Processing images... batch {i+1}/{len(images)}",end="\r")
    new_images.append(embed_image(batch))

new_images=np.vstack(new_images)
contents=np.zeros((contents.shape[0],1536))

images_mask=(df["is_image"]==1).to_numpy()
contents[images_mask]=new_images

print("\n")
for i,batch in enumerate(texts):
    print(f"\033[K Processing texts... batch {i+1}/{len(texts)}",end="\r")
    new_texts.append(translate(batch))


new_texts=np.vstack(new_texts)




texts_mask=(df["is_image"]==0).to_numpy()

contents[texts_mask,:768]=new_texts


with open(f"{city}/IMGMODEL/data_10+10/CONTENTS", 'wb') as handle:
    pickle.dump(contents, handle)
