# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()

from genericpath import exists
from os import makedirs, mkdir
import pickle
import PIL
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np 
import requests
from io import BytesIO
from time import sleep
import sys

def extract_user_id(raw_user_id):
    return raw_user_id.split('_')[1][:-4]

city="ferrol_test"

makedirs(f"{city}/IMGMODEL/data_10+10/",exist_ok=True)
makedirs(f"{city}/IMGMODEL/original_take/",exist_ok=True)

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
#Add text/image flag
df=df[df["content"].notna()]
df["is_image"]=df.apply(lambda x: 1 if x.content[:5]=="https" else 0,axis=1)

#Separate the contents to start translation, drop column from df
contents=df["content"].to_numpy()
df = df.drop('content',axis=1)

if sys.argv[1]=="embeds":
    from transformers import BertTokenizer, TFBertModel
    from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
    from keras.layers import Input
    from keras.preprocessing import image
    from PIL import Image
    from keras.models import Model

    # Initialize BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
    text_model = TFBertModel.from_pretrained("bert-base-uncased",output_hidden_states=True)

    # Initialize INCEPTION model
    base_model = InceptionResNetV2(weights='imagenet', include_top=True)
    image_model = Model(inputs=base_model.input,outputs=base_model.get_layer('avg_pool').output)






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

else:
    def separate_train_test(data):
        counts_images = data[df["is_image"]==1]["user_id"].value_counts()
        counts_texts = data[df["is_image"]==0]["user_id"].value_counts()
        morethan2=data[((data["user_id"].isin(counts_images.index[counts_images.gt(1)]))&(data["is_image"]==1))|((data["user_id"].isin(counts_texts.index[counts_texts.gt(1)]))&(data["is_image"]==0))]
        test=morethan2.groupby(['user_id','is_image']).nth(0).reset_index()
        print(test)
        train=pd.concat([data,test]).drop_duplicates(keep=False)
        return train, test
    
    train_and_val,test=separate_train_test(df)
    train,val=separate_train_test(train_and_val)
    
    train["num_images"]=1
    train_and_val["num_images"]=1
    val["num_images"]=1
    test["num_images"]=1
    df["num_images"]=1
    
    information=""
    information+="#################################\n"
    information+=str(city+"\n")
    information+="#################################\n"
    information+="\n"
    information+="====== BEFORE OVERSAMPLING ======\n"
    information+="          REVIEWS    USERS    RESURANTS\n"
    information+=str("ALL: "+ str(df.shape[0])+" "+ str(pd.unique(df["user_id"]).shape[0])+ " "+ str(pd.unique(df["restaurant_id"]).shape[0])+"\n")
    information+=str("TRAIN: "+ str(train.shape[0])+" "+ str(pd.unique(train["user_id"]).shape[0])+ " "+ str(pd.unique(train["restaurant_id"]).shape[0])+"\n")
    information+=str("TRAIN_DEV: "+ str(train_and_val.shape[0])+" "+ str(pd.unique(train_and_val["user_id"]).shape[0])+ " "+ str(pd.unique(train_and_val["restaurant_id"]).shape[0])+"\n")
    information+=str("DEV: "+ str(val.shape[0])+" "+ str(pd.unique(val["user_id"]).shape[0])+ " "+ str(pd.unique(val["restaurant_id"]).shape[0])+"\n")
    information+=str("TEST: "+ str(test.shape[0])+" "+ str(pd.unique(test["user_id"]).shape[0])+ " "+ str(pd.unique(test["restaurant_id"]).shape[0])+"\n")
    
    with open(city+"/IMGMODEL/original_take/TRAIN_DEV", "wb") as f:
        pickle.dump(train_and_val, f)
    with open(city+"/IMGMODEL/original_take/TRAIN_TEST", "wb") as f:
        pickle.dump(df, f)
    with open(city+"/IMGMODEL/original_take/DEV", "wb") as f:
        pickle.dump(val, f)
    with open(city+"/IMGMODEL/original_take/TEST", "wb") as f:
        pickle.dump(test, f)
        
    
    #==================================
    #Mark all existing reviews as positive samples
    #==================================
    train["take"]=1
    train_and_val["take"]=1
    val["is_dev"]=1
    test["is_dev"]=1
    print(test.shape)
    
    #==================================
    #Oversample training sets (TRAIN and TRAIN+VAL)
    #==================================
    def oversample_trainset(train):
        rows=[]
        newtrain=train.copy()
        i=0
        for index, row in train.iterrows():
            # print(row)
            #For each review (u,r), add then negative samples (u,r') where r' is a photo of the **same** restaurant taken by a different user u'
            same_restaurant=(train.loc[(train["user_id"]!=row["user_id"]) & (train["restaurant_id"]==row["restaurant_id"])]).copy()
            if not same_restaurant.empty: same_restaurant=same_restaurant.sample(n=10,replace=True)
            same_restaurant["rvw_id"]=row["rvw_id"]
            same_restaurant["take"]=0
            same_restaurant=same_restaurant.to_dict(orient='records')
            # print(same_restaurant)
            for e in same_restaurant:
                rows.append(e)
            if i%10000==0: print(i)
            # newtrain=newtrain.append(same_restaurant,ignore_index=True)
            
            #For each review (u,r), add then negative samples (u,r') where r' is a photo of a **different** restaurant taken by a different user u'
            different_restaurant=(train.loc[(train["user_id"]!=row["user_id"]) & (train["restaurant_id"]!=row["restaurant_id"])]).copy()
            different_restaurant=different_restaurant.sample(n=10,replace=True)
            different_restaurant["rvw_id"]=row["rvw_id"]
            different_restaurant["take"]=0
            
            different_restaurant=different_restaurant.to_dict(orient='records')
            for e in different_restaurant:
                rows.append(e)
                
            #Oversample the original review to compensate for the negative samples added
            for r in range(19):
                rows.append(row.to_dict())
            i+=1
        aux=pd.DataFrame.from_records(rows)
        return newtrain.append(pd.DataFrame(rows),ignore_index=True)
        
        
    print(train.shape)
    oversampled_train=oversample_trainset(train)
    os.makedirs(city+"/data_10+10",exist_ok=True)
    print("hey")
    with open(city+"/data_10+10/TRAIN_TXT", "wb") as f:
        pickle.dump(oversampled_train, f)
    print(oversampled_train.shape)
    
    print(train_and_val)
    oversampled_train_and_val=oversample_trainset(train_and_val)
    with open(city+"/data_10+10/TRAIN_DEV_TXT", "wb") as f:
        pickle.dump(oversampled_train_and_val, f)
    
    #==================================
    # Oversample Test sets (VAL and TEST)
    #==================================
    
    def oversample_testset(test,train):
        rows=[]
        id_test=0
        for index, row in test.iterrows():
            id_test+=1
            r=row.to_dict()
            r["id_test"]=id_test
            
            
            #For each review (u,r), add negative samples (u,r') for all photos r' taken of the **same** restaurant by a different user.
            same_restaurant=(train.loc[(train["user_id"]!=row["user_id"]) & (train["restaurant_id"]==row["restaurant_id"])]).copy()
            
            if same_restaurant.shape[0]>100: same_restaurant=same_restaurant.sample(n=100)
            if id_test%10000==0: print
            same_restaurant["user_id"]=row["user_id"]
            same_restaurant.rename(columns={'take':'is_dev'}, inplace=True)
            same_restaurant["is_dev"]=0
            same_restaurant["id_test"]=id_test
            if same_restaurant.shape[0]!=0: add=True 
            else: add=False
            same_restaurant=same_restaurant.to_dict(orient='records')
            if add:
                rows.append(r)
                for e in same_restaurant:
                    rows.append(e)
            
        print("finished oversampling")
        aux=pd.DataFrame.from_records(rows)
        print("created df")
        return aux
        
    print(val.shape)
    oversampled_val=oversample_testset(val,train)
    print("done")
    
    with open(city+"/data_10+10/DEV_TXT", "wb") as f:
        pickle.dump(oversampled_val, f)
    print(oversampled_val.shape)
    
    print(test.shape)
    oversampled_test=oversample_testset(test,train_and_val)
    with open(city+"/data_10+10/TEST_TXT", "wb") as f:
        pickle.dump(oversampled_test, f)
    print(oversampled_test.shape)
    
    information+="\n"
    information+="====== AFTER OVERSAMPLING ======\n"
    information+="          REVIEWS     POSITIVE     NEGATIVE\n"
    information+=str("TRAIN: "+ str(oversampled_train.shape[0])+" "+ str(oversampled_train.loc[oversampled_train["take"]==1].shape[0])+ " "+ str(oversampled_train.loc[oversampled_train["take"]==0].shape[0])+"\n")
    information+=str("TRAIN_DEV: "+ str(oversampled_train_and_val.shape[0])+" "+ str(oversampled_train_and_val.loc[oversampled_train_and_val["take"]==1].shape[0])+ " "+ str(oversampled_train_and_val.loc[oversampled_train_and_val["take"]==0].shape[0])+"\n")
    information+=str("DEV: "+ str(oversampled_val.shape[0])+" "+ str(oversampled_val.loc[oversampled_val["is_dev"]==1].shape[0])+ " "+ str(oversampled_val.loc[oversampled_val["is_dev"]==0].shape[0])+"\n")
    information+=str("TEST: "+ str(oversampled_test.shape[0])+" "+ str(oversampled_test.loc[oversampled_test["is_dev"]==1].shape[0])+ " "+ str(oversampled_test.loc[oversampled_test["is_dev"]==0].shape[0])+"\n")
    
    #==================================
    # Save pickles for the Embedding Size and Unique user counts
    #==================================
    v_img=768
    with open(city+"/data_10+10/V_TXT", "wb") as f:
        pickle.dump(v_img, f)
        
    n_usr=pd.unique(data["user_id"]).shape[0]
    with open(city+"/data_10+10/N_USR", "wb") as f:
        pickle.dump(n_usr, f)
