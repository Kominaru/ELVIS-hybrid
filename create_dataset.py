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
from random import choice

def extract_user_id(raw_user_id):
    return raw_user_id.split('_')[1][:-4]

city="delhi"

makedirs(f"{city}/IMGMODEL/data_10+10/",exist_ok=True)
makedirs(f"{city}/IMGMODEL/original_take/",exist_ok=True)

#Read CSV
df=pd.read_csv(f"{city}.csv")
df["photos"]=pd.Series(dtype=object)
print(df["photos"])
#Drop irrelevant columns
df=df.drop(['parse_count','author','sample','rating_review','title_review','review_preview','url_review','date','city','url_restaurant'],axis=1)
print(f"Total review count after removing invalid userids: {df.shape[0]} -> ",end="")
df = df[df['user_id'].notna()] #Purge invalid user ids
print(f"{df.shape[0]}")
# df = df[df['rating_review']>3]

#Created fabricated user_ids
df["user_id"]=df["user_id"].apply(extract_user_id)
df = df.assign(user_id=(df["user_id"]).astype('category').cat.codes)
print(f"Total unique users: {df['user_id'].unique().shape[0]}")
#Create fabricated restaurant IDs
df = df.assign(restaurant_id=(df["restaurant_name"]).astype('category').cat.codes)
df = df.drop('restaurant_name',axis=1)
print(f"Total unique restaurants: {df['restaurant_id'].unique().shape[0]}")

#Split photos column into actual lists
# df = df[df["user_id"].isin(df[df["photos"].notna()]['user_id'].unique())]
df = df.assign(photos=df.photos.str.split(','))
df["photos"]=df["photos"].apply(lambda x: [""] if x is NaN else x+[""] ) #We add an additional element to each list so we can have a row for the review as well

#Explode the dataset to have one row per image or text
df=pd.DataFrame({
  col:np.repeat(df[col].values, df["photos"].str.len())
 for col in df.columns.difference(["photos"])
  }).assign(**{"photos":np.concatenate(df["photos"].values)})[df.columns.tolist()]

print("="*50)
print("Exploding dataset...")

print("Total contents: ", df.shape[0], " | Total images: ",  df[df["photos"]!=""].shape[0], " | Total texts: ",  df[(df["photos"]=="") & (df["review_full"].notna())].shape[0])
#Collapse the review and image columns into one content column
print("="*50)
print("Creating is_image flag, merging image url and texts into content column...")
df["content"]=df.apply(lambda x : x.review_full if x["photos"]=="" else x["photos"],axis=1)
df = df.drop(["review_full","photos"],axis=1)

#Fabricate review IDs
print(f"Total review count after removing invalid userids: {df.shape[0]} -> ",end="")
df=df[df["content"].notna()]
df["content_id"]=np.arange(0,df.shape[0]).tolist()
df = df.drop('review_id',axis=1)
print(df.shape[0])
#Add text/image flag

df["is_image"]=df.apply(lambda x: 1 if x.content[:5]=="https" else 0,axis=1)

#Separate the contents to start translation, drop column from df
contents=df["content"].to_numpy()
df = df.drop('content',axis=1)


#Get separate image and text arrays
images=contents[df["is_image"]==1]
texts=contents[df["is_image"]==0]

print("Final contents: ", df.shape[0], " | Final images: ",  images.shape[0], " | Final texts: ", texts.shape[0])

if sys.argv[1]=="embeds":
    print("Creating ",contents.shape[0], " embeds")
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







    def translate(text): 
        return text_model(tokenizer(text, return_tensors='tf',padding=True, truncation=True))[1].numpy()

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

    def embed_image(img):
        return image_model.predict(loadImage(img))

    new_texts=[]
    new_images=[]

    for i,batch in enumerate(images):
        print(f"\033[K Processing images... image {i+1}/{len(images)}",end="\r")
        new_images.append(embed_image(batch))

    # new_images=np.vstack(new_images)
    contents=np.zeros((contents.shape[0],1536))

    # images_mask=(df["is_image"]==1).to_numpy()
    # contents[images_mask]=new_images

    print("\n")
    for i,batch in enumerate(texts):
        print(f"\033[K Processing texts... text {batch[:50]} ({i+1}/{len(texts)})",end="\r")
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
        print(f"Splitting {data.shape[0]} rows. Eligible for test: {counts_images[counts_images.gt(1)].shape[0]} im-users and {counts_texts[counts_texts.gt(1)].shape[0]} txt-users")
        morethan2=data[((data["user_id"].isin(counts_images.index[counts_images.gt(1)]))&(data["is_image"]==1))|((data["user_id"].isin(counts_texts.index[counts_texts.gt(1)]))&(data["is_image"]==0))]
        test=morethan2.groupby(['user_id','is_image']).nth(0).reset_index()
        print(f"Total test samples: {test.shape[0]}")
        input()
        train=pd.concat([data,test]).drop_duplicates(keep=False)
        return train, test
    
    df=df[df["is_image"]==0]
    
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
    
    print(information)

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
    train=train.drop(["num_images"],axis=1)
    train_and_val=train_and_val.drop(["num_images"],axis=1)
    val=val.drop(["num_images"],axis=1)
    test=test.drop(["num_images"],axis=1)

    train["take"]=1
    train_and_val["take"]=1
    val["is_dev"]=1
    test["is_dev"]=1
    print(test.shape)
    

    merge=train_and_val.merge(test,left_on=["user_id","restaurant_id","content_id"], right_on=["user_id","restaurant_id","content_id"])
    merge1=train.merge(val,left_on=["user_id","restaurant_id","content_id"], right_on=["user_id","restaurant_id","content_id"])
    print(f"Overlapping samples: {merge.shape[0]} (TRAIN_DEV - TEST), {merge1.shape[0]} (TRAIN - DEV)")

    #==================================
    #Oversample training sets (TRAIN and TRAIN+VAL)
    #==================================
    def oversample_trainset_new(train):

        restaurant_groups = train.groupby(["restaurant_id","is_image"])
        newtrain=train.copy()
        i=0
        rows=[]
        image_restaurant_ids=pd.unique(train[train["is_image"]==1]["restaurant_id"].to_list())
        text_restaurant_ids=pd.unique(train[train["is_image"]==0]["restaurant_id"].to_list())
        for index, row in train.iterrows():
            if index%10000==0: print(i)
            # print(row)
            #For each review (u,r), add then negative samples (u,r') where r' is a photo of the **same** restaurant taken by a different user u'
            # same_restaurant=(train.loc[(train["user_id"]!=row["user_id"]) & (train["restaurant_id"]==row["restaurant_id"])]).copy()ç
            sampled_rows=restaurant_groups.get_group((row["restaurant_id"],row["is_image"])).copy()
            sampled_rows=sampled_rows[sampled_rows["user_id"]!=row["user_id"]]
            if not sampled_rows.empty: 
                sampled_rows=sampled_rows.sample(n=10,replace=True)
                sampled_rows["content_id"]=row["content_id"]
                sampled_rows["take"]=0
                sampled_rows=sampled_rows.to_dict(orient='records')

                for e in sampled_rows:
                    rows.append(e)
            
            #For each review (u,r), add then negative samples (u,r') where r' is a photo of a **different** restaurant taken by a different user u'
            different_restaurant_id=row["restaurant_id"]
            for _ in range(10):
                while different_restaurant_id==row["restaurant_id"]:
                    different_restaurant_id = choice(image_restaurant_ids) if row["is_image"] else choice(text_restaurant_ids)
                sampled_rows=restaurant_groups.get_group((different_restaurant_id,row["is_image"])).copy()
                sampled_rows=sampled_rows[sampled_rows["user_id"]!=row["user_id"]]
                if not sampled_rows.empty:
                    sampled_rows=sampled_rows.sample(n=1)
                    sampled_rows["content_id"]=row["content_id"]
                    sampled_rows["take"]=0
                    sampled_rows=sampled_rows.to_dict(orient='records')
                    for e in sampled_rows:
                        rows.append(e)    
                
            #Oversample the original review to compensate for the negative samples added
            row=row.to_dict()
            for _ in range(19):
                rows.append(row)
            i+=1
        return newtrain.append(pd.DataFrame(rows),ignore_index=True)
        
    def oversample_trainset_old(train):
        print("Overampling train set with traditional method...")
        rows=[]
        newtrain=train.copy()
        i=0
        for index, row in train.iterrows():
            # print(row)
            #For each review (u,r), add then negative samples (u,r') where r' is a photo of the **same** restaurant taken by a different user u'
            same_restaurant=(train.loc[(train["user_id"]!=row["user_id"]) & (train["restaurant_id"]==row["restaurant_id"]) & (train["is_image"]==row["is_image"])]).copy()
            if not same_restaurant.empty: same_restaurant=same_restaurant.sample(n=10,replace=True)
            same_restaurant["content_id"]=row["content_id"]
            same_restaurant["take"]=0
            same_restaurant=same_restaurant.to_dict(orient='records')
            # print(same_restaurant)
            for e in same_restaurant:
                rows.append(e)
            if i%10000==0: print(i)
            # newtrain=newtrain.append(same_restaurant,ignore_index=True)
            
            #For each review (u,r), add then negative samples (u,r') where r' is a photo of a **different** restaurant taken by a different user u'
            different_restaurant=(train.loc[(train["user_id"]!=row["user_id"]) & (train["restaurant_id"]!=row["restaurant_id"]) & (train["is_image"]==row["is_image"])]).copy()
            different_restaurant=different_restaurant.sample(n=10,replace=True)
            different_restaurant["content_id"]=row["content_id"]
            different_restaurant["take"]=0
            
            different_restaurant=different_restaurant.to_dict(orient='records')
            for e in different_restaurant:
                rows.append(e)
                
            #Oversample the original review to compensate for the negative samples added
            for r in range(19):
                rows.append(row.to_dict())
            i+=1
        return newtrain.append(pd.DataFrame(rows),ignore_index=True)
        
    print(f"Train (ORIGINAL): {train.shape[0]} samples")
    oversampled_train=oversample_trainset_old(train)
    print(f"Train (OVERSAMPLE): {oversampled_train.shape[0]} samples")
    print("FINISHED OVERSAMPLING TRAIN")
    with open(city+"/IMGMODEL/data_10+10/TRAIN_TXT", "wb") as f:
        pickle.dump(oversampled_train, f)
    print(oversampled_train.shape)
    
    print(train_and_val)
    oversampled_train_and_val=oversample_trainset_old(train_and_val)
    with open(city+"/IMGMODEL/data_10+10/TRAIN_DEV_TXT", "wb") as f:
        pickle.dump(oversampled_train_and_val, f)
    
    #==================================
    # Oversample Test sets (VAL and TEST)
    #==================================
    
    def oversample_testset_new(test,train):
        restaurant_groups = train.groupby(["restaurant_id","is_image"])
        i=0
        rows=[]
        id_test=0
        image_restaurant_ids=pd.unique(train[train["is_image"]==1]["restaurant_id"].to_list())
        text_restaurant_ids=pd.unique(train[train["is_image"]==0]["restaurant_id"].to_list())
        for _, row in test.iterrows():
            if id_test%10000==0: print
            id_test+=1
            
            if (row["restaurant_id"],row["is_image"]) in restaurant_groups.groups:
                restaurant=restaurant_groups.get_group((row["restaurant_id"],row["is_image"])).copy()
                restaurant=restaurant[restaurant["user_id"]!=row["user_id"]]
                if restaurant.shape[0]>100: restaurant=restaurant.sample(n=100)
                restaurant["user_id"]=row["user_id"]
                restaurant.rename(columns={'take':'is_dev'}, inplace=True)
                restaurant["is_dev"]=0
                restaurant["id_test"]=id_test
                r=row.to_dict()
                r["id_test"]=id_test
                if restaurant.shape[0]!=0:
                    rows.append(r)
                    restaurant=restaurant.to_dict(orient='records')
                    for e in restaurant:
                        rows.append(e)

        print("finished oversampling")
        aux=pd.DataFrame.from_records(rows)
        print("created df")
        return aux

    def oversample_testset_old(test,train):
        rows=[]
        id_test=0
        for _, row in test.iterrows():
            id_test+=1
            r=row.to_dict()
            r["id_test"]=id_test
            
            
            #For each review (u,r), add negative samples (u,r') for all photos r' taken of the **same** restaurant by a different user.
            same_restaurant=(train.loc[(train["user_id"]!=row["user_id"]) & (train["restaurant_id"]==row["restaurant_id"]) & (train["is_image"]==row["is_image"])]).copy()
            
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
    oversampled_val=oversample_testset_old(val,train)
    print("done")
    
    with open(city+"/IMGMODEL/data_10+10/DEV_TXT", "wb") as f:
        pickle.dump(oversampled_val, f)
    print(oversampled_val.shape)
    
    print(test.shape)
    oversampled_test=oversample_testset_old(test,train_and_val)
    with open(city+"/IMGMODEL/data_10+10/TEST_TXT", "wb") as f:
        pickle.dump(oversampled_test, f)
    print(oversampled_test.shape)
    
    information+="\n"
    information+="====== AFTER OVERSAMPLING ======\n"
    information+="          REVIEWS     POSITIVE     NEGATIVE\n"
    information+=str("TRAIN: "+ str(oversampled_train.shape[0])+" "+ str(oversampled_train.loc[oversampled_train["take"]==1].shape[0])+ " "+ str(oversampled_train.loc[oversampled_train["take"]==0].shape[0])+"\n")
    information+=str("TRAIN_DEV: "+ str(oversampled_train_and_val.shape[0])+" "+ str(oversampled_train_and_val.loc[oversampled_train_and_val["take"]==1].shape[0])+ " "+ str(oversampled_train_and_val.loc[oversampled_train_and_val["take"]==0].shape[0])+"\n")
    information+=str("DEV: "+ str(oversampled_val.shape[0])+" "+ str(oversampled_val.loc[oversampled_val["is_dev"]==1].shape[0])+ " "+ str(oversampled_val.loc[oversampled_val["is_dev"]==0].shape[0])+"\n")
    information+=str("TEST: "+ str(oversampled_test.shape[0])+" "+ str(oversampled_test.loc[oversampled_test["is_dev"]==1].shape[0])+ " "+ str(oversampled_test.loc[oversampled_test["is_dev"]==0].shape[0])+"\n")
    
    print(information)

    merge=oversampled_train_and_val.merge(oversampled_test,left_on=["user_id","restaurant_id","content_id"], right_on=["user_id","restaurant_id","content_id"])
    print(merge[(merge["take"]==1) & (merge["is_dev"]==1)])
    print(merge)
    #==================================
    # Save pickles for the Embedding Size and Unique user counts
    #==================================
    v_img=768
    with open(city+"/IMGMODEL/data_10+10/V_TXT", "wb") as f:
        pickle.dump(v_img, f)
        
    n_usr=pd.unique(df["user_id"]).shape[0]
    with open(city+"/IMGMODEL/data_10+10/N_USR", "wb") as f:
        pickle.dump(n_usr, f)
