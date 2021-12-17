# from transformers import BertTokenizer, TFBertModel
import pandas as pd
import pickle
import numpy as np
import math
import os
# from googletrans import Translator
# translator = Translator()
information=""
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
# model = TFBertModel.from_pretrained("bert-base-uncased",output_hidden_states=True)
# def plot_hist(data, column, title="Histogram", title_x="X Axis", title_y="Y Axis", bins=10, save=None):

    # plt.ioff()

    # items = bins

    # plt.hist(data[str(column)], bins=range(1, items + 2), edgecolor='black',
             # align="left")  # arguments are passed to np.histogram
    # labels = list(map(lambda x: str(x), range(1, items + 1)))
    # labels[-1] = "≥" + labels[-1]
    # plt.xticks(range(1, items + 1), labels)
    # plt.title(str(title))

    # plt.xlabel(title_x)
    # plt.ylabel(title_y)

    # if save is None:
        # plt.show()
    # else:
        # plt.savefig(str(save))

    # plt.close()

f = open("datasetinfo.txt", "a")
	
def csv_pandas_to_pickle(city):
	df = pd.read_csv(city+".csv", skiprows=0, na_values=['no info', '.'],encoding='utf-8',sep=',',error_bad_lines=False,warn_bad_lines=False)
	print(df.shape)
	df.to_pickle(city+".pkl")

# with open("TXT", 'rb') as f:
		# data = pickle.load(f)
		# data=np.array(data.tolist())
		# data=np.squeeze(data)
		# print(data)
		# print(data.shape)
		# with open("TXT", "wb") as f:
			# pickle.dump(data, f)



# def translate(text): 
	# return model(tokenizer(str(text), return_tensors='tf',padding=True, truncation=True))[1].numpy()

# translate_vec = np.vectorize(translate)

def get_userid(s):
	if "UID" not in str(s): return "nan"
	# print(s)
	a=str(s).split('_')[-2][:-4]
	# print(a)
	return a
get_userid_vec=np.vectorize(get_userid)

# with open('coruna_dropped.pkl', 'rb') as f:
	# data = pickle.load(f)
	# print(data["review_full"].values)
	# reviews=truncate_vec(data["review_full"].values)

	# translated=np.empty(reviews.shape,dtype=object)
	# print(translated.shape)
	# for i in range(reviews.shape[0]):
		
		# translated[i]=translate(reviews[i])
		# print(i)
	# print(translated)
	# data["img_emb"]=translated.tolist()
	# data.to_pickle("coruna_with_embeds.pkl")
	# data=data.drop(['parse_count','author','rating_review','sample','title_review','review_preview','url_review','date','city','url_restaurant'],axis=1)
	# data.to_pickle("coruna_dropped.pkl")


#csv_pandas_to_pickle()

# with open('coruna_with_embeds.pkl', 'rb') as f:
	# data = pickle.load(f)
	
	# users=data["user_id"].values
	# strippedusrs=np.empty(users.shape,dtype=object)
	# strippedusrs=get_userid_vec(users)
	# data["user_id"]=strippedusrs.tolist()
	# print(data["user_id"][10084])
	# data = data.assign(restaurant_id=(data["restaurant_name"]).astype('category').cat.codes)
	# data = data.assign(user_id=(data["user_id"]).astype('category').cat.codes)
	# data=data.drop(['restaurant_name'],axis=1)
	# print(data.columns.values)
	# print(data.shape)
	# data.to_pickle("coruna_new_userids.pkl")
	
# with open("coruna_new_userids.pkl", 'rb') as f:
	# df = pickle.load(f)
	# df = df.assign(review_id=(df["review_id"]).astype('category').cat.codes)
	# imgs_emb=df["img_emb"].to_numpy()
	# imgs_text=df["review_full"].to_numpy()
	# print(imgs_emb,imgs_text)
	# df=df.drop(['review_full','img_emb'],axis=1)
	# print(df)
	# v = df["user_id"].value_counts()
	# print(v.shape)
	# morethan2=df[df["user_id"].isin(v.index[v.gt(1)])]
	# test=morethan2.groupby('user_id').nth(0).reset_index()
	# train= pd.concat([df,test]).drop_duplicates(keep=False)
	# print(test)
	# print(train)
	# print(df)
	
def preprocess(city):
	folder=f"old_oversamples/{city}/IMGMODEL/"
	global information
	with open(city+".pkl", 'rb') as f:
		data = pickle.load(f)
	
	if data.shape[0]>500000: data=data.head(500000)
	print(data)
	#==================================
	#Drop irrelevant columns
	#==================================
	data=data.drop(['parse_count','author','rating_review','sample','title_review','review_preview','url_review','date','city','url_restaurant','photos'],axis=1)
	
	#==================================
	#Strip the user ID's
	#==================================
	users=data["user_id"].values
	strippedusrs=np.empty(users.shape,dtype=object)
	strippedusrs=get_userid_vec(users)
	data["user_id"]=strippedusrs.tolist()
	
	#==================================
	#Create fabricated Restaurant ID's, drop Restaurant names
	#==================================
	data = data.assign(restaurant_id=(data["restaurant_name"]).astype('category').cat.codes)
	data = data.drop('restaurant_name',axis=1)
	
	#==================================
	#Create fabricated User ID's
	#==================================
	data = data.assign(user_id=(data["user_id"]).astype('category').cat.codes)
	
	#==================================
	#Create fabricated review ID's based on the Index column, drop old Review ID's
	#==================================
	data["rvw_id"]=np.arange(0,data.shape[0]).tolist()
	data = data.drop('review_id',axis=1)
	
	#==================================
	#Separate Review Texts to separate array, drop review_full from the dataframe
	#==================================
	full_reviews=data["review_full"].to_numpy()
	data = data.drop('review_full',axis=1)
	
	print(data.shape[0])
	#==================================
	#Truncate and translate reviews, save the result array into the TXT pickle
	#==================================
	
	# truncated_reviews=full_reviews
	# translated_reviews=np.empty(truncated_reviews.shape,dtype=object)
	
	# print(truncated_reviews[0])
	# print(truncated_reviews.shape)
	# for i in range(truncated_reviews.shape[0]):
		
		# translated_reviews[i]=translate(truncated_reviews[i])
		# print(i)
		
	# with open("TXT", "wb") as f:
		# translated_reviews=np.array(translated_reviews.tolist())
		# translated_reviews=np.squeeze(translated_reviews)
		# pickle.dump(translated_reviews, f)
		
	# print(translated_reviews)
	# print(translated_reviews.shape)
	
	#==================================
	#Separate TRAIN, DEV and TEST datasets (also TRAIN+DEV)
	#==================================

	def separate_train_test(data):
		v = data["user_id"].value_counts()
		morethan2=data[data["user_id"].isin(v.index[v.gt(1)])]
		test=morethan2.groupby('user_id').nth(0).reset_index()
		train=pd.concat([data,test]).drop_duplicates(keep=False)
		return train, test
	
	train_and_val,test=separate_train_test(data)
	train,val=separate_train_test(train_and_val)
	
	train["num_images"]=1
	train_and_val["num_images"]=1
	val["num_images"]=1
	test["num_images"]=1
	data["num_images"]=1
	
	
	information+="#################################\n"
	information+=str(city+"\n")
	information+="#################################\n"
	information+="\n"
	information+="====== BEFORE OVERSAMPLING ======\n"
	information+="          REVIEWS    USERS    RESURANTS\n"
	information+=str("ALL: "+ str(data.shape[0])+" "+ str(pd.unique(data["user_id"]).shape[0])+ " "+ str(pd.unique(data["restaurant_id"]).shape[0])+"\n")
	information+=str("TRAIN: "+ str(train.shape[0])+" "+ str(pd.unique(train["user_id"]).shape[0])+ " "+ str(pd.unique(train["restaurant_id"]).shape[0])+"\n")
	information+=str("TRAIN_DEV: "+ str(train_and_val.shape[0])+" "+ str(pd.unique(train_and_val["user_id"]).shape[0])+ " "+ str(pd.unique(train_and_val["restaurant_id"]).shape[0])+"\n")
	information+=str("DEV: "+ str(val.shape[0])+" "+ str(pd.unique(val["user_id"]).shape[0])+ " "+ str(pd.unique(val["restaurant_id"]).shape[0])+"\n")
	information+=str("TEST: "+ str(test.shape[0])+" "+ str(pd.unique(test["user_id"]).shape[0])+ " "+ str(pd.unique(test["restaurant_id"]).shape[0])+"\n")
	
	os.makedirs(f"{folder}/original_take",exist_ok=True)
	with open(f"{folder}/original_take/TRAIN_DEV", "wb") as f:
		pickle.dump(train_and_val, f)
	with open(f"{folder}/original_take/TRAIN_TEST", "wb") as f:
		pickle.dump(data, f)
	with open(f"{folder}/original_take/DEV", "wb") as f:
		pickle.dump(val, f)
	with open(f"{folder}/original_take/TEST", "wb") as f:
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
	os.makedirs(f"{folder}/data_10+10",exist_ok=True)
	print("hey")
	with open(f"{folder}/data_10+10/TRAIN_TXT", "wb") as f:
		pickle.dump(oversampled_train, f)
	print(oversampled_train.shape)
	
	print(train_and_val)
	oversampled_train_and_val=oversample_trainset(train_and_val)
	with open(f"{folder}/data_10+10/TRAIN_DEV_TXT", "wb") as f:
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
	
	with open(f"{folder}/data_10+10/DEV_TXT", "wb") as f:
		pickle.dump(oversampled_val, f)
	print(oversampled_val.shape)
	
	print(test.shape)
	oversampled_test=oversample_testset(test,train_and_val)
	with open(f"{folder}/data_10+10/TEST_TXT", "wb") as f:
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
	with open(f"{folder}/data_10+10/V_TXT", "wb") as f:
		pickle.dump(v_img, f)
		
	n_usr=pd.unique(data["user_id"]).shape[0]
	with open(f"{folder}/data_10+10/N_USR", "wb") as f:
		pickle.dump(n_usr, f)
if __name__ == '__main__': 
	
	
	for city in ["losangeles"]:
	# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	# model = TFBertModel.from_pretrained("bert-base-uncased",output_hidden_states=True)
	# text = "me gusta el arroz con cosas"
	# encoded_input = tokenizer(text, return_tensors='tf',padding=True, truncation=True, max_length=512, model_max_length=512)
	# output = model(encoded_input)
		csv_pandas_to_pickle(city)
		preprocess(city)
		f.write(information)
		information=""
	f.close()