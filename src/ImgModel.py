# -*- coding: utf-8 -*-

from src.ModelClass import *

import time

import itertools as it
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
from scipy import spatial
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import *
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Concatenate, Dropout, Dot, Lambda, Add, Input, Multiply
from keras.initializers import Constant

########################################################################################################################


def get_img(url, save=None):
	def center_crop(im, new_width=None, new_height=None):
		width, height = im.size  # Get dimensions

		left = (width - new_width) / 2
		top = (height - new_height) / 2
		right = (width + new_width) / 2
		bottom = (height + new_height) / 2

		im = im.crop((left, top, right, bottom))

		return im

	response = requests.get(url)
	img = Image.open(BytesIO(response.content))

	if save is not None:
		img = center_crop(img, new_width=150, new_height=150)
		img.save(save, "JPEG")

	return img


def get_image_pos(data):
	# Ordenar los restaurantes por predición
	data = data.sort_values(["id_test", "prediction"])

	# Crear una lista de posiciones en función del número de imágenes de cada restaurante
	ids = data.groupby("id_test").apply(lambda x: np.flip(np.array(range(len(x))) + 1, 0))
	ids = np.concatenate(ids.values).ravel()
	data["model_pos"] = ids

	# Obtener el número de imágenes de cada restaurante
	rest_data = data.groupby("restaurant_id").model_pos.max().reset_index(name="n_photos")
	print(rest_data)
	
	data_dev = data.loc[data.is_dev == 1]
	data_dev = data_dev.merge(rest_data, on="restaurant_id")

	data_dev["PCNT-1_MDL"] = (data_dev["model_pos"] - 1) / data_dev["n_photos"]

	data_dev = data_dev[["user_id", "restaurant_id", "id_test", "model_pos", "n_photos", "PCNT-1_MDL"]]

	return data_dev, data_dev["PCNT-1_MDL"].mean(), data_dev["PCNT-1_MDL"].median()


########################################################################################################################


class ImgModel(ModelClass):

	def __init__(self, city, config, seed=2, model_name="imgModel", load=None):

		ModelClass.__init__(self, city, config, model_name, seed=seed, load=load)

	def stop(self):

		if self.SESSION is not None:
			K.clear_session()
			print_w("-" * 50, title=False)
			print_w("Closing session...")
			print_w("-" * 50, title=False)
			self.SESSION.close()

		if self.MODEL is not None:
			tf.reset_default_graph()

	def get_model1(self):

		# Fijar las semillas de numpy y TF
		np.random.seed(self.SEED)
		random.seed(self.SEED)
		tf.compat.v1.set_random_seed(self.SEED)

		emb_size = 256

		usr_emb_size = emb_size
		img_emb_size = emb_size

		model_u = Sequential()
		model_u.add(Embedding(self.DATA["N_USR"], usr_emb_size, input_shape=(1,), name="in_usr"))
		model_u.add(Flatten())

		model_i = Sequential()
		model_i.add(Dense(img_emb_size, input_shape=(self.DATA["V_TXT"],), name="in_img"))
		

		concat = Concatenate(axis=1)([model_u.output, model_i.output])
		concat = Activation("relu")(concat)
		concat = Dropout(self.CONFIG["dropout"])(concat)

		concat = Dense(emb_size)(concat)
		concat = Activation("relu")(concat)
		concat = Dropout(self.CONFIG["dropout"])(concat)

		concat = Dense(1)(concat)
		conc_take_out = Activation("sigmoid")(concat)

		opt = Adam(lr=self.CONFIG["learning_rate"])

		model_take = Model(inputs=[model_u.input, model_i.input], outputs=conc_take_out)
		model_take.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

		return model_take
	

	def get_model2(self):

		# Fijar las semillas de numpy y TF
		np.random.seed(self.SEED)
		random.seed(self.SEED)
		tf.compat.v1.set_random_seed(self.SEED)

		emb_size = 256

		usr_emb_size = emb_size
		img_emb_size = emb_size

		print(self.DATA["V_TXT"])
		print(self.DATA["N_USR"])
		
		#USERS EMBEDDING
		model_u = Sequential()
		model_u.add(Embedding(self.DATA["N_USR"], usr_emb_size, input_shape=(1,), name="in_usr"))
		print(model_u.output.shape)
		model_u.add(Flatten())

		#IMAGE EMBEDDING
		model_i = Sequential()
		model_i.add(Dense(img_emb_size, input_shape=(1536,), name="in_img"))

		
		print(model_u.input.shape, model_i.input.shape)
		print(model_u.output.shape, model_i.output.shape)
		
		model_t = Sequential()
		model_t.add(Dense(img_emb_size, input_shape=(768,), name="in_txt"))
		
		model_flag=Input(shape=(1,),name="flag")
		model_flag1=Input(shape=(1,),name="flag1")
		
		#DOT PRODUCT
		concat_img=Dot(axes=1)([model_u.output, model_i.output])
		concat_txt=Dot(axes=1)([model_u.output, model_t.output])
		
		concat_img=Multiply()([concat_img,model_flag])
		concat_txt=Multiply()([concat_txt,model_flag1])
		
		
		concat = Add()([concat_img,concat_txt])
		
		conc_take_out = Activation("sigmoid")(concat)

		opt = Adam(lr=self.CONFIG["learning_rate"])

		model_take = Model(inputs=[model_u.input, model_i.input, model_t.input, model_flag, model_flag1], outputs=conc_take_out)
		model_take.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

		return model_take

	def get_model(self):

		# Fijar las semillas de numpy y TF
		np.random.seed(self.SEED)
		random.seed(self.SEED)
		tf.compat.v1.set_random_seed(self.SEED)

		emb_size = 256

		usr_emb_size = emb_size
		img_emb_size = emb_size

		print(self.DATA["V_TXT"])
		print(self.DATA["N_USR"])
		
		#USERS EMBEDDING
		model_u = Sequential()
		model_u.add(Embedding(self.DATA["N_USR"], usr_emb_size, input_shape=(1,), name="in_usr"))
		print(model_u.output.shape)
		model_u.add(Flatten())

		#IMAGE EMBEDDING
		model_i = Sequential()
		model_i.add(Dense(img_emb_size, input_shape=(1536,), name="in_img"))

		
		print(model_u.input.shape, model_i.input.shape)
		print(model_u.output.shape, model_i.output.shape)
		
		#DOT PRODUCT
		concat=Dot(axes=1)([model_u.output, model_i.output])
		
		# concat = Concatenate(axis=1)([model_u.output, model_i.output])
		# concat = Activation("relu")(concat)
		# concat = Dropout(self.CONFIG["dropout"])(concat)

		# concat = Dense(emb_size)(concat)
		# concat = Activation("relu")(concat)
		# concat = Dropout(self.CONFIG["dropout"])(concat)

		# concat = Dense(1)(concat)
		conc_take_out = Activation("sigmoid")(concat)

		opt = Adam(lr=self.CONFIG["learning_rate"])

		model_take = Model(inputs=[model_u.input, model_i.input], outputs=conc_take_out)
		model_take.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

		return model_take

	def train(self, sq_take,val_take):

		def _train(model, data,valdata):
			hist_ret = model.fit_generator(data,
										   epochs=1,
										   steps_per_epoch=data.__len__(),
										   shuffle=True,
										   use_multiprocessing=False,
										   workers=6,
										   verbose=0,
										   validation_data=valdata)
			return hist_ret

		# ---------------------------------------------------------------------------------------------------------------

		ret = {}

		val = K.get_value(
			tf.compat.v1.train.linear_cosine_decay(self.CONFIG["learning_rate"], self.CURRENT_EPOCH, self.CONFIG["epochs"]))
		K.set_value(self.MODEL.optimizer.lr, val)
		ret["LDECAY"] = val / self.CONFIG["learning_rate"]

		take_ret = _train(self.MODEL, sq_take,val_take)
		ret["T_LOSS"] = take_ret.history['loss'][0]

		preds=self.MODEL.predict_generator(val_take, steps=sq_take.__len__(), use_multiprocessing=False, workers=6)[:,0]
		return take_ret.history

	def dev(self, sq_take):

		def _dev(model, data,valdata):
			ret = model.predict_generator(data, steps=data.__len__(), use_multiprocessing=False, workers=6)
			tmp_like = data.DATA
			tmp_like["prediction"] = ret[:, 0]

			return tmp_like

		# ---------------------------------------------------------------------------------------------------------------

		ret = {}

		take_ret = _dev(self.MODEL, sq_take,val_take)
		_, pcnt1_model, pcnt1_model_median = get_image_pos(take_ret)
		ret["T_AVG"] = pcnt1_model
		ret["T_MDN"] = pcnt1_model_median

		return ret

	def test(self):

		def get_img_probs(model):
			test_sequence_take = self.DevSequenceTake(self.DATA["TEST_TXT"], self.CONFIG['batch_size'], self)

			dts = self.DATA["TEST_TXT"].copy()
		
			prs = model.predict_generator(test_sequence_take, steps=test_sequence_take.__len__(),
										  use_multiprocessing=False, workers=6)

			dts["prediction"] = prs

			return dts

		# ---------------------------------------------------------------------------------------------------------------
		# Cargar los modelos
		# ---------------------------------------------------------------------------------------------------------------

		if not os.path.exists(self.MODEL_PATH): print_e("The model doesn't exist."); exit()
		model_take = load_model(self.MODEL_PATH + "/model_1")

		# ---------------------------------------------------------------------------------------------------------------
		# Evaluar en TEST
		# ---------------------------------------------------------------------------------------------------------------

		RET = pd.DataFrame(
			columns=["user_id", "restaurant_id", "n_photos", "n_photos_dev", "model_pos", "pcnt_model", "pcnt1_model"])

		probs = get_img_probs(model_take)

		return probs

	####################################################################################################################

	class TrainSequenceTake(Sequence):

		def __init__(self, data, batch_size, model):
			self.DATA = data
			self.MODEL = model
			self.BATCHES = np.array_split(data, len(data) // batch_size)

			self.BATCH_SIZE = batch_size

		def __len__(self):
			return len(self.BATCHES)

		def __getitem__(self, idx):
			data_ids = self.BATCHES[idx]
			
			imgs = self.MODEL.DATA["TXT"][data_ids.content_id.values]
			# print(np.array(data_ids.user_id.values).shape)
			# print(np.array(data_ids[["take"]].values.shape))


			take_info=np.array(data_ids["take"].values)

			assert 	not np.any(np.all((imgs == 0), axis=1))
			return 	([np.array(data_ids.user_id.values),imgs],
					[take_info])

	class DevSequenceTake(Sequence):

		def __init__(self, data, batch_size, model):
			print(data.shape)
			self.DATA = data
			self.MODEL = model
			self.BATCHES = np.array_split(data, len(data) // batch_size)
			self.BATCH_SIZE = batch_size

		def __len__(self):
			return len(self.BATCHES)

		def __getitem__(self, idx):
			data_ids = self.BATCHES[idx]
			imgs = self.MODEL.DATA["TXT"][data_ids.content_id.values][:,:1536]
			txts = self.MODEL.DATA["TXT"][data_ids.content_id.values][:,:768]

			assert 	not np.any(np.all((imgs == 0), axis=1))
			return ([np.array(data_ids.user_id.values),imgs],
					[np.array(data_ids[["is_dev"]].values)])


	####################################################################################################################

	def testloss(self, sq_take):
		m=self.MODEL
		ret = m.evaluate_generator(sq_take, steps=sq_take.__len__(), use_multiprocessing=False, workers=6)
		return ret

	def grid_search(self, params, max_epochs=[100], start_n_epochs=5, last_n_epochs=5):

		def createCombs():

			def flatten(lst):
				return sum(([x] if not isinstance(x, list) else flatten(x)
							for x in lst), [])

			combs = []
			level = 0
			for v in params.values():
				if (len(combs) == 0):
					combs = v			 
				else:
					combs = list(it.product(combs, v))
				level += 1

				if (level > 1):
					for i in range(len(combs)):
						combs[i] = flatten(combs[i])

			return pd.DataFrame(combs, columns=params.keys())

		def configNet(comb):
			comb.pop("Index")

			for k in comb.keys():
				assert (k in self.CONFIG.keys())
				self.CONFIG[k] = comb[k]

		# -----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

		def get_train():

			train = get_pickle(self.DATA_PATH + "original_take/", "TRAIN_DEV")
			# Add imgs
			train = train.merge(self.DATA["IMG"][["review", "content_id"]], left_on="reviewId", right_on="review")
			train = train.drop(columns=['review'])

			train = train.groupby("user_id").apply(lambda x: pd.Series({"n_photos_train": len(x.content_id.unique()),
																		"n_rest_train": len(
																			x.restaurant_id.unique())})).reset_index()

			return train

		print("-" * 50)
		combs = createCombs()
		print("There are " + str(len(combs)) + " possible combinations")

		for c in combs.itertuples():

			stop_param = []

			c = dict(c._asdict())

			# Configurar la red
			configNet(c)

			# Crear el modelo
			self.MODEL = self.get_model()

			# Imprimir la configuración
			self.print_config(filter_dt=c.keys())

			# Configurar y crear sesion
			self.config_session()
                                             
			# Conjuntos de entrenamiento
			keras.backend.get_session().run(tf.global_variables_initializer())

			# data=get_train()
			# data = data.loc[data["n_photos_train"] >= 50]
			# devel=self.DATA["DEV_TXT"]
			# devel=devel.merge(data["user_id"], left_on="user_id", right_on="user_id")
		
			td=self.DATA["TRAIN_TXT"]
			print("td size:", td.shape)
			devel1=self.DATA["DEV_TXT"]
			# print(self.DATA["TEST_TXT"])
			print(devel1.shape)
			td=td.merge(devel1,left_on=["user_id","restaurant_id","content_id"], right_on=["user_id","restaurant_id","content_id"])
			print(td.shape)
			# print(td.columns.values)
			train_sequence_take = self.TrainSequenceTake(self.DATA["TRAIN_TXT"][self.DATA["TRAIN_TXT"]["is_image"]==0], self.CONFIG['batch_size'], self)
			dev_sequence_take = self.DevSequenceTake(self.DATA["DEV_TXT"], self.CONFIG['batch_size'], self)
			
			# test=devel1.merge(self.DATA["TRAIN_TXT"],left_on="content_id", right_on="content_id")
			# print(test.shape)
			results=np.zeros((2,c["epochs"]))
			x=np.linspace(0,c["epochs"]-1,c["epochs"])


			print("TARGING")
			for e in range(c["epochs"]):

				tes = time.time()
				self.CURRENT_EPOCH = e
	
				train_ret = self.train(train_sequence_take,dev_sequence_take)
				# print(train_ret)
				train_ret={"V_LOSS":train_ret["val_loss"][0],"T_LOSS":train_ret["loss"][0],"V_ACC":train_ret["val_accuracy"][0],"T_ACC":train_ret["accuracy"][0]}
				# dev_ret = self.testloss(dev_sequence_take)
				# dev_ret={"V_LOSS": dev_ret[0], "T_ACC": dev_ret[1]}
				results[0,e]=train_ret["V_LOSS"]
				results[1,e]=train_ret["T_LOSS"]
				
				if (e != c["epochs"] - 1): self.grid_search_print(e, time.time() - tes, train_ret, {})
			
			plt.plot(x,results[0,:],'b')
			plt.plot(x,results[1,:],'r')
			plt.legend(["Validation_Loss","Train_Loss"])
			plt.title("learning rate: "+str(c["learning_rate"]))
			print("LAST VAL LOSS: ", results[0,-1])
			save_path = "stats/" + self.CITY.lower() + "/"
			os.makedirs(save_path, exist_ok=True)
		
			plt.savefig(save_path + self.CITY.lower() + "_train_val_loss.pdf")
			#self.grid_search_print(e, time.time() - tes, train_ret, dev_ret)
			K.clear_session()

	def grid_search_print(self, epoch, elapsed, train, dev):

		def frm(item):
			if isinstance(item,list): item=item[0]
			return '{:6.3f}'.format(item)

		# --------------------------------------------------------------------------------------------------------------
		header = ["E", "E_TIME"]
		header.extend(train.keys())
		header.extend(dev.keys())

		if (epoch == 0): print("\t".join(header))

		line = [elapsed]
		line.extend(train.values())
		line.extend(dev.values())
		line = list(map(frm, line))
		line.insert(0, str(epoch))

		print("\t".join(line))

		return None

	def final_train(self, epochs=1, save=False):

		def get_train():

			train = get_pickle(self.DATA_PATH + "original_take/", "TRAIN_DEV")
			# Add imgs
			train = train.merge(self.DATA["IMG"][["review", "content_id"]], left_on="reviewId", right_on="review")
			train = train.drop(columns=['review'])

			train = train.groupby("user_id").apply(lambda x: pd.Series({"n_photos_train": len(x.content_id.unique()),
																		"n_rest_train": len(
																			x.restaurant_id.unique())})).reset_index()

			return train
		

			# Crear el modelo
		self.MODEL = self.get_model()

		# Imprimir la configuración

		# Configurar y crear sesion
		self.config_session()
										 
		# Conjuntos de entrenamiento
		keras.backend.get_session().run(tf.global_variables_initializer())

		# data=get_train()
		# data = data.loc[data["n_photos_train"] >= 50]
		# devel=self.DATA["DEV_TXT"]
		# devel=devel.merge(data["user_id"], left_on="user_id", right_on="user_id")
	
		td=self.DATA["TRAIN_TXT"]
		# print("td size:", td.shape)
		devel1=self.DATA["DEV_TXT"]
		
		# print(devel1.shape)
		td=td.merge(devel1,left_on=["user_id","restaurant_id","content_id"], right_on=["user_id","restaurant_id","content_id"])
		print(td.shape)
		# print(td.columns.values)
		train_sequence_take = self.TrainSequenceTake(self.DATA["TRAIN_DEV_TXT"], self.CONFIG['batch_size'], self)
		dev_sequence_take = self.DevSequenceTake(self.DATA["TEST_TXT"], self.CONFIG['batch_size'], self)
			
		# test=devel1.merge(self.DATA["TRAIN_TXT"],left_on="content_id", right_on="content_id")
		# print(test.shape)
		results=np.zeros((2,epochs))
		x=np.linspace(0,epochs-1,epochs)
		
		for e in range(epochs):

			tes = time.time()
			self.CURRENT_EPOCH = e

			train_ret = self.train(train_sequence_take,dev_sequence_take)
			# print(train_ret)
			train_ret={"V_LOSS":train_ret["val_loss"][0],"T_LOSS":train_ret["loss"][0],"V_ACC":train_ret["val_accuracy"][0],"T_ACC":train_ret["accuracy"][0]}
			# dev_ret = self.testloss(dev_sequence_take)
			# dev_ret={"V_LOSS": dev_ret[0], "T_ACC": dev_ret[1]}
			results[0,e]=train_ret["V_LOSS"]
			results[1,e]=train_ret["T_LOSS"]
			
			if (e != epochs - 1): self.grid_search_print(e, time.time() - tes, train_ret, {})
			
		plt.plot(x,results[0,:],'b')
		plt.plot(x,results[1,:],'r')
		save_path = "stats/" + self.CITY.lower() + "/"
		os.makedirs(save_path, exist_ok=True)
		
		print("LAST TRAIN LOSS: ", results[0,-1])
		plt.savefig(save_path + self.CITY.lower() + "_traindev_test_loss.pdf")
		# dev_ret = self.dev(test_sequence_take)
		# self.grid_search_print(e, time.time() - tes, train_ret, dev_ret)

		if (save):
			path = self.MODEL_PATH + "/"
			os.makedirs(path,exist_ok=True)
			self.MODEL.save(path + self.MODEL.name)

		K.clear_session()

	def take_baseline(self, test=False):

		def getTrain():

			file_path = self.PATH + self.MODEL_NAME.upper()
			path = file_path + "/original_take/"

			if (test == False):
				TRAIN = get_pickle(path, "TRAIN_DEV")
				
			else:
				TRAIN = get_pickle(path, "TRAIN_DEV")

			# Añadir las imágenes a TRAIN
			# TRAIN = TRAIN.merge(self.DATA["IMG"][["review", "content_id"]], left_on="reviewId", right_on="review")
			# TRAIN = TRAIN.drop(columns=['review'])

			return TRAIN

		def getCentroids(TRAIN):

			RST_CNTS = pd.DataFrame(columns=["restaurant_id", "vector"])

			for i, g in TRAIN.groupby("restaurant_id"):
				all_c = self.DATA["TXT"][g.content_id.values, :]
				cnt = np.mean(all_c, axis=0)

				RST_CNTS = RST_CNTS.append({"restaurant_id": i, "vector": cnt}, ignore_index=True)

			return RST_CNTS

		def getPos(data, ITEMS):
		
			g = data.copy()
			# print(g)
			id_rest = g.restaurant_id.unique()[0]
			# print(id_rest)
			item = ITEMS.loc[ITEMS.restaurant_id == id_rest].vector.values[0]

			rst_imgs = self.DATA["TXT"][g.content_id.values, :]
			dsts = spatial.distance.cdist(rst_imgs, [item], 'euclidean')

			g["dsts"] = dsts

			g = g.sort_values("dsts").reset_index(drop=True)

			return min(g.loc[g.is_dev == 1].index.values) + 1

		def getRndPos(data):

			g = data.copy()
			pos = []

			g["prob"] = np.random.random_sample(len(g))
			g = g.sort_values("prob").reset_index(drop=True)

			return len(g) - max(g.loc[g.is_dev == 1].index.values)

		# Fijar semilla
		# ---------------------------------------------------------------------------------------------------------------
		np.random.seed(self.SEED)

		# Obtener el conjunto de TRAIN original
		# ---------------------------------------------------------------------------------------------------------------

		TRAIN = getTrain()

		# Obtener los centroides de los restaurantes en train y los random
		# ---------------------------------------------------------------------------------------------------------------

		RST_CNTS = getCentroids(TRAIN)

		print(RST_CNTS)
		# Para cada caso de DEV, calcular el resultado de los baselines
		# ---------------------------------------------------------------------------------------------------------------

		ret = []
		rpt = 10

		if (test == False):
			ITEMS = self.DATA["DEV_TXT"]
		else:
			ITEMS = self.DATA["TEST_TXT"]

		for i, g in ITEMS.groupby("id_test"):
			
			cnt_pos = getPos(g, RST_CNTS)
			rnd_pos = []

			for r in range(rpt): rnd_pos.append(getRndPos(g))

			d = [g.user_id.values[0], g.restaurant_id.values[0], i, len(g), cnt_pos]
			d.extend(rnd_pos)

			ret.append(d)

		RET = pd.DataFrame(ret)

		title = ["user_id", "restaurant_id", "id_test", "n_photos", "cnt_pos"]
		title.extend(list(map(lambda x: "rnd_pos_" + str(x), range(10))))

		RET.columns = title

		RET["PCNT-1_CNT"] = RET.apply(lambda x: (x.cnt_pos - 1) / x.n_photos, axis=1)
		for r in range(rpt): RET["PCNT-1_RND_" + str(r)] = RET.apply(
			lambda x: (x["rnd_pos_" + str(r)] - 1) / x.n_photos, axis=1)

		ret_path = "docs/Baselines"
		os.makedirs(ret_path, exist_ok=True)

		RET.to_excel(ret_path + "/Take_" + self.CITY + ("_TEST" if test else "_DEV") + ".xlsx")

		# print("RND\t%f\t%f" % (RET["PCNT-1_RND"].mean(), RET["PCNT-1_RND"].median()))
		avg = [];
		mdn = []
		for r in range(rpt): avg.append(RET["PCNT-1_RND_" + str(r)].mean()); mdn.append(
			RET["PCNT-1_RND_" + str(r)].median())

		print("RND\t%f\t%f" % (np.mean(avg), np.mean(mdn)))
		print("CNT\t%f\t%f" % (RET["PCNT-1_CNT"].mean(), RET["PCNT-1_CNT"].median()))

		return RET

	def centroid_baseline(self, restaurant=0, id_test=0):

		def getTrain():
			file_path = self.PATH + self.MODEL_NAME.upper()
			path = file_path + "/original_take/"

			TRAIN = get_pickle(path, "TRAIN_DEV")

			# Añadir las imágenes a TRAIN
			TRAIN = TRAIN.merge(self.DATA["IMG"][["review", "content_id"]], left_on="reviewId", right_on="review")
			TRAIN = TRAIN.drop(columns=['review'])

			return TRAIN

		def getCentroids(TRAIN, restaurant):
			RST_CNTS = pd.DataFrame(columns=["restaurant_id", "vector"])

			for i, g in TRAIN.groupby("restaurant_id"):
				all_c = self.DATA["TXT"][g.content_id.values, :]
				cnt = np.mean(all_c, axis=0)

				RST_CNTS = RST_CNTS.append({"restaurant_id": i, "vector": cnt}, ignore_index=True)

			return RST_CNTS

		def getPos(data, ITEMS):
			g = data.copy()
			# print(g)
			id_rest = g.restaurant_id.unique()[0]
			item = ITEMS.loc[ITEMS.restaurant_id == id_rest].vector.values[0]

			rst_imgs = self.DATA["TXT"][g.content_id.values, :]
			dsts = spatial.distance.cdist(rst_imgs, [item], 'euclidean')

			g["dsts"] = dsts

			g = g.sort_values("dsts").reset_index(drop=True)

			return g

		# Fijar la semilla
		# ---------------------------------------------------------------------------------------------------------------

		np.random.seed(self.SEED)

		# Obtener el conjunto de TRAIN original
		# ---------------------------------------------------------------------------------------------------------------

		TRAIN = getTrain()
		TRAIN = TRAIN.loc[TRAIN.restaurant_id == restaurant]

		# Obtener los centroides de los restaurantes
		# ---------------------------------------------------------------------------------------------------------------

		RST_CNTS = getCentroids(TRAIN, restaurant)

		# Obtener ranking en TEST
		# ---------------------------------------------------------------------------------------------------------------

		ITEMS = self.DATA["TEST_TXT"]
		ITEMS = ITEMS.loc[ITEMS.id_test == id_test]

		return getPos(ITEMS, RST_CNTS)

	def get_detailed_results(self):

		def get_train():

			train = get_pickle(self.DATA_PATH + "original_take/", "TRAIN_DEV")
			
			# Add imgs
			# train = train.merge(self.DATA["IMG"][["review", "content_id"]], left_on="reviewId", right_on="review")
			# train = train.drop(columns=['review'])

			train = train.groupby("user_id").apply(lambda x: pd.Series({"n_photos_train": len(x.content_id.unique()),
																		"n_rest_train": len(
																			x.restaurant_id.unique())})).reset_index()

			return train

		def op(in_data, op_type="average"):

			if "average" in op_type:
				return np.mean(in_data.values)
			else:
				return np.median(in_data.values)

		# ---------------------------------------------------------------------------------------------------------------
		# EVALUATE IN TEST
		# ---------------------------------------------------------------------------------------------------------------

		self.config_session()

		probs = self.test()

		pcnt, _, _ = get_image_pos(probs)

		blines = self.take_baseline(test=True)

		train_data = get_train()  # Cargar todos los datos con los que se entrena (train+DEV)

		ret = pcnt.merge(train_data, right_on="user_id", left_on="user_id")
		ret = ret.merge(blines, right_on="id_test", left_on="id_test")

		ret = ret.sort_values("n_photos_train", ascending=False).reset_index(drop=True)

		# -----------------------------------------------------------------------------------------------------------
		# DROP RESTAURANTS WITH LESS THAN 10 IMAGES IN TEST (EASY TO GUESS)
		# -----------------------------------------------------------------------------------------------------------

		n_rest_img = 10

		ret10 = ret.loc[(ret["n_photos_x"] >= n_rest_img)]

		method = "median"

		random_cols = [col for col in ret10.columns if 'PCNT-1_RND_' in col]

		graph_data = []

		print("-" * 100)
		print(method.upper())
		print("-" * 100)

		header = ["N_FOTOS_TRAIN_USR (>=)", "N_ITEMS", "%ITEMS", "RANDOM", "CENTROIDE", "MODELO"]

		print("\t".join(header))

		for itr in range(100, 0, -1):
			data = ret10.loc[ret10["n_photos_train"] >= itr]

			line = (itr, len(data), len(data) / len(ret10),
					op(data[random_cols].mean(axis=1), op_type=method),
					op(data["PCNT-1_CNT"], op_type=method),
					op(data["PCNT-1_MDL"], op_type=method))

			print("%d\t%d\t%f\t%f\t%f\t%f" % line)

			graph_data.append(line)

		print("-" * 100)

		# -----------------------------------------------------------------------------------------------------------
		# PLOT A GRAFIC WHITH "graph_data"
		# -----------------------------------------------------------------------------------------------------------

		def plot_series(the_plt, the_data, xcol, ycol, color, label, marker):
			the_plt.plot(xcol, ycol, data=the_data, linestyle='-', color=color, label=label, marker=marker, zorder=2)

		save_path = "stats/" + self.CITY.lower() + "/"
		os.makedirs(save_path, exist_ok=True)

		graph_data = pd.DataFrame(graph_data)
		graph_data.columns = header

		matplotlib.style.use("default")

		plt.rcParams.update({'font.size': 15})
		plt.rc('axes', axisbelow=True)

		fig, ax1 = plt.subplots(figsize=(12, 5), facecolor='w', edgecolor='k')

		if "gijon" in self.CITY.lower():
			plt.title("Gijón", fontweight="bold")
		else:
			plt.title(self.CITY, fontweight="bold")

		ax1.grid(True, linestyle='--', color="lightgray")

		ax1.set_xlim([0, 101])
		ax1.set_ylim([0.00, 1.0])

		xtks = np.arange(0, 101, step=5)
		xtks[0] = 1

		ytks = np.arange(0.0, 1.1, step=0.1)
		ytks_lbls = list(map(lambda x: str(int(x)) + "%", ytks * 100))

		ax1.set_xticks(xtks)
		ax1.set_yticks(ytks)
		ax1.set_yticklabels(ytks_lbls)

		ax2 = ax1.twinx()
		ax2.set_ylim([0, 2000])
		ax2.set_yticks(np.arange(0, 2001, step=200))

		plot_series(ax1, graph_data, xcol='N_FOTOS_TRAIN_USR (>=)', ycol='CENTROIDE', color="rebeccapurple",
					label="CNT",
					marker="o")
		plot_series(ax1, graph_data, xcol='N_FOTOS_TRAIN_USR (>=)', ycol='RANDOM', color="firebrick",
					label="RND",
					marker=">")
		plot_series(ax1, graph_data, xcol='N_FOTOS_TRAIN_USR (>=)', ycol='MODELO', color="black",
					label="Model",
					marker="d")
		ax1.set_xlabel("Users with x or more text reviews in the training set")
		ax1.set_ylabel("Percentile (median)")

		plot_series(ax2, graph_data, xcol='N_FOTOS_TRAIN_USR (>=)', ycol='N_ITEMS', color="mediumseagreen",
					label="Test cases", marker="$\\ast$")
		ax2.set_ylabel('Number of test cases')

		if "gijon" in self.CITY.lower():

			def unite_legends(axes):
				h, l = [], []
				for ax in axes:
					tmp = ax.get_legend_handles_labels()
					h.extend(tmp[0])
					l.extend(tmp[1])
				return h, l

			handles1, labels1 = unite_legends([ax1])
			handles2, labels2 = unite_legends([ax2])

			lg1 = ax2.legend(handles1, labels1, loc='upper left', ncol=3)
			ax2.legend(handles2, labels2, loc='upper right')
			ax2.add_artist(lg1)

		fig.tight_layout()

		plt.savefig(save_path + self.CITY.lower() + "_graph.pdf")

		# -----------------------------------------------------------------------------------------------------------
		# DROP USERS WITH LESS THAN 10 IMAGES (POOR KNOWLEDGE ABOUT THEM)
		# -----------------------------------------------------------------------------------------------------------

		n_user_img = 10

		ret_f = ret10.loc[(ret10["n_photos_train"] >= n_user_img)]
		# ret_f=ret10
		# -----------------------------------------------------------------------------------------------------------
		# TIMES IN EACH POSITION
		# -----------------------------------------------------------------------------------------------------------
		print("================================================")
		print("TEST CASES: ", ret_f.shape[0])
		print("================================================")
		print("\tRND\tCNT\tMOD")

		positions = np.array(range(10)) + 1

		random_cols = [col for col in ret10.columns if 'rnd_pos_' in col]

		for p in positions:
			rnd = np.average(np.sum(ret_f[random_cols] == p))
			cnt = len(ret_f.loc[ret_f.cnt_pos == p])
			mdl = len(ret_f.loc[ret_f.model_pos == p])
			print("%d\t%f\t%d\t%d" % (p, rnd, cnt, mdl))

		print("-" * 100)

		# -----------------------------------------------------------------------------------------------------------
		# TOP (TIMES IN EACH POSITION)
		# -----------------------------------------------------------------------------------------------------------

		print("\tRND\tCNT\tMOD")

		positions = np.array(range(10)) + 1

		random_cols = [col for col in ret10.columns if 'rnd_pos_' in col]

		rnd = 0
		cnt = 0
		mdl = 0

		n_test_items = len(ret_f)

		for p in positions:
			rnd += np.average(np.sum(ret_f[random_cols] == p))*100
			cnt += len(ret_f.loc[ret_f.cnt_pos == p])*100
			mdl += len(ret_f.loc[ret_f.model_pos == p])*100
			print("%d\t%f\t%f\t%f" % (p, np.round(rnd/n_test_items,1), np.round(cnt/n_test_items,1), np.round(mdl/n_test_items,1)))

		print("-" * 100)

		ret.to_excel("docs/" + self.MODEL_NAME + "_" + self.CITY + "_TAKE.xlsx")  # Por usuario

	def get_sample_ranking(self):

		# -----------------------------------------------------------------------------------------------------------

		self.config_session()  # Configure and create session

		preds = self.test()
		urls = self.DATA["IMG"]
		print(urls.shape)
		original_train = get_pickle(self.DATA_PATH + "original_take/", "TRAIN_DEV")

		# Adding images to TRAIN set
		original_train = original_train.merge(self.DATA["IMG"][["review", "content_id"]], left_on="reviewId",
											  right_on="review")
		original_train = original_train.drop(columns=['review'])

		# -----------------------------------------------------------------------------------------------------------
		# Drop restaurants with less than 10 images in test (easy to guess)
		# -----------------------------------------------------------------------------------------------------------

		rsts = original_train.groupby("restaurant_id").content_id.count().reset_index(name="n_images")
		rsts = rsts.loc[rsts.n_images >= 10].restaurant_id.values
		preds = preds.loc[preds.restaurant_id.isin(rsts)]
		
		# randomtest=[random.choice(preds.groupby("id_test"))[0] for i in range(5)]
		# randomtest=[random.choice(preds.id_test.values) for i in range(5)]
		randomtest=[4517]
		# print(random.choice(preds.groupby("id_test").values))
		# -----------------------------------------------------------------------------------------------------------
		# Drop users with less than 10 images in train (poor knownledge about them)
		# -----------------------------------------------------------------------------------------------------------

		# usrs = original_train.groupby("user_id").content_id.count().reset_index(name="n_images")
		# usrs = usrs.loc[usrs.n_images>=10].user_id.values
		# preds = preds.loc[preds.user_id.isin(usrs)]

		for i, mdl in preds.groupby("id_test"):

			ids = range(0,12000)  # BARCELONA => 11537: 13588 2769 (Exteriores)
			if (i not in ids): continue
			print(mdl.user_id.values[0], mdl.restaurant_id.values[0])

			n_imgs = 11

			# Ordering the prediction of the particular restaurant
			mdl = mdl.sort_values("prediction", ascending=False).reset_index(drop=True)
			usr_train = original_train.loc[original_train.user_id == mdl.user_id.values[0]]
			usr_urls = urls.iloc[usr_train.content_id.unique(), :].url.values
			
			if len(usr_train)<8: continue
			# Order using the centroid
			cnt = self.centroid_baseline(restaurant=mdl.restaurant_id.values[0], id_test=i)

			# Obtaining the image urls
			mdl_urls = urls.iloc[mdl.content_id.values].url.values
			cnt_urls = urls.iloc[cnt.content_id.values].url.values

			mdl_pos = mdl.loc[mdl.is_dev == 1].index.values[0]
			cnt_pos = cnt.loc[cnt.is_dev == 1].index.values[0]

			# User data in TRAIN


			# Plot creation
			filas_usr = int(np.ceil(len(usr_urls) / n_imgs))
			filas_mdl = int(np.ceil(len(mdl_urls) / n_imgs))
			filas_cnt = int(np.ceil(len(cnt_urls) / n_imgs))

			fig, ax = plt.subplots(filas_usr + filas_mdl + filas_cnt, n_imgs,
								   figsize=(15, filas_usr + filas_mdl + filas_cnt + 2))

			usr_path = "imgs/%s/%d/user/" % (self.CITY, mdl.restaurant_id.values[0])
			top_model_path = "imgs/%s/%d/top-model/" % (self.CITY, mdl.restaurant_id.values[0])
			top_centr_path = "imgs/%s/%d/top-centroid/" % (self.CITY, mdl.restaurant_id.values[0])

			os.makedirs(usr_path, exist_ok=True)
			os.makedirs(top_model_path, exist_ok=True)
			os.makedirs(top_centr_path, exist_ok=True)

			# Positive user images in TRAIN
			for j in range(len(usr_urls)):
				f = int(np.ceil((j + 1) / n_imgs)) - 1

				img_url = usr_urls[j]
				img_path = usr_path + (str(j).zfill(3) + ".jpg")

				ax[f, j % n_imgs].imshow(get_img(img_url, save=img_path), cmap='seismic', interpolation='bilinear')

			# Model images
			for j in range(len(mdl_urls)):

				f = filas_usr + (int(np.ceil((j + 1) / n_imgs)) - 1)

				img_url = mdl_urls[j]
				img_path = top_model_path + (str(j + 1).zfill(3) + ".jpg")
				if j == mdl_pos: img_path = top_model_path + (str(j + 1).zfill(3) + "_dev.jpg")

				ax[f, j % n_imgs].imshow(get_img(img_url, save=img_path), cmap='seismic', interpolation='bilinear')

				if j == mdl_pos:
					ax[f, j % n_imgs].set_title('DEV', color="red")
				else:
					ax[f, j % n_imgs].set_title(str(j + 1))

			# Centroid images
			for j in range(len(cnt_urls)):

				f = filas_usr + filas_mdl + (int(np.ceil((j + 1) / n_imgs)) - 1)

				img_url = cnt_urls[j]
				img_path = top_centr_path + (str(j + 1).zfill(3) + ".jpg")
				if (j == cnt_pos): img_path = top_centr_path + (str(j + 1).zfill(3) + "_dev.jpg")

				ax[f, j % n_imgs].imshow(get_img(img_url, save=img_path), cmap='seismic', interpolation='bilinear')

				if (j == cnt_pos):
					ax[f, j % n_imgs].set_title('DEV', color="red")
				else:
					ax[f, j % n_imgs].set_title(str(j + 1))

			# Remove axis:
			[axi.set_axis_off() for axi in ax.ravel()]

			plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1, wspace=0.02, hspace=0.45)
			plt.show()

	def most_popular_in_rest(self, restaurant=0, which_users="all"):

		self.config_session()  # Configuring and creating session

		original_train = get_pickle(self.DATA_PATH + "original_take/", "TRAIN_DEV")

		# Adding images to TRAIN set
		original_train = original_train.merge(self.DATA["IMG"][["review", "content_id"]], left_on="reviewId",
											  right_on="review")
		original_train = original_train.drop(columns=['review'])

		# Obtaining restaurant data
		rst_data = original_train.loc[original_train.restaurant_id == restaurant]

		if "all" in which_users:
			users = list(range(self.DATA["N_USR"]))  # All the users
		elif "own" in which_users:
			users = rst_data.user_id.unique()  # Only restaurant users
		elif "pos" in which_users:
			users = rst_data.loc[
				rst_data.like == 1].user_id.unique()  # Only restaurant users with positive reviews
		elif "neg" in which_users:
			users = rst_data.loc[
				rst_data.like == 0].user_id.unique()  # Only restaurant users with negative reviews

		images = rst_data.content_id.unique()

		# Creation of the evaluation set (all users evaluate all images)
		data = pd.DataFrame()
		data["user_id"] = np.repeat(users, len(images))
		data["content_id"] = np.reshape([images] * len(users), -1)

		if not os.path.exists(self.MODEL_PATH): print_e("The model doesn't exist."); exit()
		model_take = load_model(self.MODEL_PATH + "/model_1")

		sequence = self.DevSequenceTake(data=data, batch_size=min(2 ** 14, len(data)), model=self)
		data["pred"] = model_take.predict_generator(sequence, steps=sequence.__len__())

		# Group by image and sum probabilities
		data_g = data.groupby("content_id").pred.mean().reset_index(name="pred")
		data_g = data_g.sort_values("pred", ascending=False).reset_index(drop=True)

		min_items = 30
		rows = 3

		if len(data_g) < min_items:
			print_e("Only %d photos. %d required." % (len(data_g), min_items))
			exit()

		best_img = data_g.content_id.values[0]
		best_url = self.DATA["IMG"].iloc[best_img].url

		popular_path = "imgs/%s/%d/popular_%s/" % (self.CITY, restaurant, which_users)
		os.makedirs(popular_path, exist_ok=True)

		# Plot
		fig, ax = plt.subplots(1, min(len(data_g), min_items), figsize=(18, 3))

		for i, r in data_g.iterrows():

			val = str(np.round(r.pred, decimals=3))

			img_url = self.DATA["IMG"].iloc[int(r.content_id)].url
			img_path = popular_path + (str(i).zfill(3) + "_[" + val + "].jpg")

			ax[i].imshow(get_img(img_url, save=img_path), cmap='seismic', interpolation='bilinear')
			ax[i].set_title(val)

			if (i + 1 == min_items): break

		# Remove the axis
		[axi.set_axis_off() for axi in ax.ravel()]

		plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1, wspace=0.02, hspace=0.0)
		plt.show()

	####################################################################################################################

	def config_session(self):
		# Configurar y crear sesion
		cfg = tf.compat.v1.ConfigProto()
		cfg.gpu_options.allow_growth = True
		sess = tf.compat.v1.Session(config=cfg)
		K.set_session(sess)

		self.SESSION = sess

	def get_data(self, load):

		if load is None:
			load = ["TRAIN_TXT", "TRAIN_DEV_TXT", "DEV_TXT", "TEST_TXT", "TXT", "N_USR", "V_TXT"]

		# Look if data already exists
		# --------------------------------------------------------------------------------------------------------------

		file_path = self.PATH + self.MODEL_NAME.upper()
		file_path += "/data_" + str(self.CONFIG['neg_images']) + "/"
		
		print(file_path)
		print(os.path.exists(file_path))

		if os.path.exists(file_path) and len(os.listdir(file_path)) == len(load):

			print_g("Loading previous generated data...")

			ret_dict = {}

			for d in load:
				if os.path.exists(file_path + d):
					ret_dict[d] = get_pickle(file_path, d)

			return ret_dict

		else:
			print_e("Data incomplete or nonexistent")
			exit()

	def get_data_stats(self):

		def get_numbers(data, title=""):
			print("%s & %d & %d & %d" % (title, len(data.user_id.unique()), len(data.restaurant_id.unique()),
										 data.num_images.sum()))

		file_path = self.PATH + self.MODEL_NAME.upper()
		split_file_path_take = file_path + "/original_take/"

		print("\n")
		print(self.CITY.upper())
		print("-" * 50)

		print("SET\t#USERS\t#RESTAURANTS\t#IMAGES")
		all_data = get_pickle(split_file_path_take, "TRAIN_TEST")
		get_numbers(all_data, title="TRAIN_TEST")
		get_numbers(get_pickle(split_file_path_take, "TRAIN_DEV"), title="TRAIN_DEV")
		get_numbers(get_pickle(split_file_path_take, "DEV"), title="DEV")
		get_numbers(get_pickle(split_file_path_take, "TEST"), title="TEST")

		print("-" * 50)

		save_path = "stats/" + self.CITY.lower() + "/"
		os.makedirs(save_path, exist_ok=True)

		# Review number of each user
		sts0 = all_data.groupby('user_id').apply(
			lambda x: pd.Series({"reviews": len(x.restaurant_id.unique())})).reset_index()
		plot_hist(sts0, "reviews", title="", title_x="Num. of reviews", title_y="Num. of users", bins=20,
				  save=save_path + self.CITY.lower() + "_hist_rvws_pr_usr.pdf")

		# Review number of each restaurant
		sts1 = all_data.groupby('restaurant_id').apply(
			lambda x: pd.Series({"reviews": len(x.user_id.unique())})).reset_index()
		plot_hist(sts1, "reviews", title="", title_x="Num. of reviews", title_y="Num. of restaurants", bins=20,
				  save=save_path + self.CITY.lower() + "_hist_rvws_pr_rst.pdf")

		# Image number of each review
		sts2 = all_data.groupby(['user_id', 'restaurant_id']).num_images.sum().reset_index(name="fotos")
		plot_hist(sts2, "fotos", title="", title_x="Num. of photos", title_y="Num. of reviews", bins=4,
				  save=save_path + self.CITY.lower() + "_hist_photos_pr_rvw.pdf")

		# Image number of each restaurant
		sts3 = all_data.groupby('restaurant_id').num_images.sum().reset_index(name="fotos")
		plot_hist(sts3, "fotos", title="", bins=20, title_x="Num. of photos", title_y="Num. of restaurants",
				  save=save_path + self.CITY.lower() + "_hist_photos_pr_rst.pdf")

		# Image number of each user
		sts4 = all_data.groupby('user_id').num_images.sum().reset_index(name="fotos")
		plot_hist(sts4, "fotos", title="", bins=20, title_x="Num. of photos", title_y="Num. of users",
				  save=save_path + self.CITY.lower() + "_hist_photos_pr_usr.pdf")

	def get_precision_at(self):

		data = pd.read_excel("docs/" + self.MODEL_NAME + "_" + self.CITY + "_TAKE.xlsx")  # Por usuario

		apply_filter = True
		if apply_filter:
			# Drop restaurants with less than 10 images in test (easy to guess)
			data = data.loc[(data["n_photos_x"] >= 10)]
			# Drop users with less than 10 images in train (poor knownledge about them)
			data = data.loc[(data["n_photos_train"] >= 10)]

		cols = ["model_pos", "cnt_pos", 'rnd_pos_0', 'rnd_pos_1', 'rnd_pos_2', 'rnd_pos_3', 'rnd_pos_4', 'rnd_pos_5',
				'rnd_pos_6', 'rnd_pos_7', 'rnd_pos_8', 'rnd_pos_9']
		ret = []

		for i in range(100):
			tm = [i + 1]
			for c in cols:
				times = len(data.loc[data[c] == i + 1])
				tm.append(times)
			ret.append(tm)

		n_cols = ["pos"]
		n_cols.extend(cols)

		ret = pd.DataFrame(ret)
		ret.columns = n_cols
		ret["rnd_pos"] = ret[
			['rnd_pos_0', 'rnd_pos_1', 'rnd_pos_2', 'rnd_pos_3', 'rnd_pos_4', 'rnd_pos_5', 'rnd_pos_6', 'rnd_pos_7',
			 'rnd_pos_8', 'rnd_pos_9']].mean(axis=1)
		ret["rnd_pos_std"] = ret[
			['rnd_pos_0', 'rnd_pos_1', 'rnd_pos_2', 'rnd_pos_3', 'rnd_pos_4', 'rnd_pos_5', 'rnd_pos_6', 'rnd_pos_7',
			 'rnd_pos_8', 'rnd_pos_9']].std(axis=1)

		ret = ret[["pos", "model_pos", "cnt_pos", "rnd_pos", "rnd_pos_std"]]

		ret["model_pos_p@"] = ret["model_pos"].cumsum()
		ret["cnt_pos_p@"] = ret["cnt_pos"].cumsum()
		ret["rnd_pos_p@"] = ret["rnd_pos"].cumsum()

		ret["model_pos_p@%"] = ret["model_pos_p@"] / len(data)
		ret["cnt_pos_p@%"] = ret["cnt_pos_p@"] / len(data)
		ret["rnd_pos_p@%"] = ret["rnd_pos_p@"] / len(data)

		ret.to_excel("docs/" + self.CITY + "_p@" + ("_all" if not apply_filter else "") + ".xlsx")
