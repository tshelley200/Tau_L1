from utils import *

seed = 1111

data_dir = "../data/merged/"
model_out = "../models/mar6/deepset_track_extra_endcap.h5"
track_extra = True
calo_extra = False
hgcal_extra = False
test_frac = 0.3
pt_cut = 15.
pt_norm = False
pt_weights = True
eta_min = 1.5
eta_max = 4.0
load = False
#model_fn = class_model
model_fn = class_model_quant
#model_fn = cnn
#model_fn = deepset_cnn


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.config.experimental.set_virtual_device_configuration(
				gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
	except RuntimeError as e:
		print(e)

random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)


data = create_train_test_data(data_dir, train=True, pt_cut = pt_cut, pt_norm = pt_norm,
        test_frac = test_frac, eta_min = eta_min)

X_train, y_train_jetID, y_train_pt_regress, pt  = data['X'], data['Y_jetID'], data['Y_pt_corr'], data['reco_pt']
y_train = y_train_jetID


#X_train_calo = data['ecal_extra'].reshape(y_train_jetID.shape[0], 10, -1)
#X_train_calo = X_train_calo[:,:,:len(hgcal_feat_names)]
#X_train = X_train_calo

#X_train = data['track_extra']

#X_train_calo = data['ecal_extra'].reshape(-1, 10, 3, 15)
#X_train = data['hcal_extra'].reshape(-1, 10, 3, 3)
#X_train = [X_train, X_train_calo]

if(track_extra):
    print(X_train.shape, data['track_extra'].shape)
    X_train = np.concatenate((X_train, data['track_extra']), axis = -1)
if(calo_extra):
    X_train = np.concatenate((X_train, data['ecal_extra'], data['hcal_extra']), axis = -1)
if(hgcal_extra):
    X_train = np.concatenate((X_train, data['ecal_extra']), axis = -1)

#Each candidate as a channel
if(model_fn == class_model_quant): 
    print("Deepset model!")
    print(X_train.shape)
elif(model_fn == cnn):
    print("CNN model!")
    print(X_train.shape)
elif(model_fn == deepset_cnn):
    #X_train_calo = data['ecal_extra'].reshape(-1, 10, 3, 15)
    X_train_calo = data['hcal_extra'].reshape(-1, 10, 3, 3)
    X_train = [X_train, X_train_calo]
else:
    print("Dense model")
    X_train = X_train.reshape(X_train.shape[0], -1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    joblib.dump(scaler, model_out.replace(".h5", "_scaler.pkl"))

#X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=123)

#reweight so signal equal contribution in training

if(pt_weights):
	class_labels = {
		'bkg' : 0,
		'sig' : 1,
	}
	weights = train_weights(y_train, pt, class_labels)
else:
	weights = np.ones_like(y_train)
	nSig = np.sum(y_train)
	nBkg = y_train.shape[0] - nSig
	print("Sig rw %.2f" % (nBkg/nSig))

	weights[y_train > 0.9] = nBkg / nSig


batch_size = 256
epochs = 50
validation_split = 0.2 # 20% of training set will be used for validation set. 



if(load):
    model = load_qmodel(model_out)
else:
    if(model_fn == deepset_cnn): 
        model = model_fn(point_shape = X_train[0].shape[1:], cnn_shape = X_train_calo.shape[1:])
    else: 
        model = model_fn(input_shape = X_train.shape[1:])

opt = optimizers.Adam()
model.compile(optimizer = opt, loss='binary_crossentropy', weighted_metrics = ['accuracy', 'AUC'])
model.summary()



callbacks = [ tf.keras.callbacks.EarlyStopping(verbose = 1, patience=10, restore_best_weights = True, monitor = 'val_loss', min_delta=1e-4),
              tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
            ]

history = model.fit(X_train, y_train, 
        sample_weight = weights,
        epochs=epochs,
        batch_size = batch_size,
        verbose =1, 
        validation_split = validation_split, 
        callbacks = callbacks, 
        shuffle = True)

model.save(model_out)
