import numpy as np
from imutils import paths
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.applications import ResNet50
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator


def plot_training(H, N, plot_path):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	# plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plot_path)


train_path = '../../../../Datasets/Birds/training'
test_path = '../../../../Datasets/Birds/test'
batch_size = 32
nclasses = 200

total_train = len(list(paths.list_images(train_path)))
total_test = len(list(paths.list_images(test_path)))

train_aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

test_aug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
train_aug.mean = mean
test_aug.mean = mean

train_gen = train_aug.flow_from_directory(
	train_path,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=batch_size)

test_gen = test_aug.flow_from_directory(
	test_path,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=batch_size)

base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

head_model = base_model.output
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(2048, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(nclasses, activation="softmax")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

for layer in base_model.layers:
	layer.trainable = False

print(">>> Compiling model...")
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print(">>> Training head...")
H = model.fit_generator(train_gen, steps_per_epoch=total_train // batch_size, epochs=50)

print("[INFO] evaluating after fine-tuning network head...")
test_gen.reset()
pred_idxs = model.predict_generator(test_gen, steps=(total_test // batch_size) + 1)
pred_idxs = np.argmax(pred_idxs, axis=1)
print(classification_report(test_gen.classes, pred_idxs, target_names=test_gen.class_indices.keys()))
plot_training(H, 500, 'results_frozen.png')

train_gen.reset()
for layer in base_model.layers[15:]:
	layer.trainable = True

for layer in base_model.layers:
	print("{}: {}".format(layer, layer.trainable))

print("[INFO] re-compiling model...")
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit_generator(train_gen, steps_per_epoch=total_train // batch_size, epochs=20)

print("[INFO] evaluating after fine-tuning network...")
test_gen.reset()
pred_idxs = model.predict_generator(test_gen, steps=(total_test // batch_size) + 1)
pred_idxs = np.argmax(pred_idxs, axis=1)
print(classification_report(test_gen.classes, pred_idxs, target_names=test_gen.class_indices.keys()))
plot_training(H, 200, 'results_unfrozen.png')

print("[INFO] serializing network...")
model.save('fn_birds_model.h5')
