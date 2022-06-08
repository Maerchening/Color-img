from PIL import Image
import os

path = "../resize"
os.chdir(path)
files = os.listdir(path)
i=0
for file in files:
    img = Image.open(file)
    #img_resize  = img.resize((256,256))
    img_resize = img.transpose(Image.FLIP_LEFT_RIGHT)
    name='../resize/' + str(i) + '.png'
    i += 1
    img_resize.save(name)





'''
imgGray = img_resize.convert('L')
imgGray.save('resize.png')
'''


'''
X = rgb2gray(image).reshape(1, 512, 512, 1)
Y = image.reshape(1, 512, 512, 3)

print(np.min(Y))
print(np.max(Y))
Y /= 128

X_test = rgb2gray(image_2).reshape(1,400, 400, 1)

model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))

model.compile(optimizer='rmsprop',loss='mse')
model.fit(x=X, 
    y=Y,
    batch_size=1,
    epochs=100000)
print(model.evaluate(X, Y, batch_size=1))
output = model.predict(X_test)
output *= 128

output = output.reshape(512, 512, 3)
imsave("img_result.png", output)
'''