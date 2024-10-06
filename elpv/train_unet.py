from utils.elpv_reader import load_dataset

images, proba, types = load_dataset()

print(images.shape)
print(images.dtype)
print(proba.shape)
print(proba.dtype)
print(types.shape)
print(types[0])