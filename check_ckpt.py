import torch
ckpt = torch.load("models/car_brand_classifier.pth", map_location="cpu")
print("architecture:", ckpt["architecture"])
print("image_size:", ckpt["image_size"])
print("val_accuracy:", ckpt["val_accuracy"])
names = ckpt["class_names"]
print(len(names), names)
print("Chevrolet index:", names.index("Chevrolet") if "Chevrolet" in names else "NOT IN LIST")
print("BYD index:", names.index("BYD") if "BYD" in names else "NOT IN LIST")