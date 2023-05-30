from  package import *

if __name__ == "__main__":
    image_path = "../datasets/vote/images"
    image_list = os.listdir(image_path)

    train, val = train_test_split(image_list, test_size=0.2, random_state=74)

    for img in train:
        name = img.split(".jpg")[0]
        shutil.copy(f"../datasets/vote/images/{name}.jpg", f"../datasets/vote/train/images/{name}.jpg")
        shutil.copy(f"./datasets/vote/labels/{name}.txt", f"../datasets/vote/train/labels/{name}.txt")

    for img in val:
        name = img.split(".jpg")[0]
        shutil.copy(f"../datasets/vote/images/{name}.jpg", f"../datasets/vote/val/images/{name}.jpg")
        shutil.copy(f"../datasets/vote/labels/{name}.txt", f"../datasets/vote/val/labels/{name}.txt")