import os
import shutil
# datset_dir =  C:\Users\Avinash\Desktop\New folder\OCR\dataset
def train_reformat(dataset_dir,image_limiter):
    
    train_dir = os.path.join(dataset_dir,'train')
    os.mkdir(train_dir)
    dataset_dir = os.path.join(dataset_dir,r'by_class')
    
    for classes in os.scandir(dataset_dir):
        class_file = os.path.join(train_dir,classes.name)
        os.mkdir(class_file)
        for train in os.scandir(classes.path):
            if('train' in train.name):
                count=0
                for img in os.listdir(train.path):
                    if(count==image_limiter):
                        break
                    shutil.copy(os.path.join(train.path,img),class_file)
                    count+=1
def test_reformat(dataset_dir,image_limiter):
    test_dir = os.path.join(dataset_dir,'test')
    os.mkdir(test_dir)
    dataset_dir = os.path.join(dataset_dir,'by_class')

    for classes in os.scandir(dataset_dir):
        class_file = os.path.join(test_dir,classes.name)
        os.mkdir(class_file)
        for test in os.scandir(classes.path):
            if('hsf' in test.name and test.is_dir()):
                count = 0
                for img in os.listdir(test.path):
                    if(count==image_limiter or not img.endswith('.png')):
                        break
                    shutil.copy(os.path.join(test.path,img),class_file)
                    count+=1

if __name__=='__main__':
    dataset_dir = r'C:\Users\Avinash\Desktop\New folder\OCR\dataset'
    train,test = int(input("Enter training images limiter")), int(input("Enter testing images limiter"))
    train_reformat(dataset_dir,train)
    test_reformat(dataset_dir,test)         