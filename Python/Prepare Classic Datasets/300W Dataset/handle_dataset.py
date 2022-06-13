import os, PIL, math, time, wandb, torch, pickle, random, torchvision, faulthandler
import numpy as np
from PIL import Image
from skimage import io
from tqdm.auto import tqdm

###############
# Data Helper #
###############
# Augmentation for face
class FaceAugmentation:
    def __init__(self, image_dim, brightness,    
                 contrast, saturation, hue,
                 face_offset, crop_offset):
        
        self.image_dim = image_dim
        self.face_offset = face_offset
        self.crop_offset = crop_offset
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
    
    def offset_crop(self, image, landmarks, crops_coordinates):
        left = int(crops_coordinates['left']) - self.face_offset
        top = int(crops_coordinates['top']) - self.face_offset
        width = int(crops_coordinates['width']) + (2 * self.face_offset)
        height = int(crops_coordinates['height']) + (2 * self.face_offset)

        image = TF.crop(image, top, left, height, width)
        landmarks = landmarks - np.array([[left, top]])
        new_dim = self.image_dim + self.crop_offset
        image = TF.resize(image, (new_dim, new_dim))
        landmarks[:, 0] *= new_dim / width
        landmarks[:, 1] *= new_dim / height
        return image, landmarks
    
    def random_face_crop(self, image, landmarks):
        image = np.array(image)
        h, w = image.shape[:2]
        top = np.random.randint(0, h - self.image_dim)
        left = np.random.randint(0, w - self.image_dim)
        image = image[top: top + self.image_dim, left: left + self.image_dim]
        landmarks = landmarks - np.array([[left, top]])
        return TF.to_pil_image(image), landmarks
    
    def __call__(self, image, landmarks, crops_coordinates):
        image, landmarks = self.offset_crop(image, landmarks, crops_coordinates)
        image, landmarks = self.random_face_crop(image, landmarks)
        return self.transform(image), landmarks

# Augmentation for landmarks
class LandmarksAugmentation:
    def __init__(self, rotation_limit):
        self.rotation_limit = rotation_limit

    def random_rotation(self, image, landmarks):
        angle = np.random.uniform(-self.rotation_limit, self.rotation_limit)
        landmarks_transformation = np.array([
            [+np.cos(np.radians(angle)), -np.sin(np.radians(angle))], 
            [+np.sin(np.radians(angle)), +np.cos(np.radians(angle))]
        ])
        image = TF.rotate(image, angle)
        landmarks = landmarks - 0.5
        transformed_landmarks = np.matmul(landmarks, landmarks_transformation)
        transformed_landmarks = transformed_landmarks + 0.5

        return image, transformed_landmarks
    
    def __call__(self, image, landmarks):
        image, landmarks = self.random_rotation(image, landmarks)
        return image, landmarks

# Data Preprocessing
class Preprocessor:
    def __init__(self, image_dim, brightness, contrast, saturation,
                 hue, angle, face_offset, crop_offset):
        
        self.image_dim = image_dim
        self.landmarks_augmentation = LandmarksAugmentation(rotation_limit=angle)
        self.face_augmentation      = FaceAugmentation(image_dim, brightness, contrast, 
            saturation, hue, face_offset, crop_offset)
    
    def __call__(self, image, landmarks, crops_coordinates):
        image = TF.to_pil_image(image)
        image, landmarks = self.face_augmentation(image, landmarks, crops_coordinates)
        landmarks = landmarks / np.array([*image.size])
        image, landmarks = self.landmarks_augmentation(image, landmarks)
        image = TF.to_grayscale(image)
        image = TF.to_tensor(image)
        image = (image - image.min())/(image.max() - image.min())
        image = (2 * image) - 1
        return image, torch.FloatTensor(landmarks.reshape(-1) - 0.5)

# Creating the dataset class
class LandmarksDataset(Dataset):
    def __init__(self, preprocessor, root_dir, train):
        self.root_dir = root_dir
        
        self.image_paths = []
        self.landmarks = []
        self.crops = []
        self.preprocessor = preprocessor
        self.train = train
        
        tree = ElementTree.parse(os.path.join(self.root_dir, f'labels_ibug_300W_{"train" if train else "test"}.xml'))
        root = tree.getroot()
        
        for filename in root[2]:
            self.image_paths.append(os.path.join(self.root_dir, filename.attrib['file']))
            self.crops.append(filename[0].attrib)
            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')
        assert len(self.image_paths) == len(self.landmarks)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = io.imread(self.image_paths[index], as_gray = False)
        landmarks = self.landmarks[index]
        image, landmarks = self.preprocessor(image, landmarks, self.crops[index])
        return image, landmarks

class LandmarksTestset(Dataset):
    def __init__(self, preprocessor, root_dir, train):
        self.root_dir = root_dir
        
        self.image_paths = []
        self.landmarks = []
        self.crops = []
        self.preprocessor = preprocessor
        self.train = train
        
        #tree = ElementTree.parse(os.path.join(self.root_dir, f'orig_300W_test/{"common" if common else "challenging"}'))
        #tree = ElementTree.parse(os.path.join(self.root_dir, f'labels_ibug_300W_{"train" if train else "test"}.xml'))
        tree = ElementTree.parse(os.path.join(self.root_dir, f'labels_ibug_300W.xml'))
        root = tree.getroot()
     
        for filename in root[2]:
            img_name = filename.attrib['file']
            if "ibug/" in img_name:
            #if "lfpw/testset/" in img_name or "helen/testset/" in img_name:
                self.image_paths.append(os.path.join(self.root_dir, img_name))
                self.crops.append(filename[0].attrib)
                landmark = []
                for num in range(68):
                    x_coordinate = int(filename[0][num].attrib['x'])
                    y_coordinate = int(filename[0][num].attrib['y'])
                    landmark.append([x_coordinate, y_coordinate])
                self.landmarks.append(landmark)
               
        self.landmarks = np.array(self.landmarks).astype('float32')
        assert len(self.image_paths) == len(self.landmarks)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = io.imread(self.image_paths[index], as_gray = False)
        landmarks = self.landmarks[index]
        image, landmarks = self.preprocessor(image, landmarks, self.crops[index])
        return image, landmarks

#################
# How to use it #
#################

data_path = '/your_file_path/300W'
preprocessor = Preprocessor(image_dim = 128, brightness = 0.24, saturation = 0.3,
    contrast = 0.15, hue = 0.14, angle = 14, face_offset = 32, crop_offset = 16)

train_images = LandmarksDataset(preprocessor, root_dir = data_path, train = True)
test_images  = LandmarksDataset(preprocessor, root_dir = data_path, train = False)
    
