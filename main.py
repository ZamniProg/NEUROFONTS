import os
import json
import shutil
import datasets
import numpy as np
import tensorflow as tf
from datasets import DatasetDict
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pycocotools.coco import COCO
from typing import Union, List, Tuple, Dict, Iterable
from tensorflow.keras import layers, models


class COLOR:
    """
    Class to handle colored text output.

    **get**: method that return the color
    """
    def __init__(self):
        self.__colors = {
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
            'empty': '\033[0m'
        }

    def get(self, color: str = 'empty'):
        """Get the escape sequence for a specific color."""
        return self.__colors.get(color, self.__colors['empty'])


class Dataset:
    """
    Class for handling dataset operations.

    **Methods:**
    \n- **datasetDownloader**: download dataset to user directory.
    \n- **datasetLoader**: load a dataset from path.
    """
    def __init__(self):
        pass

    @staticmethod
    def datasetDownloader(path: str = './school_notebooks_RU') -> None:
        """
        Downloads the dataset to a specified directory.

        :param path: Directory where the dataset will be saved.
        :return: None
        """
        try:
            dataset = datasets.load_dataset("ai-forever/school_notebooks_RU", trust_remote_code=True)
            if not os.path.exists(path):
                os.makedirs(path)
                print(f'{COLOR().get("green")}[OK]{COLOR().get()}'
                      f' New directory created by path: {path}')

            dataset.save_to_disk(path)

            print(f"{COLOR().get(color='green')}[OK]{COLOR().get()} "
                  f"Dataset has been download to path: "
                  f"{COLOR().get('magenta')}{path}{COLOR().get()}")

        except Exception as e:
            print(f"{COLOR().get('red')}[!] Error: {e}{COLOR().get()}")

    @staticmethod
    def datasetLoader(path: str = './school_notebooks_RU',
                      show_images: bool = True) -> Union[datasets.Dataset, datasets.DatasetDict]:
        """
        Load dataset from user directory

        :param show_images: bool var that used to show one image for every sets
        :param path: path to folder where dataset to exist
        :return: full dataset's with functional that writen in library "datasets"
        """
        try:
            dataset = datasets.load_from_disk(path)
            print(f'Structure of dataset: {dataset}')

            if show_images:
                print(f'{COLOR().get("green")}[OK]{COLOR().get()} Train check: {dataset["train"][0]}')
                plt.imshow(dataset["train"][0]['image'])
                plt.axis('off')
                plt.show()

                print(f'{COLOR().get("green")}[OK]{COLOR().get()} Test check: {dataset["test"][0]}')
                plt.imshow(dataset["test"][0]['image'])
                plt.axis('off')
                plt.show()

                print(f'{COLOR().get("green")}[OK]{COLOR().get()} Validation check: {dataset["validation"][0]}')
                plt.imshow(dataset["validation"][0]['image'])
                plt.axis('off')
                plt.show()
            else:
                print(f'{COLOR().get("green")}[OK]{COLOR().get()} Train check: {dataset["train"][0]}')
                print(f'{COLOR().get("green")}[OK]{COLOR().get()} Test check: {dataset["test"][0]}')
                print(f'{COLOR().get("green")}[OK]{COLOR().get()} Validation check: {dataset["validation"][0]}')

            return dataset

        except Exception as e:
            print(f'{COLOR().get("red")}[!] Error: {e}{COLOR().get()}')


class Annotation(Dataset):
    """
    Class for work with the annotations: folders clear, copying, rename and annotations read.

    **Methods:**
    \n- **clearFolder**: clearing folder.
    \n- **replaceAnnotations**: replace annotations files in new folder.
    \n- **renameAnnotations**: rename annotations files in the given order.
    \n- **getAnnotations_path**: return a path to annotations folder.
    \n- **readAnnotations**: reading annotation JSON-file.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def clearFolder(path_to_clear: str) -> None:
        """
        Clears the directory

        :param path_to_clear: path to directory for clear
        :return: None
        """
        try:
            if not os.path.exists(path_to_clear):
                print(f"{COLOR().get('red')}[!]{COLOR().get()} Path is doesn't exists")
                return

            for item in os.listdir(path_to_clear):
                item_path = os.path.join(path_to_clear, item)

                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"{COLOR().get('green')}[OK]{COLOR().get()} File deleted: {item_path}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"{COLOR().get('green')}[OK]{COLOR().get()} Folder deleted: {item_path}")

            print(f"{COLOR().get('green')}[OK]{COLOR().get()} Folder {path_to_clear} has been cleared.")
        except Exception as e:
            print(f"{COLOR().get('red')}[!] Error: {e}{COLOR().get()}")

    @staticmethod
    def replaceAnnotations(path_to_datafolder: str, path_to_save: str = '.\\school_notebooks_RU\\annotations') -> None:
        """
        Replaces annotations int first folder to second folder

        :param path_to_datafolder: path to directory for clear
        :param path_to_save: path to directory for save annotations
        :return: None
        """
        try:
            if not os.path.exists(path_to_datafolder):
                print(f"[!] Error: Datafolder is doesn't exist")
                raise ValueError("Datafolder doesn't exist")

            os.makedirs(path_to_save, exist_ok=True)

            for filename in os.listdir(path_to_datafolder):
                full_path = os.path.join(path_to_datafolder, filename)

                if os.path.isfile(full_path):
                    file_size = os.path.getsize(full_path)
                    if 280000 * 1024 > file_size > 1024 * 5:
                        shutil.copy(full_path, path_to_save)
                        print(f"File copied successfully from {full_path} to {path_to_save} with size: {file_size}")
        except Exception as e:
            print(f"[!] Error: {e}")

    @staticmethod
    def renameAnnotations(path_to_annotations: str) -> None:
        """
        Function for rename the annotations to more real names

        :param path_to_annotations: path to directory with annotations
        :return: None
        """
        try:
            if not os.path.exists(path_to_annotations):
                print(f"[!] Error: path is doesn't exist")
                raise ValueError("Path doesn't exist")

            dict_of_sizes = {}
            list_of_annotations = ['annotations_val.json', 'annotations_test.json', 'annotations_train.json']

            for filename in os.listdir(path_to_annotations):
                full_path = os.path.join(path_to_annotations, filename)
                dict_of_sizes[full_path] = os.path.getsize(full_path)

            sorted_dict = dict(sorted(dict_of_sizes.items(), key=lambda x: x[1]))

            for i, (key, _) in enumerate(sorted_dict.items()):
                if key != os.path.join(path_to_annotations, list_of_annotations[i]):
                    os.rename(key, os.path.join(path_to_annotations, list_of_annotations[i]))
                    print(f'renamed: {key} to {os.path.join(path_to_annotations, list_of_annotations[i])}')

        except Exception as e:
            print(f"[!] Error: {e}")

    @staticmethod
    def getAnnotationsPath() -> Union[str, None]:
        """
        Function for obtaining full path to annotations

        :return: path to annotations
        """
        try:
            base_cache_dir = os.getenv('USERPROFILE', '')
            if not base_cache_dir:
                raise EnvironmentError("Не удалось определить домашнюю директорию пользователя")

            cache_path = os.path.join(base_cache_dir, '.cache', "huggingface",
                                      "hub", "datasets--ai-forever--school_notebooks_RU", "blobs")

            second_path = os.path.join('./school_notebooks_RU', 'annotations')
            if os.path.exists(second_path) and 'annotations_test.json' in os.listdir(second_path):
                print(f"{COLOR().get('green')}[OK]{COLOR().get()} Annotation file been searched in main"
                      f" directory: {second_path}")
                return second_path

            if not os.path.exists(cache_path):
                raise FileNotFoundError(f'Папка с аннтоациями не найдена: {cache_path}')

            return cache_path
        except Exception as e:
            print(f"[!] Error: {e}")
            return None

    @staticmethod
    def loadAnnotation(annotation_folder_path: str, annotation: str = 'train') -> Union[COCO, None]:
        """
            Load COCO dataset annotations.

            :param annotation: name of annotation (train/test/val)
            :param annotation_folder_path: path to COCO annotations file
        """
        try:
            annot = COCO(os.path.join(annotation_folder_path, f'annotations_{annotation}.json'))
            return annot
        except Exception as e:
            print(f"Error loading COCO annotations: {e}")
            return None

    def getAnnotation(self):
        pass

class Preprocess(Annotation):
    """
    Class for data preprocessing.

    **Methods:**
    \n- **bbox**: preprocess for one bbox
    \n- **allBboxes**: preprocess for every bbox in tuple or list with annotations
    \n- **image**: preprocess for one image
    \n- **allImages**: preprocess for every image in tuple or list with annotations
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def getImagesGenerator(annotation: COCO, image_dir: str,
                           resize_to: Tuple[int, int] = (256, 256)) -> Iterable[Image.Image]:
        """
        Generator for reading and resizing images from the COCO dataset.

        :param annotation: COCO object
        :param image_dir: directory with images
        :param resize_to: target size for resizing images
        :yield: resized PIL.Image object
        """
        image_ids = annotation.getImgIds()
        for img_id in tqdm(image_ids, ascii=True, desc="Images reading"):
            img_info = annotation.loadImgs(img_id)[0]
            img_path = f"{image_dir}/{img_info['file_name']}"
            try:
                img = Image.open(img_path).convert('RGB')
                yield img.resize(resize_to)
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")

    @staticmethod
    def bbox(bbox: List[float],
             original_size: Tuple[int, int], target_size: Tuple[int, int]) -> Union[List[float], None]:
        """
        Bbox preprocessing

        :param bbox: list with coordinates (x_min, y_min, width, height)
        :param original_size: start size of image
        :param target_size: target size for image
        :return: new computed bbox
        """
        try:
            if len(bbox) != 4:
                raise ValueError("Bounding box should have exactly 4 elements.")

            x_min, y_min, width, height = bbox
            W_orig, H_orig = original_size
            W_new, H_new = target_size

            # Scale coordinates
            x_min_new = float(x_min * (W_new / W_orig))
            y_min_new = float(y_min * (H_new / H_orig))
            width_new = float(width * (W_new / W_orig))
            height_new = float(height * (H_new / H_orig))

            return [x_min_new, y_min_new, width_new, height_new]

        except Exception as e:
            print(f"Error: {e}")
            return None

    def allBboxes(self, annotation: COCO, target_size: Tuple[int, int]) -> Union[np.ndarray, None]:
        """
        Preprocessing for every bbox in COCO annotations.

        :param annotation: COCO object
        :param target_size: target size for images
        :return: np.NDArray with reworked bboxes
        """
        try:
            all_bboxes = []
            original_sizes = []

            for img_id in tqdm(annotation.getImgIds(), ascii=True, desc="Bboxes preprocessing"):
                img_info = annotation.loadImgs(img_id)[0]
                ann_ids = annotation.getAnnIds(imgIds=img_id)
                anns = annotation.loadAnns(ann_ids)

                original_size = (img_info['width'], img_info['height'])

                for ann in anns:
                    new_bbox = self.bbox(ann['bbox'], original_size, target_size)
                    if new_bbox is not None:
                        all_bboxes.append(new_bbox)
                        original_sizes.append(original_size)

            if not all_bboxes:
                print("Error: No valid bboxes were processed!")
                return None

            return np.array(all_bboxes, dtype=np.float32)

        except Exception as e:
            print(f"Error during bboxes preprocessing: {e}")
            return None

    @staticmethod
    def image(image_obj: Image.Image, target_size: Tuple[int, int]) -> Union[Tuple[Tuple[int, int], np.ndarray], None]:
        """
        Image preprocessing

        :param image_obj: PIL.Image object
        :param target_size: new size for image
        :return: tuple with original size of image and np.ndarray with normalized image
        """
        try:
            original_size = image_obj.size
            image = image_obj.resize(target_size).convert('L')
            image_array = np.array(image, dtype=np.float32) / 255.0

            return original_size, np.expand_dims(image_array, axis=-1)
        except Exception as e:
            print(f"Error: {e}")
            return None

    def allImages(self, image_generator: Iterable[Image.Image],
                  target_size: Union[Tuple[int, int], List[int]]) -> Union[
        Tuple[List[Tuple[int, int]], np.ndarray], None]:
        """
        Preprocessing for every image from a generator.

        :param image_generator: generator or iterable of PIL.Image objects
        :param target_size: new size for images (width, height)
        :return: tuple with list of original sizes and np.ndarray with processed images
        """
        try:
            processed_images = []
            sizes = []

            for img in tqdm(image_generator, ascii=True, desc="Images preprocessing"):
                if not isinstance(img, Image.Image):
                    print("Warning: Skipping invalid image object.")
                    continue

                result = self.image(img, target_size)
                if result is not None:
                    size, image = result
                    processed_images.append(image)
                    sizes.append(size)

            if not processed_images:
                print("Error: No valid images were processed!")
                return None

            return sizes, np.array(processed_images, dtype=np.float32)

        except Exception as e:
            print(f"Error during images preprocessing: {e}")
            return None

    def fullPreprocess(self, annotation: COCO, image_dir: str, target_size: Tuple[int, int]) -> Union[
        Tuple[np.ndarray, np.ndarray], None]:
        """
        Preprocessing for every image and bbox in COCO dataset.

        :param annotation: COCO object
        :param image_dir: directory with images
        :param target_size: target size for images
        :return: tuple with reworked bboxes and images
        """
        try:
            generator = self.getImagesGenerator(annotation, image_dir, resize_to=target_size)
            sizes, reworked_images = self.allImages(generator, target_size)
            reworked_bboxes = self.allBboxes(annotation, target_size)

            return reworked_bboxes, reworked_images
        except Exception as e:
            print(f"Error: {e}")
            return None

    # @staticmethod
    # def getImagesGenerator(dataset: Union[Dataset, DatasetDict],
    #                        type_of_images: str = 'train',
    #                        resize_to: Tuple[int, int] = (256, 256)) -> Iterable[Image.Image]:
    #     """
    #     Generator for reading and resizing images from the dataset.
    #
    #     :param dataset: dataset with images
    #     :param type_of_images: type of dataset split (e.g., train/val/test)
    #     :param resize_to: target size for resizing images
    #     :yield: resized PIL.Image object
    #     """
    #     try:
    #         for item in tqdm(dataset[type_of_images], ascii=True, desc="Images reading"):
    #             img = item.get('image')
    #             if img and isinstance(img, Image.Image):
    #                 yield img.resize(resize_to)
    #             else:
    #                 print(f"Warning: Skipping invalid image object: {item}")
    #     except KeyError as e:
    #         print(f"Error: Dataset type '{type_of_images}' not found: {e}")
    #     except Exception as e:
    #         print(f"Unexpected error while reading images: {e}")
    #
    # @staticmethod
    # def bbox(bbox: Union[List[float], Tuple[float, ...]],
    #          original_size: Tuple[int, int], target_size: Tuple[int, int]) -> Union[List[float], None]:
    #     """
    #     Bbox preprocessing
    #
    #     :param bbox: list or tuple with coordinates
    #     :param original_size: start size of image
    #     :param target_size: target size for image
    #     :return: new computed bbox
    #     """
    #     try:
    #         if len(bbox) != 4:
    #             raise ValueError(f"{COLOR().get('red')}[!]{COLOR().get()} "
    #                              f"Bounding box should have exactly 4 elements.")
    #
    #         x_min, y_min, x_max, y_max = bbox
    #         W_orig, H_orig = original_size
    #         W_new, H_new = target_size
    #
    #         # Проверка на деление на ноль
    #         if W_orig == 0 or H_orig == 0 or W_new == 0 or H_new == 0:
    #             raise ValueError(f"{COLOR().get('red')}[!]{COLOR().get()} "
    #                              f"Original or target size cannot have zero dimensions.")
    #
    #         # Масштабировать координаты
    #         x_min_new = float(x_min * (W_new / W_orig))
    #         y_min_new = float(y_min * (H_new / H_orig))
    #         x_max_new = float(x_max * (W_new / W_orig))
    #         y_max_new = float(y_max * (H_new / H_orig))
    #
    #         return [x_min_new, y_min_new, x_max_new, y_max_new]
    #
    #     except Exception as e:
    #         print(f"{COLOR().get('red')}[!] Error: {e}{COLOR().get()}")
    #         return None
    #
    # def allBboxes(self, bboxes: Union[List[List[float]], List[Tuple[float, ...]]],
    #               original_sizes: List[Tuple[int, int]], target_size: Tuple[int, int]) -> Union[np.ndarray, None]:
    #     """
    #     Preprocessing for every bbox in array.
    #
    #     :param bboxes: list with bboxes
    #     :param original_sizes: original image sizes
    #     :param target_size: target size for images
    #     :return: np.NDArray with reworked bboxes
    #     """
    #     try:
    #         if len(bboxes) != len(original_sizes):
    #             raise ValueError("Mismatch between number of bboxes and original sizes!")
    #
    #         all_bboxes = []
    #         for i, bbox in tqdm(enumerate(bboxes), ascii=True, desc="Bboxes preprocessing"):
    #             if len(bbox) != 4:
    #                 print(f"Warning: Skipping invalid bbox at index {i}: {bbox}")
    #                 continue
    #
    #             new_bbox = self.bbox(bbox, original_sizes[i], target_size)
    #             if new_bbox is not None:
    #                 all_bboxes.append(new_bbox)
    #             else:
    #                 print(f"Error: Failed to process bbox {i + 1}.")
    #
    #         if not all_bboxes:
    #             print("Error: No valid bboxes were processed!")
    #             return None
    #
    #         return np.array(all_bboxes, dtype=np.float32)
    #
    #     except Exception as e:
    #         print(f"Error during bboxes preprocessing: {e}")
    #         return None
    #
    # @staticmethod
    # def image(image_obj: Image.Image, target_size: Union[Tuple[int, int], List[int]]) -> Union[
    #     Tuple[Tuple[int, int], np.ndarray], None]:
    #     """
    #     Image preprocessing
    #
    #     :param image_obj: PIL.Image object
    #     :param target_size: new size for image
    #     :return: tuple with original size of image and np.ndarray with normalized image
    #     """
    #     try:
    #         if len(target_size) != 2:
    #             raise ValueError(f"{COLOR().get('red')}[!] Error: len of target size is not equals 2!{COLOR().get()}")
    #
    #         # Ensure the image is in grayscale mode
    #         image = image_obj.convert('L')
    #
    #         original_size = image.size
    #         image = image.resize(target_size)
    #         image_array = np.array(image, dtype=np.float32) / 255.0
    #
    #         return original_size, np.expand_dims(image_array, axis=-1)
    #     except Exception as e:
    #         print(f"{COLOR().get('red')}[!] Error: {e}{COLOR().get()}")
    #         return None
    #
    #
    # def fullPreprocess(self, dataset: Union[Dataset, DatasetDict],
    #                    bboxes: Union[List[List[float]], List[Tuple[float, ...]]],
    #                    target_size: Tuple[int, int],
    #                    type_of_data: str) -> Union[Tuple[NDArray, NDArray], None]:
    #     """
    #     Preprocessing for every image and bbox
    #
    #     :param dataset:
    #     :param type_of_data:
    #     :param bboxes: list with bboxes
    #     :param target_size: target size for images
    #     :return: tuple with reworked bboxes and images
    #     """
    #     try:
    #         if len(target_size) != 2:
    #             raise f"{COLOR().get('red')}[!] Error: len of target size is not equals 2!{COLOR().get()}"
    #
    #         generator = self.getImagesGenerator(dataset, type_of_data)
    #         start_sizes, reworked_images = self.allImages(generator, target_size)
    #         reworked_bboxes = self.allBboxes(bboxes, start_sizes, target_size)
    #
    #         return reworked_bboxes, reworked_images
    #     except Exception as e:
    #         print(f"{COLOR().get('red')}[!] Error: {e}{COLOR().get()}")
    #         return None


class ComfortableUsage(Preprocess):
    """
    Class for user-friendly usage
    """
    def __init__(self):
        super().__init__()
        self.dataset = None
        self._is_ok = True
        self._path_to_annotations = ''
        self.path_to_dataset = ''
        self.path_to_annotations = ''

    def initAll(self, new_path_to_dataset: str = './school_notebooks_RU',
                    new_path_to_annotations: str = './school_notebooks_RU/annotations',
                    replace_annotations: bool = False,
                    download_dataset: bool = True) -> None:
        if download_dataset:
            self.datasetDownloader(new_path_to_dataset)

        self.dataset = self.datasetLoader(new_path_to_dataset, show_images=False)

        if self.dataset is None:
            print(f"{COLOR().get('red')}[!] Error: dataset is not loaded, please retry! {COLOR().get()}")
            self._is_ok = False

        try:
            self.path_to_dataset = new_path_to_dataset
            self.path_to_annotations = self.getAnnotationsPath()

            if replace_annotations:
                self.replaceAnnotations(self.path_to_annotations, new_path_to_annotations)
                self.clearFolder(self.path_to_annotations)
                self.path_to_annotations = new_path_to_annotations

            self.renameAnnotations(self.path_to_annotations)

            annotations = self.loadAnnotation(self.path_to_annotations,
                                              "test")

            if annotations is None:
                print(f"{COLOR().get('red')}[!] Error: annotations is not been readed, please retry! {COLOR().get()}")
                self._is_ok = False

            return None
        except Exception as e:
            print(f"{COLOR().get('red')}[!] Error: {e}{COLOR().get()}")
            return None

    def preprocessAnnotation(self, items: str = 'train'): #-> Union[NDArray, None]:
        try:
            if not self._is_ok:
                raise EnvironmentError(f"{COLOR().get('red')}[!] Error: dataset is not loaded, please"
                                       f" use initAll() to load dataset{COLOR().get()}")

            self.loadAnnotation(self.path_to_annotations, items)


        except Exception as e:
            print(f"{COLOR().get('red')}[!] Error: {e}{COLOR().get()}")
            return None



class WordsRecognizeNetwork:
    def __init__(self, input_shape: tuple[int, int, int] = (250, 250, 1), conv_filter_size: int = 32,
                 kernel_size: tuple[int, int] = (3, 3), pool_size: tuple[int, int] = (2, 2)):
        self.model = models.Sequential([
            # Start layer
            layers.Input(shape=input_shape),

            # Layers for image recognize
            layers.Conv2D(conv_filter_size, kernel_size, activation='relu'),
            layers.MaxPooling2D(pool_size=pool_size),
            layers.Dropout(0.2),
            layers.Conv2D(conv_filter_size * 2, kernel_size, activation='relu'),
            layers.MaxPooling2D(pool_size=pool_size),
            layers.Dropout(0.3),
            layers.Conv2D(conv_filter_size * 4, kernel_size, activation='relu'),
            layers.MaxPooling2D(pool_size=pool_size),
            layers.Dropout(0.4),

            # Layer for Flatten
            layers.Flatten(),

            layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dropout(0.5),

            # End layer for coordinates predicts (like: normalized [x_min, y_min, x_max, y_max])
            layers.Dense(4, activation='sigmoid')
        ])

    def train(self, images_labels: tuple, epochs: int=100):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.Huber(),
                           metrics=['accuracy'])
        self.model.fit(images_labels[0], images_labels[1], epochs=epochs)

    def test(self, images_labels: tuple):
        test_loss, test_acc = self.model.evaluate(images_labels[0], images_labels[1], verbose=2)
        print(f"Test loss: {test_loss}\nTest accuracy: {test_acc}")


def main():
    usags = ComfortableUsage()
    usags.initAll(download_dataset=False)
    test = usags.getAnnotation()
    print(test['images'][0].keys())

if __name__ == '__main__':
    main()