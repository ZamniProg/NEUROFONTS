import torch

from imports import *


class COLOR:
    """
    A utility class for handling colored text output in the terminal.

    Provides ANSI escape sequences for different colors to format text with colors.

    **Available Colors:**
    \n- `red`: Red color.
    \n- `green`: Green color.
    \n- `yellow`: Yellow color.
    \n- `blue`: Blue color.
    \n- `magenta`: Magenta color.
    \n- `cyan`: Cyan color.
    \n- `white`: White color.
    \n- `empty`: Reset to default terminal color (no color).

    \n**Methods:**
    - **get**: Retrieve the ANSI escape sequence for the specified color.
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
        """
        Retrieve the ANSI escape sequence for a specified color.

        :param color: The name of the color. Defaults to 'empty' (no color).
        :return: The ANSI escape code for the color or the reset code if the color is not found.
        """
        return self.__colors.get(color, self.__colors['empty'])


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Dataset:
    """
    Class for managing dataset operations including downloading and loading datasets.

    \n**Methods:**
    \n- **datasetDownloader(path)**: Downloads a dataset to the specified directory.
    \n- **datasetLoader(path, show_images)**: Loads a dataset from the specified directory and optionally displays images.
    """
    def __init__(self):
        pass

    @staticmethod
    def datasetDownloader(path: str = './school_notebooks_RU') -> None:
        """
        Download a dataset from Hugging Face and save it to the specified directory.

        :param path: Directory where the dataset will be downloaded. Defaults to './school_notebooks_RU'.
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
        Load a dataset from the specified directory. Optionally displays sample images.

        :param path: Directory where the dataset is stored. Defaults to './school_notebooks_RU'.
        :param show_images: If True, displays sample images from the train, test, and validation sets.
        :return: Loaded dataset object (Dataset or DatasetDict).
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
    A class for managing annotation files and directories. This includes tasks such as clearing folders, copying
    annotations, renaming files, and reading or modifying annotation JSON files.

    **Methods:**
    \n- **clearFolder**: Clears the contents of a specified folder.
    \n- **replaceAnnotations**: Copies annotation files from one folder to another.
    \n- **renameAnnotations**: Renames annotation files in a specified directory to predefined names.
    \n- **getAnnotationsPath**: Returns the path to the folder containing annotations.
    \n- **loadAnnotation**: Loads COCO dataset annotations from a file.
    \n- **update_annotations**: Adds unique IDs to annotations in a JSON file.
    \n- **getAnnotation**: Reads and returns the contents of an annotation JSON file.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def clearFolder(path_to_clear: str) -> None:
        """
        Clears the contents of the specified directory by deleting all files and subdirectories.

        :param path_to_clear: Path to the directory that needs to be cleared.
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
        Copies annotation files from a source folder to a target folder. Only files within a specific size range are copied.

        :param path_to_datafolder: Path to the source directory containing annotation files.
        :param path_to_save: Path to the target directory where files will be saved.
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
        Renames annotation files in the given folder to predefined names based on their sizes.

        :param path_to_annotations: Path to the directory containing annotation files.
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
        Returns the path to the directory containing annotations. Checks both the cache directory and a local folder.

        :return: Path to the annotations folder if found, otherwise None.
        """
        try:
            base_cache_dir = os.getenv('USERPROFILE', '')
            if not base_cache_dir:
                raise EnvironmentError("Не удалось определить домашнюю директорию пользователя")

            cache_path = os.path.join(base_cache_dir, '.cache', "huggingface",
                                      "hub", "datasets--ai-forever--school_notebooks_RU", "blobs")

            second_path = os.path.join('../school_notebooks_RU', 'annotations')
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
        Loads COCO annotations from a specified file.

        :param annotation_folder_path: Path to the folder containing annotation files.
        :param annotation: Name of the annotation file to load (e.g., 'train', 'test', 'val').
        :return: COCO object if successful, None otherwise.
        """
        try:
            annot = COCO(os.path.join(annotation_folder_path, f'annotations_{annotation}.json'))
            return annot
        except Exception as e:
            print(f"Error loading COCO annotations: {e}")
            return None

    @staticmethod
    def update_annotations(annotation_path):
        """
        Updates the annotations file by adding a unique ID to each annotation.

        :param annotation_path: Path to the annotation JSON file to update.
        :return: None
        """
        try:
            # Загружаем JSON
            with open(annotation_path, 'r') as f:
                data = json.load(f)

            # Добавляем уникальный ID для каждой аннотации
            for i, annotation in enumerate(data['annotations']):
                annotation['id'] = i + 1

            # Сохраняем обновленный JSON
            with open(annotation_path, 'w') as f:
                json.dump(data, f, indent=4)

            print(f"Annotation: {annotation_path} updated successfully!")
        except Exception as e:
            print(f"Error updating annotations: {e}")

    @staticmethod
    def getAnnotation(path_to_annotation: str = 'school_notebooks_RU/annotations/annotations_test.json'):
        """
        Reads and returns the contents of an annotation JSON file.

        :param path_to_annotation: Path to the annotation file to read.
        :return: Parsed JSON data as a dictionary.
        """
        with open(path_to_annotation, 'r') as f:
            file = json.load(f)
            return file


class Preprocess(Annotation):
    """
    Class for data preprocessing.

    This class provides methods for preprocessing images and bounding boxes, which are common tasks
    in computer vision datasets. It supports reading and resizing images, calculating bounding boxes
    from segmentation data, and preprocessing multiple images and bounding boxes at once.

    **Methods:**
    \n- **bbox**: Preprocess a single bounding box from segmentation coordinates.
    \n- **allBboxes**: Preprocess all bounding boxes in a dataset (e.g., COCO annotations).
    \n- **image**: Preprocess a single image, including resizing and normalization.
    \n- **allImages**: Preprocess multiple images, including resizing and normalization.
    """

    def __init__(self):
        """
        Initializes the Preprocess class.

        Inherits from Annotation and prepares the methods for handling preprocessing of images and bounding boxes.
        """
        super().__init__()

    @staticmethod
    def getImagesGenerator(dataset: Dataset, resize_to: Tuple[int, int] = (256, 256)) -> Iterable[Image.Image]:
        """
        Generator for reading and resizing images from a Hugging Face Dataset.

        This method iterates over the dataset and resizes each image to the specified size.

        :param dataset: Dataset object containing images.
        :param resize_to: Target size for resizing images.
        :yield: Resized PIL.Image object.
        """
        try:
            for item in dataset:
                img = item.get('image')
                if isinstance(img, Image.Image):
                    yield img.resize(resize_to)
                else:
                    print(f"Warning: Skipping invalid image object: {item}")
        except Exception as e:
            print(f"Error while reading images: {e}")

    @staticmethod
    def bbox(segmentation: List[List[float]],
                               original_size: Tuple[int, int],
                               target_size: Tuple[int, int]) -> Union[List[float], None]:
        """
        Compute a bounding box from segmentation coordinates.

        This method calculates the bounding box coordinates from a list of segmentation points,
        and scales it to match the target image size.

        :param segmentation: List of points [[x1, y1, x2, y2, ...], ...] representing the segmentation.
        :param original_size: Original size of the image (width, height).
        :param target_size: Target size of the image (width, height).
        :return: List with bounding box [x_min, y_min, width, height].
        """
        try:
            # Flatten the list of coordinates
            flat_points = [coord for segment in segmentation for coord in segment]

            if len(flat_points) % 2 != 0:
                raise ValueError("Segmentation coordinates should be in pairs of (x, y).")

            x_coords = flat_points[::2]  # Extract x coordinates
            y_coords = flat_points[1::2]  # Extract y coordinates

            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            # Original and target sizes
            W_orig, H_orig = original_size
            W_new, H_new = target_size

            # Scale coordinates
            x_min_new = float(x_min * (W_new / W_orig))
            y_min_new = float(y_min * (H_new / H_orig))
            x_max_new = float(x_max * (W_new / W_orig))
            y_max_new = float(y_max * (H_new / H_orig))

            return [x_min_new, y_min_new, x_max_new, y_max_new]

        except Exception as e:
            print(f"Error: {e}")
            return None

    def allBboxes(self, annotation: COCO, target_size: Tuple[int, int]) -> Union[Tuple, None]:
        """
        Preprocess every bounding box in COCO annotations.

        This method processes all bounding boxes for a given annotation object (e.g., COCO annotations),
        resizes them, and returns them in a structured format.

        :param annotation: COCO object containing annotation data.
        :param target_size: Target size for images.
        :return: A tensor with reworked bounding boxes.
        """
        try:
            all_bboxes = []
            one_bbox = []
            image_id_prev = None

            for img_id in tqdm(annotation.getImgIds(), desc="Bboxes preprocessing"):
                img_info = annotation.loadImgs(img_id)[0]
                ann_ids = annotation.getAnnIds(imgIds=img_id)
                anns = annotation.loadAnns(ann_ids)

                original_size = (img_info['width'], img_info['height'])

                for ann in anns:
                    new_bbox = self.bbox(ann['segmentation'], original_size, target_size)
                    translation = ann.get('attributes', {}).get('translation', None)
                    image_id = ann['image_id']

                    if new_bbox is not None:
                        if image_id == image_id_prev:
                            one_bbox.append((new_bbox[0], new_bbox[1],
                                             new_bbox[2], new_bbox[3],
                                             translation, image_id))
                        else:
                            if one_bbox:
                                all_bboxes.append(one_bbox.copy())
                            one_bbox = [(new_bbox[0], new_bbox[1],
                                         new_bbox[2], new_bbox[3],
                                         translation, image_id)]
                            image_id_prev = image_id

            if one_bbox:
                all_bboxes.append(one_bbox)

            if not all_bboxes:
                print("Error: No valid bboxes were processed!")
                return None

            # Find the maximum number of bboxes in any group
            max_bboxes = max(len(group) for group in all_bboxes)

            bboxes_tensor = []
            translations = []
            image_ids = []

            for group in all_bboxes:
                bbox_group = []
                for bbox in group:
                    x_min, y_min, x_max, y_max, translation, image_id = bbox
                    bbox_group.append([x_min, y_min, x_max, y_max])
                    translations.append(translation)
                    image_ids.append(image_id)

                # Pad the bbox group to match the max number of bboxes
                while len(bbox_group) < max_bboxes:
                    bbox_group.append([0.0, 0.0, 0.0, 0.0])  # Padding with zeros

                # Add confidence (1) as the last element for each bbox
                for i in range(len(bbox_group)):
                    bbox_group[i].append(1.0)  # Confidence = 1 for each box

                bboxes_tensor.append(torch.tensor(bbox_group, dtype=torch.float32))

            print("BBoxes: ", len(bboxes_tensor), len(translations), len(image_ids))

            return bboxes_tensor, translations, image_ids

        except Exception as e:
            print(f"Error during bboxes preprocessing: {e}")
            return None

    @staticmethod
    def image(image_obj: Image.Image, target_size: Tuple[int, int]) -> Union[
        Tuple[Tuple[int, int], torch.Tensor], None]:
        try:
            original_size = image_obj.size
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(target_size),
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
            ])
            image_tensor = transform(image_obj)
            return original_size, image_tensor
        except Exception as e:
            print(f"Error: {e}")
            return None

    def allImages(self, image_generator: Iterable[Image.Image],
                  target_size: Union[Tuple[int, int], List[int]]) -> Union[
        Tuple[List[Tuple[int, int]], torch.Tensor], None]:
        """
        Preprocess every image from a generator.

        This method processes multiple images by resizing and normalizing them, and returns the results.

        :param image_generator: Iterable of PIL.Image objects.
        :param target_size: New size for the images (width, height).
        :return: A tuple with a list of original sizes and a numpy array with processed images.
        """
        try:
            processed_images = []
            sizes = []

            for img in tqdm(image_generator, desc="Images preprocessing", unit="image"):
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
            print(len(sizes), len(processed_images))
            # Использование torch.stack для объединения тензоров в один
            return sizes, torch.stack(processed_images).float()

        except Exception as e:
            print(f"Error during images preprocessing: {e}")
            return None

    def fullPreprocess(self, dataset: DatasetDict, target_size: Tuple[int, int],
                       annotations_folder: str = 'school_notebooks_RU/annotations',
                       split: str = 'train') -> Union[
        Tuple[torch.types.Tensor, torch.types.Tensor, List, List], None]:
        """
        Full preprocessing for every image and bounding box in the dataset.

        This method preprocesses both the images and bounding boxes from a given dataset split.

        :param dataset: DatasetDict object containing datasets for train/validation/test splits.
        :param target_size: Target size for images.
        :param split: The split of the dataset to preprocess (e.g., 'train', 'test', 'validation').
        :return: A tuple with processed bounding boxes and images.
        """
        try:
            if split not in dataset:
                raise ValueError(f"Split '{split}' not found in DatasetDict. Available splits: {list(dataset.keys())}")

            # Генератор изображений из выбранного сплита
            generator = self.getImagesGenerator(dataset[split], resize_to=target_size)

            # Предобработка изображений
            sizes, reworked_images = self.allImages(generator, target_size)

            annot = self.loadAnnotation(annotations_folder, split)

            original_sizes = [item['image'].size for item in dataset[split]]
            reworked_bboxes, translations, images_ids = self.allBboxes(annot, target_size)

            return reworked_bboxes, reworked_images, translations, images_ids
        except Exception as e:
            print(f"Error: {e}")
            return None