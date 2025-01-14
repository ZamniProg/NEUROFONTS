from imports import *
from annotations import Preprocess, COLOR
from tensorflow.keras import layers, models


class ComfortableUsage(Preprocess):
    """
    A user-friendly class that facilitates easy usage of dataset preprocessing.

    Inherits from Preprocess and provides methods for initializing and preprocessing datasets,
    as well as managing annotations and downloading datasets if necessary.
    """

    def __init__(self):
        """
        Initializes the ComfortableUsage class.

        Sets up the dataset and annotation paths, and initializes necessary flags.
        """
        super().__init__()
        self.dataset = None
        self._is_ok = True
        self._path_to_annotations = ''
        self.path_to_dataset = ''
        self.path_to_annotations = ''

    def initAll(self, new_path_to_dataset: str = './school_notebooks_RU',
                new_path_to_annotations: str = './school_notebooks_RU/annotations',
                replace_annotations: bool = False,
                download_dataset: bool = True,
                rework: bool = False) -> None:
        """
        Initializes the dataset and annotations for the project.

        This method handles downloading the dataset, loading it, replacing annotations if needed,
        and reworking the annotations. It also verifies that the dataset and annotations have been
        properly loaded.

        :param new_path_to_dataset: Path to the dataset directory.
        :param new_path_to_annotations: Path to the annotations directory.
        :param replace_annotations: Whether to replace the existing annotations with new ones.
        :param download_dataset: Whether to download the dataset.
        :param rework: Whether to rework existing annotations.
        """
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

            annotations = self.loadAnnotation(self.path_to_annotations, "test")

            if rework:
                for filename in os.listdir(self.path_to_annotations):
                    self.update_annotations(os.path.join(self.path_to_annotations, filename))

            if annotations is None:
                print(f"{COLOR().get('red')}[!] Error: annotations are not loaded, please retry! {COLOR().get()}")
                self._is_ok = False

            return None
        except Exception as e:
            print(f"{COLOR().get('red')}[!] Error: {e}{COLOR().get()}")
            return None

    def preprocessAnnotation(self, items: str = 'train', target_size: Tuple[int, int] = (250, 250)) \
            -> Union[Tuple, None]:
        """
        Preprocesses the annotations and images for a given dataset split (train/test).

        This method loads the annotations and preprocesses the images and bounding boxes
        according to the target size. It returns the preprocessed bounding boxes and images.

        :param items: The dataset split to preprocess (e.g., 'train' or 'test').
        :param target_size: The target size for the images after preprocessing (width, height).
        :return: A tuple containing preprocessed bounding boxes and images.
        """
        try:
            if not self._is_ok:
                raise EnvironmentError(f"{COLOR().get('red')}[!] Error: dataset is not loaded, please"
                                       f" use initAll() to load dataset{COLOR().get()}")

            annotation = self.loadAnnotation(self.path_to_annotations, items)
            bboxes, images = self.fullPreprocess(dataset=self.dataset, target_size=target_size, split=items)

            return bboxes, images
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
    usags.initAll(download_dataset=False, rework=False)
    usags.preprocessAnnotation(items='test')


if __name__ == '__main__':
    main()