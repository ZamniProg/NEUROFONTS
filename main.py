from imports import *
from annotations import *

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
            bboxes, images, translations, ids = self.fullPreprocess(dataset=self.dataset, target_size=target_size, split=items)

            return bboxes, images, translations, ids
        except Exception as e:
            print(f"{COLOR().get('red')}[!] Error: {e}{COLOR().get()}")
            return None


class BboxModel(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int] = (250, 250, 1), conv_filter_size: int = 32,
                 kernel_size: tuple[int, int] = (3, 3), pool_size: tuple[int, int] = (2, 2),
                 num_anchors: int = 9):
        super(BboxModel, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], conv_filter_size, kernel_size)
        self.bn1 = nn.BatchNorm2d(conv_filter_size)
        self.mpl1 = nn.MaxPool2d(pool_size)
        self.dropout1 = nn.Dropout(0.2)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)

        self.conv2 = nn.Conv2d(conv_filter_size, conv_filter_size * 2,  kernel_size)
        self.bn2 = nn.BatchNorm2d(conv_filter_size * 2)
        self.mpl2 = nn.MaxPool2d(pool_size)
        self.dropout2 = nn.Dropout(0.3)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)

        self.conv3 = nn.Conv2d(conv_filter_size * 2, conv_filter_size * 4, kernel_size)
        self.bn3 = nn.BatchNorm2d(conv_filter_size * 4)
        self.mpl3 = nn.MaxPool2d(pool_size)
        self.dropout3 = nn.Dropout(0.4)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.01)

        self.conv_out = nn.Conv2d(conv_filter_size * 4, 1024, kernel_size=3, stride=1, padding=1)

        self.bboxes_head = nn.Conv2d(1024, num_anchors * 4, kernel_size=1)
        self.confid_head = nn.Conv2d(1024, num_anchors, kernel_size=1)

        self.bbox_criterion = nn.SmoothL1Loss()
        self.conf_criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.leaky_relu1(self.bn1(self.conv1(x)))
        x = self.mpl1(x)
        x = self.dropout1(x)

        x = self.leaky_relu2(self.bn2(self.conv2(x)))
        x = self.mpl2(x)
        x = self.dropout2(x)

        x = self.leaky_relu3(self.bn3(self.conv3(x)))
        x = self.mpl3(x)
        x = self.dropout3(x)

        x = self.conv_out(x)

        bboxes_pred = self.bboxes_head(x).permute(0, 2, 3, 1).contiguous()
        confid_pred = self.confid_head(x).permute(0, 2, 3, 1).contiguous()

        return bboxes_pred, confid_pred

    @staticmethod
    def generate_anchors(feature_map, scales, ratios, stride):
        anchors = []
        for i in range(feature_map[0]):
            for j in range(feature_map[1]):
                cx, cy = j * stride, i * stride
                for scale in scales:
                    for ratio in ratios:
                        w = scale * (ratio ** 0.5)
                        h = scale / (ratio ** 0.5)
                        anchors.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

        return torch.tensor(anchors)

    @staticmethod
    def decode_preds(bboxes_pred, confid_pred, anchors, conf_tresh: float = 0.5, iou_trash: float = 0.4):
        confid_pred = torch.sigmoid(confid_pred)
        keep = confid_pred > conf_tresh

        filtered_bboxes = bboxes_pred[keep]
        filtered_scores = confid_pred[keep]

        indeces = torchvision.ops.nms(filtered_bboxes, filtered_scores, iou_threshold=iou_trash)
        return filtered_bboxes[indeces], filtered_scores[indeces]


class WordsRecognizeNetwork:
    def __init__(self, bbox_model: nn.Module, text_rec_model: str = 's', device: str = 'cuda'):
        self.device = device
        self.bbox_model = bbox_model.to(self.device)
        self.bbox_optimizer = optim.Adam(bbox_model.parameters(), lr=0.001)

    def train_bbox(self, dataloader: torch.utils.data.DataLoader, epochs: int=100):
        self.bbox_model.train()
        loss = None

        for epoch in range(epochs):
            epoch_loss = 0.0
            for image, label in dataloader:
                images = image.to(self.device)
                labels = label.to(self.device)

                self.bbox_optimizer.zero_grad()

                bboxes_pred, confid_pred = self.bbox_model(images)

                targets_bboxes = labels[..., :4]
                targets_confid = labels[..., 4]

                bbox_loss = self.bbox_model.bbox_criterion(bboxes_pred, targets_bboxes)
                confid_loss = self.bbox_model.confid_criterion(confid_pred, targets_confid)

                loss = bbox_loss + confid_loss

                loss.backward()

                self.bbox_optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    def test(self, images_labels: tuple):
        pass


def main():
    usags = ComfortableUsage()
    usags.initAll(download_dataset=False, rework=False)
    labels, images, _, _ = usags.preprocessAnnotation(items='test')
    dataset = CustomDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    network = BboxModel()
    nn = WordsRecognizeNetwork(network, device='cpu')
    nn.train_bbox(dataloader)

if __name__ == '__main__':
    main()