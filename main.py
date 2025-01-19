import time
from imports import *
from models import *
from prework import *


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
            bboxes, images, translations, ids = self.fullPreprocess(dataset=self.dataset,
                                                                    target_size=target_size,
                                                                    split=items)

            return bboxes, images, translations, ids
        except Exception as e:
            print(f"{COLOR().get('red')}[!] Error: {e}{COLOR().get()}")
            return None


class TEST:
    def __init__(self):
        pass

    @staticmethod
    def test_iou():
        helper = Anchors((300, 300), (15, 15))

        # Генерация якорей
        anchors = helper.generate_anchors(
            feature_map_size=(15, 15),
            sizes=[32, 64],
            ratios=[1.0, 2.0, 0.5],
            stride=16
        )
        print(f"Всего якорей: {anchors.shape[0]}")
        print(f"Первые 5 якорей: {anchors[:5]}")

        # Пример ground truth (истинных коробок)
        gt_boxes = torch.tensor([
            [50, 50, 100, 100],  # Пример 1 (xmin, ymin, xmax, ymax)
            [120, 120, 180, 180]  # Пример 2
        ], dtype=torch.float32)
        print(f"Ground truth: {gt_boxes}")

        # Вычисление IoU и меток
        labels, max_overlaps, gt_argmax_overlaps = helper.iou_calc(anchors, gt_boxes)

        print("Метки для якорей:")
        print(labels)
        print("Максимальные перекрытия:")
        print(max_overlaps)
        print("Индексы лучших совпадений с gt:")
        print(gt_argmax_overlaps)

        # # Вывод информации о якорях, которые попадают в определенные категории
        # print("\nЯкоря с меткой 1 (IoU >= 0.7):")
        # for i, label in enumerate(labels):
        #     if label == 1:
        #         print(f"Якорь {i}: {anchors[i]}")
        #
        # print("\nЯкоря с меткой 0 (IoU <= 0.3):")
        # for i, label in enumerate(labels):
        #     if label == 0:
        #         print(f"Якорь {i}: {anchors[i]}")
        #
        # print("\nЯкоря с меткой -1 (0.3 < IoU < 0.7):")
        # for i, label in enumerate(labels):
        #     if label == -1:
        #         print(f"Якорь {i}: {anchors[i]}")

        # Проверка корректности
        assert torch.all(labels >= -1)  # Метки могут быть -1, 0 или 1
        assert torch.all(labels <= 1)  # Метки могут быть -1, 0 или 1

        print("Тест завершён успешно!")

    @staticmethod
    def anchors_generation():
        helper = Anchors((300, 300), (15, 15))
        w_stride, h_stride = helper.recalculating()
        print(f"Шаги: {w_stride}, {h_stride}")
        shifts = helper.calculate_points()
        print(f"Количество точек: {shifts.shape[0]}")
        print(shifts[:5])  # Посмотрим первые 5 точек
        anchors = helper.generate_anchors(
            feature_map_size=(15, 15),  # Сетка 15x15
            sizes=[32, 64, 128],  # Размеры якорей
            ratios=[1.0, 2.0, 0.5],  # Соотношения сторон
            stride=16  # Шаг между центрами
        )

        print(f"Количество якорей: {anchors.shape[0]}")
        print(anchors[:5])  # Первые 5 якорей

        anchors = helper.generate_anchors(
            feature_map_size=(15, 15),
            sizes=[64],
            ratios=[1.0],
            stride=32
        )
        filtered_anchors = helper.remove_outside_image(anchors)

        print(f"Изначальное количество якорей: {anchors.shape[0]}")
        print(f"Количество якорей внутри изображения: {filtered_anchors.shape[0]}")

        image = torch.zeros((300, 300, 3))  # Пустое изображение 300x300
        anchors = helper.generate_anchors(
            feature_map_size=(15, 15),
            sizes=[32],
            ratios=[1.0],
            stride=20
        )

        plt.imshow(image)
        for anchor in anchors:
            x_min, y_min, x_max, y_max = anchor
            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                color="red"
            )
            plt.gca().add_patch(rect)

        plt.show()


class HELP:
    def __init__(self, inp_size: Tuple[int, int], out_size: Tuple[int, int]):
        self.inp_size = inp_size
        self.out_size = out_size

    def recalculating(self) -> Tuple[float, float]:
        # Расчёт шагов
        w_stride = self.inp_size[0] / self.out_size[0]
        h_stride = self.inp_size[1] / self.out_size[1]
        return w_stride, h_stride

    def calculate_points(self) -> torch.Tensor:
        w_stride, h_stride = self.recalculating()

        # Генерация координат сетки
        shift_x = torch.arange(0, self.out_size[1]) * w_stride
        shift_y = torch.arange(0, self.out_size[0]) * h_stride

        # Создание сетки
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing='ij')

        # Объединение в формате (xmin, ymin, xmax, ymax)
        shifts = torch.stack(
            (shift_x.flatten(), shift_y.flatten(), shift_x.flatten(), shift_y.flatten()),
            dim=1
        )

        return shifts

    @staticmethod
    def generate_anchors(feature_map_size: List[int],
                         sizes: List[int],
                         ratios: List[float],
                         stride: int) -> torch.Tensor:
        anchors = []
        for h in range(feature_map_size[0]):
            for w in range(feature_map_size[1]):
                cx, cy = h * stride + stride / 2, w * stride + stride / 2
                for size in sizes:
                    area = torch.tensor(size ** 2, dtype=torch.float32)  # Преобразуем в Tensor
                    for ratio in ratios:
                        ratio = torch.tensor(ratio, dtype=torch.float32)  # Преобразуем в Tensor
                        w_1 = torch.sqrt(area / ratio)  # Теперь всё корректно
                        h_1 = w_1 * ratio
                        x_min, y_min = cx - w_1 / 2, cy - h_1 / 2
                        x_max, y_max = cx + w_1 / 2, cy + h_1 / 2

                        anchors.append([x_min, y_min, x_max, y_max])
        return torch.tensor(anchors, dtype=torch.float32)

    def remove_outside_image(self, anchors, border=0):
        inds_inside = torch.where(
            (anchors[:, 0] >= -border) &
            (anchors[:, 1] >= -border) &
            (anchors[:, 2] < self.inp_size[0] + border) &
            (anchors[:, 3] < self.inp_size[1] + border)
        )

        return anchors[inds_inside]


class WordsRecognizeNetwork:
    def __init__(self, cnn_model: nn.Module, rpn_model: nn.Module, device: str = 'cuda', roi_pool=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device=='cuda' else "cpu")

        self.cnn_model = cnn_model.to(self.device)
        self.cnn_optimizer = optim.Adam(self.cnn_model.parameters(), lr=0.001)

        self.rpn_model = rpn_model.to(self.device)
        self.rpn_optimizer = optim.Adam(self.rpn_model.parameters(), lr=0.001)
        # Параметры якорей
        self.scales = [32, 64, 128]
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.stride = 16

        self.roi_pool_layer = roi_pool

    def train_bbox(self, dataloader, epochs=100):
        self.cnn_model.train()
        self.rpn_model.train()

        class_loss_fn = nn.BCELoss()
        reg_loss_fn = nn.SmoothL1Loss()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for images, gt_boxes in tqdm(dataloader):
                images = images.to(self.device)
                gt_boxes = [gt.to(self.device) for gt in gt_boxes]

                self.cnn_optimizer.zero_grad()
                self.rpn_optimizer.zero_grad()

                feature_map = self.cnn_model(images)
                print(feature_map.shape)

                anchors = Anchors().generate_anchors(
                    feature_map_size=(18, 18),
                    sizes=self.scales,
                    ratios=self.aspect_ratios,
                    stride=self.stride
                ).to(self.device)

                anchors = Anchors().remove_outside_image(anchors)

                deltas_pred, scores_pred = self.rpn_model(feature_map)

                time_s = time.time()
                labels, max_overlaps, _ = Anchors().iou_calc(anchors, gt_boxes)
                print(labels.shape, max_overlaps.shape)

                print(labels[:2], max_overlaps[:2], sep='\n')

                # # Привести метки к типу float и нужному размеру
                # labels = labels.float().view(scores_pred.size())
                #
                # labels = Anchors().subsample(labels, batch_size=256)
                # time_e = time.time()

                # rois = ...
                # pooled_deature = self.roi_pool_layer(feature_map, rois)

                # Расчет потерь
                cls_loss = class_loss_fn(scores_pred.squeeze(), labels)
                reg_loss = reg_loss_fn(deltas_pred.squeeze(), max_overlaps)

                total_loss = cls_loss + reg_loss

                total_loss.backward()

                self.cnn_optimizer.step()
                self.rpn_optimizer.step()

                epoch_loss += total_loss.item()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")


def main():
    usags = ComfortableUsage()
    usags.initAll(download_dataset=False, rework=False)
    labels, images, _, _ = usags.preprocessAnnotation(items='test',
                                                      target_size=(300, 300))
    dataset = CustomDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    network1 = ConvolutionNN()
    network2 = RPN(1024)
    roi = ROIAlignLayer((7,7))
    nn = WordsRecognizeNetwork(network1, network2, device='cpu')
    nn.train_bbox(dataloader)
    conv = ConvolutionNN()
    mapa = conv(images)
    # print(mapa.shape)
    # print(dataset.X.shape, len(dataset.y), dataset.y, sep='\n')
    # TEST().anchors_generation()
    # TEST().test_iou()

if __name__ == '__main__':
    main()