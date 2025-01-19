from imports import *

class Anchors:
    """ UPDATE TO BATCH PROCESS """
    def __init__(self, inp_img_size: Tuple[int, int]=(300, 300),
                 out_img_size: Tuple[int, int]=(18, 18)):
        self.inp_size = inp_img_size
        self.out_size = out_img_size

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
    def generate_anchors(feature_map_size: Tuple[int, int],
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

    @staticmethod
    def _compute_iou(anchors, gts):
        xa, ya, xb, yb = anchors[0]
        xg, yg, xg_max, yg_max, _ = gts[0]

        xi1 = max(xa, xg)
        yi1 = max(ya, yg)
        xi2 = min(xb, xg_max)
        yi2 = min(yb, yg_max)

        inter_w = torch.relu(xi2 - xi1 + 1)
        inter_h = torch.relu(yi2 - yi1 + 1)

        intersection = inter_w * inter_h
        area_a = (xb - xa + 1) * (yb - ya + 1)
        area_b = (xg_max - xg + 1) * (yg_max - yg + 1)

        union = area_a + area_b - intersection

        iou = intersection / union
        return iou

    def iou_calc(self, anchors, gt_boxes):
        labels = torch.full((anchors.size(0), ), -1, dtype=torch.float32)

        max_overlaps = torch.zeros(anchors.size(0), dtype=torch.float32)
        gt_argmax_overlaps = torch.zeros(anchors.size(0), dtype=torch.long)

        print(anchors.shape)

        for i, anchor in enumerate(anchors):
            max_iou = 0
            best_gt_idx = -1
            for gt in gt_boxes:
                for j, mini_gt in enumerate(gt):
                    if mini_gt[0] == mini_gt[2] == 0:
                        break
                    iou = self._compute_iou(anchor.unsqueeze(0), mini_gt.unsqueeze(0)).item()
                    if iou > max_iou:
                        max_iou = iou
                        best_gt_idx = j

            max_overlaps[i] = max_iou
            gt_argmax_overlaps[i] = best_gt_idx

            if max_iou >= .7:
                labels[i] = 1
            elif max_iou <= .3:
                labels[i] = 0

        return labels, max_overlaps, gt_argmax_overlaps

    @staticmethod
    def subsample(labels: torch.Tensor, batch_size=256, fg_fraction=0.4):
        # batch_size = map[0] * map[1] * len(sizes) * len(ratios)
        num_fg = int(fg_fraction * batch_size)
        fg_inds = torch.where(labels == 1)[0]

        if len(fg_inds) > num_fg:
            disable_inds = torch.randperm(len(fg_inds))[:len(fg_inds) - num_fg]
            labels[fg_inds[disable_inds]] = -1

        num_bg = batch_size - torch.sum(labels == 1)
        bg_inds = torch.where(labels == 0)[0]

        if len(bg_inds) > num_bg:
            disable_inds = torch.randperm(len(bg_inds))[:len(bg_inds) - num_bg]
            labels[bg_inds[disable_inds]] = -1

        return labels

    @staticmethod
    def prepare_batches(labels: torch.Tensor, map_size, anchors=9):
        batch_inds = torch.where(labels != -1)[0]
        batch_pos = (batch_inds // anchors).int()
        return batch_inds, batch_pos

    @staticmethod
    def create_tiles(batch_inds, feat_map, width):
        # metki (batch_size, 1, 1, 9)
        # reg (batch_size, 1, 1, 4 * 9)
        padded_map = torch.nn.functional.pad(feat_map, (1, 1, 1, 1), mode="constant")
        batch_tiles = []
        for ind in batch_inds:
            x = ind % width
            y = ind // width
            tile = padded_map[:, y:y+3, x:x+3]
            batch_tiles.append(tile)
        return torch.stack(batch_tiles)