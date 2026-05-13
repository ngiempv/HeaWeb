import os
import argparse
import pandas as pd
import xml.etree.ElementTree as ET

class AnnotationConverter:
    """
    Chuyển đổi annotations từ dataset-master (XML hoặc CSV) sang format chuẩn
    cho model detection (YOLO/Sparse-RCNN)
    """
    def __init__(self, annotations_dir, output_dir, classes):
        """
        :param annotations_dir: folder chứa file XML (Pascal VOC) hoặc CSV
        :param output_dir: folder lưu annotations convert
        :param classes: list tên class, ví dụ ["NEUTROPHIL", "LYMPHOCYTE", "MONOCYTE", "EOSINOPHIL"]
        """
        self.annotations_dir = annotations_dir
        self.output_dir = output_dir
        self.classes = classes
        os.makedirs(output_dir, exist_ok=True)

    def xml_to_csv(self, xml_file):
        """
        Chuyển 1 file Pascal VOC XML sang CSV format
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        records = []
        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in self.classes:
                continue
            cls_id = self.classes.index(cls)
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            records.append([cls_id, xmin, ymin, xmax, ymax])
        return records

    def convert_all_xml(self):
        """
        Convert tất cả file XML trong folder annotations_dir sang CSV
        """
        all_data = []
        for file_name in os.listdir(self.annotations_dir):
            if not file_name.endswith('.xml'):
                continue
            xml_path = os.path.join(self.annotations_dir, file_name)
            records = self.xml_to_csv(xml_path)
            img_name = file_name.replace('.xml', '.jpg')
            for rec in records:
                all_data.append([img_name] + rec)
        df = pd.DataFrame(all_data, columns=['image','class_id','xmin','ymin','xmax','ymax'])
        csv_file = os.path.join(self.output_dir, 'annotations_converted.csv')
        df.to_csv(csv_file, index=False)
        print(f"[INFO] Saved converted annotations to {csv_file}")
        return df

    def csv_to_yolo(self, csv_file, img_dir):
        """
        Chuyển CSV annotations sang format YOLO: class x_center y_center width height (normalized)
        :param csv_file: đường dẫn CSV
        :param img_dir: folder chứa ảnh để lấy kích thước
        """
        import cv2
        df = pd.read_csv(csv_file)
        for img_name in df['image'].unique():
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            df_img = df[df['image'] == img_name]
            lines = []
            for _, row in df_img.iterrows():
                x_center = ((row['xmin'] + row['xmax']) / 2) / w
                y_center = ((row['ymin'] + row['ymax']) / 2) / h
                width = (row['xmax'] - row['xmin']) / w
                height = (row['ymax'] - row['ymin']) / h
                lines.append(f"{row['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            txt_file = os.path.join(self.output_dir, img_name.replace('.jpg','.txt'))
            with open(txt_file, 'w') as f:
                f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description="Convert RBC XML annotations to CSV/YOLO labels")
    parser.add_argument(
        "--annotations_dir",
        default="dataset-master/dataset-master/Annotations",
        help="Folder chua file XML Pascal VOC",
    )
    parser.add_argument(
        "--output_dir",
        default="dataset-master/dataset-master/YOLO_labels",
        help="Folder luu annotation da convert",
    )
    parser.add_argument(
        "--img_dir",
        default="dataset-master/dataset-master/JPEGImages",
        help="Folder anh dung de lay kich thuoc khi convert YOLO txt",
    )
    parser.add_argument("--classes", default="RBC", help="Danh sach class cach nhau bang dau phay")
    parser.add_argument("--make_yolo_txt", action="store_true", help="Tao them file label .txt theo format YOLO")
    args = parser.parse_args()

    classes = [name.strip() for name in args.classes.split(",") if name.strip()]
    converter = AnnotationConverter(
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        classes=classes,
    )
    df = converter.convert_all_xml()
    csv_file = os.path.join(args.output_dir, "annotations_converted.csv")
    print(df["class_id"].value_counts().sort_index())

    if args.make_yolo_txt:
        converter.csv_to_yolo(csv_file, args.img_dir)
        print(f"[INFO] Saved YOLO txt labels to {args.output_dir}")


if __name__ == "__main__":
    main()
