import fiftyone as fo
import fiftyone.core.fields as fof
import torch


class FiftyOneLogger:
    def __init__(
        self,
        dataset_name: str = "seagrass",
        n_classes: int = 2,
    ):
        self.dataset_name = dataset_name
        self.n_classes = n_classes

        if fo.dataset_exists(self.dataset_name):
            self.dataset = fo.load_dataset(self.dataset_name)
        else:
            self.dataset = fo.Dataset(name=self.dataset_name, overwrite=True, persistent=True)

        if "ground_truth" not in self.dataset.get_field_schema():
            self.dataset.add_sample_field(
                "ground_truth", fof.EmbeddedDocumentField, embedded_doc_type=fo.Classification
            )
        if "predictions" not in self.dataset.get_field_schema():
            self.dataset.add_sample_field("predictions", fof.EmbeddedDocumentField, embedded_doc_type=fo.Classification)

    def add_batch_samples(
        self, inputs: torch.Tensor, input_path: str, labels: torch.Tensor, predicted: torch.Tensor
    ) -> None:
        for i in range(len(inputs)):
            self.add_sample(image_path=input_path[i], label=labels[i], prediction=predicted[i])

    def add_image(self, image_path: str, attributes: dict) -> None:
        if self.dataset.match({"filepath": image_path}).count() > 0:
            return

        sample = fo.Sample(filepath=image_path)

        # TODO make this safer
        for key, att in attributes.items():
            sample[key] = att

        self.dataset.add_sample(sample, expand_schema=True)
        self.dataset.save()

    def add_sample(self, image_path: str, label: torch.Tensor, prediction: torch.Tensor) -> None:
        if self.dataset.match({"filepath": image_path}).count() > 0:
            return

        sample = fo.Sample(filepath=image_path)

        label_str = "seagrass" if label == 0 else "background"
        prediction_str = "seagrass" if prediction == 0 else "background"

        sample["ground_truth"] = fo.Classification(label=label_str)
        sample["predictions"] = fo.Classification(label=prediction_str)

        self.dataset.add_sample(sample, expand_schema=True)
        self.dataset.save()

    def evaluate(self) -> None:
        results = self.dataset.evaluate_classifications("predictions", gt_field="ground_truth")
        results.print_report()

    def visualize(self) -> None:
        session = fo.launch_app(self.dataset, port=5154)
        session.wait()
        return session


if __name__ == "__main__":
    fiftyone_logger = FiftyOneLogger()
    fiftyone_logger.evaluate()
    fiftyone_logger.visualize()
